"""Validation-only model selection and seed->user paired bootstrap for E1."""
from pathlib import Path
import argparse
import json
import hashlib
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_logged_pilot import atomic_json
from muslate.eval.integrity import validate_protocols,checked_returns

SELECTION_RULE='validation common-prefix sid_hier mean return, lower weight tie break'


def normalized_contract(settings):
    contract=json.loads(json.dumps(settings))
    contract.pop('arm')
    for key in ('seed','w_score','score_supervision'):contract['model'].pop(key)
    contract['log'].pop('seed')
    for key in ('validation_seed','test_seed','validation_prefix_seed','test_prefix_seed','planner_seed','device'):
        contract.pop(key,None)
    return contract


def validate_results(results,splits=('validation','test')):
    if not results:raise ValueError('no completed results')
    source=results[0]['identity']['source']
    contract=normalized_contract(results[0]['identity']['config'])
    seen=set();seed_configs={};case_configs={}
    for result in results:
        settings=result['identity']['config'];arm=settings['arm']
        validate_protocols(result['protocols'],settings,splits)
        condition,seed,weight=arm['condition'],arm['seed'],arm['w_score']
        key=(condition,seed,weight)
        if key in seen:raise ValueError('duplicate result')
        seen.add(key)
        if result['identity']['source']!=source or normalized_contract(settings)!=contract:
            raise ValueError('incompatible experiment source/full configuration')
        expected_mode={'L0':'none','L1':'exposed','U1':'oracle_full'}.get(condition)
        if (expected_mode is None or settings['model']['score_supervision']!=expected_mode
                or settings['model']['oracle_state'] or settings['model']['seed']!=seed
                or settings['model']['w_score']!=weight):raise ValueError('mislabeled information condition')
        per_seed={k:settings[k] for k in ('log','validation_seed','test_seed',
                                          'validation_prefix_seed','test_prefix_seed','planner_seed') if k in settings}
        if seed in seed_configs and per_seed!=seed_configs[seed]:raise ValueError('unpaired training/evaluation seeds')
        seed_configs[seed]=per_seed
        for split in splits:
            for protocol in ('full_episode','common_prefix'):
                data=result['protocols'][split][protocol]
                user_ids=data['user_ids'];n=settings['users']
                if len(user_ids)!=n or len(set(user_ids))!=n:raise ValueError('invalid user identities')
                case={k:data[k] for k in ('user_ids','episode_seeds','prefix_returns')}
                if any(len(v)!=n for v in case.values()):raise ValueError('case length mismatch')
                case_key=(seed,split,protocol)
                if case_key in case_configs and case!=case_configs[case_key]:raise ValueError('unpaired validation/test cases')
                case_configs[case_key]=case
                for row in data['rows'].values():
                    if len(row['returns'])!=n or not np.isfinite(row['returns']).all():
                        raise ValueError('invalid return vector')
    return source,contract


def paired_interval(left, right, draws=10000, seed=20261008):
    left,right=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    if left.shape!=right.shape or left.ndim!=2 or min(left.shape)<1:
        raise ValueError('paired arrays must share [independent seeds, users] shape')
    delta=left-right
    if not np.isfinite(delta).all():raise ValueError('nonfinite returns')
    rng=np.random.default_rng(seed);n_seeds,n_users=delta.shape
    seed_index=rng.integers(n_seeds,size=(draws,n_seeds))
    user_index=rng.integers(n_users,size=(draws,n_seeds,n_users))
    replicates=delta[seed_index[:,:,None],user_index].mean(axis=(1,2))
    return dict(mean=float(delta.mean()),ci95=np.quantile(replicates,[.025,.975]).tolist(),
                seed_means=delta.mean(axis=1).tolist(),n_seeds=n_seeds,n_users=n_users,
                note='exploratory three-seed interval' if n_seeds<5 else 'hierarchical paired interval')


def select_weights(results, seeds=(1,2,3)):
    selected={'L0':0}
    for condition in ('L1','U1'):
        values={}
        for result in results:
            arm=result['identity']['config']['arm']
            if arm['condition']!=condition or arm['seed'] not in seeds:continue
            weight=arm['w_score'];values.setdefault(weight,{})
            if arm['seed'] in values[weight]:raise ValueError('duplicate arm')
            values[weight][arm['seed']]=float(checked_returns(result['protocols']['validation']['common_prefix']['rows']['sid_hier']).mean())
        if set(values)!={1,10,100} or any(set(v)!=set(seeds) for v in values.values()):
            raise ValueError(f'incomplete validation matrix for {condition}')
        selected[condition]=min(values,key=lambda w:(-np.mean(list(values[w].values())),w))
    return selected


def build_summary(results, seeds=(1,2,3), frozen_selection=None,frozen_selection_sha256=None,validation_root=None):
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError('seeds must be unique nonnegative integer repeat IDs')
    source,contract=validate_results(results)
    if contract.get('protocol_version',1)>=2:
        if frozen_selection is None or not isinstance(frozen_selection_sha256,str) or len(frozen_selection_sha256)!=64:
            raise ValueError('stage1 report requires verified frozen selection file hash')
        from scripts.reevaluate_stage1 import validate_frozen_selection
        validate_frozen_selection(frozen_selection,source,validation_root)
        if any(r.get('metadata',{}).get('selection_sha256')!=frozen_selection_sha256 for r in results):
            raise ValueError('test result is bound to another frozen selection file')
    if frozen_selection is not None:
        if (frozen_selection.get('source')!=source or frozen_selection.get('contract')!=contract
                or frozen_selection.get('selection_rule')!=SELECTION_RULE):
            raise ValueError('frozen selection lacks compatible source/protocol provenance')
        selection=frozen_selection['selection']
    else: selection=select_weights(results)
    if (set(selection)!={'L0','L1','U1'} or selection['L0']!=0
            or any(selection[c] not in (1,10,100) for c in ('L1','U1'))):
        raise ValueError('invalid frozen weights')
    grouped={}
    for result in results:
        arm=result['identity']['config']['arm']
        if arm['seed'] not in seeds or arm['w_score']!=selection[arm['condition']]:continue
        key=(arm['condition'],arm['seed'])
        if key in grouped:raise ValueError('duplicate selected result')
        grouped[key]=result
    if set(grouped)!={(c,s) for c in selection for s in seeds}:
        raise ValueError('selected model matrix incomplete')
    planners=('reactive','flat_uniform','flat_policy','sid_flat','sid_hier')
    protocols={}
    for protocol in ('full_episode','common_prefix'):
        for seed in seeds:
            reference=grouped[('L0',seed)]['protocols']['test'][protocol]
            for condition in selection:
                other=grouped[(condition,seed)]['protocols']['test'][protocol]
                for key in ('episode_seeds','user_ids','prefix_returns'):
                    if reference[key]!=other[key]:raise ValueError(f'unpaired {key}')
        arrays={}
        rows={};comparisons={}
        for condition in selection:
            rows[condition]={}
            for planner in planners:
                entries=[grouped[(condition,s)]['protocols']['test'][protocol]['rows'][planner] for s in seeds]
                array=np.asarray([e['returns'] for e in entries]);arrays[(condition,planner)]=array
                rows[condition][planner]=dict(**paired_interval(array,np.zeros_like(array)),
                                              latency_ms=float(np.mean([e['latency_ms'] for e in entries])),
                                              slate_evals_per_decision=float(np.mean([e['slate_evals_per_decision'] for e in entries])),
                                              score_calls_per_decision=float(np.mean([e['score_calls']/e['n_decisions'] for e in entries])),
                                              closed_loop_spearman=float(np.mean([e['closed_loop_spearman'] for e in entries])))
            for baseline in ('flat_uniform','flat_policy','sid_flat'):
                comparisons[f'{condition}:sid_hier-{baseline}']=paired_interval(arrays[(condition,'sid_hier')],arrays[(condition,baseline)])
        for left,right in (('L1','L0'),('U1','L1')):
            for planner in planners:
                comparisons[f'{left}-{right}:{planner}']=paired_interval(arrays[(left,planner)],arrays[(right,planner)])
        protocols[protocol]=dict(rows=rows,comparisons=comparisons)
    diagnostics={c:dict(logged_spearman=float(np.mean([grouped[(c,s)]['diagnostics']['test']['spearman_mean'] for s in seeds])),
                        reward_mae=float(np.mean([grouped[(c,s)]['diagnostics']['test']['reward_mae'] for s in seeds])),
                        train_seconds=float(np.mean([grouped[(c,s)]['metadata']['train_seconds'] for s in seeds]))) for c in selection}
    return dict(selection=selection,seeds=list(seeds),n_complete_results=len(results),diagnostics=diagnostics,
                selection_sha256=frozen_selection_sha256,
                protocols=protocols,selection_rule=SELECTION_RULE,
                bootstrap=dict(draws=10000,seed=20261008,unit='seed then paired user'),
                source=source,contract=contract,limitations=['synthetic continuous served-item feedback','fixed catalog/shared user population',
                                             'nominal proposal budget; no hard total-operation budget','shared GPU latency is descriptive'])


def markdown_report(summary):
    lines=['# E1 实验结果', '', f"独立 seeds：{summary['seeds']}；验证集选定权重：{summary['selection']}。",
           '权重仅由验证集公共前缀 sid_hier 平均回报选择；下表为冻结选择后的测试集结果。',
           '区间按 seed→用户进行层次配对 bootstrap；三 seed 结果仅作探索。','']
    for protocol,data in summary['protocols'].items():
        lines += [f'## {protocol}','', '| 条件 | reactive | flat_uniform | flat_policy | sid_flat | sid_hier |',
                  '|---|---:|---:|---:|---:|---:|']
        for condition,rows in data['rows'].items():
            lines.append('| '+condition+' | '+' | '.join(f'{r["mean"]:.4f}' for r in rows.values())+' |')
        lines += ['', '| 配对差值 | 均值 | 95% CI |','|---|---:|---|']
        for name,result in data['comparisons'].items():
            if name.startswith('L1:') or name.endswith(':sid_hier'):
                lo,hi=result['ci95'];lines.append(f'| {name} | {result["mean"]:.4f} | [{lo:.4f}, {hi:.4f}] |')
        lines += ['']
    lines += ['## 诊断','','| 条件 | logged Spearman | reward MAE | 训练秒数 |','|---|---:|---:|---:|']
    for c,d in summary['diagnostics'].items():
        lines.append(f'| {c} | {d["logged_spearman"]:.4f} | {d["reward_mae"]:.4f} | {d["train_seconds"]:.1f} |')
    lines += ['', '预算与延迟的完整数值见同名 JSON。实际调用不同的比较不能称为严格等预算比较。',
              '本轮是连续合成反馈、固定目录和用户群的受控实验；真实点击、跨目录/用户泛化及硬预算验证仍待完成。']
    return '\n'.join(lines)+'\n'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',required=True);ap.add_argument('--report',required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]);ap.add_argument('--frozen-selection')
    ap.add_argument('--validation-root',help='optional local root for downloaded pinned validation files')
    args=ap.parse_args();results=[]
    for p in sorted(Path(args.out_dir).glob('*/result.json')):
        result=json.loads(p.read_text(encoding='utf-8'))
        if result['status']!='complete':raise ValueError(f'incomplete result {p}')
        results.append(result)
    selection=None
    selection_hash=None
    if args.frozen_selection:
        raw=Path(args.frozen_selection).read_bytes();selection=json.loads(raw)
        selection_hash=hashlib.sha256(raw).hexdigest()
    summary=build_summary(results,tuple(args.seeds),selection,selection_hash,args.validation_root)
    report=Path(args.report);atomic_json(report.with_suffix('.json'),summary)
    with report.with_suffix('.md').open('x',encoding='utf-8') as stream:stream.write(markdown_report(summary))
    print(json.dumps(dict(selection=summary['selection'],diagnostics=summary['diagnostics']),ensure_ascii=False))


if __name__=='__main__':main()
