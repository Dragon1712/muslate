"""Generate Chinese R2-C comparisons exclusively from unchanged verified inputs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from ops.r2c_run import digest,atomic


def verify_report_inputs(root):
    root=Path(root);manifest=json.loads((root/'verified_manifest.json').read_text())
    if manifest['status']!='verified': raise ValueError('verified manifest required')
    for relative,sha in manifest['input_sha256'].items():
        if digest(root/relative)!=sha: raise ValueError(f'verified input changed:{relative}')
    return manifest


def paired_ci(values):
    a=np.asarray(values,dtype=float)
    if a.shape!=(5,): raise ValueError('five training seeds required for paired CI')
    mean=float(a.mean());half=2.7764451051977987*float(a.std(ddof=1))/np.sqrt(5)
    return dict(per_seed=a.tolist(),mean=mean,ci95=[float(mean-half),float(mean+half)],n_training_seeds=5)


def generate(root,output):
    root=Path(root);verified=verify_report_inputs(root);protocol=json.loads((root/'protocol.json').read_text())
    if protocol['smoke']: raise ValueError('smoke cannot generate full results report')
    rows=[json.loads((root/'models'/f'{label}.json').read_text()) for label in protocol['expected_models']]
    comparisons={};lines=['# R2-C：固定序列预测诊断', '',
        '结论仅针对固定均匀动作 bank 的预测误差；teacher forcing 使用模拟器提供的过去观测，不能部署为未知未来的前瞻策略。此比较同时涉及历史编码器、reward head 和 learned dynamics，不能识别单一 transition 模块的因果责任。', '',
        f'已审计 {verified["model_count"]} 个模型、{verified["case_count"]} 个初始状态；每状态 64 条长度 5 序列，horizon 1/3/5 共享嵌套前缀；每个 reached state 16 个共享 probe slate。Source commit `{protocol["source_commit"]}`。原始结果路径 `{root}`。', '',
        '独立重复单位是五个训练 seed。95% 区间是 seed 配对差值的 t 区间（df=4），用户与序列只参与模型内平均。下表 TF−OL 为负表示 teacher forcing 平均误差更小。探索性比较没有因果解释或规划收益保证。', '',
        '| 条件 | horizon | OL reward MAE | TF reward MAE | OL return MAE | TF return MAE | TF−OL return MAE [95% CI] |',
        '|---|---:|---:|---:|---:|---:|---|']
    for condition in ('L0','L1'):
        selected=sorted((r for r in rows if r['condition']==condition),key=lambda r:r['seed'])
        comparisons[condition]={}
        for h in ('1','3','5'):
            a=[r['aggregate']['horizons'][h] for r in selected]
            values={mode:{key:float(np.mean([x[mode][key] for x in a])) for key in ('reward_mae','return_mae','selected_overestimation')}
                    for mode in ('open_loop','teacher_forced')}
            delta={key:paired_ci([x['teacher_forced'][key]-x['open_loop'][key] for x in a])
                   for key in ('reward_mae','return_mae')}
            comparisons[condition][h]=dict(mean=values,teacher_forced_minus_open_loop=delta)
            c=delta['return_mae'];ol=values['open_loop'];tf=values['teacher_forced']
            lines.append(f'| {condition} | {h} | {ol["reward_mae"]:.6f} | {tf["reward_mae"]:.6f} | {ol["return_mae"]:.6f} | {tf["return_mae"]:.6f} | {c["mean"]:.6f} [{c["ci95"][0]:.6f}, {c["ci95"][1]:.6f}] |')
    lines+=['','## 五个 seed 配对原值','',
            '| 条件 | horizon | 指标 | seed 1–5 的 TF−OL | 均值 [95% CI] |','|---|---:|---|---|---|']
    for condition in ('L0','L1'):
        for h in ('1','3','5'):
            for metric in ('reward_mae','return_mae'):
                c=comparisons[condition][h]['teacher_forced_minus_open_loop'][metric]
                lines.append(f'| {condition} | {h} | {metric} | '+', '.join(f'{v:.6f}' for v in c['per_seed'])+f' | {c["mean"]:.6f} [{c["ci95"][0]:.6f}, {c["ci95"][1]:.6f}] |')
    lines+=['','## 每步 reward 与函数性 probe','',
            '| 条件 | 步 | OL reward MAE | TF reward MAE | probe OL MAE | probe TF MAE | probe 两模式绝对差 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for condition in ('L0','L1'):
        a=[r['aggregate'] for r in rows if r['condition']==condition]
        for t in range(5):
            values=[np.mean([x['reward_mae_by_step'][m][t] for x in a]) for m in ('open_loop','teacher_forced')]
            values+=[np.mean([x['probe'][k][t] for x in a]) for k in ('open_loop_mae_by_step','teacher_forced_mae_by_step','functional_difference_by_step')]
            lines.append(f'| {condition} | {t+1} | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['','## 状态内排序与同序列过估计','',
            '先在每状态对 64 条序列算排序相关，再跨用户与 seed 平均。并列为平均秩；恒定向量产生 undefined 并排除于相关均值。选择 index 总由 OL 决定，TF 评分使用同一个 index；下列过估计是预测回报减真回报，正值代表过估计。','',
            '| 条件 | horizon | 模式 | 相关均值 | 有效/undefined | 同序列过估计 |','|---|---:|---|---:|---|---:|']
    for condition in ('L0','L1'):
        for h in ('1','3','5'):
            for mode in ('open_loop','teacher_forced'):
                a=[r['aggregate']['horizons'][h][mode] for r in rows if r['condition']==condition]
                n=sum(x['rank_defined'] for x in a);u=sum(x['rank_undefined'] for x in a)
                rank=sum(x['rank_correlation_mean']*x['rank_defined'] for x in a if x['rank_correlation_mean'] is not None)/n if n else None
                ranktext='undefined' if rank is None else f'{rank:.6f}'
                lines.append(f'| {condition} | {h} | {mode} | {ranktext} | {n}/{u} | {np.mean([x["selected_overestimation"] for x in a]):.6f} |')
    lines+=['','## 解释边界与下一步','']
    for condition in ('L0','L1'):
        c=comparisons[condition]['5']['teacher_forced_minus_open_loop']['return_mae']
        text='误差下降' if c['mean']<0 else '误差上升' if c['mean']>0 else '平均误差相同'
        exclusion='该区间不跨零' if c['ci95'][0]>0 or c['ci95'][1]<0 else '该区间跨零或包含零'
        lines.append(f'{condition} 在 horizon 5 的 teacher forcing 相对 OL {text}，差值 {c["mean"]:.6f}，95% CI [{c["ci95"][0]:.6f}, {c["ci95"][1]:.6f}]；{exclusion}。这是固定 bank 上的探索性预测诊断，不能转换为可执行策略收益或 transition 独立贡献。')
        lines.append('')
    lines+=['Horizon 1 两模式使用同一前缀，逐 reward/probe 数值相等是 chronology sanity check。Horizon 3/5 的差值、排序和同序列过估计用于描述误差累积；probe 衡量相同 evaluator 状态上 latent 分支导致的 reward 函数偏离。', '',
            '后续应独立预注册 encoder、reward head、dynamics 的受控消融，并把新观测约束下可部署的重编码策略单独评价。本次不因结果改变 bank、指标或主要比较。', '',
            f'审计通过：checkpoint/condition/seed/arm 标签与 hash；source/protocol/bank hash；十模型完整性；全部 bank/模拟器真值重放；原始指标与聚合重算；同序列 selection；有限值与 {verified["undefined_rank_count"]} 个明确 undefined 相关；horizon 1 equality；GPU/命令记录。', '',
            '所有输入 hash 记录在 verified_manifest.json。原始奖励/预测/probe、初始状态和历史、reached-state hash 可独立重算，启动命令与运行时位于 protocol.json、各模型 JSON 和 seed 日志。']
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic(root/'comparisons.json',dict(source_commit=protocol['source_commit'],comparisons=comparisons,
          verified_manifest_sha256=digest(root/'verified_manifest.json'),report_sha256=digest(output)))
    return comparisons


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);args=p.parse_args();generate(args.root,args.output)
