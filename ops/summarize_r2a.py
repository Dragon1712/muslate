"""Generate a compact Chinese research report from audited R2-A artifacts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def mean(values):return sum(values)/len(values)


def quantile(values,p):
    x=sorted(values);index=(len(x)-1)*p;left=int(index);right=min(left+1,len(x)-1)
    return x[left]+(x[right]-x[left])*(index-left)


def fmt(value):return f'{value:.3f}'


def generate(root,report):
    root=Path(root);report=Path(report)
    verified=json.loads((root/'verified_manifest.json').read_text())
    if verified['status']!='verified':raise ValueError('supplementary identity verification required')
    for name,sha in verified['input_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise ValueError(f'verified input changed:{name}')
    audit=json.loads((root/'audit.json').read_text());contract=json.loads((root/'protocol.json').read_text())
    if (audit['complete_cells']!=80 or audit['missing'] or audit['extra'] or audit['missing_probes']
        or audit['extra_probes'] or audit['probe_states']!=80 or audit['queue_failed'] or not audit['tiny_complete']):
        raise ValueError('complete audited80cell/80probe-state result required')
    cells=[json.loads(p.read_text()) for p in sorted((root/'cells').glob('*.json'))]
    for p in (root/'cells').glob('*.json'):
        if hashlib.sha256(p.read_bytes()).hexdigest()!=audit['files'][p.name]:raise ValueError('cell changed after audit')
    groups=defaultdict(list)
    for row in cells:groups[row['condition'],row['proposal'],row['depth']].append(row)
    summary=[]
    for (condition,proposal,depth),rows in sorted(groups.items()):
        records=[r for row in rows for r in row['decisions']]
        costs={k:mean([r['cost'][k] for r in records]) for k in ('reward_only_calls','transition_calls','score_queries','proposal_candidates')}
        lat=[r['latency_ms'] for r in records];enc=[r['encoding_ms'] for r in records]
        summary.append(dict(condition=condition,proposal=proposal,depth=depth,return_mean=mean([r['return_mean'] for r in rows]),
                            tail_mean=mean([r['tail_mean'] for r in rows]),cost_per_decision=costs,
                            latency_ms=dict(mean=mean(lat),p50=quantile(lat,.5),p95=quantile(lat,.95)),
                            encoding_ms_mean=mean(enc),
                            selected_sequence_overestimation_mean=mean([r['selected_sequence_overestimation'] for r in records])))
    for row in summary:
        base=next(r for r in summary if (r['condition'],r['proposal'],r['depth'])==(row['condition'],row['proposal'],0))
        row['ratio_to_depth0']=dict(transition_count=row['cost_per_decision']['transition_calls']/base['cost_per_decision']['transition_calls'],
                                    score_count=row['cost_per_decision']['score_queries']/base['cost_per_decision']['score_queries'],
                                    proposal_count=row['cost_per_decision']['proposal_candidates']/base['cost_per_decision']['proposal_candidates'],
                                    latency_mean=row['latency_ms']['mean']/base['latency_ms']['mean'])
    probes=defaultdict(list)
    for path in sorted((root/'probes').glob('*.json')):
        payload=json.loads(path.read_text());condition=payload['identity']['training_identity']['config']['arm']['condition']
        proposal=path.stem.split('_',2)[2];probes[condition,proposal]+=payload['rows']
    probe_summary=[]
    for (condition,proposal),rows in sorted(probes.items()):
        for beam in (3,8,16):
            selected=[next(s for s in r['searches'] if s['depth']==1 and s['beam']==beam) for r in rows]
            probe_summary.append(dict(condition=condition,proposal=proposal,beam=beam,n=len(selected),
                                      reference_best_pruned_fraction=mean([int(s['reference_best_pruned']) for s in selected]),
                                      learned_candidate_scope_regret_mean=mean([s['candidate_scope_regret'] for s in selected]),
                                      learned_candidate_scope_regret_max=max(s['candidate_scope_regret'] for s in selected),
                                      search_latency_ms_mean=mean([s['latency_ms'] for s in selected]),
                                      cache_generation_latency_ms_mean=mean([r['cache_generation_latency_ms'] for r in rows])))
    source_sha=hashlib.sha256(json.dumps(contract['source'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    checkpoint_ids={f"{r['condition']}_s{r['seed']}":r['identity']['checkpoint_sha256'] for r in cells}
    analysis=dict(scope='exploratory fixed nominal B128 no equal-compute claim',group_summary=summary,
                  probe_summary=probe_summary,paired_depth_effects=audit['paired_depth_effects'],
                  source_manifest_sha256=source_sha,protocol_sha256=contract['protocol_sha256'],checkpoint_sha256=checkpoint_ids)
    (root/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    lines=['# R2-A：完整一步规划、深度与剪枝实验结果','',
           '2026-10-09。本轮已实际复用10个既有检查点完成80/80共享前缀配置、20/20探针文件（80个固定状态）及4个小环境穷举状态。'+
           f"独立重放{audit['replay_decisions']:,}个闭环执行决策，奖励、已选择动作序列的真值回报和折扣episode回报一致；补充核验再生成80个缓存根候选池，绑定配置标签、检查点、共享轨迹和全部报告输入SHA。无缺失配置、失败job或缺失探针。",
           '', '主要发现：12个depth相对0的主95%区间都跨零，现有B128/beam3没有稳定的深度增益证据；这不能证明深度收益为零。实际计算与延迟随深度明显增加，下一步先检验强一步基线、等总计算和reward/transition诊断，暂不实施Adaptive-H。',
           '',f'依据：[冻结协议](D:/Coding/next_codex/docs/research/2026-10-09-r2a-protocol.md)、[原始审计](D:/Coding/next_codex/results/r2a_20261009/full/audit.json)、[汇总数据](D:/Coding/next_codex/results/r2a_20261009/full/analysis.json)。',
           '', '这是探索性结果：固定L0/L1现有权重与16个既有用户，未重训、未调参、未使用40M确认轨迹区域。'+
           '每训练seed使用30M新轨迹偏移与共享5步行为前缀，后15步闭环执行；depth=0使用同一proposal和完整即时奖励目标，depth=1/2/3对应最多2/3/4个动作，均按剩余episode长度截断。B128是固定提案设置，实际计算量不同。',
           '', '## 配对深度效应','',
           '下表为depth减同模型/同proposal的depth0；先计算每seed的16用户平均差，再对5个独立训练seed计算主Student-t95%区间。补充区间按预先声明的seed→配对用户分层bootstrap10,000次计算；两者均报告，不选择较有利结果。',
           '', '| 模型 | Proposal | 深度 | 均值差 | 主95%区间 | 补充分层bootstrap95% | 五seed差值 |',
           '|---|---|---:|---:|---|---|---|']
    for e in audit['paired_depth_effects']:
        ci=e['ci95'];boot=e['supplementary_hierarchical_bootstrap_ci95']
        lines.append(f"| {e['condition']} | {e['proposal']} | {e['depth']} | {fmt(e['mean'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | [{fmt(boot[0])}, {fmt(boot[1])}] | "+', '.join(fmt(x) for x in e['seed_differences'])+' |')
    lines+=['','这些决策记录不是独立统计复制；独立训练复制仍只有5个。没有多重比较校正或确认性成功判据。',
            '', '## 回报与实际成本','',
            '表中回报包含共享前缀；配对效应中前缀抵消。操作数量是每决策平均：reward为独立reward-only样本，transition为联合奖励/转移样本，score为物品打分查询，proposal为实际生成/细化候选尝试。SID候选生成内部reward与score已计入。proposal合并基础生成与细化尝试，本轮没有单独的细化计数，不能从合并值恢复两者；这比协议中单独列出细化计数的文字范围少了一项，明确披露。',
            '', '| 模型 | Proposal | 深度 | 平均回报 | reward | transition | score | proposal | 延迟均值/p50/p95 ms | 选定序列高估均值 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in summary:
        c=r['cost_per_decision'];lat=r['latency_ms']
        lines.append(f"| {r['condition']} | {r['proposal']} | {r['depth']} | {fmt(r['return_mean'])} | {fmt(c['reward_only_calls'])} | {fmt(c['transition_calls'])} | {fmt(c['score_queries'])} | {fmt(c['proposal_candidates'])} | {fmt(lat['mean'])}/{fmt(lat['p50'])}/{fmt(lat['p95'])} | {fmt(r['selected_sequence_overestimation_mean'])} |")
    lines+=['','延迟包含提案、细化和模型查询，并在GPU同步后测量；每GPU仅有本seed的一个进程串行决策，seed间在5张GPU并行。'+
            '未进行独立控制的最终部署延迟复测，也未校准等总计算预算曲线。belief编码独立记录于原始数据与analysis.json，未并入planning延迟。',
            '', '“选定序列高估”比较的是规划器保存的同一个open-loop动作序列的预测折扣回报与模拟器真值回报。'+
            '后续闭环会重新编码并可能改变动作，因此该量不能解释为整个闭环策略的价值预测误差；不同深度的序列长度也不同。真值只在动作选择之后评价，未进入learned搜索。',
            '', '联合transition样本数与score查询数相对depth0的倍率，在depth0/1/2/3分别为1、3.8、11.6、33.2；SID内部reward和proposal数量随状态稍有变化。不同操作类型未经成本校准，不能简单相加或称为同一种计算单位。平均planning延迟倍率如下；跨GPU同时运行共享主机资源，尾延迟的突增也保留在原始数据中。',
            '', '| 模型 | Proposal | depth0 | depth1 | depth2 | depth3 |',
            '|---|---|---:|---:|---:|---:|']
    for condition in ('L0','L1'):
        for name in ('flat_uniform','sid_hier'):
            rows=sorted([r for r in summary if (r['condition'],r['proposal'])==(condition,name)],key=lambda r:r['depth'])
            lines.append(f'| {condition} | {name} | '+' | '.join(f"{r['ratio_to_depth0']['latency_mean']:.2f}×" for r in rows)+' |')
    lines+=['', '## 固定状态的剪枝诊断','',
            '每检查点/proposal使用前4用户的公共前缀终点。缓存并逐字节复用根候选池；每根动作的完整后继候选池由ordered祖先动作寻址，独立于depth、beam和访问顺序。'+
            'depth1参考评价全部根候选及各自全部后继候选；beam3/8/16是这个learned候选范围参考的精确子集。它不是200-item全局Oracle。',
            '', '| 模型 | Proposal | beam | 状态数 | 参考最好根被剪比例 | 模型候选范围regret均值/最大 | 搜索延迟ms | 缓存生成ms |',
            '|---|---|---:|---:|---:|---|---:|---:|']
    for r in probe_summary:
        lines.append(f"| {r['condition']} | {r['proposal']} | {r['beam']} | {r['n']} | {100*r['reference_best_pruned_fraction']:.1f}% | {fmt(r['learned_candidate_scope_regret_mean'])}/{fmt(r['learned_candidate_scope_regret_max'])} | {fmt(r['search_latency_ms_mean'])} | {fmt(r['cache_generation_latency_ms_mean'])} |")
    lines+=['','regret是相同模型、相同候选范围内的预测价值损失，不能直接当作真值策略收益。参考最好根若有同价值并列，剪掉其中一个也可能产生零regret。缓存生成只做一次，成本单独报告；closed-loop每决策实际生成候选，并完整计费。4用户探针不代表全部16用户。',
            '', '## 小环境穷举Oracle','',
            '4-item、K2、H3环境枚举12种合法有序slate及全部12³动作序列；真状态/真动力学只用于机制诊断。完整beam12与独立穷举一致。',
            '', '| 用户 | 精确Oracle回报 | myopic回报 | beam3回报 | beam8回报 | Oracle−myopic |',
            '|---:|---:|---:|---:|---:|---:|']
    tiny=json.loads((root/'tiny_oracle.json').read_text())
    for r in tiny['rows']:
        a,b,c,_=r['trials'];lines.append(f"| {r['user_id']} | {fmt(r['exact_value'])} | {fmt(a['return_value'])} | {fmt(b['return_value'])} | {fmt(c['return_value'])} | {fmt(r['exact_value']-a['return_value'])} |")
    lines+=['','小环境隐藏状态Oracle与learned belief的信息条件不同；小目录、短时域和4个状态也不能证明200-item环境具有相同长期机会。',
            '', '## 后续投入判断','']
    effects=audit['paired_depth_effects'];positive=[e for e in effects if e['ci95'][0]>0];l1positive=[e for e in positive if e['condition']=='L1']
    if l1positive:
        lines.append('L1中存在主区间完全为正的固定设置深度效应（'+', '.join(f"{e['proposal']}/depth{e['depth']}" for e in l1positive)+'）。这支持继续检查收益状态异质性与搜索改进；由于成本未匹配，尚不足以进入Adaptive-H优越性的正式主张。')
    else:
        lines.append('本轮L1没有主95%区间完全为正的深度效应。这表示尚缺稳定增益证据，不能证明深度收益为零。当前不进入Adaptive-H或更大型planner升级；先检验搜索剪枝、候选覆盖与模型多步误差，再决定是否增加训练复杂度。')
    l1sid3=next(e for e in effects if e['condition']=='L1' and e['proposal']=='sid_hier' and e['depth']==3)
    hier0=next(r for r in summary if (r['condition'],r['proposal'],r['depth'])==('L1','sid_hier',0))
    hier3=next(r for r in summary if (r['condition'],r['proposal'],r['depth'])==('L1','sid_hier',3))
    flat0=next(r for r in summary if (r['condition'],r['proposal'],r['depth'])==('L1','flat_uniform',0))
    lines.append(f"L1 sid_hier的depth3−0只有{fmt(l1sid3['mean'])}，主区间[{fmt(l1sid3['ci95'][0])}, {fmt(l1sid3['ci95'][1])}]；平均planning延迟从{fmt(hier0['latency_ms']['mean'])}ms升到{fmt(hier3['latency_ms']['mean'])}ms。这条未等计算曲线尚不支持增加深度的成本收益主张。")
    l1flatprobe=next(r for r in probe_summary if (r['condition'],r['proposal'],r['beam'])==('L1','flat_uniform',3))
    l1hierprobe=next(r for r in probe_summary if (r['condition'],r['proposal'],r['beam'])==('L1','sid_hier',3))
    lines.append(f"固定状态中L1 flat_uniform beam3仅剪掉{100*l1flatprobe['reference_best_pruned_fraction']:.0f}%的模型参考最好根，平均模型regret为{fmt(l1flatprobe['learned_candidate_scope_regret_mean'])}；因此不能把它的深度效应不足全归于beam3。L1 sid_hier对应剪掉{100*l1hierprobe['reference_best_pruned_fraction']:.0f}%，值得进一步检查剪枝，但这仍不是放宽beam会改善真实闭环回报的证据。")
    lines.append(f"L1层次proposal在depth0就比flat_uniform的平均回报高{fmt(hier0['return_mean']-flat0['return_mean'])}（事后描述，不作新的确认性检验），支持先检验列表内组合优化解释。优先增加具有同等验证资源的flat_coordinate_refine/分组对照，以及总操作预算校准；并补1/3/5步open-loop/re-encoded诊断。此次结果没有完成反馈历史模型消融、全episode稳健性或64用户确认，这些明确留待后续。")
    lines+=['','## 产物与身份','',f"冻结协议SHA256：`{contract['protocol_sha256']}`；源代码清单规范JSON的SHA256：`{source_sha}`。10个检查点SHA及原训练身份保存在每cell与analysis.json。",
            '', '运行目录：[完整结果](D:/Coding/next_codex/results/r2a_20261009/full/protocol.json)、[身份核验与输入哈希](D:/Coding/next_codex/results/r2a_20261009/full/verified_manifest.json)。烟雾测试最终版本完成16配置/240决策；首次烟雾版本独立保存，未混入完整矩阵。E1冻结代码、协议与结果保留。',
            '', '执行器采取整轮重启策略：不支持已有目录的部分resume，任何失败均保留failure/queue记录并使用全新目录重跑。补充核验器允许checkpoint-root覆盖服务器绝对路径，同时严格核对原记录SHA，方便跨机器复核。']
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    delivered=dict(report_path=str(report.resolve()),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                   analysis_sha256=hashlib.sha256((root/'analysis.json').read_bytes()).hexdigest(),
                   summary_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   verified_manifest_sha256=hashlib.sha256((root/'verified_manifest.json').read_bytes()).hexdigest())
    (root/'deliverables_manifest.json').write_text(json.dumps(delivered,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(report),groups=len(summary),probe_groups=len(probe_summary),positive_seed_t_effects=len(positive)),ensure_ascii=False))
    return analysis


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('report');a=p.parse_args();generate(a.root,a.report)
