"""Render the stage-one research report from verified numeric artifacts."""
from pathlib import Path
import json,numpy as np

def main():
    root=Path(__file__).resolve().parents[1];base=root/'results/e1_stage1_20261009';raw=base/'artifacts'
    s=json.loads((base/'e1_summary_local.json').read_text());assert s==json.loads((raw/'e1_summary.json').read_text())
    receipt=json.loads((raw/'frozen_selection_receipt.json').read_text());audit=json.loads((raw/'final_integrity_audit.json').read_text())
    assert audit['status']=='passed' and audit['n_replayed_decisions']==117600 and audit['max_reward_error']==0
    cost=json.loads((base/'figures/stage1_cost.json').read_text())['protocols']
    archive=json.loads((base/'local_archive_verification.json').read_text())
    for p in [base/'final_local_tests.json',raw/'final_remote_tests_verified.json']:
        t=json.loads(p.read_text());assert t['total_tests']==140 and all(x['exit_code']==0 for x in t['files'])
    def link(path,label=None):
        p=root/path;return f'[{label or p.name}]({p.as_posix()})'
    def interval(r):return f"{r['mean']:.4f} | [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]"
    primary=s['protocols']['common_prefix']['comparisons']['L1-L0:sid_hier']
    aux=s['protocols']['full_episode']['comparisons']['L1-L0:sid_hier']
    lines=['# MuSlate 第一阶段修复版 E1：结果、代码逻辑与验收','',
        f"2026-10-09。第一阶段修改1–5及完整E1重评已完成。预先指定的公共前缀主比较中，L1 − L0 的 SID hierarchical 折扣回报差为 **{primary['mean']:.4f}，95% CI [{primary['ci95'][0]:.4f}, {primary['ci95'][1]:.4f}]**；五个独立训练seed均为正。完整episode辅助比较为 {aux['mean']:.4f} [{aux['ci95'][0]:.4f}, {aux['ci95'][1]:.4f}]。",
        '', '这支持当前受控合成设置中已曝光逐项辅助监督的决策收益。它不构成真实点击、跨目录/新用户泛化、自适应算法有效性或严格等总预算效率的证明。旧受非法slate及超时域影响的结果继续保留为探索性资料。','',
        '## 1. 本轮范围与代码逻辑','',
        '保留原世界模型架构、损失及27个checkpoint参数，没有重新训练。先复用21个pilot checkpoint重新验证选参，再复用已存在的选中条件seeds4/5，完成15个模型的独立新测试。第二阶段反馈进入GRU、完整ensemble坐标修复、机制矩阵及Adaptive-H未实施。','',
        '模拟器用用户兴趣与类别饱和状态生成逐项价值、列表pairwise惩罚及下一状态。日志记录执行slate、整体reward与已曝光逐项反馈。世界模型GRU仅编码此前的slate和整体reward；本轮逐项反馈用于辅助损失，仍未直接进入GRU。item_scores与reward的unary部分共享参数，因此L0的item评分也接受整体reward损失的间接训练；不能将L0解释为完全未训练的提案头。当前checkpoint已有pairwise reward模块。','',
        '规划由历史belief出发，用flat或SID proposal生成合法候选，通过reward/dynamics评价当前及最多两步未来回报，beam=3。SID hierarchical在粗组提案后进行合法逐位置细化。reactive按unary评分直接选slate。执行后加入观察历史，下一轮重新编码belief。测试诊断访问的模拟器真值仅用于验收，不作为L0/L1部署输入。','',
        '修复后的每次决策遵循：历史编码 → 传入剩余动作数 → 合法提案与候选池去重 → 截断有限时域并计量 → 执行同一被评价动作 → 保存逐步奖励及成本。末步depth=0仍表示当前一步；终点continuation为零。模拟器是动作合法性的最终检查层。','',
        '| 第一阶段要求 | 实施与验收 |','|---|---|',
        '| 禁止重复item | 共用整数/K/范围/唯一性验证；refinement跳过重复替换；容量不足显式失败；合法候选池去重 |',
        '| 剩余时域 | 五类planner均支持remaining_steps；有效depth=min(configured_depth,remaining_steps−1)；修复MCTS重访末叶bootstrap |',
        '| 单样本/批量奖励一致 | scalar委托统一batch核心，包含diversity和位置折扣 |',
        '| 实际成本 | reward-only、联合reward/transition、score、提案尝试等分开计数；硬件校准估计成本及GPU同步规划延迟 |',
        '| 选参与结果一致性 | 从returns/逐步reward重算；21验证文件逐个SHA绑定；冻结选择后方可test；最终汇总核对冻结文件身份 |','',
        '完整变更与独立审查记录：'+link('docs/research/2026-10-09-stage1-changes.md')+'。冻结协议：'+link('docs/research/2026-10-09-stage1-protocol.md')+'。','',
        '## 2. 设计与重新选参','',
        '固定catalogue=200、K=5、episode=20、gamma=.95、lt=3、div=.8、位置折扣=.8。每模型16个既有用户，两种协议（完整episode、共享行为前缀5步），闭环评估执行noise=False以使用无噪声奖励；训练/反馈噪声沿用旧协议。每split另有128条诊断episode。模型latent=64、n_ensemble=1、oracle_state=False、pairwise=True，原训练3000条日志/40epochs。','',
        'L0不使用逐项辅助标签；L1只监督已曝光item的连续unary值加std=.02噪声；U1使用所有item真值辅助监督，是特权参考。三组均使用相同模型结构及既有训练配置。U1不是保证优于L1的理论回报上界。','',
        '先在training seeds1/2/3的21个pilot模型上重算公共前缀sid_hier验证回报，最大值选权重，精确并列选较小权重。新验证seed base=10,000,000+2,000×training_seed；新测试base=20,000,000+2,000×training_seed，实际用户episode偏移保存在结果中，与旧评估区间分离。','',
        '| 条件 | w=1 验证均值 | w=10 验证均值 | w=100 验证均值 | 重新选择 |','|---|---:|---:|---:|---:|']
    for c in ('L1','U1'):
        values=receipt['validation_scores'][c];lines.append(f"| {c} | "+' | '.join(f"{values[str(w)]['mean']:.4f}" for w in (1,10,100))+f" | {s['selection'][c]} |")
    lines += ['',f"L0固定w=0。虽然选择仍是100，但依据是修复后重新计算的验证结果。冻结时共有21份验证证据、**0份新test**；冻结时间UTC {receipt['file_mtime_utc']}。seeds4/5的6份新增验证只用于确认记录，不参与选参。共有27份验证、15份测试及15份最终result，无队列失败。",'',
        '## 3. 新测试回报','',
        '以下为training seeds1–5均值，回报越大越好；负值来自当前环境奖励/饱和惩罚定义。各条件与方法共享配对初始轨迹，公共前缀的总回报含相同前缀贡献，故配对差值等于其后策略贡献之差。']
    for protocol,title in [('common_prefix','共享5步前缀（主协议）'),('full_episode','完整episode（辅助协议）')]:
        d=s['protocols'][protocol]
        lines += ['',f'### {title}','','| 条件 | reactive | flat_uniform | flat_policy | sid_flat | sid_hier |','|---|---:|---:|---:|---:|---:|']
        for c,rows in d['rows'].items():lines.append('| '+c+' | '+' | '.join(f"{r['mean']:.4f}" for r in rows.values())+' |')
        lines += ['','| 配对差值 | 平均效应（回报单位） | 95% CI |','|---|---:|---|']
        for n in ('L1-L0:sid_hier','U1-L1:sid_hier','L1:sid_hier-flat_uniform','L1:sid_hier-flat_policy','L1:sid_hier-sid_flat'):
            lines.append('| '+n+' | '+interval(d['comparisons'][n])+' |')
    lines += ['','### 主比较的训练seed复制','','| Training seed | 公共前缀 L1−L0 | 完整episode L1−L0 |','|---|---:|---:|']
    for i,seed in enumerate(s['seeds']):lines.append(f"| {seed} | {primary['seed_means'][i]:.4f} | {aux['seed_means'][i]:.4f} |")
    lines += ['','区间采用预先指定的层次配对bootstrap：先重采样训练seed，再在每个seed内重采样配对用户；10,000 draws，固定随机种子20261008。训练seed是主要独立复制单位；16用户不是额外16个训练复制，95%区间也不代表跨catalogue泛化。次要比较未做多重校正，按探索性机制证据解释。','',
        f"![主比较及各seed差值]({(base/'figures/stage1_primary.png').as_posix()})",'',
        f"![层次搜索相对基线]({(base/'figures/stage1_search.png').as_posix()})",'',
        '## 4. 调用成本与延迟','',
        'B512继续是提案参数，不是总模型评估数。以下只列L1公共前缀的每决策均值；全部条件/协议、候选尝试与交付数量见 '+link('results/e1_stage1_20261009/figures/stage1_cost.md')+' 及同名JSON。','',
        '| Planner | Reward-only | 联合transition | Score queries | 校准估计ms | 实测规划ms |','|---|---:|---:|---:|---:|---:|']
    for n,r in cost['common_prefix']['L1'].items():
        lines.append('| '+n+' | '+' | '.join(f'{r[k]:.3f}' for k in ('reward_only_calls','transition_calls','score_queries','estimated_compute_ms','latency_ms'))+' |')
    lines += ['',
        '计数中联合transition同时得到reward和next belief，与reward-only不重叠；score按被查询状态数计。候选生成尝试包括之后去重/裁剪的候选。unique_slates是每决策中出现过的不同有序item tuple，并不表示不同(state,slate)对的数量。proposal时间扣除其中嵌套模型调用，避免重复计时。','',
        '校准在RTX5090/WZLcosmos运行，model batch=1/16/128/512、20次重复，参考batch=128；保留median/q10/q90及提案器独占时间。线性估计受batch摊销影响，例如reactive估计约.002ms而实际约.267ms，不能视作准确墙钟预测。硬预算约束该明确估计，操作前不足即失败；24项组合验收通过，未计量外部proposal/adjust被拒绝。E1没有施加严格相同总硬预算，不能据此完成E4效率主张。','',
        '实测延迟包含给定belief后的完整planner.plan与GPU同步，不含历史belief编码、模拟器或诊断。每卡一个worker，六卡并发；延迟是该运行环境的描述性测量，没有独立串行延迟复评。SID hierarchical在本轮较sid_flat回报更高且延迟较低，但比flat_uniform慢，因此也不是全方法普遍效率支配。','',
        f"![描述性回报与成本]({(base/'figures/stage1_cost.png').as_posix()})",'',
        '## 5. 正确性与归档验收','',
        '- 最终本地和服务器均140测试/20文件通过；24项预算组合通过。4 items/K2/H3穷举参考核对有限时域Oracle，小规模测试排除相应搜索边界缺陷。',
        '- 逐个校验27原checkpoint的字节SHA、信息条件、40epoch历史与有限参数；旧训练源26文件身份独立保存。所有参数复用，未把新评估源伪装成旧训练源。',
        '- 全部27验证+15测试共42splits、117,600实际执行决策，动作合法、时域及分项成本一致。按原case seed恢复前缀并逐步真实模拟器重放，奖励最大差为0，returns/prefix/tail/汇总均一致。',
        '- 本地离线分析重新校验21个选参文件及冻结SHA，生成汇总与远程JSON完全一致。209文件原始结果包下载SHA一致，其中208个数据/源码/checkpoint文件逐个校验（manifest自身除外）。',
        '- 所有E1执行指纹34文件仍与冻结相同。独立Oracle probe补传remaining_steps及对应测试作为两文件补充包保存；该入口不参与E1，核心冻结139测试，加补充后140测试。','',
        '| 证据 | 路径 / SHA256 |','|---|---|',
        '| 本地完整原始包 | '+link('results/e1_stage1_20261009/e1_stage1_20261009_bundle.tar.gz')+' |',
        '| 原始包SHA256 | `'+archive['archive_sha256']+'` |',
        '| 冻结执行source.zip SHA256 | `303311d01c1aa6bc049150e8fce04532adc52949ee1ac4974404df3749b41daa` |',
        '| 冻结选参SHA256 | `'+receipt['selection_sha256']+'` |',
        '| 完整重放审计 | '+link('results/e1_stage1_20261009/artifacts/final_integrity_audit.json')+' |',
        '| 最终本地测试 | '+link('results/e1_stage1_20261009/final_local_tests.json')+' |',
        '| 最终服务器测试 | '+link('results/e1_stage1_20261009/artifacts/final_remote_tests_verified.json')+' |',
        '| 原始选参文件及receipt | '+link('results/e1_stage1_20261009/artifacts/frozen_selection_receipt.json')+' |',
        '| 完整统计JSON | '+link('results/e1_stage1_20261009/e1_summary_local.json')+' |','',
        '远程执行工作区：`/home/xys/code/WZL/next/results/e1_stage1_fixed_20261009`。远程输出：`/home/xys/code/WZL/next/results/e1_stage1_20261009`；原始包在其同级results目录。原项目、旧模型及旧结果未覆盖。图表提供PNG/PDF；论文文档、最终图表和当前分析helper另包归档为 '+link('results/e1_stage1_20261009/stage1_analysis_delivery.zip')+'，原始结果包保持不变。','',
        '## 6. 结论边界与下一阶段','',
        '合法监督可行性满足本轮继续主线的依据：L1相比L0在固定合成环境的主比较及五seed复制中有稳定回报收益。U1−L1区间跨零，不足以证明两者等效；应报告为本轮未观察到可靠差异。logged reward MAE分别为L0 .2855、L1 .1305、U1 .1260，是行为分布下的一步诊断，不是任意候选或多步误差上界。','',
        '不能据本轮判定收益主要来自proposal、reward还是dynamics；L1辅助损失共享参数，会影响多个模块。各搜索器配置depth=2，而reactive不包含完整pairwise slate目标，因此搜索优于reactive也不能单独证明增加深度有效。SID hierarchical优于sid_flat尚不能区分语义分组与局部细化贡献；需要flat_coordinate_refine和随机分组对照。本轮未修复完整ensemble、Spearman并列rank等后续审计项，不能宣称全系统无问题。','',
        '下一阶段按用户路线：先增加L1-B/L1-C历史反馈变体并重新训练；补多步误差/重编码belief诊断；用小环境Oracle、饱和与漂移拆分、proposal/reward/transition分离干预识别机制；在严格总成本控制下补E4；再修复ensemble/误差代理并实现Adaptive-H。尚未启动这些扩展或据测试重新选参。','',
        'KuaiRand-27K路径已确认；10个CSV表头含user/video/time/click/play_time等逐项字段，未发现原生request/session/position/slate字段。当前证据仅目录/表头，未全文校验、建立真实slate或训练模型；后续应先确定会话/曝光构造、时间拆分及反馈合法性。证据：'+link('docs/research/kuairand_27k_inventory_20261009.json')+'。','',
        '## 7. 复现入口','',
        '原始结果包含source.zip、sha256.json、probe_fix.zip、27个model.pt、旧训练源码包、全部阶段JSON和queue日志。先解压source.zip到独立代码目录；如需当前完整140测试，再叠加probe_fix.zip。旧checkpoint身份与新评估source必须分别保留。','',
        '已有结果的离线统计复现（从项目根目录执行，无需GPU）：','',
        '```powershell',
        'python scripts/analyze_logged_pilot.py --out-dir results/e1_stage1_20261009/artifacts --seeds 1 2 3 4 5 --frozen-selection results/e1_stage1_20261009/artifacts/frozen_selection.json --validation-root results/e1_stage1_20261009/artifacts --report results/e1_stage1_20261009/reproduced_summary',
        'python ops/run_native_tests.py --root . --out reproduced_tests.json',
        '```','',
        '输出路径须新建，脚本拒绝覆盖报告。统计bootstrap依赖版本改变时数值末位可能不同；当前本地与远程环境已验证JSON完全一致。重新执行GPU实验使用归档ops/stage1_queue.py及其保存的validation/test queue协议，不应混入后续模型改动。归档中test_runner_wrong_root_noop.json只记录一次路径错误导致的零文件调用，不是测试验收；以final_remote_tests_verified.json为准。','']
    target=root/'docs/research/2026-10-09-stage1-results.md'
    with target.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    print(target)

if __name__=='__main__':main()
