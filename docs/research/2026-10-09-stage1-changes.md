# MuSlate 第一阶段修改、逻辑与验收

本轮实施用户方案的第一阶段修改1–5。WorldModel 的结构、GRU 输入、损失与训练超参数保持原协议；旧 checkpoint 参数不变。第二阶段反馈进入历史编码、完整 Ensemble 修复、机制矩阵和 Adaptive Planner 尚未实施。

## 修改后的一次决策

1. 用当前动作之前的 slate/reward 历史编码 belief；评估程序传入 `remaining_steps = horizon - t`。
2. 规划入口建立本次 `PlanningCost`。若传硬预算，先检查是计数包装与受支持的同计数器提案器；外部未计量 penalty/proposal 直接拒绝。
3. 提案器生成合法、无重复 item 的有序 slate。完整重复候选去重；容量不足显式报错，不用重复 item 补位。
4. 规划深度限制为 `min(configured_depth, remaining_steps - 1)`；depth=0仍包含当前动作的一步奖励。模型 rollout 只累加实际剩余时域内的奖励。
5. 执行被模型评价过的动作；模拟器再次校验K、整数索引、范围和唯一性。没有预测以后删重复/补位的动作变换。
6. 保存执行动作、reward、时间位置、有效深度和分项成本。结果验收由这些 reward 重算折扣回报，与原始 returns、prefix/tail 和均值核对。

## 五项实现

| 要求 | 实现位置 | 具体行为 |
|---|---|---|
| A01 合法 slate | `planning/validity.py`、`proposal.py`、`abstractions.py`、`env/simulator.py` | 共用 validator；flat 使用实际未选 item 集合；SID 逐位置容量约束与合法替换；池去重且不补重复候选；模拟器拒绝非法动作 |
| A02 有限时域 | `planning/search.py`、`eval/rollout.py` | Reactive/Lookahead/Shooting/CEM/MCTS 接口统一；剩余时域沿递归/序列/树传递；末步只评价当前奖励；MCTS 重访叶保持相同 bootstrap |
| 单样本/批量奖励 | `env/simulator.py` | scalar 委托 batch，无噪声核心统一，包括 diversity pairwise 和位置折扣；噪声在 scalar 外层添加 |
| A14 成本 | `planning/cost.py`、`latent_model.py`、`scripts/calibrate_planning_cost.py` | reward-only、joint transition、score query、候选生成尝试分开；每决策重置、旧累计计数保留；提案独占时间与总规划时间分别记录；实际硬件校准 |
| 选参及结果一致性 | `eval/integrity.py`、`analyze_logged_pilot.py`、`reevaluate_stage1.py` | returns 重算均值，完整动作/时域/成本及prefix/tail检验；validation阶段不生成test；21份证据重算选择并冻结；test和最终报告绑定冻结文件SHA |

SID refinement 在每个给定状态内缓存已经评价过的合法 slate；无效替换和原位不变跳过，重复新候选不重复查询 reward。不同状态下同一 slate 的预测不能复用。`unique_slates` 是单次决策内出现过的 distinct ordered item tuples；`emitted_candidates` 是各次提案实际交付数量，跨节点仍会相加。

批量提案需要矩形数组：不同状态的去重池若大小不同，裁到共同最小长度，不补重复项。这个选择可能减少候选数，实际交付量已记录；不能把名义 `out_size` 当作有效池大小。

## 成本与预算的准确含义

`transition_calls` 表示同时预测 reward 和 next belief 的联合调用，和 `reward_only_calls` 不重叠。`score_queries` 按状态数计；`proposal_candidates` 按生成尝试计，即使候选重复或之后被裁剪也仍收费。CEM 内部采样同样纳入提案时间/有效池记录，并去重完整 first-action slate；其 continuation 仍是每个候选采样一条随机轨迹。

校准提供 batch=1/16/128/512 的 model median/q10/q90，以及各提案器候选生成的独占时间分布。参考 batch=128；所有模型权重由实测得到，没有设为1。估计成本是

`reward_ms × reward_only_calls + transition_ms × transition_calls + score_ms × score_queries + proposal_ms × proposal_candidates`。

它是线性近似的操作成本毫秒，受 batch/硬件影响；另报真实 planner 调用延迟。硬预算限制此估计，不是 wall-clock deadline；不足时在对应操作前抛 `BudgetExceeded`，不执行免费补救动作。未计量外部 proposal 和 adjust hook 不支持硬预算；计量后的误差代理接入留给后续阶段。E1 的 B512 仍是提案参数，本轮不宣称严格等总计算预算或 SID 计算效率优势。

延迟测量范围为给定 belief 后的 `planner.plan`（同步GPU），不包含历史 belief 编码、模拟器执行或测试诊断。它不是整个推荐服务的端到端响应时间。

## 独立审查及补充修复

独立只读审查发现五项 Important、四项 Minor，无 Critical；全部在正式实验前统一修复。

| 审查问题 | 修复与证据 |
|---|---|
| GPU ordinal 进入 immutable identity，导致validation→test恢复失败 | 逻辑合同只存 device type；具体GPU/runtime随split保存并冻结复用。真实checkpoint的cuda:0→cuda:1验证恢复成功 |
| proposal绕过CountingModel | 硬预算入口在任何工作之前检查内置提案实现及同计数器绑定；raw inner、另一counter、未知外部proposal均有拒绝回归 |
| CEM遗漏emitted与proposal time | 内部采样共用计量装饰器；候选池去重，尝试数保留 |
| 冻结证据链不完整 | 检查21唯一文件、准确arm矩阵、source/contract和raw选择；最终汇总检查各test的冻结文件SHA；离线汇总可用`--validation-root`重新定位下载文件 |
| 队列不能恢复 | 稳定队列合同核验复用；每次运行创建独立attempt，保留旧日志和完成记录 |
| 非法动作测试因缺字段假通过 | 先校验完整合法V2 fixture，再分别修改slate和末步depth，断言明确错误 |
| 分组概率下溢误报容量不足 | 每位置只在仍有容量的组内重新稳定softmax；极端[1000,0,0,0,0]小catalog回归通过 |
| 任意seed允许区间重叠 | 第一阶段seed限1–5；validation CLI限1–3，confirmation限1–5 |
| 补训seed4/5继承seed1 planner_seed | 显式使用60000+training seed，所有条件保持配对 |

独立Oracle探针 `scripts/probe_environment.py` 的闭环循环也补齐了remaining_steps传参。该脚本不参与E1执行，E1冻结source字典保持一致；补充源码/测试与哈希另存于 `snapshots/20261009_stage1_supplement/`，原执行源码包保留。

## 验收记录

- 修改前103项测试；修改后核心版本本地与服务器各139项；补齐独立Oracle探针入口后各140项，20文件全部通过。混合测试文件须逐个运行原生入口；仅用 unittest discovery 会漏掉函数式测试。
- 另有24项小模型预算验收组合：全部五类planner及flat/SID提案，在零、有限和充足预算下，实际记录估计与已用预算一致，无超额执行；这是计量验收，不是硬件性能比较。
- 五类 planner 有末步奖励陷阱、最多两步、MCTS重访叶测试；4 items/K2/H3小环境枚举所有合法slate序列，与穷举Oracle核对。
- 七条件最终GPU短评估通过，未产生test。旧重复反例重放15决策全部合法，t=18/19有效深度为1/0。
- 源码包SHA256：`303311d01c1aa6bc049150e8fce04532adc52949ee1ac4974404df3749b41daa`；逐文件清单在 `snapshots/20261009_stage1_final/sha256.json`。
- 原始远程项目、旧27模型和旧结果未覆盖；新工作区为 `/home/xys/code/WZL/next/results/e1_stage1_fixed_20261009`，新产物根目录为 `/home/xys/code/WZL/next/results/e1_stage1_20261009`。
- 正式重评完成：21 pilot验证重新选择L0=0/L1=100/U1=100；所选seeds4/5模型已存在，未重新训练。共27验证、15测试、15最终result，队列无失败；冻结选择前新test数量为0。
- 全部42splits、117,600决策通过动作/时域/成本检查及真实模拟器逐步重放，奖励最大差为0。最终本地和服务器重新运行各140项测试通过。
- 原始结果/源码/27checkpoint包209文件已下载，外层及208个清单文件SHA一致；本地离线重算统计与服务器JSON完全一致。原始包SHA256为`e8115a55dd84acf25d2bff85dbde036c5f88372929a05b9036854773821ce868`。
- 公共前缀主比较L1−L0 sid_hier为4.2197 [3.7823,4.5999]；完整episode为5.6180 [4.8922,6.2544]。这是固定合成环境中的合法逐项辅助监督收益，不是严格等成本、深度机制或真实数据有效性的证明。

完整修复版 E1 结果另见同日 `stage1-results` 报告；冻结设计见 `2026-10-09-stage1-protocol.md`。KuaiRand-27K 本轮只做目录与CSV表头清点，未执行真实数据训练。
