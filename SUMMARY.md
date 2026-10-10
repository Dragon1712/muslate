> 2026-10-10 R2-B/R2-C 已完成：R2-B 中预算50配置、12,000次决策全部重放通过，无缺失或决策错误；R2-C十模型、80个共享状态（160个模型-状态case）完成bank真值/原始指标审计，十模型user0的40个预测数组复算完全一致。R2-B主配对区间显示扁平细化−均匀搜索为0.942 [0.493,1.390]；真实SID层次−扁平细化为−0.448 [−0.910,0.013]，区间跨零；真实SID层次−随机分组层次为−0.541 [−0.904,−0.179]。尚不支持语义层次普遍更优。预算为校准的估计成本cap 6.735ms，不是真实延迟硬上限。R2-C中L1的五步回报MAE从0.329降至0.258，TF−OL为−0.0704 [−0.0863,−0.0544]；这是回顾性预测诊断，不能证明可部署规划收益或单一transition因果责任。
> 源代码冻结提交：R2-B 462c787、R2-C 3c41d1e；[R2-B完整结果](D:/Coding/next_codex/docs/research/2026-10-10-r2b-results.md)、[R2-C完整结果](D:/Coding/next_codex/docs/research/2026-10-10-r2c-results.md)。三预算曲线、跨组替换对照、反馈历史模型训练、真实数据训练及40M确认轨迹尚未执行。本轮保持五训练seed、单目录、16用户的探索性解释。

> 2026-10-09 R2-A已实际完成：80/80共享前缀配置、80固定状态探针及小环境穷举Oracle；19,200决策完整重放、缓存根候选再生成与身份/输入SHA核验通过，失败及缺失均为0。12个depth相对0的主95%区间均跨零；L1 sid_hier depth3−0为0.095 [-0.036,0.225]，暂不足以支持Adaptive-H投入。B128为固定提案设置，不是等总计算；确认轨迹、等预算、64用户和模型消融尚未执行。[完整新结果与成本/剪枝诊断](D:/Coding/next_codex/docs/research/2026-10-09-r2a-results.md)。

> 2026-10-09 实验状态更正：旧SID refinement可能产生重复item，旧深规划未截断episode剩余时域；2026-10-08及之前受影响回报保留为探索性结果。第一阶段五项修复及新E1已完成：重新验证21模型选参，选中条件5训练seed共15测试模型；本地/服务器各140项测试通过，117,600决策完整重放奖励误差为0。主比较（共享5步前缀）L1−L0 sid_hier为4.2197，95% CI [3.7823,4.5999]；五seed差值均正。当前E1模型pairwise=True，与下方历史v2的纯unary描述不同。完整新结论、表图、成本解释及归档见 `docs/research/2026-10-09-stage1-results.md`；代码逻辑见 `docs/research/2026-10-09-stage1-changes.md`；冻结协议见 `docs/research/2026-10-09-stage1-protocol.md`。真实数据训练、自适应和机制分离尚未完成。

# MuSlate 研究总结：思路、方法、结果

> **版本说明（v2）**：本版在 v1（实验结果汇总）基础上做了三处重要更新：
> 1. **修正了对 learned Gate 3 失效的解释**——新增的关键认识是当前 world model 的 reward head
>    是**纯 unary 的**，结构上无法表示环境里人为加入的 **pairwise** div 项（第 2.3、3.3 节）。
>    这比原先「语义质量不够」的归因更根本，也是当前最优先要修的结构性缺陷。
> 2. **论文定位升级**——从「我们提出一个 world model + search 的系统」改为
>    **「何时该深规划、该用层次动作、该保守」的适用性理论**（第 1.1、1.4 节）。
> 3. **后续实验按新的因果依赖重排**（第五部分）。
>
> 数据来源：`results/*.json`（本地 CPU）与 `results/server/*.json`（6 × RTX 5090 大规模）。
> 标有 ⚠️ 的段落是本版新增或修正的内容。

---

# 第一部分　研究思路

## 1.1 论文要回答的问题

⚠️ **定位修正**。最初的写法是「world model + Semantic-ID + MCTS 的组合系统」。但这三个组件
各自都已有近邻工作占据（world model: Pseudo Dyna-Q / UA-TWM；generative slate:
TIGER / HiGR / OGR；MCTS: SAPIENT / MF-MDP），因此「组合」只能作为系统形式，不能作为
最核心的科学贡献。

本研究实验真正指向的问题更漂亮，也更难被推翻：

> **When is deep planning over a learned world model actually useful for slate recommendation?**
>
> 在离线 slate 推荐中，多步规划与层次动作抽象**什么时候**真正有价值？
> 当世界模型存在误差时，应如何**自适应**地控制搜索深度与动作抽象，避免 planner 主动利用模型错误？

三个 innovation 因此从「贡献」降级为「被检验的机制」，而**适用条件是真正的贡献**：

| 维度 | 判断 |
|---|---|
| 科学问题价值 | 高 |
| 原始方法创新 | 中（组件各有近邻） |
| 实验设计思想 | 高 |
| 当前证据完整度 | 中（合成、单 seed、模型有已知结构缺陷） |
| 补完关键实验后潜力 | 高 |

## 1.2 为什么先做 gate，而不是直接搭系统

「组合是否产生新东西」不能用组件的新颖性论证，只能用实验论证。更关键的是存在两个可能
直接击穿方案的已知结论：

* **MF-MDP 报告「1 步 lookahead 已拿走大部分收益」** → 若在 slate 场景同样成立，整个
  「多步」框架退回 UA-TWM / OGR 的领地。
* **UA-TWM 已占据「slate 未来状态 + uncertainty + fallback」** → 剩下的独占区只有
  「多步 + 层次搜索 + 生成式 proposal」。

所以第一阶段**不搭系统，而是给每个 innovation 设一道可以判死的门**：

| Gate | 问题 | 为什么是生死线 |
|---|---|---|
| **Gate 2** | 多步规划是否真比 1 步好？ | 直接对抗 MF-MDP 的结论 |
| **Gate 3** | Semantic-ID 层次在等预算下是否比扁平动作强？ | 决定 Innovation 2 是不是装饰 |
| **Q3** | 树是否比普通 beam / rollout search 强？ | 直接对抗「普通搜索已够」的质疑 |
| **Gate 4** | 保守规划能否阻止 planner 利用模型误差？ | 决定 Innovation 3 是否成立 |

## 1.3 验证路线：把「是否成立」变成「在什么条件下成立」

单纯 gate 只能给 yes/no，而预期是**条件性成立**。因此必须在环境里放**可扫的旋钮**，
把 gate 变成「随参数变化的曲线」。

当前环境有两个**正交**旋钮：

* `lt`（long_term_strength）：缩放两个**延迟**耦合机制 → 「多步是否值钱」
* `div`（div_penalty）：缩放一个**即时组合**项 → 「层次搜索是否值得」

正交性是刻意的：Gate 2 只扫 `lt`，Gate 3 只扫 `div`，两个结论互不污染
（`test_diversity_is_immediate_not_delayed` 锁定该性质）。

这带来一个比「我们的方法更好」强得多的 claim：

> 多步规划需要长程耦合（`lt`），层次搜索需要 slate 内组合结构（`div`）。
> **两者都是数据集的可测属性，不是建模选择。**

## 1.4 ⚠️ 目标形态：规划适用性的 phase diagram

在 `lt` / `div` 之外，实验还暴露出**第三个**决定成败的因素：**语义层次的可信度**
（`q_sem`）。四者各决定一件事：

| 旋钮 | 决定 | 低值时的正确行为 |
|---|---|---|
| `lt`（long-term coupling） | 是否值得深规划 | 退回 reactive / one-step |
| `div`（slate interaction） | 是否需要组合式 slate 搜索 | 逐位置贪心即可 |
| `q_prior`（**先验排序质量**） | 所有 score-driven 搜索（含结构化）的共同上限 | 模型差到一定程度时，**随机池反而最好** |
| `U`（model reliability） | 是否允许长 rollout | 缩短 horizon / 加保守惩罚 |

⚠️ **v3 修正（第二次）**：这一栏最初写「`q_sem` 语义保真度」，v3 初稿改成
「`q_prior`，低则退回 flat」。purity sweep 实测**把这个推论也推翻了**：
先验退化时 `flat_policy`（非结构化、同样吃 scores）掉得**比** `sid_hier` 快一倍
（Δ≈−22.6 vs −10.3），即**结构化搜索在先验差时是更鲁棒的一方，不该退回 flat**。

正确的表述是：`q_prior` 是**所有依赖先验的搜索的共同上限**——先验一旦差过某个点，
**不用模型的随机池会赢**（服务器实测 `lt=3`：`flat_uniform −11.13` 优于
`sid_hier −15.40` 优于 `flat_policy −16.77`）。这一轴限制的是**搜索**，
但它的修法在**模型**（Innovation 1），不在搜索结构（Innovation 2）。

最终希望给出的是**规划机制的条件理论**，而不是一个固定配置的算法：

```
                  Prior quality ↑
                        │  Hierarchical deep planning
                        │
     flat search ───────┼──────────────────→
                        │            long-term coupling
```

⚠️ **范围决定**：尽管有三/四个旋钮，主线**先做二维**（`lt` × `q_sem`），`div` 固定在一个
足够大的值。理由：高维 phase diagram 的采样量与图表复杂度都难控制，而二维已足以支撑
「条件性 claim」。`div` 的扫描留作补充。

## 1.5 三个 innovation 的耦合方式

论文主张「三个模块是因果链而非技术堆叠」。本研究把这句话变成了**可测的**：

* Innovation 2（层次搜索）在**真值动力学**下最优，在**学习动力学**下排名完全镜像
  → **层次搜索的收益以 world model 的语义质量为前提**（1 与 2 的耦合）。
* Innovation 3 的惩罚项是目前唯一能抑制这种失效的机制（3 与 1、2 的耦合）。
* 因此最关键的实验是把 3 用到 2 上（P2 闭环）。

⚠️ 而 3.3 节进一步指出：learned 下 hierarchy 失效的**首要原因不是语义质量，而是
reward head 的函数形式排除了组合项**。这把耦合关系从「1 → 2」精确化为
「1 的**结构正确性** → 2 的收益」。

---

# 第二部分　实验方法

## 2.1 可控合成 slate 环境

**代码**：`muslate/env/catalog.py`、`muslate/env/simulator.py`

**为什么自建**：需要 ground-truth 动力学才能把「规划无用」与「模型太弱」分开。
KuaiRand 没有天然的 request-level slate transition，也没有真值动力学。

**Item catalogue**：两级 Semantic ID，`N = C1 × C2 = 10 × 20 = 200`。
`z_i = normalize(w_c·A[c1] + w_f·B[c2])`，`w_c=1.0, w_f=0.5`（粗粒度主导）。

**MDP**：

```
state_t   = (h_t, sat_t)
action_t  = slate_t = (i_1, ..., i_K)，K=5，有序
reward_t  = <h_t, mean_w(z_slate)>                       即时、位置可分离（unary）
            - lt * sat_penalty * <w, sat_t[cat(slate)]>   延迟、类别级饱和
            - div * Σ_{j<k} w_j w_k <z_j, z_k>            即时、组合（pairwise）
            + noise
state_{t+1} = ( normalize(drift(h_t, slate_t)), decay(sat_t, slate_t) )
```

| 项 | 性质 | 作用 |
|---|---|---|
| `<h, mean_w(z)>` | 即时、可分离、依赖 h | 基本偏好结构 |
| `-lt·sat_penalty·sat` | **延迟**（用 t−1 的曝光）、类别级 | 长程依赖；`lt=0` 退化为 bandit |
| `-div·Σ w_j w_k <z_j,z_k>` | **即时、组合、不依赖 h** | 破坏可分离性，使逐位置贪心不再最优 |

⚠️ **最后一行加粗的「不依赖 h」是本版新增的关键观察**：div 项是 item 对的固定相似度，
与用户状态无关。这一点在第 2.3 节决定 reward head 该用什么形式。

## 2.2 离线交互日志

**代码**：`muslate/env/logger.py`

behavior policy 是 temperature 控制的 myopic greedy + ε 探索（`ε=0.25, temperature=1.0`），
记录 `(episode_id, step, slate, reward)`，真值 latent `(h, sat)` 单独存下（仅供
`oracle_state` 监督与诊断，planner 看不到）。

真实感来自它的**偏斜**：足够强以致含有信息，又足够 myopic 以致朴素 OPE 有偏。

⚠️ 日志覆盖度（`ε`）在 P1 中升级为**实验旋钮**（`ε ∈ {.05,.1,.25,.5,.8}`），
因为它直接决定 world model 的可靠性与安全规划深度。

## 2.3 Latent world model

**代码**：`muslate/model/world_model.py`、`train.py`、`neural_model.py`

```
(history of slates, rewards) --GRU--> h_t
(h_t, slate) --dyn--> h_{t+1}
(h_t, slate) --reward head--> r_t
```

### ⚠️ 2.3.1 当前 reward head 的结构性缺陷（本版最重要的修正）

**当前实现**（`world_model.py::predict_reward`）：

```
r̂(h, A) = Σ_k w_k · [ <g(h), e_{i_k}> + q(h)[i_k] ]
```

这是**纯 unary 的**——每一项只看单个 item，**没有任何 item 对的交叉项**。

**环境的 div 项**（`simulator.py::reward_batch`）：

```
- div · Σ_{j<k} w_j w_k <z_j, z_k>          # 纯 pairwise，且不依赖 h
```

因此从函数表达能力上：

```
r̂_unary  ⊉  Σ_{j<k} g(i_j, i_k)
```

**当前模型在结构上无法表示我们专门设计出来的 div 惩罚。**

这不是「训练不够」或「容量不足」，而是**函数形式的错配（reward misspecification）**。
它很可能直接解释 learned 下 hierarchy 的崩塌（详见 3.3）：层次搜索正在寻找组合结构，
而 world model 根本没有足够的形式去描述组合结构。

**修复方案**：

```
r̂ =  Σ_k w_k u_θ(h, i_k)                     ← unary，与 item_scores 共享参数（保留原设计）
    + Σ_{j<k} w_j w_k · p_θ(i_j, i_k)         ← pairwise，新增
```

`p_θ` 的形式选择（第一版）：

| 形式 | 参数 | 匹配 div？ | 判断 |
|---|---|---|---|
| **(a) 常数双线性 `e_jᵀ W e_k`，W 对称** | `de²` | ✅ 精确（`<z_j,z_k> = z_jᵀ I z_k` 特例） | **推荐** |
| (b) h-依赖 `e_jᵀ W(h) e_k` | 更大 | 过度 | 学一个本不该依赖 h 的项，浪费容量、易过拟合 |
| (c) 固定 `P[i,j]` | `N²/2 ≈ 2 万` | ✅ | 丢失 embedding 泛化，且无法与 `item_scores` 共享 |
| (d) DeepSets / Set Transformer | 大 | — | 丢弃「已知是 pairwise」的归纳偏置 |

**推荐 (a)**：`W ← (W + Wᵀ)/2` 强制对称，pair 权重复用现成的 `_pair_w`；
`<z_j,z_k>` 不依赖 h，所以常数 `W` 才是与真实结构对齐的对照。
作为「排除 misspecification」的诊断实验，它是最干净的。

**消融设计**：

| Reward head | unary | pairwise | 预期 |
|---|---|---|---|
| Monolithic MLP（最早的版本） | implicit | implicit | 泛化差，被 planner 利用 |
| 当前 item-decomposed | ✅ | ❌ | 强 OOD unary，但学不到组合 |
| **Unary + Pairwise** | ✅ | ✅ | 主方法 |

### ⚠️ 2.3.2 修复带来的连锁影响（必须先想清楚）

之前刻意做了「reward 与 `item_scores` 共享参数」的修复，并有测试锁定
（`test_reward_is_decomposable_into_item_scores`）。加 pairwise 后该性质**只能部分保留**：

* `item_scores` 继续**只由 unary 部分**定义（proposal 的 myopic 先验本就该无组合信息）
* `predict_reward` = unary 加权和 + pairwise 加权和
* 测试改为「**unary 部分**与 `item_scores` 一致」，而非整个 reward

这不是退步而是更准确：**proposal 不该知道 pair 惩罚（那正是搜索要解决的问题），
而 value 必须知道**。

### 2.3.3 其余设计（保留）

* **顺序感知 slate 编码**（用于 dynamics）：位置折扣 + 可学习 position embedding
* **多步训练**：dynamics 通过自身 rollout 预测未来若干步 reward，而非只 1 步

⚠️ **待增强（P3）**：加入**与规划 horizon 一致**的损失
`L = Σ_k ω_k ||ĥ_{t+k} − h_{t+k}||²`，因为真正的消费方是 `H=3,5` 的 planner，
而不是 one-step predictor。

* **异构 ensemble（已证伪）**：`latent_dim` 必须一致，只异构
  `slate_emb_dim / hidden_dim / rollout_depth / lr`（见 4.2）

## 2.4 Planner 家族

**代码**：`muslate/planning/search.py`

统一接口 `plan(states[B,D]) -> slates[B,K]`，且都接受 `adjust` 钩子
（`(states, slates, rewards) -> adjusted_rewards`），这是保守规划的接入点。

| Planner | 机制 | 用途 |
|---|---|---|
| `ReactiveGreedy` | 按 item score 取 top-K，零搜索 | 下界基线 |
| `Lookahead` | 深度受限 best-first + beam，`depth=H` | Gate 2 主工具 |
| `RandomShooting` | 采样整条轨迹取最优首动作 | Q3 对照 |
| `MCTS` | PUCT，`Q` 逐节点 min-max 归一化，未访问乐观初始化 | Q3 |
| `CEM` | 每位置 item 分布上的 CEM | Q3 |

**两次直接影响结论的实现细节**：

* `Lookahead` 在视界处用该层候选的 `r.max()` 做 **bootstrap**。
* `MCTS` 的叶子价值**必须使用同一 bootstrap**。第一版用硬 0，会系统性低估浅路径、
  把探索偏向「更深」而非「更好」，导致「MCTS 从不胜出」的**错误结论**。
  `--no-bootstrap` 保留为 ablation。

## 2.5 动作抽象（Gate 3）与 SID 质量旋钮

**代码**：`muslate/planning/abstractions.py`

`SIDProposal` 是可插拔的候选生成器（与 `ProposalPolicy` 同接口），搜索算法不变，
只有动作空间变化：

* `refine_positions=0` → 扁平单层：采样 coarse 类别计划并贪心补全
* `refine_positions>0` → **progressive widening**：只把最好的 `widen` 个 coarse 计划
  带入细化，在固定类别结构内逐位置坐标上升

**预算核算**：`out_size + proposal_cost()` 在变体间保持相等，否则比较被预算差混淆。

### ⚠️ 2.5.1 新增旋钮：SID semantic purity（P0b）

当前 catalogue 的 SID 就是 `(c1, c2)`，而 item embedding 正是 `A[c1] + B[c2]`
—— **搜索树的层次与环境真实 item 几何同源生成**。因此 oracle 下 `sid_hier` 表现好
**是构造性的必然，不能作为「层次搜索有效」的证据**。必须补一个 semantic purity sweep。

**实现要点**：把「category 映射」做成 `SIDProposal` 的**参数**，而不是写死 catalogue：

| 变体 | 构造 |
|---|---|
| `oracle` | 真实 `(c1, c2)` |
| `learned` | 对 item embedding（或 learned embedding）做 k-means |
| `noisy(p)` | 以概率 p 把 coarse 标签打乱（纯度 = 1−p） |
| `random` | 完全随机的树 |

**两个必须注意的点**：

1. **纯度指标**：横轴用「同组 item 平均相似度 vs 组间」这一连续量，
   它同时覆盖四种变体，不必人为对齐 p。
2. **`noisy` 只能打乱树的层级结构，补全仍用真实 item score**。否则
   `SIDProposal._complete` 的前提（同组内按 score 排序）也被破坏，
   测到的是「补全坏没坏」而不是「树好不好」。

**实现结果（⚠️ 已跑，结论见 3.3）**：`run_purity.py` 实测的曲线是
`purity` **几乎不影响** `hier − flat`（purity 1.00 → 0.25 时只从 +2.370 变到 +2.287），
所以「purity 阈值」这个预期**不成立**。取而代之的实测规律是：
**`hier − flat` 由先验质量而非树质量决定**。这条轴因此从 phase diagram 的
「语义保真度」改写为「**先验排序质量**」。

## 2.6 保守规划（Gate 4）

**代码**：`muslate/planning/conservative.py`、`controlled.py`、`muslate/model/error_model.py`

惩罚形式：`Q_plan = Q̂ − λ·U − β·D`，通过 `adjust` 钩子注入，**搜索代码完全不变**。

四种不确定性信号，互为对照：

| 信号 | 实现 | 说明 |
|---|---|---|
| `ensemble_std` | ensemble 成员 reward 预测的 std | 间接代理，实测弱 |
| `state_ood` | `LatentSupport`：到日志中最近 belief 的距离 | belief 空间密度 |
| `item_ood` | `BehaviorSupport`：item 频率的 **z-score** | 原始 NLL 无区分度（std<0.02），必须归一化 |
| **`error_model`** | `E(h,a) ~ |r̂(h,a) − r_true|`，**holdout 监督** | 直接监督，实测 ρ=0.81 |

**error model 训练**：在 holdout 日志上用 `NeuralLatentModel.encode_sequence`
（一次前向得到全部时刻 belief）构造 `(belief, slate) → 真实绝对误差`，
在 **log 空间**做 MSE 回归，输出经 softplus 保证非负。

**受控实验** `BiasedOracleModel`：在完美动力学上注入
`bias = strength · <w, mean(z_A)> · (1 + coupling·Σsat)`。`coupling` 是让 bias
**可被规划利用**的关键——否则它只是常数偏差，多步规划仍会叠加正常收益。
`uncertainty()` 返回 `|bias|`，即**完美不确定性**。

### ⚠️ 2.6.1 新增：error model 的 calibration-by-shift（P1）

当前 ρ=0.81 **只在 logged distribution 上成立**。真正危险的是 planner 选出的动作
不再服从 logged 分布，所以必须分分布测：

| 分布 | 构造 | 意义 |
|---|---|---|
| logged | holdout 日志 | 基线 ρ≈0.81 |
| proposal | proposal policy 候选 | 中间态 |
| flat-selected / SID-selected | 短闭环收集 planner 实际选出的 slate | **部署分布** |
| 刻意 OOD | 全同类 / 极端组合 / 随机 slate | 上界压力 |

**判据**：若 ρ 掉到 0.3 以下，说明 error model 在部署分布失准。此时有两条修法：

* **不建议**用 simulator 给 OOD slate 标注来增强训练——那等于把「合成环境专属能力」
  当成方法，真实部署没有 simulator。
* **建议**让 error model 的预测随「距 logged 支撑的距离」单调增长
  （输入加 `state_ood` 特征，或显式加外推惩罚），或用 **conformal prediction**
  （OOD 时区间自动变宽）。

⚠️ 但**先只测不修**：衰减曲线本身就是可写进论文的结果。

## 2.7 评测方法

**代码**：`muslate/eval/rollout.py`、`calibration.py`

* **闭环评估**：`OracleModel`（模拟器）与 `NeuralLatentModel`（学习模型）实现
  **同一 `LatentModel` 接口**，同一份 planner 跑在两者上 —— 差距只能归因于模型。
* **warm-start 指标**：冷启动（`t=0,1,2`）的信念误差对所有 `H` 是共同噪声，
  会淹没 `H` 的对比，故 `--warmup N` 额外报告排除前 N 步的指标。
* **预算计量**：`CountingModel` 统计**实际模型评估次数**。不同 planner 的预算分配
  结构不同，名义预算不可比。
* **校准诊断**：`signal_report` 在**独立 holdout** 上算每个信号与真实误差的
  Pearson / Spearman。
* ⚠️ **新增 shift 维度**：见 2.6.1。

## 2.8 实验矩阵与代码映射

| 实验 | 脚本 | 关键参数 | 状态 |
|---|---|---|---|
| 环境信号探针 | `probe_environment.py` | lt × H | ✅ |
| Calibration | `run_calibration.py` | 4 成员 × 20000 ep | ✅ |
| Gate 2 | `run_gate2.py` | lt × H，oracle/learned | ✅ |
| Gate 3 | `run_gate3.py` | 等预算 512，depth 2，div 0.8 | ✅ |
| Q3 | `run_q3.py` | 5 planner，实测预算 | ✅ |
| Gate 4 受控 | `run_gate4_controlled.py` | bias × λ | ✅ |
| Gate 4 端到端 | `run_gate4.py` | `--unc-source` | ✅ |
| **P0a reward head 修复** | `world_model.py` + `run_gate3.py` | unary → unary+pairwise | ✅ |
| **P0b SID purity + 先验纯度** | `sid_map.py` + `perturb.py` + `run_purity.py` | 双维度 sweep | ✅ |
| P0a 诊断 | `diagnose_pairwise.py` | head × div cross-over | ✅ |
| 语义诊断 D1–D4 | `diagnose_semantics.py` | embedding/先验/pairwise/coarse | ✅ |
| **P1a λ sweep** | `run_gate4.py` 扩展 | λ × 双纵轴 | ⬜ |
| **P1b OOD 校准** | `calibration.py` 扩展 | 5 种分布 | ⬜ |
| 批量入口 | `server_run_all.sh` | 6 卡轮转 + 自检 | ✅ |

---

# 第三部分　实验结果

## 3.1 Calibration：不确定性信号是否校准

20000 episodes/成员，error model 用 5000×2 holdout，在**第三个独立 holdout**（500 回合）评测：

| signal | 同构 ensemble | 异构 ensemble |
|---|---|---|
| `ensemble_std` (Pearson ρ) | 0.0856 | 0.0221 |
| `ensemble_std` (Spearman) | 0.0626 | 0.0045 |
| **`error_model` (Pearson ρ)** | **0.814** | **0.833** |
| **`error_model` (Spearman)** | **0.757** | **0.771** |
| `state_ood` | −0.1149 | −0.0316 |
| `item_ood` | 0.0004 | −0.0123 |
| target: mean abs error | 0.398 | 0.386 |

**结论**

1. 监督式 error model 达 **ρ=0.81**（本地小规模 0.64 → 大规模 0.81，随数据提升）。
2. **ensemble 分歧本质无用**（ρ≈0.09）。与 offline MBRL 的观察一致：
   分歧度量的是「模型之间的差异」，不是「模型的对错」。
3. **异构 ensemble 是失败的修法**（ρ=0.02，比同构还差）。
4. ⚠️ 但 ρ=0.81 **仅在 logged 分布上成立**，planner 会走到 OOD（见 2.6.1）。

## 3.2 Gate 2：多步规划的价值边界

`gain = return(H) − return(H=0)`，32 users，warmup 5，20000 episodes。

**真值动力学（oracle）**

| `lt` | H=1 | H=2 | H=3 | H=5 |
|---|---|---|---|---|
| 0.0 | +0.000 | +0.000 | +0.000 | +0.000 |
| 0.5 | +0.017 | +0.090 | +0.077 | +0.075 |
| 1.0 | +0.123 | +0.239 | +0.176 | +0.194 |
| 2.0 | +0.472 | +0.327 | +0.317 | +0.537 |
| 3.0 | +0.422 | +0.739 | +0.841 | **+1.067** |
| 5.0 | +0.256 | +0.482 | +1.040 | **+1.368** |

**学习动力学（learned）**

| `lt` | H=1 | H=2 | H=3 | H=5 |
|---|---|---|---|---|
| 0.0 | −0.223 | +0.345 | +0.178 | +0.284 |
| 0.5 | +0.011 | +0.031 | +0.419 | −0.316 |
| 1.0 | **−0.447** | −0.026 | −0.220 | +0.083 |
| 2.0 | **−0.573** | **−0.405** | **−0.215** | **−0.458** |
| 3.0 | +0.425 | +0.720 | +0.135 | +0.261 |
| 5.0 | +0.086 | +0.583 | +0.313 | +0.192 |

**模型质量（H=0 绝对值）**

| `lt` | oracle | learned | 差距 |
|---|---|---|---|
| 0.0 | 15.464 | 5.240 | 10.2 |
| 1.0 | 13.916 | 7.986 | 5.9 |
| 3.0 | 8.543 | 6.130 | 2.4 |
| 5.0 | 7.822 | 5.247 | 2.6 |

**结论**

1. `lt=0` 时 oracle 增益**逐位为 0** → 复现 MF-MDP「1 步就够」的**适用区间**。
   这正面回应了 MF-MDP：其结论不是普遍定律，而是低长程耦合区的特例。
2. 有耦合时增益随 `lt` 增大、最优深度右移（H=2 → H=5）。
3. **learned 在 `lt=1.0/2.0` 出现负增益（exploit）**，`lt=3/5` 恢复为正。
4. ⚠️ 唯一可写的表述是「多步规划**条件性**成立」，不能写「多步总比单步好」；
   当 `lt=0` 时应诚实报告 `H*=0/1`。

## 3.3 Gate 3：动作抽象（本版重新解释）

等预算 512、depth 2、`div=0.8`、32 users。

**真值动力学（oracle）**

| `lt` | reactive | flat_uniform | flat_policy | sid_flat | **sid_hier** |
|---|---|---|---|---|---|
| 0.0 | −10.43 | 8.65 | 5.96 | 6.76 | **11.83** |
| 1.0 | −2.33 | 7.22 | 6.05 | 7.70 | **9.00** |
| 3.0 | −16.10 | −0.91 | −1.25 | −0.27 | **1.49** |
| 5.0 | −22.81 | −11.15 | −10.00 | −10.01 | **−7.67** |

**学习动力学（learned）**

| `lt` | reactive | **flat_uniform** | flat_policy | sid_flat | sid_hier |
|---|---|---|---|---|---|
| 0.0 | −7.65 | **−3.75** | −6.91 | −6.53 | −8.03 |
| 1.0 | −13.15 | **−6.01** | −14.41 | −10.14 | −12.92 |
| 3.0 | −16.09 | **−11.13** | −16.77 | −11.43 | −15.40 |
| 5.0 | −23.81 | **−18.40** | −19.25 | −19.25 | −21.84 |

### ⚠️ 解释修正：两个原因叠加，且第一个更根本

**原先的解释**（v1）：oracle 下 hierarchy 好、learned 下崩 → 「world model 语义质量不够」。

**修正后的解释**（v2）：

**oracle 下 `sid_hier` 全胜，部分是构造性的必然**：
1. **同源泄漏**：搜索树的 SID 就是 `(c1, c2)`，而 item embedding 正是 `A[c1] + B[c2]`
   —— 树的结构与环境的真实几何同源（见 2.5.1）。这必须用 purity sweep 排除。
2. **oracle 没有模型误差**，真值动力学当然能表达 pairwise div 项，所以层次搜索确实
   能利用组合结构。

### ⚠️ 解释修正（v3：由 P0a 与 purity sweeps 实测决定，v2 的归因被部分推翻）

**第一步（P0a）：reward head 的 pairwise 缺失是真的，但修好它只解决一半。**

诊断实验给出干净的 cross-over：`div=0` 时 unary 与 pairwise 两版持平（−0.009）；
`div=0.8` 时 unary 的预测误差是无 div 环境的 **3.8 倍**（0.303 vs 0.080），
加 pairwise 后降到 0.179（**−41%**）。planning 层面（3 个模型种子）：

| arm | reactive | flat_uniform | flat_policy | sid_flat | sid_hier |
|---|---|---|---|---|---|
| pairwise ON | −22.51 | **−7.75** | **−12.81** | **−8.22** | **−10.84** |
| pairwise OFF | −23.64 | −12.14 | −23.59 | −14.48 | −16.37 |
| 改善 | +1.1 | +4.39 | +10.78 | +6.26 | +5.53 |

`reactive` 几乎不动（+1.1）是关键证据：pairwise **不是让模型普遍变好，而是专门修复组合结构**
（`reactive` 只吃 unary 的 item score，若只是参数变多带来的普遍改善，它也该同步提升）。
但 `sid_hier − flat_uniform` 只从 −4.23 缩到 **−3.09**，**仍未转正**。

**第二步（purity sweeps）：v2 的「层次依赖语义」被推翻——但 v3 初稿的替代说法也是错的。**

在真值动力学下同时扫两个维度（budget 512、depth 2、16 users、`lt=3`、`div=0.8`）：

*Sweep A — 树的质量：*

| tree | purity | sid_flat | **sid_hier** | **hier − flat** |
|---|---|---|---|---|
| oracle | 1.000 | −0.225 | 1.474 | **+2.370** |
| noisy(p=0.50) | 0.590 | −0.372 | 1.496 | +2.392 |
| noisy(p=0.75) | 0.355 | −1.198 | 1.332 | +2.228 |
| **random** | **0.250** | −1.156 | **1.391** | **+2.287** |
| learned(env-emb) | 0.800 | −0.102 | 1.388 | +2.284 |

**`sid_hier` 的优势几乎与树的质量无关**：从完美树到完全随机树，
`hier − flat` 只从 +2.370 变到 +2.287（波动 0.08）。
而 `sid_flat`（无 refinement）确实随树质量下降（−0.225 → −1.156）。
→ **「层次搜索需要语义正确的树」被证伪**。

*Sweep B — 先验（`item_scores`）的质量：*

| prior noise | flat_uniform | flat_policy | sid_flat | **sid_hier** | hier−flat | **hier−policy** |
|---|---|---|---|---|---|---|
| 0 | −0.896 | −1.306 | −0.225 | 1.474 | +2.370 | **+2.780** |
| 0.25 | −0.896 | −5.182 | −0.638 | 1.294 | +2.191 | **+6.476** |
| 0.5 | −0.896 | −11.066 | −1.478 | 0.426 | +1.322 | **+11.492** |
| 1 | −0.896 | −18.221 | −3.741 | −1.709 | −0.813 | **+16.512** |
| 2 | −0.896 | −23.887 | −10.478 | −8.806 | −7.910 | **+15.081** |

⚠️ **读这张表的关键（也是本版修正的一处实质错误）**：
`flat_uniform` **恒为 −0.896**，因为它走 `ProposalPolicy(mode="uniform")`，
纯随机采样、**根本不读 `item_scores`** —— 它是**免疫基线**，先验好坏与它无关。
因此 `hier − flat`（取 `max(flat_uniform, flat_policy)`）在 Sweep B 里
**退化成 `sid_hier` 的常数平移，不提供任何相对信息**。
先验这一维上唯一可比的基线是 **`flat_policy`**（它和 `sid_hier` 一样吃 `scores`）。

**用正确基线（`hier − policy`）读出来的结论与初稿相反**：

> 先验退化时**所有 score-driven planner 都变差**，但**结构化的 `sid_hier` 变差得更慢**
> （Δ ≈ −10.3）**than `flat_policy`（Δ ≈ −22.6）**，于是它**相对于同类 flat 搜索的优势反而扩大**：
> `hier − policy` 从 **+2.780 升到 +16.512**。
>
> 也就是说：**结构化搜索在先验不可靠时更鲁棒，而不是更脆弱。**

另外，`sid_flat` 的 Δ（−10.25）与 `sid_hier` 的 Δ（−10.28）**几乎相同** ——
所以「脆弱点在 refinement」这个初稿说法**没有数据支持**（两者对先验退化的敏感度一样）。

**最终归因（v3 定稿）**

* `sid_hier` 的价值来自「**分组约束 + 组内细化**」，**不是**「语义层次」——
  树退化成随机依然有效（Sweep A）
* 它**不特别依赖先验**；恰恰相反，先验退化时它相对同类搜索的**优势扩大**（Sweep B）
* ⚠️ **learned 下 `flat_uniform` 领先的真正原因（v3 第三次修正）**：
  `flat_uniform` **不是「不用模型」**——它只是**候选池随机**，而**选择**仍然走
  `Lookahead` + `model.reward`。错误地把「候选池随机」读成「不用模型」，
  会得出「模型整体有害」这个**过强的结论**。

  P2 实测（learned、budget 512、depth 2、`lt=3`、`div=0.8`）：

  | planner | 用了什么 | return |
  |---|---|---|
  | `reactive` | 只用 `item_scores`、零搜索 | **−24.70** |
  | `sid_hier` | `item_scores` 生成 + `reward` 搜索 | −10.64 |
  | `flat_policy` | `item_scores` 生成 + `reward` 搜索 | −10.00 |
  | `flat_uniform` | **随机**生成 + `reward` 搜索 | **−7.29** |

  → **`item_scores` 已不可信（只用它最差），而 `model.reward` 仍然有用**。
  **搜索的价值恰恰在于用 `reward` 纠正 `scores` 的错误**——这也解释了
  为什么候选池的质量影响不大（搜索能纠正它），而 `reactive` 会崩。

**对 Innovation 2 的修正**：卖点应从「Semantic-ID 的语义」改为
「**结构化分组 + 组内细化**」，且它在先验退化时是**更鲁棒**而非更脆弱的一方。
真正的瓶颈是 **world model 的先验质量**（Innovation 1），而不是搜索结构（Innovation 2）。

## 3.4 Q3：树是否必要

oracle、`lt=3`、`div=0.8`、depth 2、pool 32、32 users，预算为**实测模型评估次数**：

| planner | return | evals/decision |
|---|---|---|
| reactive | −16.095 | 0 |
| shooting | −3.709 | 96 |
| beam | −2.919 | 416 |
| cem | −5.358 | 288 |
| **mcts** | **−2.512** | 1056 |

**结论**：MCTS 最优；CEM 最弱（每位置独立分布忽略 slate 的组合约束）。

⚠️ **但当前只能算初步证据**：MCTS 预算最大（1056），本地等预算对照（416 evals：
mcts −2.47 vs beam −2.79）方向正确但只有单点。正式结论需要
**budget curve**（见 P3），目标是证明「MCTS dominates the search-efficiency frontier」，
而不是「某个预算下 MCTS 数值最高」。

## 3.5 Gate 4：保守规划

### (a)(b) 受控偏置（本地，结论稳定）

| 注入 bias | no penalty (H=0) | unc-perfect (H=0) | 恢复 |
|---|---|---|---|
| 0.0 | −5.475 | −5.475 | — |
| 1.0 | −6.482 | −5.322 | +1.16 |
| 2.0 | −8.420 | **−4.884** | **+3.54** |
| 4.0 | −10.218 | **−4.539** | **+5.68** |

* **exploit 真实存在**：H=0 随 bias 单调恶化，`bias=4` 时曲线翻头（H=5 比 H=3 更差）。
* **完美信号下惩罚完全恢复损失**（回到无偏置水平）。
* **失校准信号（常量）逐位复现 no_penalty** → 惩罚的好坏完全取决于信号。

### (d)(e) 端到端（20000 ep、60 epoch、2 成员、32 users、warmup 5）

`error model` 驱动 U（ρ≈0.81）：

| config | H=0 | H=1 | H=2 | H=3 | H=5 | Δ(H0→H5) |
|---|---|---|---|---|---|---|
| oracle | −5.290 | −2.923 | −2.919 | −2.781 | −2.926 | **+2.364** |
| learned, no penalty | −21.508 | −23.147 | −22.189 | −21.835 | −22.208 | **−0.700** |
| learned, **unc penalty** | **−10.841** | −10.729 | −10.560 | −10.848 | −11.169 | −0.328 |
| learned, ood penalty | −21.508 | −19.476 | −20.612 | −19.640 | −21.266 | +0.243 |
| learned, both | **−10.841** | −10.750 | −10.905 | −10.819 | −11.239 | −0.398 |

`ensemble` 驱动 U（ρ≈0.06，**对照组**）：

| config | H=0 | H=1 | H=2 | H=3 | H=5 | Δ(H0→H5) |
|---|---|---|---|---|---|---|
| learned, no penalty | −19.168 | −20.236 | −20.141 | −20.544 | −19.441 | −0.273 |
| learned, unc penalty | −15.569 | −15.682 | −15.333 | −14.953 | −15.649 | −0.080 |
| learned, both | −15.569 | −14.318 | −14.325 | −13.772 | −14.292 | **+1.277** |

**结论**

1. **exploit 第二次独立复现**（Δ=−0.700；本地 −1.096）。
2. **强校准信号的绝对价值与 ρ 成比例**：error model +10.67，ensemble +3.60（信噪比 3×）。
3. **惩罚消除了 exploit 的形态**（H 曲线从下降变水平），**但没制造正收益**。
4. ⚠️ **这不是方法失败，而是 conservatism–planning trade-off**：惩罚越精确，
   越会同时压制多步规划可利用的空间（规划要选「模型认为收益更高」的动作，
   而那些恰是误差大的）。对照组反向印证：弱惩罚绝对性能差，但 `both` 的 Δ 转正。
   → **`λ*_return ≠ λ*_planning`**，必须扫 λ（P1a）。
5. **模型误差不随数据量改善**：5000→20000 episodes 仅让 mean abs error 从 0.34 降至 0.32
   → **瓶颈是模型结构/容量，不是数据量**。

## 3.6 结论汇总

| # | 结论 | 强度 | v2 修正 |
|---|---|---|---|
| 1 | `lt=0` 时多步规划严格无用；有耦合时增益随 `lt` 增长 | 强 | — |
| 2 | learned 下低 `lt` 出现负增益（exploit） | 强 | — |
| 3 | 层次动作空间在 oracle + 组合奖励下最优 | 强 | ⚠️ 需用 purity sweep 排除同源泄漏 |
| 4 | 层次结构在 learned 下失效 | 强 | ⚠️ 归因修正：首要是 reward head misspecification |
| 5 | MCTS 在组合奖励下胜过 beam / shooting / CEM | **中** | ⚠️ 需 budget curve 才能定论 |
| 6 | ensemble 分歧不可用作不确定性信号；异构修法无效 | 强 | — |
| 7 | 监督式 error model 达 ρ=0.81 | 强 | ⚠️ 仅限 logged 分布，OOD 待测 |
| 8 | 惩罚机制成立；绝对提升 +10.7，但 Δ 仍为负 | 强 | — |
| 9 | **λ 存在权衡**（return 与 planning gain 的最优值不同） | 中 | ⚠️ 待 λ sweep |
| 10 | 模型误差不随数据量改善 → 瓶颈是结构/容量 | 强 | — |
| 11 | **reward head 缺 pairwise 项**（结构缺陷） | 强（代码证实） | ⚠️ v2 新增 |
| 12 | pairwise 修复有效（组合项误差 **−41%**、planning **+4.4~10.8**），但**未翻转** hierarchy（−4.23→−3.09） | 强（3 个模型种子） | ⚠️ v3 |
| 13 | **`sid_hier` 的优势与树质量几乎无关**：purity 1.00 → 0.25 时 `hier−flat` 仅 +2.370 → +2.287 | 强（oracle 动力学） | ⚠️ v3，**与 v2 的解释相反** |
| 14 | **结构化搜索在先验退化时更鲁棒**：`hier−policy` 从 +2.780 **升到** +16.512（`flat_policy` 掉得比 `sid_hier` 快一倍） | 强（oracle 动力学） | ⚠️ v3（初稿方向写反，已修正） |
| 15 | 模型 `item_scores` 排序相关仅 **0.716**；learned 下**随机池优于任何 score-driven planner** → 模型整体已不可用 | 强（D2 + Gate 3 learned） | ⚠️ v3 |
| 16 | **`flat_uniform` 是免疫基线**（纯随机、不读 `item_scores`），不能用作「先验」维度的对照 | 强（测试锁定） | ⚠️ Review finding B1 |
| 17 | **`flat_uniform` 并非「不用模型」**：它只随机化候选池，选择仍走 `model.reward`。learned 下 `reactive`（只用 `item_scores`）−24.70 ≪ 所有 `Lookahead` 变体 ≈ −7~−10.6 | 强（P2） | ⚠️ v3 第三次修正 |
| 18 | **learned 下的主要变量是搜索预算，不是 penalty**：`--candidates 24` → −21.5，`--budget 512` → −10.0；λ 的边际效果仅 ±1 且不改变排序 | 强（P2） | ⚠️ v3 新增 |
| 19 | **score head 不是瓶颈，belief 才是**：head 见真实状态时 ρ=**0.993**，模型自身仅 0.720 → 误差 0.272 归 belief、**0.007** 归 head | 强（D5） | ⚠️ v3 新增 |
| 20 | **`hier−flat` 随 belief 精度单调上升，在 ρ≈0.93 处由负转正**（15 次运行：ρ<0.82 → 负；ρ>0.96 → +0.5~+1.4） | 强（B + B2） | ⚠️ v3 新增，**推翻此前"learned 杀死层次搜索"** |
| 21 | **决定性旋钮是监督强度而非容量**：`w_state` 1→10 提升稳态 ρ **+0.128**，`latent_dim` 64→256 仅 **+0.057**；且 `w_score`（无特权）与 `w_state` **等效** | 强（B + B2） | ⚠️ v3 新增 |

---

# 第四部分　关键发现与被推翻的假设

研究过程中有**六次自我纠错**，全部记录在案，因为它们构成方法本身的证据。

## 4.1 两个结论级翻转

**翻转一：Gate 3 的胜负（本地 → 大规模）**
本地小预算（budget 64–1024、8 users）得出「flat_policy 最优、层次无用」；
大规模（budget 512、32 users）得出「`sid_hier` 在 4 个 `lt` 上全胜」。
教训：**小规模 gate 的负面结论可能只是预算/样本不足**，最终结论必须在真实规模下复核。

**翻转二：同一 `sid_hier` 在 learned 下排名镜像**
这不是错误，而是最有价值的现象：它揭示了 Innovation 1 与 2 的耦合。
⚠️ v2 进一步把归因从「语义质量」精确到「reward head 函数形式」。

## 4.2 四个方法学问题（v2 新增第四个）

**① MCTS 的 zero leaf value（导致错误结论）**
第一版用硬 0 作叶子价值，而 `Lookahead` 在视界处用 `r.max()` bootstrap。
硬 0 系统性低估浅路径、把探索偏向「更深」。用同一 bootstrap 后，同一棵树在等预算下
从 **−4.621 变到 −2.466**（beam 为 −2.790）。
**教训：公平基线的纪律比 planner 本身更重要。**

**② ⚠️ reward head 的 misspecification（v2 新增，最重要）**
`MLP(h, slate_emb)` 版本（最早的实现）：与 `item_scores` 参数独立、排序不一致，
导致 planner 主动搜索高估区域（`Lookahead(H=0)` = −5.03，reactive greedy = −0.24）。
→ 改为 item 分解式后该问题消除。
**但它留下了一个更深的问题**：item 分解式是**纯 unary** 的，无法表示 pairwise div 项
（2.3.1）。这个缺陷直到被外部评审指出才被发现，且很可能就是 learned Gate 3 失效的原因。

**③ 异构 ensemble 的假设（证伪）**
直觉上「让成员多样化会改善分歧作为误差代理」，实测 **ρ=0.019 vs 同构 0.112**，反而更差。
原因是本质性的：**分歧度量的是「模型之间的差异」，不是「模型的对错」**。

**④ 数据驱动的设计修正**
`state_ood` 最初只在同一个 state 上探测，导致 std=0（常量）、尺度无意义；
改为在 32 个**多样**状态上探测后才有意义。同理 `BehaviorSupport` 的原始 NLL 无区分度
（std<0.02），必须做 z-score 归一化。

## 4.3 一条关于数据量的发现

**20000 episodes 相比 5000 只把误差从 0.34 降到 0.32**。
这直接改变了后续优先级：不是「跑更多数据」，而是「换更好/更合适的模型结构」。
如果不是刻意做了这个对比，很容易在「加数据」上浪费大量算力。

## 4.4 一条原则

所有上述修正都遵循同一个做法：**先设计一个可能推翻自己的对照实验，再让结果决定模型怎么改**。
最重要的例子就是 reward misspecification —— 它不是调参问题，而是**结构问题**，
必须改模型，不能靠调 `hidden_dim` 或 `λ` 掩盖。

---

# 第五部分　后续实验设计

执行顺序按**因果依赖**排列，而不是按重要性罗列。

## P0a　修复 reward head（最可能发生结论级翻转）

**做什么**：`predict_reward` 增加 pairwise 项

```
r̂ = Σ_k w_k u_θ(h, i_k)  +  Σ_{j<k} w_j w_k · e_{i_j}ᵀ W e_{i_k}      W ← (W+Wᵀ)/2
```

**连锁改动**：
* `item_scores` 只保留 unary 部分
* 测试 `test_reward_is_decomposable_into_item_scores` 改为「**unary 部分**一致」
* `_pair_w` 复用于 pair 权重

**为什么必须做**：这是唯一一个**代码可证实的结构性缺陷**，不是猜测；
它直接对应 learned Gate 3 的失效，且修复成本极低。
即使最后证明失效另有原因，这个缺陷也无论如何都该修。

**判据**：重跑 learned Gate 3，看 `sid_hier − flat_uniform` 是否从负转正。
**✅ 已实现并跑完（结论见 3.3）**

* 实现：`world_model.py` 增加 pairwise 项（`W` 常数、强制对称、复用 `pair_w` 的位置权重掩码），
  `predict_reward` 拆为 `predict_unary + pair_value`，`item_scores` 保持纯 unary。
  新增 `WorldModelConfig.pairwise` 开关用于消融。
* 诊断（`scripts/diagnose_pairwise.py`）：`div=0` 时两版持平（−0.009）；
  `div=0.8` 时 unary 误差 0.303 → pairwise **0.179（−41%）**。
* planning（3 个模型种子）：pairwise 稳定提升 **+4.4 ~ +10.8**，
  但 `sid_hier − flat_uniform` 只从 −4.23 缩到 **−3.09**，**未转正**。

**判据未满足**，但 P0b 给出了原因。

**消融**：`WorldModelConfig(pairwise=False)` 复现旧的 unary-only 模型，
有测试锁定（`test_pairwise_off_reproduces_the_unary_only_model`）。

## P0b　SID purity sweep（✅ 已实现并跑完，**结果推翻了本节的原判据**）

**做了什么**：把树参数化为 `SIDMap`（`muslate/planning/sid_map.py`，
oracle / noisy(p) / random / learned 四类），并新增 `NoisyScoreModel`
（`muslate/planning/perturb.py`）在**先验**侧注入固定的类别级偏置。
`scripts/run_purity.py` 在**真值动力学**下同时扫这两个维度。

**实测结果（budget 512、depth 2、16 users、`lt=3`、`div=0.8`）**

| 扫的维度 | 范围 | `hier − flat` 的变化 |
|---|---|---|
| **树纯度**（oracle → random） | purity 1.00 → 0.25 | +2.370 → +2.287（**波动 0.08**） |
| **先验质量**（noise 0 → 2） | — | +2.370 → **−7.910** |

**原判据不成立**：`purity ↑ ⇒ hierarchical gain ↑` 是**错的**——
树从完美退化到完全随机，`sid_hier` 的优势几乎不动。

**取而代之的实测规律**：

> **`sid_hier` 的价值来自「分组约束 + 组内细化」，不来自树的语义；
> 它唯一的脆弱点是先验（`item_scores`）的排序质量。**

**依赖关系已在实现中确认**：`oracle` / `random` 两端确实不依赖 P0a（本节先跑完），
而「先验质量」这一维才是决定性因素。

## P1　λ sweep + error model 的 OOD 校准 + 覆盖度

### P1a　λ sweep（不需要重训任何模型）

**做什么**：`λ ∈ {0, .1, .3, 1, 3}` × 两种 U 来源，画**双纵轴**：
左轴 absolute return，右轴 `Δ_planning = R(H=5) − R(H=0)`。
若出现峰值（例如 `λ*≈0.3`），再局部细扫 `{.15,.2,.3,.4,.5}`。

**假设**：存在中间 λ 同时满足「Δ > 0」且「绝对 return 优于 no-penalty」。
**关键预期**：`λ*_return ≠ λ*_planning` —— 这本身就是一个好的分析结果，
说明「最大化性能」与「让规划体现价值」是两个不同目标。

### P1b　calibration-by-shift

**做什么**：按 2.6.1 的五种分布测 error model 的 ρ。
**判据**：若 `ρ_planner ≪ ρ_log`，说明惩罚到真实部署仍有问题；
**先只测不修**，衰减曲线本身就是结果。

### P1c　日志覆盖度旋钮

**做什么**：`ε ∈ {.05,.1,.25,.5,.8}`，研究
`coverage ↑ ⇒ world model error ↓ ⇒ planning gain ↑`。
建立 `dataset coverage → model reliability → safe planning depth` 的链条。

## P2　闭环：保守的层次搜索（✅ 已跑完，**判据未达成但发现了更重要的东西**）

**做了什么**：给 `run_gate3.py` 加了 `--lambda-unc` / `--err-episodes` / `--err-epochs`，
使保守 penalty 能挂在**每一个** action abstraction 上（`adjust` 钩子对
`reactive` / `flat_*` / `sid_*` 一视同仁）。

**实测**（learned、`lt=3`、`div=0.8`、budget 512、depth 2、3000 ep / 25 epoch、16 users）：

| λ | reactive | flat_uniform | flat_policy | sid_flat | sid_hier |
|---|---|---|---|---|---|
| 0.0 | −24.704 | **−7.292** | −10.000 | −8.194 | −10.641 |
| 0.3 | −24.704 | **−7.094** | −9.013 | −9.357 | −10.342 |
| 1.0 | −24.704 | **−7.322** | −9.900 | −8.937 | −10.780 |

**三条结论**

1. **判据未达成**：penalty 的边际效果只有 ±1，且**完全不改变 planner 排序**；
   `sid_hier` 始终没能超过 `flat_uniform`。
2. **penalty 不是这里的主要变量，搜索预算才是**：Gate 4 用同一环境但
   `--candidates 24`，`learned/none` 是 −21.5；P2 的 λ=0 用 `--budget 512`
   已经是 −10.0。**把预算提上去的效果（+11）远大于任何 λ（±1）**。
3. **`reactive` 在三个 λ 下完全相同**（−24.704）——它只用 `item_scores` 排序、
   从不评估候选价值，所以 `adjust` 对它无效。这既验证了 penalty 只作用于**搜索**，
   也再次确认 `item_scores` 是 learned 下的瓶颈（它最差）。

**意义（最终版本，经 B 修正）**：penalty 不是正确的手段，**belief 精度才是**。

---

## B　提升 `item_scores` 精度（✅ 已跑完，**原思路复活**）

**动机**：P2 判定瓶颈在 `item_scores`。但"精度不够"有**两种完全不同的原因**，
修法相反，所以先用一个上界实验分开它们。

**步骤 1：D5 定位误差**（`diagnose_semantics.py::d5_head_ceiling`）

训练 `MLP(true_state) → true_item_scores`，与模型自身精度对比：

| 量 | ρ |
|---|---|
| head 看到**真实状态**（上界） | **0.993** |
| 模型自身的 `item_scores`（D2） | 0.720 |
| → 归因于 **belief** | **0.272** |
| → 归因于 **head / 目标函数** | **0.007** |

**score head 几乎不是问题**。加容量、换损失都是白费——这正否定了 P1 原本的设想。

**步骤 2：进一步分解**（D2 新增 `spearman_by_t`）

```
ρ(t):  t0=0.010   t3=0.837  t6=0.845  t9=0.842  t12=0.841  t15=0.838  t18=0.830
```

两件事性质不同：**冷启动**（t=0 信息论上不可知，不可修）
与 **稳态平台 0.84 vs 上界 0.99**（encoder 极限，**可修**）。

**步骤 3：扫描**（服务器 RTX 5090，`oracle_state=True`）

| arm | D2 steady | flat_uniform | sid_hier | **hier−flat** |
|---|---|---|---|---|
| ld64_ws1 | 0.8377 | −3.568 | −5.289 | **−1.722** |
| ld128_ws1 | 0.8594 | −3.983 | −4.196 | **−0.212** |
| ld256_ws1 | 0.8946 | −3.504 | −3.865 | **−0.361** |
| **ld64_ws10** | **0.9657** | −1.830 | −0.615 | **+1.215** |
| **ld128_ws10** | **0.9709** | −1.634 | −0.696 | **+0.938** |
| **ld256_ws10** | **0.9766** | −1.621 | −0.526 | **+1.096** |

**步骤 4：对照 —— 去掉特权状态监督**（`--no-oracle-state`，改扫 `w_score`）

| arm | D2 steady | **hier−flat** |
|---|---|---|
| ld64_score1 | 0.8121 | **−1.656** |
| ld128_score1 | 0.7841 | **−1.296** |
| ld256_score1 | 0.8073 | **−1.555** |
| **ld64_score10** | **0.9649** | **+1.015** |
| **ld128_score10** | **0.9657** | **+0.501** |
| **ld256_score10** | **0.9740** | **+0.846** |
| **ld64_score100** | **0.9868** | **+1.136** |
| **ld128_score100** | **0.9875** | **+1.429** |
| **ld256_score100** | **0.9898** | **+1.345** |

**合并 B + B2 共 15 次运行，关系干净且单调**：

```
ρ_steady < 0.82   →   hier−flat 为负（−1.3 … −1.7）
ρ_steady > 0.96   →   hier−flat 为正（+0.5 … +1.4）
翻转点约在 ρ ≈ 0.93
```

**四条结论**

1. **判据达成**：`hier−flat` 随 belief 精度**单调上升**，在 ρ≈0.93 处转正。
2. **决定性旋钮是监督强度，不是容量**：`w_state` 1→10 把稳态 ρ 提高 **+0.128**
   （0.838 → 0.966）；`latent_dim` 64→256 只提高 **+0.057**。
3. **两条监督路径等效**：`w_state`（有特权状态）与 `w_score`（**无特权**，且对真实
   系统现实可行——交互日志里 item 级反馈可观测）。两者都把 ρ 推到 ~0.97 并让
   `hier−flat` 转正。
4. **⚠️ 修正我此前的结论**：「learned 动力学杀死层次搜索」是在 `w_state=1.0`
   下测的，即 belief 精度只有 ≈0.84，**低于层次搜索需要的 ≈0.93**。
   **方法没有被否定，而是被跑在了工作区间之外**——而这个工作区间在 D5 之前
   从未被测量过。

**诚实的边界**：上界 0.99、翻转点 ~0.93，意味着**该方法要求 belief 精度很高**。
synthetic 环境提供了特权监督，生产日志未必有。**这个 headroom 就是真实数据上
唯一未解的问题**——也是"原思路能否成立"最终取决于的事。

## P3　统计强度与模型结构

* **budget curve**：`B ∈ {64,128,256,512,1024}`，每个 planner 一条
  `return vs 实际模型评估次数` 曲线，目标是证明
  **MCTS dominates the search-efficiency frontier**。
* **multi-seed**：synthetic 上 10 seeds（最低 5），主表报 `mean ± std` 与 95% CI；
  同一初始用户/环境下可做 paired Wilcoxon / paired bootstrap。
  ⚠️ 没有多 seed，Gate 3 与 Q3 的**排序结论不能作为最终结论**。
* **模型结构/容量二维消融**（不只是扩大 `hidden_dim`）：
  M0 GRU+unary → M1 GRU+unary+pairwise → M2 Transformer+pairwise →
  M3 +multi-step latent consistency → M4 可选 RSSM。
  先看 `M1−M0` 有多大；若很大，说明真正的瓶颈是 reward misspecification 而非 `latent_dim`。

## P4　真实数据（Gate 1）与 phase diagram 的外部验证

**路径**：`Synthetic → KuaiSim → KuaiRand / OPE`。

* **KuaiSim**：构造 request-level slate transition，做半真实验证
* **KuaiRand**：公开数据主要是 sequential exposure logs，适合 offline representation
  与 OPE，并非天然是完整的 request-level slate MDP

⚠️ **最关键的新贡献点**：给真实数据定义 `lt` / `div` 的**可估计 proxy**，
使 phase diagram 可以外部验证：

| Proxy | 定义 |
|---|---|
| `C_LT`（long-term coupling） | history-aware 模型相对 myopic 模型的**预测增益**：用完整曝光历史 vs 只用当前 user embedding 预测 `r_{t+1:t+H}`，两者 cross-validation 的 predictive gain |
| `C_DIV`（slate interaction） | `R²_interaction − R²_additive`：拟合 `r(A)=Σu(i)` 与 `r(A)=Σu(i)+Σ_{i<j}v(i,j)`，两者 R² 之差 |

若在真实数据上估计出的 `(C_LT, C_DIV)` 落在 phase diagram 的某区域，
且实际 planner 胜负与预测一致 —— 这将是最有说服力的验证。

⚠️ **但这两个 proxy 必须先在半合成数据上验证有效性**
（proxy 高 ⇔ 真实 `lt`/`div` 高），否则在真实数据上估出来也无法解释。

---

# 附录　工程与复现

* 服务器：`/home/xys/code/WZL/next`，conda 环境 `WZLcosmos`（torch 2.11.0+cu128，支持 sm_120）
* 批量运行：`CONDA_ENV=WZLcosmos bash scripts/server_run_all.sh`
  （环境变量 `EPISODES` / `EPOCHS` / `ENSEMBLE` / `CHECK_ONLY=1`）
* 测试：`python tests/test_*.py`（44 个，零外部依赖，可直接运行）
* 建议以 `python -u` 运行以获得实时日志（默认块缓冲会让日志延迟出现）
* 修复过的调度 bug：`$(gpu_for)` 的命令替换在子 shell 中递增，导致 6 个任务
  全部落在 GPU 0；已改为在函数内递增全局变量
* ⚠️ **待补的测试**：reward head 加 pairwise 后需新增
  「pairwise 项对称」「unary 部分仍与 item_scores 一致」两条断言

---

## 附：外部文献核验状态

⚠️ 本版采纳的技术结论**不依赖**外部文献（reward head misspecification、同源泄漏、
λ 权衡、OOD 校准风险均已用代码或数据在本仓库内证实）。但下列由外部评审引入的引用
**尚未在本环境核实**，在写入论文前需要确认其存在与结论：

* MF-MDP「one-step lookahead captures most of the gain」
* UA-TWM、HiGR、OGR、SAPIENT 的具体覆盖范围
* 用于支撑「conservatism–planning trade-off 有文献背书」与「ensemble uncertainty 失败与
  文献一致」的 offline MBRL 工作（MOBILE / SUMO / ICML 2026 相关）

这些属于**加分项而非常要条件**：即使不能核实，本文的技术结论依然成立。


## 2026-10-08：严格日志首轮实施（运行中）

根据用户批准，已新增严格可观测日志和曝光位置masked loss，L0/L1训练器不接收simulator或states；U1仍是全item真值的特权对照。修正尾batch、小样本训练，新增公共历史前缀、原始回报/checkpoint和恢复身份检查。已有源码/结果保留，本轮在远程独立源码目录 `results/e1_workspace_20261008_v2` 运行。

核心96项测试本地/远程通过；CPU/GPU微型实验通过。默认L1_s1_w10全流程耗时约317.7秒，其余20模型六卡排队运行。当前不据单模型作成功结论。验证集选择、层次配对bootstrap与至少5seed确认规则见 `docs/research/2026-10-08-e1-protocol.md`。名义B512/H2的实际slate调用为6656/决策，硬总预算仍待E4。本轮逐项反馈为连续合成响应，尚非真实点击实验。


### 首轮完成（2026-10-08）

21个pilot与6个确认模型均运行完成，主条件L0/L1/U1各5个独立seed。全流程151,200次闭环决策，103项测试通过，27checkpoint/训练历史/配置/source与paired data已终检，归档SHA256校验通过。验证集冻结选择L1/U1权重100。

五seed结果：SID-hier的L1–L0完整回报差8.0113，95%区间[7.2539,8.6063]；L1的hier–uniform差0.5302，[0.1947,0.8211]，公共前缀对应差0.2429，区间跨0；hier–sid_flat两协议均无明确正增益。真实点击/硬总预算/跨目录泛化仍未完成。

报告、图与全部模型原始归档见 `results/e1_paper_results_20261008/`；结论及下一轮控制安排见 `docs/research/2026-10-08-e1-findings-next.md`。KuaiRand-1K/Pure下载、校验和六表逐行审计已完成；不能把数据准备称为真实闭环验证。
