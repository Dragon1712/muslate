# MuSlate 第一阶段修复与重新评估 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 实施用户修改方案第一阶段的五项修改，重建合法、有限时域的 E1 验证和新测试结果。
**Architecture:** 保持 WorldModel 架构、训练损失和 checkpoint 参数不变。统一列表合法性与 planner 剩余时域；细分模型调用、提案成本和延迟，以校准系数记录估计开销。采用新的评估入口，显式记录旧 checkpoint 来源与新评估源码；先重新验证全部21模型选参，再做新测试确认。
**Tech Stack:** Python、NumPy、PyTorch；WZLcosmos / SSH。
**Spec:** 用户 2026-10-09 提供的《MuSlate 具体修改方案》第二节“第一阶段”，以及 docs/research/2026-10-08-code-logic-audit.md。

## Global Constraints

- 第一阶段仅修改1–5；不改GRU输入/世界模型训练，不实施Adaptive或ensemble坐标改造。
- `depth=0`包含当前一步；`effective_depth=min(configured_depth,remaining_steps-1)`。
- 列表内item不重复，整数索引、K长度和范围一致；候选池重复合法列表去重，不用重复填充伪造有效候选数。
- 旧27模型与结果不可覆盖，服务器原始项目与其他进程不可覆盖/停止。
- 重新选参使用原pilot21checkpoint及新验证轨迹；确认使用从未用于选择的新test轨迹，固定catalog仍须披露。
- 本地不是Git仓库，保留源zip/hash与独立远程workspace；不创建虚假提交。
- `PlanningCost`细分reward-only、joint transition、score、proposal candidates、distinct slates、proposal/总延迟；权重来自实际运行校准，不默认全1。
- 硬预算针对明确的校准估计成本，不承诺硬墙钟；不足时在相应调用前抛出可识别的异常，不能静默超额。

## Review Focus

1. 分组容量不足、小catalog=K和候选池缩短，所有执行动作仍合法。
2. 所有五类planner末端不bootstrap到终点以后；保留未知剩余时域的旧调用兼容。
3. proposal/refinement/bootstrap/可选uncertainty查询不逃逸计量，嵌套计时不重复计算proposal成本。
4. validation-only阶段不计算最终test、不默认旧选择100；旧训练与新评估provenance独立。
5. cached mean、tail、case identity、checkpoint实际配置与阶段产物一致；中断恢复不接受缺失依赖。

## Task 1: 合法动作与统一奖励

Files: 新增 `muslate/planning/validity.py`；修改 `env/simulator.py`、`planning/abstractions.py`、`proposal.py`、`model/train.py`；新增 `tests/test_slate_validity.py`、`test_reward_consistency.py`。
- [x] 写refinement重复反例、组容量不足、完整catalog替换、非法batch和scalar/batch测试。
- [x] 运行并记录RED。
- [x] 实现共享验证、合法替换、容量约束、池去重与统一奖励；不在执行后改动作。
- [x] 新测试与完整现有suite GREEN，记录源快照/ledger。

## Task 2: 所有planner有限时域

Files: `planning/search.py`、`eval/rollout.py`；新增 `tests/test_finite_horizon.py`。
Interface: `plan(states,remaining_steps=None,budget=None)`；剩余时域为已知正整数；旧调用None保持配置深度。
- [x] 写terminal奖励陷阱、2步最多两奖励、所有planner时域与MCTS重复叶反例，运行RED。
- [x] depth作为局部参数贯穿Lookahead/Shooting/CEM/MCTS；终点continuation零；修复重访叶bootstrap作为同一末端逻辑的一部分。
- [x] evaluator逐决策传horizon-state.t；所有新/旧测试GREEN。

## Task 3: 计量、校准与预算

Files: 新增 `planning/cost.py`、`scripts/calibrate_planning_cost.py`；修改 `latent_model.py`、`proposal.py`、`abstractions.py`、`search.py`；新增 `tests/test_planning_budget.py`。
Interfaces: `PlanningCost`、`CostWeights`、`PlanningBudget`、`BudgetExceeded`；`CountingModel`每决策记录，调用前预算收费。
- [x] 写reward/step/score分开计数、候选去重统计、非单位系数、硬预算拒绝、proposal与bootstrap计量测试，运行RED。
- [x] 实现分开计数和计时；uncertainty按实际成员reward调用计量；生成候选按尝试数收费，计量唯一列表数；无隐藏未计数fallback。
- [x] 保存实际硬件校准、参考batch和误差范围；报告估计compute和真实latency，不把线性估计说成FLOPs或硬墙钟。
- [x] GREEN和小规模端到端预算测试。

## Task 4: 结果完整性与两阶段选参

Files: 新增 `scripts/reevaluate_stage1.py`；修改 `analyze_logged_pilot.py`；新增 `tests/test_result_integrity.py`、`test_stage1_reevaluation.py`。
- [x] 写cached mean mismatch、缺split、旧checkpoint来源、验证阶段禁止test、新eval seed不重叠测试，运行RED。
- [x] 统一schema检验，从returns重算选参；检查tail/prefix/finite/decision counts与协议身份。
- [x] 独立新入口支持validation/test两阶段、旧ckpt+新source身份、动作日志、每决策cost、原始case IDs和CPU/GPU时域验证；最终报告包含校准与数据来源。
- [x] 验证选择先冻结；确认缺少的seed模型只按旧训练协议补训并保存新训练provenance。
- [x] 新/旧测试GREEN。

## Task 5: 独立审查、部署与正式运行

- [x] 最后一位独立只读reviewer审查全部修改；Important一次修复，RED→GREEN。
- [x] 本地与远程完整suite、7条件smoke、旧重复动作重放、小catalog穷举末端测试。
- [x] 在独立20261009目录部署，先21validation、冻结选择，补齐seeds4/5，再新test15条件模型；保存逐步动作/cost、校准、源码hash与bootstrap。所选seeds4/5模型均已有，直接复用，无需补训。
- [x] 下载结果、核对hash/合法动作/有效深度/表图、写修复版报告与阶段结论。117,600决策重放、209文件归档、离线汇总一致；主报告为docs/research/2026-10-09-stage1-results.md。
- [x] 记录KuaiRand-27K目录与schema：本轮仅文件/表头审计，不假称全文校验或真实实验完成。
