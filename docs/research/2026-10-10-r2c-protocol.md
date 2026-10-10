# R2-C 冻结诊断协议（2026-10-10）

本研究是固定动作序列上的预测诊断，不是可执行的 teacher-forcing 前瞻策略；不能把两种模式差异归因于单一 transition 模块。历史编码器、reward head 与 dynamics 的联合失配均可能产生差异。不得据此声称因果机制、规划性能改善或等计算公平性。

## 预先固定的设计

- 使用原 E1 的 L0/L1、训练 seed 1–5 共十个不可修改 checkpoint。原环境、16 个已有用户（ID 0–15），五步行为前缀。新前缀种子 `31000000+2000*seed`；两条件共享前缀及真状态。
- 在任何模型预测之前冻结所有 bank；bank 生成不接收模型或 condition。每个初始状态均匀抽样 64 条合法有序、无重复 item 的长度 5 动作序列；horizon 1/3/5 使用同一序列的嵌套前缀。
- 每条序列的每个动作前真状态额外固定 16 个均匀合法 probe slates。相同 reached state 的这 16 个 slate 由两模式共同预测，并用同一 evaluator 真奖励评分。
- 模拟器仅作为 evaluator。保存初始真状态、完整前缀历史、序列、逐步真奖励、reached-state hash、probe slate 与 probe 真奖励。保存 bank 内容 hash；不能把真状态输入 learned model。
- Open loop：编码五步前缀一次；后续仅调用 learned transitions。Teacher forcing：第 t 步编码前缀及严格先于 t 的动作/观测奖励，再预测第 t 步 reward；不输入当前或未来奖励。两模式使用完全相同动作和 evaluator outcomes。

## 预声明比较与统计

逐步 reward MAE；各 horizon 的局部折扣回报 MAE（`sum gamma**t*r_t`，t 从 0 开始）；每初始状态内 64 条序列的 Spearman 排序相关。相同数值用平均秩；任一向量恒定时相关记为 JSON null，报告有效数与 undefined 数，绝不填零。

对于各 horizon，按 open-loop 预测回报最大化选一个序列（并列取最小 bank index），两模式都报告该同一序列上的预测减真回报；不混入 teacher-forcing 自选最大值。所有状态都保留选择 index 和原始预测奖励以支持复核。

Probe 报告相同状态/slate 上的两模式预测绝对差、各自对真 reward 的 MAE，按步及整体汇总。Probe 是函数性偏离指标；不使用未校准 latent 欧氏距离。

各模型先在 16 用户/64 序列上平均；训练 seed 是独立重复单位。L0、L1 分别对各 horizon 的 teacher-forcing 减 open-loop 回报 MAE 差值及对应 reward MAE 差值，报告五个 seed 原值、均值和 95% paired t CI（df=4、临界值 2.7764451051977987）。用户/序列不充当训练重复数。结论保持探索性，不增加事后选择的主要检验。

## 可复核冻结与执行

源代码、协议与测试先经测试和独立 review，再由协调者提交。运行时显式传入该 source commit；执行使用此 commit 的 archive，不能依赖可变工作目录。先写不可覆盖的 `protocol.json`，绑定 source commit、所有 Python 源文件/本协议/测试 hash、十个 checkpoint 路径/hash/身份/环境、实际启动命令与 runtime。之后生成五个 bank，完成 `bank_manifest.json` 后才允许模型预测。新目录 `/home/xys/code/WZL/next/results/r2c_20261010`，不得覆盖旧研究结果。只使用 GPU 3、4，一卡一个 seed job，各条件串行；batch 64 条序列。

Smoke 是独立新目录、seed 1、一个用户，不能计入完整结果。完整 run 须十个模型、每模型 16 用户、每用户 64 序列/五步/16 probes 完整；所有数值有限，只有恒定秩相关允许 null。审计必须校验文件名/condition/seed/arm 标签，checkpoint hash、source/protocol/bank hash，重建 bank 和真 outcomes，重算指标/同序列 selection、runtime/命令记录。只有 `verified_manifest.json` 绑定全输入 hash 后生成中文结果报告；之后任一输入变化均拒绝出报告。

## 完成标准与后续

所有十模型完整、针对 horizon=1 的模式相等、teacher-forcing chronology 扰动、bank 确定性/合法性/嵌套前缀、折扣计算与审计负向例均通过。保留 raw 以支持独立重算。先判断误差是否随 horizon 增长、同序列 teacher forcing 是否改善，以及 probe 函数偏离是否同向；下一阶段若需要机制归因，另行预注册 encoder/reward/dynamics 的受控消融，不能从本诊断直接推断。
