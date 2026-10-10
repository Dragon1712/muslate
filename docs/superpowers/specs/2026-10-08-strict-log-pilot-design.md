# E0/E1 严格离线日志监督与 pilot 设计

日期：2026-10-08。研究背景与总体矩阵见 `EXPERIMENT_PLAN.md`。

目标：在不读取真实状态和未曝光 item 真值分数的训练条件下，比较只用整体 reward 与增加已展示 item 反馈的模型，保留现有 B2 为特权上界。先完成这一个证据缺口，后续 phase diagram、架构升级和保守模块独立开展。

## 数据与模型接口

保留旧 `OfflineLog`、旧训练接口和默认行为，避免已有结果失去复现路径。新增无 `states` 字段的 `ObservedLog`，仅包含 episode/step IDs、slates、rewards、可选 item_feedback 和数据维度 metadata；逐项反馈数组形状 `[T,K]`，缺测值为 NaN，掩码仅由有限值产生。已展示且未点击是观测0，不是缺测；未曝光 item 不创建标签。

合成生成器为实际曝光 item 生成逐项连续反馈 `unary_score + noise`；噪声默认std=.02，使用独立随机流，避免增加反馈改变原有 slate/reward/state 序列。模拟器负责响应生成、允许用真实状态；训练器只接导出的 ObservedLog。诊断模块可单独保留 OfflineLog，但不能传进严格训练器。

第一轮保持 GRU 的输入为历史 slate 和整体 reward，让 item 反馈只进入辅助损失，以隔离监督效果。后续单独比较将逐项反馈加入历史输入。连续反馈实验是受控机制验证，不能称为已完成真实点击日志验证。

严格训练入口 `train_logged_world_model(log, cfg, device, ...)` 不接受 simulator/state 目标。配置显式关闭 oracle_state；设置 `score_supervision='none'|'exposed'|'oracle_full'`，旧路径默认保留 oracle_full。严格路径拒绝 oracle_state=True/oracle_full，暴露缺测反馈时只对有限的曝光位置计算 MSE，全缺测时该项为0。使用 item_value/state→exposed IDs 而非全 item 真值目标。

公共训练循环共享 reward、多步训练、优化器及梯度裁剪；私有 tensor fitter 接收可选已观测目标/显式特权目标，两个外层接口分别负责信息审计。L0 与 L1 不分配真值 state/score tensor；U1 通过旧接口计算上界。oracle 标签函数保持只在该路径中调用。

batch_size 大于 episode 数时保证仍有一个有效 batch；批次尾部不足时仍覆盖全部训练 episode。保持模型 `true_state_dim` 仅在有state head时需要的行为。

## 评估与复现

完整闭环保留旧接口与结果。新增公共5步行为前缀：相同用户、初始状态、日志策略随机流下生成一次 history，clone state，随后交给不同 planner；折扣统一从episode起点计算。per-user 完整回报、后缀回报均保存。该协议与旧 `tail_from` 的独立历史协议分开标注。

模型 checkpoint 保存 config、state_dict、n_items、slate_size、true_state_dim，以及数据与seed metadata。测量 GPU 时间前后 synchronize；记录训练时间和评估延迟，计算评估期间已存在的 reward/slate counter。更完整的硬预算接口在 E4 实现，本 pilot 不声称已完成全操作计量。

所有结果写入新的 run 目录，拒绝覆盖同名完成任务；中断时只以原子写出的模型/结果状态恢复，遇到配置或代码hash不同不复用。

本地/远程无Git，实施前对 `muslate/ scripts/ tests/ requirements.txt` 生成只含源码的快照与SHA256。同步本次修改前比对hash；远程文件若已变化，停止覆盖并报告差异。

## 第一轮矩阵与约束

固定 `lt=3, div=.8, latent_dim=64, pairwise=True, ε=.25, episodes=3000, epochs=40, B=512, H=2`。L0 一个配置、L1权重1/10/100、U1权重1/10/100，3个独立seed，共21模型。每个seed内共用train/validation/test划分，训练、日志、planner seeds显式写出。

先CPU微型测试；再GPU0单模型小规模计时/检查；通过后运行21个pilot。每模型复用checkpoint评估五种proposal，不逐planner重训。主条件是L1，U1始终标注特权上界。真实数据目录确认后另制定对应反馈link/位置/缺测模型，不能直接套用连续监督损失。

验收：严格路径可在没有states属性的日志上完成训练；篡改未提供真值字段不影响训练；曝光掩码正确、当前/未来反馈不进入当前belief；已有72项测试仍通过；公共前缀一致；pilot状态和原始结果可恢复与追溯。

本设计与实施计划需经用户审阅后实施。已完成的只读服务器检查与旧测试不代表新接口已实现。
