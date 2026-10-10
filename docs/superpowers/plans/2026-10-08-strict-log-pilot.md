# Strict Logged-Feedback Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. 本次建议当前会话直接实施；不主动派生子代理。

**Goal:** 完成 E0/E1 的无特权日志训练、配对暖启动和21模型pilot入口。

**Architecture:** 在旧合成上界接口旁增加 ObservedLog 与严格训练接口；共享训练循环，信息选择发生在外层。首轮不改 GRU 输入，只用已曝光连续反馈辅助监督；训练checkpoint复用于多planner评估。

**Tech Stack:** Python 3.10.22、NumPy、PyTorch 2.11.0+cu128；远程 WZLcosmos，独立运行的现有 test_*.py 风格。

**Spec:** `docs/superpowers/specs/2026-10-08-strict-log-pilot-design.md`

## Global Constraints

- 保留旧 `OfflineLog`、旧训练接口和默认行为，避免已有结果失去复现路径。
- 第一轮保持 GRU 的输入为历史 slate 和整体 reward，让 item 反馈只进入辅助损失，以隔离监督效果。
- 严格训练入口不接受 simulator/state 目标。
- 固定 `lt=3, div=.8, latent_dim=64, pairwise=True, ε=.25, episodes=3000, epochs=40, B=512, H=2`。
- 所有结果写入新的 run 目录，拒绝覆盖同名完成任务。
- 本地/远程无Git，实施前对 `muslate/ scripts/ tests/ requirements.txt` 生成只含源码的快照与SHA256。

## Review Focus

- 全缺测逐项反馈：score loss有限且为0，不产生NaN梯度。
- 非点击和缺测：0保留为有效监督，NaN被排除，未曝光item无目标。
- 小样本与尾batch：episode数小于batch_size仍训练，尾部不丢样本。
- 历史时间边界：当前/未来reward与item反馈不得进入当前belief。
- 恢复/同步冲突：hash或完整配置不符时拒绝覆盖/复用。

## Task 1: 可观测日志与反馈生成

**Files:** Modify `muslate/env/logger.py`, `muslate/config.py`; Create `tests/test_observed_log.py`.

**Interfaces:** 新增 `ObservedLog` dataclass 与 `.save(path) / .load(path)`；新增 `generate_observed_log(sim: SlateSim, cfg: LogConfig) -> ObservedLog`；新增 `as_observed_log(log: OfflineLog, item_feedback: np.ndarray | None = None) -> ObservedLog`。生成器导出schema不含states；旧日志保持兼容。

- [x] 写 `test_observed_log_has_no_state_channel`：`not hasattr(log,'states')`，反馈 `[T,K]` 与 slate 对齐。
- [x] 写 `test_feedback_rng_does_not_change_logged_actions_or_rewards`：相同seed新/旧日志的slates、rewards完全相同。
- [x] 写 `test_round_trip_preserves_zero_and_missing_feedback`：保存/加载后0仍为0，NaN位置不变，无states键。
- [x] 在远程运行新测试，确认因缺少接口失败；不要把依赖或语法错误当成RED。
- [x] 实现schema/序列化及独立反馈随机流；连续观测noise std=.02，只返回已展示item响应。
- [x] 重跑新测试与旧8文件suite；记录所有退出码。

## Task 2: 无特权训练与监督模式

**Files:** Modify `muslate/model/train.py`, `muslate/config.py`; Create `tests/test_logged_training.py`.

**Interfaces:** `train_logged_world_model(log: ObservedLog, cfg: WorldModelConfig, device: torch.device, verbose: bool = True) -> WorldModel`；配置 `score_supervision: str` 默认 `oracle_full`。共享优化入口 `_fit_world_model(model: WorldModel, slates: torch.Tensor, rewards: torch.Tensor, cfg: WorldModelConfig, *, exposed_feedback: torch.Tensor | None = None, oracle_states: torch.Tensor | None = None, oracle_scores: torch.Tensor | None = None, verbose: bool = True) -> WorldModel`。曝光损失 `_exposed_feedback_loss(model: WorldModel, h: torch.Tensor, slates: torch.Tensor, target: torch.Tensor) -> torch.Tensor`，返回有限曝光位置平均MSE；全缺测返回可回传的0。旧 `train_world_model(sim, log, cfg, device, verbose)` 继续实现U1。

- [x] 写 `test_strict_training_needs_no_simulator_or_states`：两个episode/两步的ObservedLog完成一次优化，loss与参数有限。
- [x] 写 `test_strict_training_rejects_privileged_config`：oracle_state=True或oracle_full均ValueError。
- [x] 写 `test_exposed_loss_uses_only_finite_positions`：对finite曝光target计算人工MSE，0参与、NaN不参与；全NaN loss=0且梯度有限。
- [x] 写 `test_strict_training_never_calls_true_item_scores`：仅对该特权函数设置调用即抛错的哨兵，严格训练仍成功；其余模型和优化器使用真实代码。
- [x] 写 `test_small_and_partial_batches_are_trained`：episode数1与batch_size+1均有更新，所有episode进入训练，不丢尾部。
- [x] 写 `test_current_and_future_feedback_do_not_change_current_belief`：改变t及以后reward，`states_seq[:,:t+1]`保持一致；逐项反馈不进入第一轮encoder。
- [x] 运行并确认RED；实现严格外层、shared fitter与exposed loss；L0不创建score标签，L1不创建oracle state/full-score标签。
- [x] 运行新测试与全suite；确认旧上界训练仍可运行。记录兼容性改动与tiny-run配置。

## Task 3: 公共历史暖启动与原始结果

**Files:** Modify `muslate/eval/rollout.py`; Create `tests/test_warm_start.py`.

**Interfaces:** 新增 `WarmStartCase`（克隆前缀状态、slate/reward history、user_id、episode_seed）；`build_warm_start_cases(sim, n_users, warmup=5, seed=0) -> list[WarmStartCase]`；`evaluate_from_prefixes(sim, planner, latent_fn, cases, measure_latency=True) -> EvalResult`。保留原evaluate语义。

- [x] 写 `test_planners_receive_identical_cloned_prefixes`：两个planner看到相同首belief/history，一方状态更新不改变另一方case。
- [x] 写 `test_suffix_discount_matches_full_episode_clock`：固定可解析reward序列的后缀折扣与手算一致。
- [x] 写 `test_zero_warmup_matches_existing_evaluate`：同seed下无前缀deterministic planner回报一致。
- [x] 写 `test_invalid_warmup_is_rejected`：负值及不小于horizon拒绝；原始每用户return均保存。
- [x] 运行RED；实现行为策略前缀与clone/折扣；同步GPU计时仅在CUDA模型评估时启用。
- [x] 全suite回归；明确tail_from与公共前缀两种指标名称。

## Task 4: 可恢复pilot入口与远程执行准备

**Files:** Create `scripts/run_logged_pilot.py`, `tests/test_pilot_manifest.py`; 如需checkpoint辅助，Create `muslate/model/checkpoint.py`。

**Interfaces:** `build_pilot_arms(seeds=(1,2,3)) -> list[dict]`，明确21arms；`save_checkpoint(model: WorldModel, path: str | Path, metadata: dict) -> None`、`load_checkpoint(path: str | Path, device: torch.device) -> tuple[WorldModel, dict]` 保存模型config/维度/状态与metadata。runner接受 `--out-dir --arm --device --episodes --epochs --users`，单arm可恢复；固定默认矩阵按spec实现，tiny-run覆盖项进入实际manifest。

- [x] 写 `test_matrix_contains_21_distinct_arms`：L0每seed1个、L1/U1每seed3个；L0/L1无特权配置。
- [x] 写 `test_checkpoint_round_trip_preserves_predictions`：save/load的item_scores、reward、transition一致。
- [x] 写 `test_resume_rejects_changed_config_or_source_hash` 与 `test_complete_result_cannot_be_overwritten`：配置/hash冲突拒绝；只跳过完整且匹配run。
- [x] 运行RED；实现atomic checkpoint/result写入与run manifest；每用户数组、公共前缀/完整回报、seeds和运行时信息保存。
- [x] 在本地/远程保存源码SHA256并核对差异；生成实施前快照，远程改动只同步已核对文件。记录同步和快照位置。
- [x] 先CPU tiny smoke；GPU0 tiny单arm，确认信息条件/finite指标/显存和耗时；过关后扩大到默认单arm。
- [x] 根据实际单arm耗时排21arms，先GPU0顺序；检查占用后再使用其他空闲GPU，不终止既有任务。
- [x] E1结果未完成前只报告进度/诊断，不宣称无特权方法成功；完成后汇总关键差值和需要扩5seeds的配置。

## 2026-10-08 实施状态

用户已批准直接实施和运行。Task1–4 已实现；严格日志/训练、公共前缀、checkpoint、身份检查、崩溃恢复和独立评估随机流均通过回归。核心96项测试本地/远程全通过；统计分析另外7项测试通过。CPU tiny 与 GPU七组tiny均已完成，默认 L1_s1_w10 完整运行耗时317.7秒。

远程原项目在实施期间有其他修改和实验，已保留；本轮独立源码目录为 `/home/xys/code/WZL/next/results/e1_workspace_20261008_v2`，正式结果为 `results/e1_pilot_20261008_v2`。其余20个模型在六卡队列运行，原进程未终止。初轮仍属探索；后续按验证集冻结选择并至少扩至5seeds。

一次独立源码审查发现的恢复/配置/随机流问题已修复；统计审查发现的不兼容实验混合、冻结选择缺少provenance、重复seed及用户配对检查均已补回归。预算为名义B512，实际H2每决策6656 slate评估；未声称硬总预算已实现。冻结选择/统计协议见 `docs/research/2026-10-08-e1-protocol.md`。


### 首轮完成（2026-10-08）

21个pilot与6个确认模型均运行完成，主条件L0/L1/U1各5个独立seed。全流程151,200次闭环决策，103项测试通过，27checkpoint/训练历史/配置/source与paired data已终检，归档SHA256校验通过。验证集冻结选择L1/U1权重100。

五seed结果：SID-hier的L1–L0完整回报差8.0113，95%区间[7.2539,8.6063]；L1的hier–uniform差0.5302，[0.1947,0.8211]，公共前缀对应差0.2429，区间跨0；hier–sid_flat两协议均无明确正增益。真实点击/硬总预算/跨目录泛化仍未完成。

报告、图与全部模型原始归档见 `results/e1_paper_results_20261008/`；结论及下一轮控制安排见 `docs/research/2026-10-08-e1-findings-next.md`。KuaiRand-1K/Pure下载、校验和六表逐行审计已完成；不能把数据准备称为真实闭环验证。
