# 第一阶段 E1 冻结协议

本协议在查看修复后 validation/test 回报之前制定。实现范围是用户第一阶段修改1–5；训练结构、损失、日志预算与旧 checkpoint 保持一致。

## 设计与选参

- 环境：固定 catalogue 200 items、slate K=5、episode H=20、gamma=.95、lt=3、div=.8；原用户群。跨 catalogue 和新用户泛化不在本轮结论范围。
- 模型：先复用 L0(w=0)、L1/U1(w=1,10,100) × training seeds 1,2,3 的21 checkpoint。
- 逐项反馈：已曝光 item 的连续 unary + std=.02 噪声；不是真实点击。U1 为全 item 特权监督参考。
- Validation：每模型16用户；full episode 与共享行为前缀5步两种协议；独立诊断日志128 episodes。
- 验证轨迹 seed base=10,000,000+2,000×training seed；最终测试 base=20,000,000+2,000×training seed。仅使用 training seeds 1–5。这些区间与旧评估区间分离。
- 所有21 validation 完成并通过原始 returns/逐决策动作核验后，按既有规则：共同前缀 sid_hier 跨3 training seeds 平均验证 return 最大值选择 L1/U1 权重；精确并列取较小权重。不得默认100或按test更改。
- 冻结选择记录21文件 SHA256、代码 SHA256 和评估合同。确认补齐选中权重 seeds4/5，训练超参数沿用旧协议；使用全新 test，一次确认。
- 五个 planner：reactive、flat_uniform、flat_policy、sid_flat、sid_hier；configured depth=2、beam=3、原提案设置B512。

## 主比较与统计

主比较是共同前缀协议下 L1 sid_hier 与 L0 sid_hier 的配对回报差，以判断合法曝光监督能否支撑决策。full episode 同项为辅助稳健性比较。L1 sid_hier 相对 flat_uniform、flat_policy、sid_flat，以及 U1 相对 L1，均作为预先声明的次要比较，不能从中事后挑出唯一成功比较。

报告5个独立训练 seeds 的配对差值、各 seed 差值和95%层次配对 bootstrap CI（先seed再该seed内用户，10,000 draws）。16用户在不同模型/方法间共享同一轨迹初始条件和前缀；用户反应由执行动作导致不同。固定 catalogue 下模型训练 seed 是主要独立复制单位。区间不是跨 catalogue 的泛化保证；次要多重比较作探索性解释。

## 正确性与计算解释

每个执行 slate 必须具有K个不同的合法整数 item。每次决策记录 t、剩余步数、有效 depth、动作、reward、分项成本；末步只评价当前一步。最终结果由观察 reward 重算并与 returns/tail/汇总核对。

B512 是提案参数，不能宣称每决策总调用512次，也不代表严格等成本。分别记录 reward-only、joint reward+transition、score queries、候选生成尝试、有效候选与 distinct ordered slates，以及实际规划延迟。硬件实测权重构成线性估计毫秒；其 batch 标度有近似误差，不是 FLOPs 或 wall-clock deadline。硬预算接口限制此估计且在不足时显式失败；本轮 E1 不施加严格相同总硬预算。Return–成本描述可展示，但不作E4效率优势确认。

## KuaiRand-27K

服务器目录 `/home/xys/code/WZL/data/KuaiRand-27K` 已做文件及CSV表头清点，证据见 `kuairand_27k_inventory_20261009.json`。本轮未训练真实数据模型。读取到的表头包含逐项点击/观看反馈，未发现原生 request/session/position/slate 字段；后续须先制定顺序交互和会话重建协议，再开展真实数据验证。
