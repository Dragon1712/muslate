# R2-B：等估计计算的一步列表组合搜索（预先冻结）

2026-10-10，探索性协议。只复用 L1 seed1–5、w_score100 检查点，不重训。
本轮先运行一个中预算：六个命名方法 uniform、flat_coordinate_refine、
true_sid_flat、true_sid_hier、random_sid_flat、random_sid_hier；随机分组使用
三个固定种子 271、811、1459，保持真实分组容量，形成每训练seed10配置、共50cell。
16用户共享5步行为前缀，episode20步，轨迹起点31,000,000+2000*训练seed；
40M区域保留。所有方法只优化完整即时预测reward，depth0，没有transition查询。

## 计算预算与预声明校准规则

校准使用seed1/2、每seed前10用户的公共前缀终点，轨迹起点32,000,000+2000*seed，
与评估区域分离；只根据操作时间冻结预算，不查看这些状态的真实回报以选择参数。
同一GPU2串行校准，warmup后测量reward batch1–32及score batch1，保存原始重复时间、
中位数和线性拟合残差。完整队列使用GPU2/4/5，每GPU一个job，seed1/4分配GPU2、
seed2/5分配GPU4、seed3分配GPU5；共享GPU2校准系数。GPU0/1运行无关作业，校准
保存前后全GPU负载和host load average；CPU争用未被控制，真实延迟仅为该次机器条件下描述。
预算实际计费使用每个batch的中位数查表，包含模型调用开销；
拟合的固定调用开销及样本斜率仅作描述。每方法最大microbatch32。
候选构造/去重/缓存检查、初始化排序、坐标替换分别以排除模型调用的CPU代理时间校准。
CPU计费是操作代理估计，未精确逐项分解Python控制/缓存排序/容器分配全部开销；
因此“等计算”仅指声明的估计账本，不声称全部真实计算或FLOPs匹配。
所有系数必须为有限正数。中预算是20个校准状态上的固定SIDhier参考搜索
（34粗计划、最好4起点、5位置、每位置top3物品、严格改善接受）的估计总成本中位数。
保存校准输入、原始时序、估计成本、内容SHA并在run manifest绑定；不按评估回报调参。

每方法先支付相同score查询、排序/共同合法初始slate构造和reward batch1；
共同初始slate为每状态item score稳定降序topK，与方法/随机分组无关。
最小共同初始化成本不能适配预算则拒绝配置。每个构造+模型batch在执行之前检查
完整估计成本，必要时缩短batch；重复候选仍支付构造/检查成本，缓存reward不重复查询。
预算不足返回已实际评价的合法最好incumbent，平局保留较早候选。估计成本永不超cap，
记录未用预算、停机原因、异常、完整调用及样本计数。真实延迟可超过估计cap。

## 搜索与对照

uniform持续均匀无放回抽取有序slate。flat_coordinate_refine先均匀产生32起点，
true/random SID方法先产生32粗计划并按组内item score完成无重复slate。
平面SID方法继续生成粗计划直到预算耗尽。三个refine方法从相同规则选最好4起点，
按score稳定顺序切成宽8的rank窗口，每窗口按位置0..K-1、起点稳定顺序遍历，
最多4起点×8物品组成microbatch32，严格改善接受。窗口持续前进到完整目录/组，
没有top3永久候选限制；预算可能在完整扫描前耗尽。校准参考仍固定单轮top3。
flat_coordinate_refine允许全目录替换；hier只允许该位置当前item所在组替换。
每轮后若仍有改善继续同样遍历；没有改善则支付并生成32新起点，重新选择最好4。
全程使用同一缓存和全局最好incumbent，计数base attempts、refinement attempts、
duplicates、cache hits、fresh evaluations、accepted refinements和emitted candidates分别保存。
refinement attempts包含预付的合法性检查slot，另列invalid slots及legal candidates；
预算不足时在构造任何该批候选之前终止，不构造未收费的截断尾部。

补充固定状态诊断：每seed前4用户前缀终点，以共同初始slate开始，分别只运行
unrestricted、true grouped、三个random grouped坐标细化；初始化成本相同且已计费，
使用同一总中cap（因此同一额外可用预算），不增加闭环测试配置。
记录预测改善、真值改善及选择偏差（预测改善减真值改善）。不是全局Oracle。

## 身份、校验、统计

代码、协议和测试Git提交后方可完整运行；运行时传入source commit，并保存全部依赖源文件SHA、
checkpoint SHA及训练身份、环境、校准SHA、GPU/PyTorch/NumPy/Python版本、实际queue和child
命令、cwd/hostname/pid及GPU分配。source commit要求完整40位SHA；失败保留命令/源SHA/运行时上下文。
输出使用原子无覆盖发布，拒绝部分resume或替换已有产物。校准及smoke可以先做。
50个预期cell和5份固定状态诊断全部枚举，失败保留traceback及queue记录，不部分覆盖重跑。
每decision保存动作、预测/实际reward、操作账本、预算及GPU同步延迟；独立重放全部动作，
核验奖励、共享前缀、折扣episode回报和输入SHA。禁止把decision或随机分组当作独立复制。

随机分组先在每训练seed内对三个分组的同用户回报取平均；再计算16用户均值和5seed配对差。
主95%区间使用df4 Student-t；补充seed→配对用户分层bootstrap10,000次、固定seed20261010。
预声明对比为flat_coordinate_refine−uniform、true_sid_hier−true_sid_flat、
random_sid_hier−random_sid_flat、true_sid_hier−flat_coordinate_refine、
true_sid_hier−random_sid_hier、true_sid_flat−random_sid_flat。
报告全部区间、五seed差值、成本/未用预算和延迟，不作多重比较校正，不声明确认性成功。
先仅50个中预算cell；低/高预算或确认实验需依据本轮结果另行决定。
