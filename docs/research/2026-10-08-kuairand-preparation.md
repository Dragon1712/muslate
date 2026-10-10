# KuaiRand 数据准备与接入边界

本轮正式模型采用合成连续反馈。本文件是后续真实日志实验的准备记录，不代表已完成真实点击闭环验证。

下载来源：[作者仓库](https://github.com/chongminggao/KuaiRand)、[官方 Zenodo record10439422](https://zenodo.org/records/10439422)。远程目录：`/home/xys/code/WZL/next/data/external/kuairand_20261008`。Pure 校验通过：47,432,272 bytes，MD5 `0820331067a3784d9691136f772b35a7`。1K已下载、解压并校验通过：1,135,436,720 bytes，MD5 `6b0b9c8222d67fcd4c676218edca3f1f`。

实际 Pure 三张日志表逐行审计：

| 表 | 行数 | 已展示且 is_click=0 | 用户数 | 用户内逆序行 |
|---|---:|---:|---:|---:|
| random 4/22–5/08 | 1,186,059 | 977,125 | 27,285 | 0 |
| standard 4/08–4/21 | 1,141,112 | 612,267 | 26,210 | 23,938 |
| standard 4/22–5/08 | 295,497 | 164,000 | 25,877 | 18,763 |

click/play_time 在这三张表没有空字符串。字段包含 user_id、video_id、time_ms、tab、is_rand、is_click、play_time_ms、duration_ms，以及其他行为反馈；没有原生 request_id、session_id、position、slate_id。审计原始 JSON：`kuairand_pure_schema_audit.json`。

接入约束：

1. 已曝光的0是有效观测；未曝光视频不能补零。先区分 tab 场景的 click/valid-play 含义，再确定预测目标和link函数，不能直接沿用连续 unary MSE。
2. 原 CSV 的标准日志存在用户内时间逆序；历史构造须按时间排序，稳定处理相同时间戳，并在全局时间切分后禁止未来反馈进入当前输入。
3. 原始表不提供完整曝光列表或位置。KuaiSim 预处理中的 request/session 构造规则须单独记录并做敏感性分析；由时间窗或固定长度组出的列表不能描述为真实线上 slate。
4. 作者说明 Pure 删除候选池外的交互，序列不完整；使用 1K 评估历史建模，Pure仅作小规模接入/官方模拟器复现。[作者版本建议](https://github.com/chongminggao/KuaiRand#which-version-should-i-use)
5. video statistical features 含跨收集期间的聚合信息；时间外测试先使用可核验的静态 basic features 或只用训练期重算统计，避免未来统计泄漏。

下一轮真实数据协议须冻结时间分割、反馈目标、ID映射/候选覆盖、会话构造、模拟器训练与测试隔离；线上结论只能由相应线上试验支持。[KuaiSim 官方代码](https://github.com/Applied-Machine-Learning-Lab/KuaiSim)

已取得官方 `code/preprocess/KuaiRandDataset.ipynb` 的内容快照，SHA256 `7a7371375616d364d30a1299cc96b3db89e63bfa99ec63a562427ede1d0a9757`。仓库/API连接或限流失败，未取得Git commit；因此以内容hash锁定本次审计版本，不声称已完成官方代码安装。文件及manifest保存在 `kuaisim_source_audit_20261008/`，服务器亦有副本。

这份notebook会按 user_id/time_ms 排序，以用户的日期构造session，position是当天交互序号；列表大小诊断将同一用户在同一秒的交互聚在一起。它们是预处理定义，position不等于原生列表内曝光位置。[已审阅源文件](https://raw.githubusercontent.com/Applied-Machine-Learning-Lab/KuaiSim/main/code/preprocess/KuaiRandDataset.ipynb)。其他训练reader中request_id的最终构造尚须追踪，不能由README示例推断已获得真实request。


1K三张日志亦已逐行审计（`kuairand_1k_schema_audit.json`）：随机干预43,028行、早期标准5,055,984行、后期标准6,657,061行；均无空click/play_time字段、无用户内时间逆序，原生request/session/position/slate字段均不存在。三表非点击分别35,536、3,138,050、4,145,155行。时间跨度2022-04-08至2022-05-08；早期有983个用户，后期1000个。训练早期标准日志后对后期测试时，应区分17个早期未出现的用户与既有用户，不能混称同一泛化任务。

下载完成manifest及MD5记录已保存为 `kuairand_download_complete.json`。完整原始数据保留在服务器；本地保存审计与来源信息。
