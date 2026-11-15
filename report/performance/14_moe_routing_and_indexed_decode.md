# 14 MoE路由共享有效；直接索引局部快，但尚无D200结论

## 本轮结论

[计划16](../../docs/plan/16_moe_routing_and_indexed_decode.md)运行`/workspace/results/mellum_moe_optimization/20261115T121305387577Z`完整结束，实际10个GPU worker、累计1739.80秒（约29分钟），没有失败或预算截断。采集代码commit `874115af2be849dbe17471b98ae041cfad397969`；[轻量证据](../../artifacts/performance/14_moe_routing_and_indexed_decode/manifest.json)。

**shared_routing可保留为下一轮的可选优化对照。** 同批短/长输入ABBA加速1.0915×/1.0468×，几何均值1.0689×；完整D200的200条输出逐条与当前CPD相同，且当前CPD与历史CPD的200条也全部相同。它没有新增这批已观察到的生成分叉，位置96.8125%、完整序列87%维持不变。HTTP小规模回归通过。默认服务配置本轮未切换。

**indexed_decode保留为实验候选，不宣称已通过或不通过题目门槛。** 其decode32局部时间8.030→1.641ms，约4.892×，额外分配峰值约64.20→3.71MiB；但16token小集位置匹配降至S32 94.34%、L32 93.95%，触发计划16“相对CPD每组最多下降2个百分点”的保守筛选，未进入D200。

这两个小集位置分数本身仍高于90%。不能把本项目额外的“保留CPD水平”规则当成题目规则，更不能据此断言indexed_decode在完整200条上不达90%。后续若优先追求更大性能收益，值得单独补一次固定D200、按题目位置口径评估；本轮没有自行变更预设规则或追加GPU。

## 核验与运行边界

独立重算169个产物hash、236个相关源码/配置指纹及HF200参考checksum，全部一致；[核验结果](../../artifacts/performance/14_moe_routing_and_indexed_decode/integrity_audit.json)、[核验脚本](../../artifacts/performance/14_moe_routing_and_indexed_decode/verify_source_run.py)。环境仍为MI308X、Torch `2.10.0+rocm7.2.4.git3d3aa833`、HIP7.2.53211、Transformers5.2.0，缓存HF来源5.10.1未改。

全部模型worker加载审计5631/5631，missing/unexpected为空。TP/DP1、BF16/eager，关闭prefix/chunk和SWA CK/Triton优化；真实调度、分页KV、attention及采样公式保持。所有数据为正常greedy，没有强制HF历史或输出缓存推理。

四个screen配置均K4/S32/L32各跑两次，12组重复逐请求相同。ABBA又独立核验同配置、同集合的4次输出相同。D200每配置只跑一次，不把单次结果的`stable=true`当作重复测试。

原B0/CPD/R/I的实际批量均由元数据核验。D200两配置各200提交/完成、32×6+8共7组提交；实际decode32×384次、decode8×64次，prefill batch分布仍为2×1、5×1、6×4、9×1、32×5。forward计数可包括终止后的额外执行，不当输出token数。

本次轻量归档约3.6MB，保留原输出、计时、日志和trace摘要；原始trace与重复完整输入的manifest留在results，compact元数据保存原hash。原始运行未改写。本轮仅分析和归档，无新GPU或数学修改。

## 两项候选实际执行了什么

实现见[MellumTorchExperts.route/forward_routed](../../ATOM/atom/model_config/mellum_experts.py)、[optimized_experts/install_experts](../../ATOM/atom/model_config/mellum_moe_optimization.py)。公共重构后的CPD与历史CPD：S32 32/32、L32 32/32、D200 200/200完整输出相同，未观察到该重构带来的漂移。

R保留L（QKV/O/router线性）的按请求shape，以及prefill/decode专家GEMM的按请求shape，只在整个真实批次上共享计算一次softmax/topk/renormalize。它没有撤销此前有效的CPD联合控制。

I在R上仅替换decode专家线性计算，用[Triton indexed GEMV](../../ATOM/atom/model_config/mellum_indexed_gemv.py)直接读取canonical专家权重，保留Torch的BF16 SiLU乘法、路由权重乘法与top-k求和。FP32点积归约次序与原BMM不同，不承诺逐位相同。

layer0运行审计确认：R的prefill/decode均`requestwise_gemm_shared_route`；I的prefill同样为该路径，decode实际为`indexed_triton`。局部GPU trace也确有`_indexed_expert_gemv`，不是静默回退后得到的速度。

## 局部MoE微基准

只加载真实layer0权重，复用四条保存的HF prefill输入/路由logits。decode-shaped组用它们的末行循环构成1/8/32行：属于真实数值的decode形状回放，**不是32条独立自然decode轨迹**。重复expert与缓存复用程度可能不同，不能直接外推整网收益。

| 局部负载 | CPD中位ms | R中位ms | I中位ms | I相对CPD的relative L2 |
| --- | --- | --- | --- | --- |
| ragged prefill，8请求/150行 | 13.605 | 13.281 | 13.112 | 0（I此时仍走R） |
| decode1 | 0.319 | 0.324 | 0.273 | 0 |
| decode8 | 2.051 | 1.542 | 0.552 | 0.0000701 |
| decode32 | 8.030 | 6.113 | 1.641 | 0.0000701 |

R在这四个局部回放中全部与CPD精确相同。I在decode8/32的最大绝对差为0.00012207，语义边界探针通过、输出有限；这些检查支持继续做生成验证，但不能替代它。

decode32额外allocated峰值：CPD64.200MiB、R64.202MiB、I3.706MiB。此处是计时调用相对调用前的Torch allocated峰值，非整模型峰值、非进程全部显存，也非直接测量的HBM流量。CPD已按请求执行，因此不能将本轮数字误写成从同时物化3GiB降至4MiB。

## 小集筛选与I停止位置

| 配置 | K4位置/完整序列 | S32位置/完整序列 | L32位置/完整序列 |
| --- | --- | --- | --- |
| 原B0 | 62/64，3/4 | 478/512=93.36%，28/32 | 456/512=89.06%，23/32 |
| CPD | 64/64，4/4 | 501/512=97.85%，31/32 | 510/512=99.61%，31/32 |
| R | 64/64，4/4 | 501/512=97.85%，31/32 | 510/512=99.61%，31/32 |
| I | 62/64，3/4 | 483/512=94.34%，27/32 | 481/512=93.95%，26/32 |

I相对CPD：S32下降3.5156个百分点，L32下降5.6641个百分点，均超过预设2个百分点筛选。因此本轮只选择R进入D200。根`decisions.skipped=[]`并不表示所有候选晋级：该版runner只记录了`selected_d200`，没有逐条写出小集淘汰理由；本报告依据未改动的`screen_ok`函数与原数据补充解释。

I的screen wall也较低（S32约9.353秒、L32约21.055秒），显示整网收益潜力，但screen并非ABBA配对计时，输出内容也改变，不能以这些数值冒充已验证的正式加速比。尚无I的D200、ABBA或HTTP结论。

## R完整D200：输出相同，不只汇总分数相同

| 分组 | 数量 | CPD及R共同的位置匹配 | CPD及R共同的完整序列匹配 |
| --- | --- | --- | --- |
| 全体 | 200 | 12392/12800=96.8125% | 174/200=87% |
| 输入<1024 | 168 | 10478/10752=97.4516% | 148/168=88.0952% |
| 输入≥1024 | 32 | 1914/2048=93.4570% | 26/32=81.25% |
| HumanEval来源 | 164 | 10222/10496=97.3895% | 144/164=87.8049% |
| 场景A来源 | 36 | 2170/2304=94.1840% | 30/36=83.3333% |

独立按name对齐200个actual_ids，R与本批CPD **200/200逐条完全相同**。CPD历史输出复现及相同HF来源已核验，不能把这个结论写成R与HF200/200相同；它保留了CPD原有的26条分叉。

D200记录的执行wall分别310.91/279.37秒，属于不同worker的一次完整执行记录，不以此单次比值作为主要性能证据。HumanEval来源列也仍只覆盖前64token生成，不是完整代码功能pass@1。

## R无probe ABBA计时

两个ABBA周期，S32/L32交替；同worker加载一次，CPD与R各组4次，段前预热、greedy16。无张量probe、无活动profiler、关闭MoE audit写盘；两边均承担相同的控制文件与layer0 batch元数据审计开销。下表是开发负载的wall中位数：

| 输入组 | CPD秒 | R秒 | CPD/R | R耗时下降 |
| --- | --- | --- | --- | --- |
| S32 | 13.3992 | 12.2755 | **1.09154×** | 8.39% |
| L32 | 25.0981 | 23.9771 | **1.04675×** | 4.47% |

两组加速几何均值1.06891×，各组四次输出稳定、真实batch32。满足预设“几何均值≥1.05、每组≥0.95”的收益筛选；这不是置信区间或正式A/C成绩，也不是相对题目官方Baseline的收益。

**收益有效，但尚未解决大部分CPD开销。** 不能把约7%的本地平均收益描述成已经追回相对原B0的约2.44倍差距；本轮没有在ABBA中重新配对原B0作第三方计时。

## 真实子进程trace与服务回归

最佳R候选另采S32、输出2token的实际ModelRunner子进程trace，rank0 start/stop回执、GPU kernel与模块标记通过；没有将前端提交侧trace当GPU执行证据。该短trace不是长输入trace。

相对上批同类CPD短trace，kernel总数204024→185796，少18228个（8.93%）；experts累计kernel1792.77→1662.90ms，减少约7.24%。其中prefill experts1446.52→1389.37ms，decode experts346.24→273.53ms。不同批次profile耗时只作归因旁证，不代替本批ABBA。

R中experts仍占累计GPU kernel时间约75.98%；prefill中占82.56%，decode中占54.09%。prefill归因experts/`aten::mm`的发射仍为96582次、约1261.90ms；decode专家`aten::index`/`aten::bmm`各3584次发射仍在。这与本轮只合并route辅助操作、未改变专家GEMM/权重索引的源码设计一致；top-launch列表为截断排名，不能据列表中不见topk就声称没有任何topk kernel。

HTTP同配置R：顺序completion/SSE、usage、两个并发请求均与短HF参考匹配，实际decode2、pending最大2；chat保留已知reasoning首换行差异，按原展示归一分类通过。没有在此宣称32路HTTP压测或全HumanEval通过。

## 后续取舍

1. 保留R作为后续可选性能对照，保留CPD作回退；本轮不切生产默认，也不重跑已完成的完整shape矩阵。
2. I具有显著局部速度与分配收益，小集位置仍高于90%。建议下一轮**只增加一次固定D200**，显式按题目位置门槛判断，不把原先自设2个百分点筛选伪装成正式要求；先报告结果，再决定是否做ABBA/服务/功能评估。不提前宣称I通过，也不直接丢弃。
3. 若I完整D200不具备条件，则停止该版本，不展开无限轮浮点对齐；依据trace转向prefill小GEMM发射/按请求专家调度的有界优化。R对prefill主要成本的改善有限，源码与trace已明确这一点。
4. 公共集身份/计数口径、完整HumanEval以及正式A/C仍为最终交付工作，不用本地前64token或局部误差替代。R与已验证CPD输出相同是有用证据，但并没有新增这些项目的测试覆盖。

下一轮GPU操作应对应上述有界问题，不默认再运行整个12-worker批次。本轮无追加GPU命令。
