# 13 批量准确性自动实验：CPD恢复位置一致性，MoE成为首要性能问题

## 本轮结论与决策

[计划15](../../docs/plan/15_batch_accuracy_campaign.md)的一次启动批次已完整执行：14个GPU worker，累计1888.67秒（约31.48分钟），没有预算截断、失败job或probe截断。原始运行`/workspace/results/mellum_batch_campaign/20261115T043721420603Z`，代码commit `4259e4badaa68ed3d3a7367be2a4354884bdea41`；[轻量归档](../../artifacts/performance/13_batch_accuracy_campaign/manifest.json)保留原始来源与hash。

最有效方案为**CPD：prefill和decode均按真实请求边界调用L/E**，其中L=QKV、O、router，E=experts。它保留真实batch提交、分页KV、调度、正常greedy和原有BF16公式，并没有将前端请求改成串行提交。200条64-token开发对照达到**12392/12800=96.8125%位置一致，174/200=87%完整序列一致**；实际decode32，200条全部完成，K4和HTTP小规模回归通过。

CPD可保留为**批量正确性的可选回退方案及后续优化对照**，但不能称为高效实现：同批关闭张量probe、profiler未活动的S32计时，原B0为5.489秒，CPD为13.405秒，约慢2.442倍。默认配置本轮未切换；本轮只核验、分析和归档，无新GPU/功能修改。

后续主线应转为**在CPD对照下减少MoE和逐请求调用的开销**，不再为bitwise一致继续无限细分。仍需明确：本地开发集位置口径超过90%，序列口径87%未达到90%；公开集身份/计数口径未获出题方确认，HumanEval完整pass@1及16k批量/正式A/C不在本批覆盖范围。`RESULT complete`表示实验及其预置分支完成，并非题目全部验收通过。

## 完整性与可复现性

- 独立重算585个job产物hash、232个相关源码/配置指纹与缓存HF参考hash，全部一致；[核验结果](../../artifacts/performance/13_batch_accuracy_campaign/integrity_audit.json)、[核验脚本](../../artifacts/performance/13_batch_accuracy_campaign/verify_source_run.py)。
- Torch `2.10.0+rocm7.2.4.git3d3aa833`、Transformers5.2.0、HIP7.2.53211、MI308X；HF参考仍为5.10.1。原HF200 checksum `407cab507b32e45191f6cd5431709824a5b9f845442936dfc018c3668a4d7df1`，不重新生成、不替换参考。
- 各worker权重审计5631/5631，missing/unexpected为空。SWA CK/Triton优化关闭，TP/DP1、BF16/eager，prefix/chunk关闭，maxseq32及真实分页cache保持。
- A与B阶段各S32/L32重复两次，D计时K4/原B0/候选各重复两次，21组重复输出逐请求核验相同。D200每个候选只跑一次，CX_cache各cohort也只有一次；它们的脚本`stable=true`不能解释为已经做了重复实验。
- 两个D200候选均200提交/完成，7组提交（32×6+8），没有未知、重复或丢失完成。实际decode32×384次、decode8×64次，prefill batch为2×1、5×1、6×4、9×1、32×5。forward次数可包含终止后的额外执行，不能直接当输出token数。
- 根manifest/summary各约29MB，含重复完整输入。本次归档采用明确标注的compact元数据、原文件hash及必要结果，共约4.4MB；原始tensor/trace和完整manifest留在results。原始运行文件未改写。

## A/B矩阵：联合控制有效，单一控制不能替代

下表均为原始自然生成前16token，位置一致率；原始详细矩阵见[CSV](../../artifacts/performance/13_batch_accuracy_campaign/matrix.csv)。S32为前32条HumanEval来源，L32为全部32条输入≥1024的scenario A来源。

| 方案 | S32位置 | L32位置 | S32完整序列 | L32完整序列 |
| --- | --- | --- | --- | --- |
| 原B0 bs1 | 100% | 100% | 32/32 | 32/32 |
| 原B0 bs2 | 91.80% | 94.14% | 25/32 | 26/32 |
| 原B0 bs8 | 89.65% | 91.80% | 24/32 | 25/32 |
| 原B0 bs32 | 93.36% | 89.06% | 28/32 | 23/32 |
| CP：仅prefill L/E | 92.97% | 92.77% | 27/32 | 26/32 |
| CD：仅decode L/E | 94.34% | 90.82% | 28/32 | 26/32 |
| CPD：两阶段L/E | **97.85%** | **99.61%** | **31/32** | **31/32** |
| CL：两阶段仅L | 86.52% | 91.80% | 23/32 | 24/32 |
| CE：两阶段仅E | 94.92% | 93.16% | 29/32 | 26/32 |

由CP/CD/CPD选择的CL/CE隔离phase为prefill+decode；实际配置已记录，不能把CL理解成仅prefill。小集重复稳定，但batch越大并非单调越不一致，不能只据此断言显存/索引规模越大越错。

联合改变L/E的请求shape显著提高生成匹配，单独L或E没有同等效果，支持它们的数值影响会共同传播。这个实验没有隔离每个GEMM的舍入步骤，也不证明其他运算或所有缓存逻辑都不存在问题。

按预置规则CPD与CE进入D200，CE在小集的改善未泛化：200位置仅72.1563%、完整41%，低于此前B0的74.8359%/44%。因此弃用CE作为下一阶段候选，不从小集结果直接宣称优化有效。

## C阶段：200条完整结果

| 分组 | 数量 | 原B0位置一致率（历史同HF对照） | CPD位置一致率 | CPD完整序列 |
| --- | --- | --- | --- | --- |
| 全体 | 200 | 74.8359% | **12392/12800=96.8125%** | 174/200=87% |
| 输入<1024 | 168 | 77.5577% | 10478/10752=97.4516% | 148/168=88.0952% |
| 输入≥1024 | 32 | 60.5469% | 1914/2048=93.4570% | 26/32=81.25% |
| HumanEval来源 | 164 | 78.0393% | 10222/10496=97.3895% | 144/164=87.8049% |
| 场景A来源 | 36 | 60.2431% | 2170/2304=94.1840% | 30/36=83.3333% |

原B0历史来源为[性能12](12_b0_batch200.md)，输入、参考与预算相同，但不是本批同步重跑的D200性能对照。逐题匹配位置相对其103条改善、93条相同、4条下降。CPD仍有26条分叉，其中humaneval_2第5个0起位置、scenario_a_25第14个位置较早，其余24条首差位置均≥29。不会删除这些失败条目或提前EOS来改变分母。

HumanEval来源97.3895%仅是思考/生成前64token对照，不能换算为完整代码pass@1。这里还没有CPD完整HumanEval功能分数。

## 有限首差与真实KV观察

[首差记录](../../artifacts/performance/13_batch_accuracy_campaign/first_difference.json)只比较四个目标、前8步、repeat0、**实际历史与HF相同**的bs1/32对。共27对；bs1对应HF全词表logits最大绝对差全部为0，支持参考复现条件。本记录是所观察末行边界的首差，未遍历全部prefill token，不是全模型任意位置的绝对首差。

首个观察差异分布：9对`00.qkv.0`、1对`01.qkv.0`、14对`00.attention_context.0`、3对`01.attention_context.0`。部分QKV差异仅约1e-7，不能把任何非零都解释为有意义的错误。

明确的排序翻转例：humaneval_13、0起step6，历史相同；bs1 top2 token IDs为9729/302、logits31.125/31.0，bs32为302/9729、30.875/30.75。两边top1 margin均0.125，小数值变化确实可改变正常greedy选择。这是原B0探针，不是对CPD残余分叉的逐层证明。

自动CX配额用于layer1真实paged KV跟进，没有同时展开norm/head额外搜索。bs1/32自然生成分别复现原A结果；同历史decode中目标层cache已有差异，例如scenario_a_1的K最大绝对差0.1640625、V为0.0092163。**缓存值不同并不等于缓存页归属错误**：不同prefill中间结果也会写成不同KV。当前证据不能从该差异单独推出索引bug；联合L/E控制在不改分页逻辑下显著改善，下一步优先处理算术执行开销。

对应实现为[Campaign.begin/end、split_call](../../ATOM/atom/model_config/mellum_campaign.py)、[ModelRunner.run_model](../../ATOM/atom/model_engine/model_runner.py)和[drive_window](../../scripts/validate_mellum_batch200.py)。实际请求映射在dispatch前发布；CPD按真实`ScheduledBatch.num_scheduled_tokens`切分，不读取HF期望token决定推理计算。

## D阶段：正确性收益有明显成本

独立S32计时在同worker、同模型、同32条输入、greedy16、预热后各两次，张量观察器关闭，profiler尚未启动。两边均承担控制文件读取和边界管理开销，因此是开发代价对照，不是正式A/B/C：

| 配置 | 两次wall秒 | 均值秒 | 位置匹配 |
| --- | --- | --- | --- |
| 原B0 | 5.4942 / 5.4842 | 5.4892 | 478/512 |
| CPD | 13.4060 / 13.4036 | 13.4048 | 501/512 |

CPD耗时为2.442倍，固定输出预算下吞吐约为原B0的40.95%。原B0准确性不足，所以不能把它作为合格性能成绩；同样不能把CPD人为较慢的耗时当作官方Baseline来放大后续加速比。

K4四例（含1023/1025边界）64/64位置、4/4完整序列，重复一致。D阶段K4只按当前顺序提交；这次没有重跑旧batch脚本中的反序与迟到请求场景，不额外宣称它们也已覆盖。

HTTP使用同一CPD控制，实际观察decode2：顺序completion/SSE、usage、两个并发请求均与所用HF短参考匹配，pending最大2；chat仅原有reasoning首换行差异，按既定展示归一分类通过，完整content未作宽松归一。HTTP是小规模回归，不是32路服务压测。

## Profile为下一轮提供的方向

trace由ATOM原生start/stop profile、rank0回执在真实ModelRunner子进程采集，且在无活动profiler的计时之后单独运行；不是前端提交侧trace。[摘要](../../artifacts/performance/13_batch_accuracy_campaign/jobs/D_timing_trace/trace_analysis.md)、[机器数据](../../artifacts/performance/13_batch_accuracy_campaign/jobs/D_timing_trace/trace_analysis.json)。

短trace为S32、输出2token，观测prefill1次、decode2次；204024个GPU kernel、模块归因99.9475%。累计kernel2318.42ms，其中experts1792.77ms（77.33%）；prefill experts1446.52/1740.44ms（83.11%），decode experts346.24/577.79ms（59.93%）。这些是累计kernel归因，不能直接当端到端占比或套成承诺加速上限。

prefill中归到experts/`aten::mm`的GPU kernel发射96582次、累计1263.16ms；decode experts中`aten::index`和`aten::bmm`各归因3584次GPU kernel、约107.70ms/107.92ms。结合[MellumTorchExperts.forward](../../ATOM/atom/model_config/mellum_experts.py)的逐请求路由、grouped_mm、选中专家权重索引和BMM，优先研究MoE调度/发射与数据搬运具有源码与trace依据。不能只根据算子名字断言底层具体实现或哪一个自定义kernel必然更快。

kernel span约5975.38ms，活跃并集2318.42ms，空隙约61.2%；profile明显有插桩影响，空隙也不自动等于纯CPU瓶颈。需要关闭profiler的配对计时判断实际收益。

## 下一轮建议与停止条件

1. 固定CPD为可选正确性回退，保留原B0作消融；不切默认、不另花一轮追求所有logits精确。
2. 以MoE为下一优化主线，先拆清prefill大量小GEMM发射与decode权重选择/路由开销。优先减少重复路由、索引和Python调用，尽量保留已验证的激活/累加顺序；合并GEMM或换AITER实现作为独立有界候选，仍要实测，不能预设精度安全。
3. 下一批一次开发和一次启动覆盖原B0/CPD/少量候选：S32+L32筛选、最多两项D200、通过者无probe ABBA计时与短真实trace。短集不泛化的CE结果已说明不能省略D200。候选必须同时报告吞吐和双一致率，不能只报kernel速度。
4. 不默认再跑当前14项完整shape矩阵；它已给出联合控制与性能代价证据。正式A/C应在候选准确性具备条件后运行；完整HumanEval功能对照需单独预算，不用前64token分数替代。

当前无需用户追加运行命令。下一次GPU启动应对应已开发好的MoE候选实验，而不是重复本轮定位。公开集计数口径与HumanEval仍需在最终交付前说明；本轮不将其混同为继续做微小浮点定位的理由。
