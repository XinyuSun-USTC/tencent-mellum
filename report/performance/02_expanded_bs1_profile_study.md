# 02 扩展bs1性能基线与优化决策依据

## 结论与范围

五种长度、三档输出及真实A/C提示词的扩展采集完成。长prefill优先研究SWA注意力；decode同时研究KV整理/RoPE重复工作与专家权重gather。短prefill的MoE占比更高，不能用长输入排序替代全部场景。当前可以结束这一轮bs1基线采集，进入受控候选实现；正式并发A与开放到达C仍需服务/批量能力与专门基准。

原始运行：`/workspace/results/mellum_profile_study/20261113T190525160560Z`；干净采集版本`63851f02fa53ee5e5ecd4122f091dac24cea31de`。协议见[09](../../docs/plan/09_expanded_profile_study.md)，[轻量归档与原始trace哈希](../../artifacts/performance/02_expanded_profile/manifest.json)。本轮只分析已有产物，未运行新GPU任务、未改模型数学。

## 完整性与测量限制

19组、132次无profiler计时、22617个输出token；另9份16输出token短trace及1份1K/4输出详细trace。10份原始trace SHA256均复核一致，同预算profile/非profile生成token逐个一致；每份实际捕获1次prefill和与输出预算相同次数的decode，包含异步末尾额外forward。加载审计见worker原始记录。

短trace模块归因覆盖98.66%–99.86%；profile请求wall为配对非profile的1.09–1.53倍，详细trace为2.61倍。性能结论采用关闭profiler的计时；GPU累计duration只用作归因，不与CPU时间相加，不把空隙全部归为可消除CPU成本。

真实样本为固定chat/thinking配置下A/C数据集输入长度的中位与长分位代表：A 2303/3857，C 10448/15007 token；每个只测3次，不能代表完整数据集。合成组通常5次，1K×256为50次。ignore_eos且固定输出预算；没有HTTP、并发32或Poisson到达，以下均非正式A/B/C成绩。

## 未开启profiler的计时

表内为中位数；单token输出TPOT没有定义。长输出使decode成为整请求的重要成本。

| 样本 | 输入 | 输出 | 次数 | TTFT ms | TPOT ms | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| ctx_512 | 512 | 1 | 5 | 190.774 | N/A | 0.201 |
| ctx_512 | 512 | 64 | 5 | 187.133 | 54.888 | 3.648 |
| ctx_512 | 512 | 256 | 5 | 183.757 | 54.772 | 14.155 |
| ctx_1024 | 1024 | 1 | 5 | 212.372 | N/A | 0.223 |
| ctx_1024 | 1024 | 64 | 5 | 216.275 | 56.233 | 3.757 |
| ctx_1024 | 1024 | 256 | 50 | 209.112 | 54.124 | 14.005 |
| ctx_4096 | 4096 | 1 | 5 | 665.126 | N/A | 0.676 |
| ctx_4096 | 4096 | 64 | 5 | 665.023 | 56.388 | 4.226 |
| ctx_4096 | 4096 | 256 | 5 | 659.814 | 56.276 | 15.013 |
| ctx_8192 | 8192 | 1 | 5 | 1299.645 | N/A | 1.311 |
| ctx_8192 | 8192 | 64 | 5 | 1299.276 | 56.299 | 4.846 |
| ctx_8192 | 8192 | 256 | 5 | 1299.841 | 56.149 | 15.629 |
| ctx_16384 | 16384 | 1 | 5 | 3995.399 | N/A | 4.007 |
| ctx_16384 | 16384 | 64 | 5 | 3996.092 | 60.766 | 7.825 |
| ctx_16384 | 16384 | 256 | 5 | 4005.998 | 60.653 | 19.471 |
| real_a_median | 2303 | 256 | 3 | 384.614 | 55.907 | 14.647 |
| real_a_long | 3857 | 256 | 3 | 641.267 | 56.216 | 14.981 |
| real_c_median | 10448 | 256 | 3 | 1864.638 | 57.381 | 16.497 |
| real_c_long | 15007 | 256 | 3 | 3272.957 | 59.692 | 18.495 |

1K×256的50次TPOT均值55.006ms、中位54.124ms、经验p90 56.185ms、经验p99 56.291ms，CV约1.88%。每10次均值依次54.166、54.056、54.452、56.171、56.183ms，存在时段变化；不将小样本经验p99解释为稳定尾部保证。其余多token组TPOT CV≤0.31%，TTFT最大CV约3.66%。后续候选与基线应交错/多轮测量，同时保留分布与时间序列。

## GPU子阶段归因

下表为phase内kernel累计时间及百分比。Attention总路径含SWA/Full残余、RoPE、K写回、KV gather、SDPA，排除独立QKV/O；新增细标签后不能只加attention_swa/full两个残余项来与性能01比较。

| 输入/样本 | prefill累计ms | prefill SDPA% | prefill专家% | decode累计ms/16次 | decode Attention总% | decode专家% |
| --- | --- | --- | --- | --- | --- | --- |
| ctx_512 | 89.38 | 4.0 | 67.2 | 313.75 | 44.0 | 31.2 |
| ctx_1024 | 127.40 | 8.5 | 58.9 | 360.64 | 51.3 | 27.1 |
| ctx_4096 | 581.73 | 48.4 | 30.7 | 392.29 | 55.1 | 24.9 |
| ctx_8192 | 1215.34 | 56.6 | 26.9 | 433.86 | 59.2 | 22.7 |
| ctx_16384 | 3902.96 | 72.2 | 15.3 | 515.45 | 65.6 | 19.1 |
| real_a_median | 302.99 | 30.9 | 41.5 | 374.50 | 53.1 | 26.1 |
| real_a_long | 557.59 | 45.6 | 33.1 | 390.56 | 55.1 | 24.9 |
| real_c_median | 1775.60 | 63.2 | 23.1 | 455.93 | 61.1 | 21.5 |
| real_c_long | 3176.00 | 73.7 | 14.6 | 501.95 | 64.7 | 19.6 |

16K prefill中21个SWA层的SDPA每层约122.12–122.37ms，7个Full层约35.82–35.90ms；两类层数不同，表述按每层比较。SWA语义窗口更小却更慢，是当前实现路径值得优先替换的实测证据，不能由此断言专用内核必然达到某加速倍数。

源码依据：[sdpa_single_prefill](../../ATOM/atom/model_config/mellum_attention.py#L43)在窗口阈值生成n×n布尔band mask，并repeat_interleave展开KV头，再调用SDPA；Full走无显式mask路径。这与实测不同执行成本相符，但本轮没有硬件计数器，不能将耗时全部归因于某一带宽或计算上限。

Decode在1K时：RoPE约10.3%、K写回7.4%、分页读取7.5%、SDPA13.1%；16K时分别7.2%、5.2%、12.4%、31.3%。[install_reference_kv_rope](../../ATOM/atom/model_config/mellum_attention.py#L92)保留original rope/cache调用再覆盖K，存在重复工作；[gather_reference_kv](../../ATOM/atom/model_config/mellum_attention.py#L106)每步从分页cache重新收集连续K/V，随后可能展开KV。候选应保留参考逆频率和舍入语义，逐项减少重复计算和物化。

MoE也有明确机会：1K decode专家路径累计97.71ms/16次，其中语义launch归因的aten::index约35.68ms、bmm约27.38ms。实际源码[MellumTorchExperts.forward](../../ATOM/atom/model_config/mellum_experts.py#L54)decode通过索引物化所选专家w13/w2后bmm；prefill则使用grouped_linear。可尝试避免权重gather的专用专家执行，但此前AITER与HF存在舍入差异，不能一次替换router、激活和归约全部逻辑。

## 资源与尚未证明的事项

短trace的PyTorch allocator峰值allocated约56.48–63.24GiB，包含既有模型及KV池，峰值在profile开始重置；不是无profile峰值，也不等于整卡全部分配。详细trace1K峰值约56.69GiB。原始started/stopped回执保留了allocated/reserved与峰值，不能据此直接推算并发32容量。

122条SMI遥测命令均成功。card1温度39–47°C、利用率采样0–100%、功率159–340W；card0也出现最高32%利用率、209W。未建立SMI编号与HIP设备的可靠映射，因此不能保证独占、认定降频或解释50次计时后段变化；15秒采样也不能代表单kernel利用率。没有测量HBM实际吞吐、occupancy、cache命中或roofline。

## 下一步顺序与验收方式

1. 长prefill SWA候选：复用现有可用Attention接口，单独验证窗口边界、RoPE及输出语义；先比较4K/16K单层，再测完整模型。保持其它后端不变，关闭profiler计时，验证题目要求的生成一致性门槛；全logits差异作为诊断而非额外硬门槛。
2. Decode候选：优先削减重复RoPE/K写回及分页gather/展开，分别做消融；同时保留MoE避免权重物化路线。以1K×256和长输入×256衡量端到端TPOT，不只看单kernel。
3. 服务/多请求接入需明确推进，才能建立场景A并发32与C到达过程的真实基线；批量接入本身若改变执行策略，应披露其贡献。单请求reference不支持多请求，不能串行伪装并发。
4. Graph/编译放在动态索引、Python取值和分配路径清理之后；当前profiler扰动明显，不以trace间隙承诺Graph收益。暂不追加同形状大规模bs1采集，也不需要安装新工具。

这些方向是选择与实现建议，尚未产生优化后成绩。正式A/B/C复现、候选消融与准确率回归仍是Step1后续工作。
