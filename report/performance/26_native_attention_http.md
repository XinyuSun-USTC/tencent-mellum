# 26 FusedMoE+完整原生Attention的HTTP性能

GPU1 20261115T231450301731Z diagnostic_complete；smoke B2/A8/C2、A64/C20均全部成功，5631加载及输入/长度/stream/实际batch检查通过。独立核验全部产物及运行源码hash；与前两轮相同小负载计划/输入逐项核对一致。

| 指标 | FusedMoE | +SWA CK prefill | +完整原生Attention |
| --- | --- | --- | --- |
| A64输出tokens/s/GPU | 57.0039 | 59.6120 | **121.4966** |
| A64 TTFT p99秒 | 10.0410 | 6.2729 | **4.9136** |
| A64 TPOT p99秒 | 0.53744 | 0.51917 | **0.24968** |
| C20输出tokens/s/GPU | 30.9853 | 34.4001 | **55.1928** |
| C20 TTFT p99秒 | 22.6893 | 3.8190 | **1.7263** |
| C20 TPOT p99秒 | 0.44394 | 0.36259 | **0.13478** |

注意后两列是同一FusedMoE基础上两种Attention实现的替代，不是把SWA CK与完整原生路径当两个累计开关。完整原生A吞吐为单MoE2.13倍、SWA组合2.04倍；相对SWA组合，C TTFT p99下降约54.8%、TPOT p99下降约62.8%。仍是单次小集观察，不是官方完整ABC或无干扰配对加速。

## 路径依据

layer0日志确认A13次/C21次aiter_varlen prefill、A768次/C1202次paged_attention_triton decode，与FusedMoE同行记录；实际A decode batch32，C12。绕过参考ragged SDPA/gather路径是更大性能信号；仅凭此次结果不能分离每个子机制的收益。B2 smoke TPOT约56.44ms，样本不足以声称正式单流性能。

C20全成功不证明C200持续0.3req/s稳定；真实共享prefix、chunked调度及Graph仍未启用。另一GPU负载与前两轮不同，最终正式结果应空闲环境复测。

[紧凑三配置指标](../../artifacts/performance/26_native_attention_http/comparison.json)，原始请求DB与native日志位于运行目录，归档逐产物SHA256。

## 下一步

性能收益足够，进入完整HumanEval候选评估（计划29）；复用HF完整164参考，不重跑HF、不以D200 token差异前置阻断。该组合不能继承单FusedMoE的147/164成绩。GPU0同时测prefix真实HTTP；若两条均有价值，再研究原生Attention与prefix接入，不盲目叠加当前互斥开关。公开集门槛与长上下文结果仍需独立核验。
