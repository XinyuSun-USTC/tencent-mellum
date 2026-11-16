# 29 原生Attention+FusedMoE+prefix组合

批次/workspace/results/mellum_native_prefix/20261116T025408685476Z complete。独立104产物及源码hash通过；Full/W4/W1024矩形Attention结构探针均max_abs0，8次真实模型off/on中4条序列相同，记录差异但不当作功能失败或通过。累计复用46592token；最终组合尚无独立完整HumanEval。

| 指标 | 原生组合无prefix（性能26） | +prefix |
| --- | --- | --- |
| A64输出tokens/s/GPU | 121.4966 | 115.9656 |
| C20输出tokens/s/GPU | 55.1928 | 56.2066 |
| C20 TTFT p99秒 | 1.72630 | 1.12319 |
| C20 TPOT p99秒 | 0.13478 | 0.12185 |

C20 20/20成功，19/20命中，共72352token；实际aiter_varlen_cached，未落回参考SDPA。观察TTFT p99下降34.94%、TPOT p99下降9.60%，吞吐+1.84%。A64全成功，但无命中，吞吐观察下降4.55%；不同GPU/负载、单次小集，暂不认定该回退由prefix必然造成，也不当正式配对加速。保留prefix为有价值的C候选，但先解决当前主线更大执行开销。

[紧凑指标及来源指纹](../../artifacts/performance/29_native_prefix/metrics.json)。原始子结果capacity/summary.json，真实输入/输出/请求/命中日志保留；本轮未补功能评分、不声称组合正式正确性已验。
