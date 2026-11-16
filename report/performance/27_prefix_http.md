# 27 FusedMoE+prefix HTTP实测

GPU0 20261115T231925241791Z diagnostic_complete；smoke、A64、C20全部成功，source/产物hash复核通过，实际输入计划与性能23单FusedMoE相等。此轮prefix开启，SWA优化关闭、Attention仍为参考ragged SDPA，不是原生Attention+prefix组合。

| 指标 | FusedMoE无缓存（GPU1旧批次） | FusedMoE+prefix（GPU0） |
| --- | --- | --- |
| A64输出tokens/s/GPU | 57.0039 | 57.0946 |
| A64 TTFT p99秒 | 10.0410 | 10.0790 |
| C20输出tokens/s/GPU | 30.9853 | 31.9333 |
| C20 TTFT p99秒 | 22.6893 | 19.5069 |
| C20 TPOT p99秒 | 0.44394 | 0.41837 |

C20 19/20请求命中，每条3808token，共72352token。自然冷启动、无特别预热；A64未观察到命中，指标近似持平。C吞吐观察+3.06%、TTFT p99下降14.03%、TPOT p99下降5.76%；GPU/干扰不同、单次小样本，不作为受控正式加速。

实际命中已证明prefix接入有效，但离线完全重复请求的大幅收益不能外推真实C；参考decode和cached矩形SDPA仍有成本，prefix不能代替Attention内核优化。

原生Attention组合（性能26）的C20 TTFT/TPOT p99 1.7263/0.13478秒、A吞吐121.50，明显更值得作为性能主线。下一步优先完整HumanEval候选侧功能评估，复用HF；缓存后续接入原生cached prefill，不能直接绕过其uncached guard或在当前服务开关下假称组合已支持。

[指标与来源指纹](../../artifacts/performance/27_prefix_http/metrics.json)。原始prefix日志、请求DB与服务日志保留原运行目录。另一个用户回传目录20261115T231450301731Z是同一原生Attention性能批次，已记性能26，不算新增复测或HumanEval结果。
