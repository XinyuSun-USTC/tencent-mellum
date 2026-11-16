# 34 整批投影+原生prefix组合性能

GPU0 20261116T101213244110Z complete。原生Attention/FusedMoE/整批投影基础上prefix=True；Full/W4/W1024矩形结构探针max_abs0，8次模型off/on中3条序列相同，实际复用46592token。自由生成相同率仅报告，不直接判定任务功能；此最终组合未获独立HumanEval。

| 指标 | 整批投影无prefix（性能31） | +prefix |
| --- | --- | --- |
| A64输出tokens/s/GPU | 439.6727 | 438.8336 |
| C20输出tokens/s/GPU | 63.5585 | 63.9490 |
| C20 TTFT p99秒 | 1.58614 | 1.01427 |
| C20 TPOT p99秒 | 0.08037 | 0.07378 |

全部请求成功；C19/20命中72352token，A无命中。A吞吐观察-0.19%近似持平，C TTFT p99下降约36.05%、TPOT p99下降约8.20%，支持prefix在最快现有配置上保留。GPU与同期负载不同、单次小集，不冒称正式配对加速或C200成绩。

独立核验全部产物/源码hash、同一输入manifest、5631加载、实际原生cached执行。结果及指纹见[指标](../../artifacts/performance/34_batched_prefix/metrics.json)。下一步与单次KV写入合并，补整体功能与最终完整ABC，不继承L控制版本147/164作为本配置成绩。
