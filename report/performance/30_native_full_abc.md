# 30 原生Attention+FusedMoE完整ABC

/workspace/results/mellum_abc_optimized/20261116T025416247034Z complete，全部B50/A512/C200成功。累计47.37分钟，原生profile、GPU1、prefix/chunked/MTP关闭；独立98产物、源码/加载与请求长度/stream/实际batch审计通过。这里是已审计客户端的题目规模本地复测，不是裁判最终三次中位数。

| 场景 | 成功 | 输出tokens/s/GPU | TTFT p50 / p99秒 | TPOT p50 / p99秒 |
| --- | --- | --- | --- | --- |
| B | 50/50 | 17.7980 | 0.21798 / 0.23388 | 0.05693 / 0.05731 |
| A | 512/512 | 121.2755 | 4.47436 / 4.87686 | 0.24635 / 0.25147 |
| C | 200/200 | 74.7039 | 1.01352 / 2.54046 | 0.16334 / 0.21940 |

A实际decode batch32，C22。相对旧R本地完整A的41.5797，吞吐观察约2.92倍；旧R C199/200超时，本轮200/200成功，服务容量明显改善。但以题目给定官方Baseline比较：A仅19.00%；C TTFT p99为官方2.557倍，TPOT p99为5.317倍。当前仍显著落后官方性能，不可把相对自己R的改善冒充官方加速。baseline.json target_g为空，不编造分数。

客户端EvalScope1.15数据适配/stream语法/归一化指数到达差异沿用性能16并保留。GPU0同期跑prefix任务，存在共享资源干扰；最终固定版本需空闲环境复测。小C20不能代替完整C200，本批提供第一份完整有效C尾延迟。

结合完整HumanEval28（144/164，HF145/164）保留原生组合主线。下一个有明确源码依据的开销是campaign groups=L仍按请求拆分投影，计划31用显式profile取消这项控制、恢复整批GEMM，先小负载性能、显著收益后功能评估。不无限数值定位，也不在版本未变时原样重跑47分钟ABC。

[完整指标与manifest](../../artifacts/performance/30_native_full_abc/metrics.json)，对应目录保留manifest及阶段产物hash，原始SQLite与日志在运行目录。
