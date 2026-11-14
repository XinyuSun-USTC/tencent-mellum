# 03 优化点1：SWA单层GPU初测

运行`/workspace/results/mellum_swa_optimization/20261114T013307034534Z`完成micro-only，五个长度均正常返回有限输出。原始结果与源码指纹见[归档](../../artifacts/performance/03_swa_micro/manifest.json)，实现协议见[计划10](../../docs/plan/10_swa_prefill_optimization.md)。本轮仅分析用户执行的GPU产物，未追加GPU任务。

## 实测

相同随机BF16 Q32/KV4/head128，window1024，三次预热后两后端交错各10次CUDA event计时。reference包含显式mask及KV展开，候选原生local attention。下表中位数，单位ms；relative L2为输出张量差异，不是token错误率。

| 输入 | reference ms | candidate ms | 局部加速 | max_abs | relative L2 |
| --- | --- | --- | --- | --- | --- |
| 1023 | 0.7183 | 0.1851 | 3.88× | 0.00390625 | 0.000060 |
| 1024 | 0.5343 | 0.1925 | 2.78× | 0.03125000 | 0.005784 |
| 1025 | 0.5814 | 0.2187 | 2.66× | 0.01562500 | 0.005733 |
| 4096 | 13.0933 | 0.7805 | 16.78× | 0.01562500 | 0.005988 |
| 16384 | 126.6280 | 3.1603 | 40.07× | 0.01562500 | 0.006244 |

## 解释与边界

4K/16K局部加速约16.8×/40.1×，支持进入真实模型对照。不是整模型加速：Full attention、MoE、投影、KV及decode都没有在这个微基准内优化。16K旧SDPA约126.6ms，与此前真实层约122ms处于相近量级；不能直接以21层乘差值当实测TTFT收益。

1024及以上relative L2约0.0057–0.0062（0.57%–0.62%），并非逐位精确。1023约0.000060，阈值变化值得保留：AITER源码会在left>=sequence length时将left归一化为无限窗口，且部分dispatch排除SWA，可能切换kernel路径；当前没有trace证明具体选路，不下确定结论。1023微基准仍用了显式reference mask，但整模型1023不启用候选，因此该行不能声称整模型短输入加速。

日志存在gfx950-only opus模块导入警告，本次所有长度已成功完成；这不表示实际执行调用了gfx950 opus。fmha_v3符号日志同样不足以证明local路径执行了该kernel。

目前只说明候选可执行且局部性能有潜力。还没有真实模型HF token匹配、HumanEval、整模型TTFT/TPOT或正式A/C成绩。下一步运行window边界HF对照，再做long及ABBA整模型性能；不能因为0.6%张量相对差异自动判通过或不通过。默认Step0不变。
