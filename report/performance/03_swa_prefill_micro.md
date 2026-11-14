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

目前只说明候选可执行且局部性能有潜力。初次micro完成时还没有真实模型HF token匹配、HumanEval、整模型TTFT/TPOT或正式A/C成绩（后续短模型验证见下节）。下一步运行window边界HF对照，再做long及ABBA整模型性能；不能因为0.6%张量相对差异自动判通过或不通过。默认Step0不变。

## 后续：真实模型窗口与长输入短回归

2026-11-14用户完成HF对照：window运行`20261114T112559495973Z`，long运行`20261114T112653344802Z`。两者均为`tokens_exact_logits_review_required`；实际MELLUM_SWA_PREFILL_OPT=1，其余Step0组合及正常生成保持，ignore_eos=False，无强制历史；加载5631/5631、missing/unexpected为空。

| 输入 | 输出检查 | token结果 | prefill logits max_abs / RMS |
| --- | --- | --- | --- |
| 1023 | 8 | 8/8一致 | 0 / 0 |
| 1024 | 8 | 8/8一致 | 5.75 / 3.8390 |
| 1025 | 8 | 8/8一致 | 1.00 / 0.2421 |
| 8191 | 4 | 4/4一致 | 3.75 / 1.7104 |
| 8192 | 4 | 4/4一致 | 4.00 / 1.5030 |
| 8193 | 4 | 4/4一致 | 3.25 / 1.3644 |
| 16384 | 4 | 4/4一致 | 4.50 / 2.3847 |

合计7/7序列、40/40 token一致。1023未启用局部候选，完整logits仍精确；其余6例的32个生成位置logits均非逐位一致，但选中的greedy token与HF一致。max_abs不能单独解释为采样错误概率，也不将5.75等差异直接称为可忽略误差。

结论是**短边界回归通过，可以进入整模型性能对照**，不是优化版完成准确率验收。当前每条仅4或8输出，可能未暴露后续分叉，尚未覆盖公开集200×64或HumanEval；后续采用既有64token正常生成覆盖核对，无需为了进入性能实测先追求所有logits bitwise一致。

下一步执行`/opt/venv/bin/python /workspace/scripts/benchmark_mellum_swa.py`，获取相同环境ABBA基线/候选的端到端TTFT、TPOT、wall及较长输出token差异，再决定收益与准确率的综合处理。原始轻量记录已加入本报告同一归档manifest，未修改GPU产物或模型代码。
