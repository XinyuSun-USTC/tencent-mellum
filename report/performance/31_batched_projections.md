# 31 恢复整批投影的HTTP性能

GPU1 /workspace/results/mellum_candidate_capacity/20261116T043117581045Z diagnostic_complete，smoke/A64/C20全部成功。独立核验产物和源码hash；与性能26逐文件ATOM源码内容完全一致、相同输入manifest。新profile只取消campaign groups=L请求级线性投影拆分，其余配置相同，保留独立QKV、FusedMoE与原生Attention。

| 指标 | 原生组合/按请求投影 | 整批投影 |
| --- | --- | --- |
| A64输出tokens/s/GPU | 121.4966 | **439.6727** |
| A64 TTFT p99秒 | 4.91357 | 4.24182 |
| A64 TPOT p99秒 | 0.24968 | **0.05862** |
| C20输出tokens/s/GPU | 55.1928 | **63.5585** |
| C20 TTFT p99秒 | 1.72630 | 1.58614 |
| C20 TPOT p99秒 | 0.13478 | **0.08037** |

A吞吐约3.619倍，实际decode batch32；C20全部成功、最大batch9，TPOT p99下降约40.4%。B2 smoke TPOT约54.51ms，与原生旧B2约56.44ms差异较小，收益主要出现在批量，与取消每请求小GEMM发射的改动范围一致。

模型源码无差异、profile仅一项控制改变，对此次开发负载的性能归因较明确；但两次时间/另一卡负载不同，且只是单次小集，不能据此宣称正式A512达到439.67或官方归一化比分。该候选仍eager、prefix关闭，没有Graph/量化收益。

此配置不能继承旧组合144/164的HumanEval成绩；整批GEMM可能改变BF16输出。按计划先实测性能，收益显著后进入完整HumanEval，复用既有HF164，不以逐token相同前置阻断。若功能达标，再整合prefix/直接KV写入和最终完整ABC，避免每项小改都重跑47分钟。

[原始指标、实际native路径与来源指纹](../../artifacts/performance/31_batched_projections/metrics.json)。原始请求DB、log、逐产物hash保留。
