# 32 原生Attention+FusedMoE+prefix完整HumanEval

GPU0 20261116T044507881892Z paired_functional_evaluation_complete，本地BF16 2pp检查通过。完整164题、关闭thinking、最多2048token；复用HF既有参考，未重跑HF生成。

| 指标 | HF | 原生prefix组合 |
| --- | --- | --- |
| 通过 | 145/164 | 147/164 |
| pass@1 | 88.4146% | 89.6341% |
| 2048预算截断 | 4 | 4 |
| 提取失败 | 5 | 5 |

绝对差1.21951pp，新增通过5题、损失3题，均保留164分母。84/164完整生成序列相同，仅作描述，不阻断功能评分。单次生成不能证明prefix提高模型能力，也不能把与旧原生144/164的差值都归因缓存。

## 独立审计

双方164唯一task_id、passed计数与官方score.json一致；HF内容与原复用参考相同；加载5631项完整、运行源码hash未变。prefix日志128次native cached事件，132个命中请求、2128token，均backend=aiter_varlen_cached，确实覆盖缓存命中路径，而非只打开空开关。HumanEval共享前缀较短，16k/长前缀证据仍需独立看性能29与后续最终回归。

**本配置仍groups=L按请求投影。** 它不等同正在推进的fused_moe_native_batched（groups为空）；不能把本分数赋给439.67tokens/s的新配置，也不能据此宣称新配置+prefix已验。后者需计划33功能结果及之后的组合回归。

[评分、逐题与命中证据](../../artifacts/performance/32_native_prefix_humaneval/summary.json)。公开200/16k独立要求及最终空闲复测仍保留。
