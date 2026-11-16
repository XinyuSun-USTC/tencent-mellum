# 22 原生FusedMoE完整HumanEval功能评估

结论：本地配对协议下，原生FusedMoE达到BF16 HumanEval pass@1与HF差距≤2个百分点的检查，可继续推进HTTP性能与组合优化。不将此结果等同整个题目验收，不再用逐token完全一致阻止该候选功能评分。

| 指标 | HF Transformers | FusedMoE候选 |
| --- | --- | --- |
| 完整题数 | 164 | 164 |
| 通过题数 | 145 | 147 |
| pass@1 | 88.4146% | 89.6341% |
| 达到2048预算题数 | 4 | 5 |
| 提取失败题数 | 5 | 6 |

候选较HF提高1.2195个百分点，绝对差距小于2pp；新增通过5题、损失3题。所有164题保留在分母中，未删除截断或提取失败题。完整输出相同92/164，说明文本分叉不能直接视为任务功能退化；本次没有测量或替代公开200位置指标。

## 协议与核验

关闭thinking，最多2048token，相同输入与统一chat/full-code提取；使用固定官方OpenAI HumanEval评分代码及隔离执行（chroot、uid65534、AF_UNIX only）。这是明确的本地协议，不声称与未公开裁判模板完全相同。分别读取双方score.json与164条唯一task_id的逐题passed，计数与summary一致。

原批次20261115T210153552943Z候选完成，HF诊断信息set序列化失败；独立恢复20261115T213706098181Z只补HF并复用候选评分，源码未变。HF加载导出修复df33885，不改数学或评分。原始结果路径：/workspace/results/mellum_humaneval_recovery/20261115T213706098181Z。

[结果与逐题证据](../../artifacts/performance/22_fused_moe_humaneval/summary.json)，同目录保留双方评分、逐题结果、评测器manifest及来源文件SHA256。

## 后续取舍

继续计划26的GPU1候选HTTP smoke/A64/C20，不为微观浮点差异再次开启无期限数值搜索。此候选旧D200位置72.36%的独立问题仍如实保留，不宣称题目全部正确性已通过。下一步结合实际HTTP吞吐/排队，决定FusedMoE+SWA组合或prefix兼容性修复；最终配置再做必要回归与正式ABC。原生MoE固定输入重复输出波动的证据未被本次单次功能评估消除，最终复测应记录。
