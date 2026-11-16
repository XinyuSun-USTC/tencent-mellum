# 33 整批投影完整HumanEval：分数提高与绝对差判定

20261116T085201556563Z paired_functional_evaluation_complete，完整164题全部计入。原生Attention+FusedMoE、整批投影groups为空、prefix关闭；保持关闭thinking/2048，复用原HF参考。

| 指标 | HF | 整批投影候选 |
| --- | --- | --- |
| 通过 | 145/164 | 149/164 |
| pass@1 | 88.4146% | 90.8537% |
| 2048预算截断 | 4 | 3 |
| 提取失败 | 5 | 3 |

候选净增4题、分数高2.4390个百分点；增6失2（损失57、134），没有剔除失败或截断题。双方逐题164唯一ID、官方score和计数核验一致，HF复用内容相同，源码hash和5631加载审计通过。实际campaign groups为空，与性能31的439.67tokens/s配置一致。

## 必须如实解释False

原题写HumanEval pass@1与参考值差距≤T，当前脚本采用绝对差，因此bf16_2pp_local_check=False。这是候选高出参考超过2pp触发，并非总体pass@1退化。当前不能把它写成严格绝对差检查已通过；也没有依据为了贴近参考降低得分。

保留原始summary和False，不修改验收门槛、不改评分/提取规则；另补signed delta=+2.4390pp及observed drop=0作为解释指标，不将后者冒充裁判认可口径。最终材料同时报告双方计数与差值，正式差距含义由裁判口径决定。

这是单轮实际生成，不证明长期统计意义上的准确率提升，不能把全部变化归因投影而忽略原生MoE波动。候选总体功能成绩和性能信号值得保留，不为这个方向性False回退到慢配置或重启无界token对齐。

[原始评分/逐题证据](../../artifacts/performance/33_batched_projection_humaneval/summary.json)；补充解释见同目录directional_metrics.json。

## 后续

以整批投影作为当前性能主线，整合已具备独立功能/性能证据的prefix，再评估最终组合。该结果不能直接赋给整批+prefix，也不能当公开200或16k独立门槛已通过。最终版本需完整ABC、功能回归及空闲重复测量，保留旧144/164原生组合与147/164原生prefix作为后备。
