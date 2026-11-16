# 28 原生Attention+FusedMoE完整HumanEval

20261116T010456387933Z paired_functional_evaluation_complete，本地BF16差距≤2pp检查通过。候选164题全部生成、评分；仅GPU1执行，HF直接复用20261115T213706098181Z已完成参考，未重新生成。

| 指标 | HF | 原生Attention+FusedMoE |
| --- | --- | --- |
| 通过题数 | 145/164 | 144/164 |
| pass@1 | 88.4146% | 87.8049% |
| 达到2048输出预算 | 4 | 5 |
| 提取失败 | 5 | 6 |

候选少通过1题，绝对差0.60976个百分点。新增通过HumanEval/75、113；损失118、132、134。全部164题保留分母，截断/提取失败未剔除。84/164完整生成序列与HF相同，文本分叉不妨碍本次任务指标达标。

## 审计

相同输入、关闭thinking、最多2048token、同一提取规则、官方OpenAI HumanEval隔离评分。双方逐题164唯一ID/通过计数与score.json重新核验一致；复用HF generation与原参考内容相同。全部运行源码hash通过，5631源张量加载完整。

实际设置MELLUM_ATTENTION_EXECUTION=native、SDPA_DECODE=0、REQUEST_BATCHING=0、MOE_BACKEND=fused_torch_route。layer0审计124次aiter_varlen prefill、2926次paged_attention_triton decode，与FusedMoE对应，未测成参考Attention回退。

[评分、逐题结果及来源指纹](../../artifacts/performance/28_native_attention_humaneval/summary.json)。这是单轮本地chat/full-code协议功能检查，不断言裁判模板完全相同，也不消除原生MoE的重复执行波动；公开200/16k指标仍独立，不宣布整题正式验收完成。

## 当前取舍

结合性能26（A64 121.50tokens/s，C20 TTFT/TPOT p99 1.7263/0.13478秒），保留原生Attention+FusedMoE为当前性能主线。单FusedMoE147/164与本组合144/164是两次实际生成，不能将3题差全部归因Attention而忽略执行波动；两者相对HF均达到本轮2pp检查。

下一步把prefix兼容路径接入原生cached prefill，测组合真实C负载，保留现成高速组合做对照。不再为该组合开展无界逐token数值追踪；最终配置仍需正式ABC、累计消融及题目要求的独立正确性检查。
