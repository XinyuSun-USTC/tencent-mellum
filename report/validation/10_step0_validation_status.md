# Step0当前验证覆盖与未完成项

最新状态：候选10例完整prefill及32个强制历史cached位置精确。下一步正常greedy短生成；默认配置、长prompt、多请求和正式Step0尚未通过。见[验证34](34_cached_decode_exact_and_normal_generation.md)。

最新覆盖：新增10例完整prefill的logits及300层状态均精确；已有四固定探针也精确。该候选仍只保证已测prefill，正在建立强制同历史decode/KV基线，Step0未完成。见[验证26](26_prefill_passed_and_forced_decode_baseline.md)。

当前里程碑：显式separate+torch MoE+CPU逆频率/SDPA候选四例完整prefill logits、120份层状态和56份子算子精确。候选不覆盖原KV/decode，Step0未完成；下一步10例新内容/边界prefill。见[验证24](24_fixed_prefill_exact_and_new_inputs.md)。

当前状态以[04 正确性协议](../../docs/plan/04_correctness_protocol.md)为准：默认torch MoE+AITER Attention，expanded仍7/11分叉，Step0未完成。下面的9输入168token表是早期AITER后端历史覆盖，不是新后端完整验收。

日期：2026-11-11。结论：已有可运行的单卡ATOM适配初版；尚未完成Step0验收，未进入性能优化。

## 已确认覆盖

| 开发集 | 样例数 | 每例新token | 对齐 |
|---|---:|---:|---|
| short | 2 | 64 | 全部精确一致 |
| window 1023/1024/1025 | 3 | 8 | 全部精确一致 |
| long 8191/8192/8193/16384 | 4 | 4 | 全部精确一致 |

以上9个固定开发输入，共168个新token，均在各自HF参考与ATOM之间匹配；是小规模开发覆盖，不是独立随机集或题目200prompt。早期重复16token smoke不重复计数。具体原始结果见[边界记录06](06_mellum_boundary_suite.md)和[RMSNorm回归09](09_mellum_rmsnorm_rounding_fix.md)。

- 独立HF5.10.1可运行真实checkpoint；ATOM仍使用冻结HF5.2.0与原ROCm PyTorch。
- ATOM模型构造、加载覆盖5631/5631、warmup、单请求prefill/decode、正常关闭均已实跑。
- 首个明确的RMSNorm顺序差异已作Mellum局部修正，首层input/Q norm实际对齐。

## 仍未闭合

1. logits/末层仍有差异；long prefill max abs2.75–6.0，RMS0.719–2.518。没有定量容差依据，不写成数值验收通过。MoE等需使用相同算子输入隔离，不能把上游漂移造成的输出差异误归因。
2. long仅4新token、window仅8，尚未覆盖更完整生成。需增加长输入输出长度及更多代码/数学prompt，区分HF内部cached/prefill的边界选择。
3. 1024窗口物理边界静态疑点尚未由独立mask/kernel试验排除，token样例通过不替代逐位置验证。
4. 多请求、不同长度/结束时间、prefix cache/chunked组合、HTTP/chat服务尚未验证。
5. 题目200prompt/reference/evaluator与HumanEval有效参考仍缺；不能以168token开发结果冒充正式匹配率或HumanEval通过。
6. 一键服务交付、最终验证说明与提交历史收尾未完成；Git作者身份尚未提供，不伪造提交。

## 后续优先顺序

先扩展生成长度和独立算子数值证据，再验证多请求/HTTP与选定缓存/调度组合。正式测试能完成的部分采用统一协议，缺项明确披露。仍以最快正确Step0为目标，不引入性能优化或更换冻结依赖。

- 最新实现状态：Mellum默认MoE已改为PyTorch正确性后端；GPU四例同输入首层MoE与HF逐元素精确一致，完整加载审计通过。旧short/window/long及expanded结果属于此前AITER后端，不能当作新后端正常生成回归。下一步short64无干预回归，见[验证17](17_official_backend_confirmed_and_step0_torch_moe.md)。

- 新torch后端short64已完成无干预正常回归：2条128新token完全匹配，logits最大差1.75/1.25，仍待数值审查。旧AITER expanded的8/11分叉尚需在新后端重测，下一步expanded；见[验证17](17_official_backend_confirmed_and_step0_torch_moe.md)。

- 新torch后端expanded真实回归：4/11全匹配、7条分叉，Step0未完成。旧AITER3/11与新torch4/11不构成整体正确结论；下一步同QKV Attention诊断，见[验证18](18_torch_expanded_and_attention_replay.md)。
