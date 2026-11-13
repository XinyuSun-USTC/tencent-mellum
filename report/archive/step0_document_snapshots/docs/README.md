# Mellum on ATOM 文档导航

当前执行点：[正常greedy候选回归](../report/validation/34_cached_decode_exact_and_normal_generation.md)。32/32强制历史logits精确，下一步两条短生成及被动cached logits，默认不变。

当前执行点：[router投影局部候选验证](../report/validation/33_router_projection_root_and_candidate.md)。索引8首差已由真实层输出重建确认，显式PyTorch gate待完整32位置回归。

当前执行点：[索引8层后半段重放](../report/validation/32_layer8_tail_replay.md)。全层观察将首差定位到Attention之后，正常模型未改。

当前执行点：[窗口首decode逐层观察](../report/validation/31_window_decode_layer_observation.md)。首层全部精确，单窗口两位置定位后续首差。

当前执行点：[滑窗首decode Q/K/V快照核查](../report/validation/30_short_decode_exact_and_window_snapshot.md)。三短输入decode已精确，剩余窗口样例不同。

当前执行点：[同实际分页KV的SDPA decode对照](../report/validation/29_kv_rope_result_and_cached_sdpa.md)。旋转/K写回单独未闭合decode，显式新候选待GPU验证。

当前执行点：[KV旋转单因素对照](../report/validation/28_forced_decode_baseline_and_kv_rope_control.md)。强制历史基线已完成，decode仍不同；默认关闭候选。

当前执行点：[decode挂起修复与HF复用恢复](../report/validation/27_decode_probe_failure_and_watchdog.md)。旧命令需Ctrl+C；探针/失败检测已修，四条decode基线未完成。

当前执行点：[强制同历史decode基线](../report/validation/26_prefill_passed_and_forced_decode_baseline.md)。10例完整prefill已精确，进入逐步cached对照，默认路径不变。

当前执行点：[协议恢复与1024边界修正](../report/validation/25_prefill_schema_recovery_and_window_threshold.md)。离线确认9/10完整prefill精确，候选边界调用修正待GPU验证。

当前执行点：[四例完整prefill精确与新集验证](../report/validation/24_fixed_prefill_exact_and_new_inputs.md)。候选已达到完整固定前向精确，下一步10例新内容/边界，不跳过decode/KV。

当前执行点：[CPU逆频率prefill候选](../report/validation/23_rope_initialization_candidate.md)。初始化来源差异已有GPU证据，候选GPU待验证，默认路径未改。

当前执行点：[RoPE系数来源核查](../report/validation/22_prefill_combination_and_rope_coefficients.md)，组合prefill尚未闭合，默认后端不变。

当前执行点：[成对QKV prefill控制](../report/validation/21_qkv_packing_and_paired_prefill.md)，默认packed不变，统一流程见[04协议](plan/04_correctness_protocol.md)。

当前执行入口：[04 Step0正确性协议与实现状态](plan/04_correctness_protocol.md)。暂停扩展生成，先完成固定投影、完整prefill及强制同历史decode/KV。

当前目标：最快完成正确的单卡Step0；功能适配尚未实施，本地HF支持已复核，独立HF5.10.1已安装并通过配置/导入检查。Step1优化在Step0完成后讨论。文档整理日期：2026-11-10。

## 推荐阅读顺序

1. [题目要求摘要](requirements/01_assessment_summary.md)与[原始PDF](requirements/2026%20CSIG云一计算加速团队实习校招面试题.pdf)：正式要求。
2. [项目指南](MELLUM_ATOM_RESEARCH_GUIDE.md)：背景、冻结环境与工作区约束。
3. [当前Step0方案](plan/01_step0.md)：当前有效实施路线与完成条件。
4. [详细实施与验证计划](plan/03_atom_step0_implementation_and_validation.md)：分批改动与逐项放行。
5. [决策记录](plan/02_decisions.md)：主要取舍与被替代的旧方案。
6. [研究索引](../report/README.md)：具体源码和环境证据。
7. [每日工作日志](../report/worklog.md)：自然日期进度。

## 文档维护

遵循[01 文档管理约定](01_document_management.md)。目录内数字前缀用于同类文档排序，不表示日期。原D0/D1等已[完整归档](../report/archive/initial_research/README.md)，不再作为当前计划。

下一项技术工作：准备真实checkpoint单卡BF16短输入HF参考推理与加载审计。小模型CPU验证已24/24通过，见[验证记录](../report/validation/02_hf_mellum_cpu.md)；尚未运行真实模型或GPU。支持调查见[参考研究](../report/research/04_correctness_reference.md)，尚未运行完整模型或GPU任务。

最新进展：真实checkpoint的HF单卡BF16短输入已跑通，加载完整性和两条cache一致性通过，见[GPU验证](../report/validation/03_hf_mellum_gpu.md)。接下来可推进ATOM必要适配与对应参考对比；长上下文和正式验收仍待完成。

当前开发进展：第一批[配置接入与比较协议](../report/validation/04_atom_step0_batch_ab.md)已完成，下一批为独立Mellum模型与权重加载。

当前执行点：模型/注册与加载审计初版已实现，CPU专项通过；下一步为[ATOM首次GPU引擎验证](../report/validation/05_atom_mellum_model_wiring.md)。尚未完成真实加载或HF/ATOM输出对比。

最新验证：ATOM在冻结环境已完成真实加载与两条短greedy对比，均与HF精确一致。下一步进入分层/teacher-forcing与窗口边界验证；见[模型验证最新结果](../report/validation/05_atom_mellum_model_wiring.md)。

下一执行点：[64token与窗口/长输入套件](../report/validation/06_mellum_boundary_suite.md)，先在普通终端运行short；三组开发输入已准备，GPU尚未执行。

最新阻碍：64token回归两条分叉，prefill logits已有差异，暂停window/long。下一步[定向同前缀与分层诊断](../report/validation/07_mellum_short_mismatch_diagnosis.md)，尚未确定根因。

当前诊断点：embedding一致，差异从第0层起累积；已准备[第0层子算子与实际权重观察](../report/validation/08_mellum_layer0_diagnosis.md)，GPU尚未执行，暂停边界测试。

最新修正：[Mellum RMSNorm顺序与舍入](../report/validation/09_mellum_rmsnorm_rounding_fix.md)已按官方HF实现，CPU26项通过；下一步短64token带层/子算子dump回归，尚未证明全部分叉消除。

最新结果：RMSNorm修正后短回归128个新token全部匹配，首层input/Q norm精确对齐；整体logits与末层仍有未解决差异。下一执行window，详情见[修正后GPU记录](../report/validation/09_mellum_rmsnorm_rounding_fix.md)。

最新覆盖：window三例24个新token全部与HF一致，logits差异和mask静态疑点仍保留；下一步long开发集，见[边界结果](../report/validation/06_mellum_boundary_suite.md)。

当前覆盖汇总：[Step0验证总表](../report/validation/10_step0_validation_status.md)。short/window/long共9个开发输入168新token匹配；不是正式验收，剩余logits、长生成、多请求和服务仍待验证。

当前下一步：[扩展回归与同输入路由诊断](../report/validation/11_mellum_expanded_and_router_replay.md)。11条开发输入统一最多64个新token，先隔离HF/AITER同logits路由；CPU24项通过，GPU待普通终端执行。

- 当前阻碍：扩展11条中8条生成分叉，Step0未完成；同输入路由集合一致，权重精度差未归因。下一步[四条短探针定位](../report/validation/12_expanded_mismatch_and_router_findings.md)。

- 当前下一步：[固定HF输入重放首层完整MoE](../report/validation/13_short_probes_and_moe_input_replay.md)。短探针仍分叉，已区分HF内部cached/prefill变化及ATOM并列最高logits，未确认剩余根因。

- 当前执行点：[首层MoE路由受控比较](../report/validation/14_moe_same_input_and_routing_study.md)。同输入/router仍有MoE输出差异，下一步隔离权重舍入与完整HF路由。

- 当前下一步：[第0层专家激活与累加舍入实验](../report/validation/15_moe_routing_results_and_expert_arithmetic.md)。路由对齐仅小幅改善，主要MoE差异仍未闭合。

- 当前下一步：[直接核验官方专家后端](../report/validation/16_expert_reduction_and_hf_dispatch.md)。FP32累加最接近参考，已更正遗漏HF专家装饰器分派的分析。

- 当前执行点：[Step0 PyTorch MoE路径GPU核验](../report/validation/17_official_backend_confirmed_and_step0_torch_moe.md)。官方后端已确认，保留加载接口接入正确性计算路径；先固定输入重放，再正常生成回归。

- 当前下一步：新torch MoE四例GPU固定输入输出精确对齐；恢复无干预short64逐层正常回归，见[验证17更新](../report/validation/17_official_backend_confirmed_and_step0_torch_moe.md)。

- 当前下一步：新torch后端short64正常生成128/128匹配，但logits仍有差异；重跑此前分叉的expanded开发集，见[验证17](../report/validation/17_official_backend_confirmed_and_step0_torch_moe.md)。

- 当前下一步：[同QKV Attention诊断](../report/validation/18_torch_expanded_and_attention_replay.md)。新torch expanded仍7条分叉，不再扩大样本，先区分旋转与Attention内核差异。

- 当前下一步：Attention同QKV参考闭环已成立，但独立HIP旋转不是首层实际融合路径；补测Triton旋转/缓存分支，见[验证18更新](../report/validation/18_torch_expanded_and_attention_replay.md)。

- 当前下一步：[受限SDPA prefill短回归](../report/validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)，仅作单变量诊断，默认AITER不变。

- 当前下一步：SDPA实验未使短生成全面改善；算术首层已精确，观察误差放大的第9层子算子，见[验证19更新](../report/validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)。

- 当前下一步：第9层权重一致、输入不同且路由边界敏感；显式SDPA+分步RoPE prefill对照待GPU执行，见[验证19](../report/validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)。

- 当前下一步：[固定输入/权重的投影重放](../report/validation/20_stepwise_rope_result_and_projection_replay.md)。分步旋转未整体改善，先区分线性内核及QKV打包影响。
