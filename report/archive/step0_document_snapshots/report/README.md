# 研究与结果索引

[38 Step0开发200条对照](validation/38_step0_developer_200_agreement.md)：200/200序列、12800token及完整逐步logits精确通过；核心适配闭环，HumanEval pass@1待补，下一主线服务测量入口与profiling。

[37 Step0长prefill增强](validation/37_step0_long_prefill_extension.md)：1023/1024/1025及8191/8192/8193/16384 GPU smoke均精确通过；16k能力已验证，长输入64token回归256位置精确通过。

[36 Step0验收审计](validation/36_step0_acceptance_audit.md)：已保存768位置重验精确，CPU49项通过；16k能力不支持、200条/HumanEval/服务缺项，不能宣布完整Step0通过。

[35 Greedy并列与调度修复](validation/35_greedy_tie_break_and_sampling_dispatch.md)：采样入口修复后10/10序列、640/640 token及640个完整逐步logits向量精确通过；长prompt/并发/服务仍待补齐。

[34 Cached闭环与正常生成](validation/34_cached_decode_exact_and_normal_generation.md)：固定32位置精确；HF隐式专家切换已对齐，普通short 128位置token/logits全精确。

[33 Router首差与局部候选](validation/33_router_projection_root_and_candidate.md)：一个logit改变MoE权重，两路真实输出重建精确；45+5项CPU通过，GPU待执行。

[32 索引8层后半段定位](validation/32_layer8_tail_replay.md)：首decode前层及该层Attention精确，轻量O/norm/router/MoE重放待GPU。

[31 窗口首decode逐层观察](validation/31_window_decode_layer_observation.md)：四例首层重建精确；全层只读成对取证已准备，GPU待执行。

[30 短decode精确与窗口快照](validation/30_short_decode_exact_and_window_snapshot.md)：25/32位置精确，轻量HF首层Q/K/V重建待GPU，不重跑完整基线。

[29 KV旋转结果与cached SDPA](validation/29_kv_rope_result_and_cached_sdpa.md)：保留实际缓存，候选对齐单tokendecode算术，合并首层同输入取证。

[28 Decode基线与KV旋转控制](validation/28_forced_decode_baseline_and_kv_rope_control.md)：4prefill精确、28decode不同，历史/position通过；正确K写回对照待GPU。

[27 Decode探针故障与运行器监测](validation/27_decode_probe_failure_and_watchdog.md)：HF已完成、worker退出造成上层等待；修正后复用参考恢复ATOM，GPU待执行。

[26 Prefill通过与强制历史decode](validation/26_prefill_passed_and_forced_decode_baseline.md)：10例logits及300层状态精确；四输入32份逐步logits基线已准备，GPU待执行。

[25 协议恢复与窗口阈值](validation/25_prefill_schema_recovery_and_window_threshold.md)：元数据问题无GPU恢复，9例精确、1024边界不同；HF mask调用修正待验证。

[24 完整prefill闭环与新增输入](validation/24_fixed_prefill_exact_and_new_inputs.md)：四例logits及全部观察精确；新增10例prefill准备完成，decode/KV仍待验证。

[23 RoPE初始化证据与prefill候选](validation/23_rope_initialization_candidate.md)：CPU/GPU来源差异已证实，CPU inv/Q/K对照精确；显式候选GPU待验证。

[22 Prefill组合与RoPE系数核查](validation/22_prefill_combination_and_rope_coefficients.md)：首层QKV/QK norm已精确、context仍不同；轻量初始化来源实验待GPU执行。

[21 QKV打包证据与成对prefill候选验证](validation/21_qkv_packing_and_paired_prefill.md)：同shape内核一致，K/V打包有差异；显式separate候选和成对一步prefill已准备。

当前正确性执行顺序与状态以[统一协议](../docs/plan/04_correctness_protocol.md)为准；投影CPU预检完成，GPU待执行，见[验证20更新](validation/20_stepwise_rope_result_and_projection_replay.md)。

[项目总导航](../docs/README.md) · [当前方案](../docs/plan/01_step0.md) · [文档管理](../docs/01_document_management.md)

## 研究文档阅读顺序

1. [01 环境与开发入口](research/01_environment.md)
2. [02 ATOM架构、模型接入与加载链](research/02_atom_architecture.md)
3. [03 Mellum模型结构、权重与候选组件](research/03_mellum_model.md)
4. [04 正确性资产与参考实现调查](research/04_correctness_reference.md)

这些页面是2026-11-10既有调查的主题整理，不代表已完成模型适配或GPU验证。保留路径、符号、关键代码及验证边界。原文证据数字节号可查归档；当前决策只以plan为准。

## 日志、结果与历史

- [worklog.md](worklog.md)：唯一按自然日期维护的简要日志。
- [01 HF参考环境建立记录](validation/01_hf_reference_environment.md)：HF5.10.1已安装，配置/导入与必需依赖检查通过，尚未执行模型推理。
- [02 HF小模型CPU验证](validation/02_hf_mellum_cpu.md)：24/24通过，尚未运行真实checkpoint或GPU。
- [03 真实checkpoint GPU参考验证](validation/03_hf_mellum_gpu.md)：普通终端真实权重BF16短输入通过，加载审计与两条cache一致性通过。
- [04 ATOM配置与比较协议](validation/04_atom_step0_batch_ab.md)：配置12项、比较器14项通过，真实HF配置字段一致；尚无ATOM模型推理。
- [05 ATOM Mellum模型接入](validation/05_atom_mellum_model_wiring.md)：真实加载/推理与两条16token HF对比已通过，跨窗/长输入与正式验收待完成。
- [06 64token与边界套件](validation/06_mellum_boundary_suite.md)：三组输入及CPU回归完成，GPU尚未执行。
- [07 64token分叉诊断](validation/07_mellum_short_mismatch_diagnosis.md)：两条发生分叉，暂停长输入；同前缀/分层诊断待GPU执行。
- [08 第0层子算子定位](validation/08_mellum_layer0_diagnosis.md)：embedding一致、误差从第0层累积；发现HF内部cached/prefill差异，下一步观察子算子与实际权重。
- [09 RMSNorm数值修正](validation/09_mellum_rmsnorm_rounding_fix.md)：同输入/权重下的首个差异已定位并局部修正，26项CPU测试通过，GPU回归待执行。
- [10 Step0验证覆盖总表](validation/10_step0_validation_status.md)：9个开发输入168新token匹配；剩余数值、长生成、多请求、服务及正式验收待完成。
- [11 扩展回归与同输入路由诊断](validation/11_mellum_expanded_and_router_replay.md)：11条输入各最多64token；路由重放与CPU24项检查已准备，GPU待执行。
- [12 扩展回归失败与路由证据](validation/12_expanded_mismatch_and_router_findings.md)：11条中8条分叉；已测同输入路由集合一致，权重精度差尚未归因，下一步四条短探针。
- [13 短探针与完整MoE同输入重放](validation/13_short_probes_and_moe_input_replay.md)：HF内部差异与ATOM并列logits已区分；准备固定HF输入的首层完整MoE诊断，GPU待执行。
- performance、submission：有实际内容时再建立，按目录内递增序号维护。
- [原始阶段研究归档](archive/initial_research/README.md)：原文与SHA256迁移清单，停止追加。

- [14 同输入MoE与路由受控比较](validation/14_moe_same_input_and_routing_study.md)：相同输入/router下MoE仍差约0.45%；准备权重舍入及HF路由两组重放。

- [15 路由结果与专家算术隔离](validation/15_moe_routing_results_and_expert_arithmetic.md)：完整HF路由后仍差约0.41%–0.43%；准备原始专家权重的激活/累加舍入实验。

- [16 专家累加与HF分派核验](validation/16_expert_reduction_and_hf_dispatch.md)：FP32累加接近保存HF；纠正此前遗漏的专家装饰器分派，准备官方后端直接重放。

- [17 官方后端确认与Step0 Torch MoE](validation/17_official_backend_confirmed_and_step0_torch_moe.md)：官方grouped_mm四例精确复现HF；接入保留加载接口的PyTorch正确性后端，CPU31项及官方CPU6组合通过，GPU待验证。

- 最新进展：[17 GPU固定输入重放](validation/17_official_backend_confirmed_and_step0_torch_moe.md)四例MoE逐元素一致，加载审计完整；下一步无干预short64正常回归。

- [18 Torch扩展回归与Attention定位](validation/18_torch_expanded_and_attention_replay.md)：4/11条完全匹配、7条分叉；准备同QKV的HF/AITER旋转与SDPA重放。

- [19 Prefill内核与SDPA受控验证](validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)：同QKV确认prefill数值差异，准备仅限短单请求的显式SDPA实验，默认AITER未改。

- [20 分步旋转结果与线性投影核查](validation/20_stepwise_rope_result_and_projection_replay.md)：分步旋转未带来整体改善，准备固定输入/权重区分QKV打包与tgemm计算。
