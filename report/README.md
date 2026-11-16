# 研究与结果索引

当前阶段：**Step0适配开发完成、本地生成一致性验证通过，进入Step1准备。** 当前结论看39、配置看05；01–38是各自配置与运行时点的证据，不以其中“待验证”作为现状。

## 当前入口

当前基线接入：[性能16](performance/16_abc_protocol_and_client.md)：EvalScope客户端安装/CPU审计通过，题目数据适配与smoke→B/A/C入口准备完成，尚无正式GPU负载成绩。

最新补证：[性能15](performance/15_indexed_decode_d200.md)：直接索引当前版本D200位置78.2266%、完整序列53.5%，未达到90%；执行/后端审计正常，停止当前版本晋级。共享路由保留，下一主线prefill专家计算开销。

最新MoE实验：[性能14](performance/14_moe_routing_and_indexed_decode.md)：共享路由D200与CPD逐条相同、短/长ABBA加速1.092×/1.047×；直接索引decode局部4.892×，但因自设小集保守筛选未测D200，不能据此判题目90%不通过。默认未切换。

最新批量实验：[性能13](performance/13_batch_accuracy_campaign.md)：一次14-worker批次完成；CPD在真实batch32的developer200位置96.8125%、完整序列87%，HTTP小规模回归通过，但同批短负载慢2.442倍。下一主线是MoE发射/数据搬运开销优化；默认配置未切换。

最新批量准确性：[性能12](performance/12_b0_batch200.md)：真实batch32固定200位置74.84%、序列44%，B0不启用点1仍未达开发门槛；先做少量首差有界定位，暂不运行正式性能。容量/HTTP执行结果见[11](performance/11_batch32_and_http_batch.md)。

扩展实测见[性能02](performance/02_expanded_bs1_profile_study.md)：19配置132次计时与10份trace已核验；长prefill优先SWA，decode关注KV/RoPE与专家权重物化。实验协议见[09](../docs/plan/09_expanded_profile_study.md)。

[性能01：首轮bs1基线](performance/01_bs1_profile_baseline.md)：GPU采集有效，ROCm镜像区间解析已离线修正；长prefill和decode指向Attention优先，正式并发A/C尚未测。

采集协议与工具见[08 单请求Profile协议](../docs/plan/08_profiling_experiment.md)，本轮bs1为诊断数据，不列正式A/B/C成绩。

下一阶段见[06 Step1优化路线](../docs/plan/06_step1_optimization_routes.md)：单卡BF16为默认评估方向，MoE/Attention与条件第三点，先形成可测服务与trace，不预定收益或凑满三点。

- [39 Step0阶段总结](validation/39_step0_summary.md)：实现、问题闭环、结果与交接。
- [05 基线与复现](../docs/plan/05_step0_baseline_and_reproduction.md)：已通过配置、命令、回归政策。
- [源码与结果指纹](validation/39_step0_manifest.json)：当前工作树与关键artifact；HEAD本身不含全部适配。
- [工作日志](worklog.md)：11月10–13日每日简要记录。
- [总导航](../docs/README.md) · [题目要求](../docs/requirements/01_assessment_summary.md) · [文档管理](../docs/01_document_management.md)

## 研究文档

| 编号 | 文档 |
| --- | --- |
| 01 | [环境与开发入口](research/01_environment.md) |
| 02 | [ATOM 架构、模型接入与加载链](research/02_atom_architecture.md) |
| 03 | [Mellum 模型结构、权重与候选组件](research/03_mellum_model.md) |
| 04 | [正确性资产与参考实现调查](research/04_correctness_reference.md) |

[性能03：SWA单层GPU初测](performance/03_swa_prefill_micro.md)：4K/16K局部收益明显，真实模型正确性及整模型收益待验证。

## 验证01–10：参考建立与初始适配

| 编号 | 文档 | 阅读说明 |
| --- | --- | --- |
| 01 | [HF参考环境建立记录](validation/01_hf_reference_environment.md) | 历史实验，按运行ID/配置理解 |
| 02 | [HF Mellum 小模型 CPU 验证](validation/02_hf_mellum_cpu.md) | 历史实验，按运行ID/配置理解 |
| 03 | [真实checkpoint单卡BF16参考验证](validation/03_hf_mellum_gpu.md) | 历史实验，按运行ID/配置理解 |
| 04 | [ATOM Step0 第一批：配置接入与参考比较协议](validation/04_atom_step0_batch_ab.md) | 历史实验，按运行ID/配置理解 |
| 05 | [ATOM Mellum 模型接入：CPU检查与首次GPU入口](validation/05_atom_mellum_model_wiring.md) | 历史实验，按运行ID/配置理解 |
| 06 | [64 token与真实窗口/上下文边界验证准备](validation/06_mellum_boundary_suite.md) | 历史实验，按运行ID/配置理解 |
| 07 | [64 token分叉：离线分析与同前缀诊断](validation/07_mellum_short_mismatch_diagnosis.md) | 历史实验，按运行ID/配置理解 |
| 08 | [同前缀分层结果：定位第0层子算子](validation/08_mellum_layer0_diagnosis.md) | 历史实验，按运行ID/配置理解 |
| 09 | [RMSNorm首个数值差异：证据与Mellum局部修正](validation/09_mellum_rmsnorm_rounding_fix.md) | 历史实验，按运行ID/配置理解 |
| 10 | [Step0当前验证覆盖与未完成项](validation/10_step0_validation_status.md) | 早期覆盖表；当前结论已由39替代 |

## 验证11–20：扩展分叉与算子隔离

| 编号 | 文档 | 阅读说明 |
| --- | --- | --- |
| 11 | [11 扩展回归与同输入路由诊断](validation/11_mellum_expanded_and_router_replay.md) | 历史实验，按运行ID/配置理解 |
| 12 | [12 扩展回归失败与路由证据](validation/12_expanded_mismatch_and_router_findings.md) | 历史实验，按运行ID/配置理解 |
| 13 | [13 短探针结果与完整MoE同输入重放](validation/13_short_probes_and_moe_input_replay.md) | 历史实验，按运行ID/配置理解 |
| 14 | [14 完整MoE同输入结果与路由受控比较](validation/14_moe_same_input_and_routing_study.md) | 历史实验，按运行ID/配置理解 |
| 15 | [15 路由受控结果与专家算术隔离](validation/15_moe_routing_results_and_expert_arithmetic.md) | 历史实验，按运行ID/配置理解 |
| 16 | [16 专家累加实验与HF实际分派核验](validation/16_expert_reduction_and_hf_dispatch.md) | 历史实验，按运行ID/配置理解 |
| 17 | [17 官方专家后端确认与Step0 PyTorch MoE接入](validation/17_official_backend_confirmed_and_step0_torch_moe.md) | 历史实验，按运行ID/配置理解 |
| 18 | [18 Torch后端扩展回归与Attention同输入定位](validation/18_torch_expanded_and_attention_replay.md) | 历史实验，按运行ID/配置理解 |
| 19 | [19 Prefill内核同输入证据与SDPA受控验证](validation/19_prefill_kernel_evidence_and_sdpa_intervention.md) | 历史实验，按运行ID/配置理解 |
| 20 | [20 分步旋转实验结果与线性投影核查](validation/20_stepwise_rope_result_and_projection_replay.md) | 历史实验，按运行ID/配置理解 |

## 验证21–30：完整prefill与实际缓存decode

| 编号 | 文档 | 阅读说明 |
| --- | --- | --- |
| 21 | [21 QKV打包证据与成对prefill候选验证](validation/21_qkv_packing_and_paired_prefill.md) | 历史实验，按运行ID/配置理解 |
| 22 | [22 Prefill组合结果与RoPE系数来源核查](validation/22_prefill_combination_and_rope_coefficients.md) | 历史实验，按运行ID/配置理解 |
| 23 | [23 RoPE初始化差异证据与匹配参考的prefill候选](validation/23_rope_initialization_candidate.md) | 历史实验，按运行ID/配置理解 |
| 24 | [24 四例完整prefill精确对齐与新增输入回归](validation/24_fixed_prefill_exact_and_new_inputs.md) | 历史实验，按运行ID/配置理解 |
| 25 | [25 Prefill协议缺项恢复与1024边界调用修正](validation/25_prefill_schema_recovery_and_window_threshold.md) | 历史实验，按运行ID/配置理解 |
| 26 | [26 新prefill集通过与强制同历史decode基线](validation/26_prefill_passed_and_forced_decode_baseline.md) | 历史实验，按运行ID/配置理解 |
| 27 | [27 Decode诊断挂起排查、探针修正与失败检测](validation/27_decode_probe_failure_and_watchdog.md) | 历史实验，按运行ID/配置理解 |
| 28 | [28 强制历史decode基线与KV旋转单因素对照](validation/28_forced_decode_baseline_and_kv_rope_control.md) | 历史实验，按运行ID/配置理解 |
| 29 | [29 KV旋转对照结果与同分页缓存SDPA decode候选](validation/29_kv_rope_result_and_cached_sdpa.md) | 历史实验，按运行ID/配置理解 |
| 30 | [30 短输入decode精确与滑窗首步快照核查](validation/30_short_decode_exact_and_window_snapshot.md) | 历史实验，按运行ID/配置理解 |

## 验证31–39：生成闭环、长输入及阶段收尾

| 编号 | 文档 | 阅读说明 |
| --- | --- | --- |
| 31 | [31 滑窗首decode的逐层成对观察](validation/31_window_decode_layer_observation.md) | 历史实验，按运行ID/配置理解 |
| 32 | [32 窗口首decode首差定位至索引8层后半段](validation/32_layer8_tail_replay.md) | 历史实验，按运行ID/配置理解 |
| 33 | [33 索引8层router投影首差与局部候选](validation/33_router_projection_root_and_candidate.md) | 历史实验，按运行ID/配置理解 |
| 34 | [34 强制同历史32位置精确与正常生成候选回归](validation/34_cached_decode_exact_and_normal_generation.md) | HF generate隐式专家切换与普通生成 |
| 35 | [35 普通多样生成：并列最大值与greedy调度修复](validation/35_greedy_tie_break_and_sampling_dispatch.md) | 精确logits并列最大值采样修复 |
| 36 | [36 Step0验收审计](validation/36_step0_acceptance_audit.md) | 增强前审计快照；16k不支持结论已被37后续结果替代 |
| 37 | [37 Step0增强：长prefill滑窗语义扩展](validation/37_step0_long_prefill_extension.md) | 窗口、8192边界及16k 64token回归通过 |
| 38 | [38 Step0：200条开发生成一致性对照](validation/38_step0_developer_200_agreement.md) | 200/200序列、12800token及完整logits精确；非HumanEval pass@1 |
| 39 | [39 Step0阶段总结：Mellum在ATOM中的单卡适配](validation/39_step0_summary.md) | 当前阶段结果主入口 |

## 历史与后续材料

- [最初研究归档](archive/initial_research/README.md)：原始批次文档与迁移清单。
- [Step0收尾前文档快照](archive/step0_document_snapshots/README.md)：追加式导航、协议与完整日志原文。
- performance和submission在实际profiling/消融及提交材料产生后建立；最终技术报告≤4页，AI使用说明1页，素材从本阶段总结与后续性能证据提取。

- [17 R配置A/B/C基线](performance/17_r_abc_baseline.md)：A512/B50全成功，A41.58tokens/s/GPU；C199/200超时，不报告有效C完整尾延迟成绩。
- [18 GPU1 prefill逐元素合并](performance/18_prefill_elementwise.md)：D200输出与R200/200相同，综合0.998×，停止当前候选。

- [研究05 优化实现源码对照](research/05_optimization_implementation_sources.md)：冻结ATOM复用点、上游版本边界、vLLM/SGLang Mellum、候选兼容性和优先级。
- [19 原生MoE/Attention两卡结果](performance/19_native_runtime_pilots.md)：批量约1.47×/1.45×，D200位置72.36%/71.38%；MoE重复生成存在差异，当前均不晋级。
- [20 固定MoE回放与Attention阶段消融](performance/20_native_phase_diagnostics.md)：排序重复但MoE输出变化，CK原子归并证据；Attention仅prefill72.55%、仅decode77.25%，均停止晋级。
- [21 MoE贡献独立归约](performance/21_isolated_moe_reduction.md)：三组局部回放12次稳定，D200位置74.18%，未达90%，跳过性能并停止晋级。

- [研究06 并行准备与C前缀复用](research/06_parallel_preparation_and_prefix.md)：服务profiles已完成CPU验证；C公共前缀3820token，结构复用量37.15%，缓存尚未启用。

- [22 FusedMoE完整HumanEval](performance/22_fused_moe_humaneval.md)：HF145/164、候选147/164，差距1.2195pp，本地功能门槛通过；其它公开集要求独立。

- [23 FusedMoE HTTP容量](performance/23_fused_moe_http_capacity.md)：A64全成功57.00tokens/s，C20全成功TTFT p99 22.69秒；原生路径确认，仍需改善C延迟，非正式ABC。

- [24 FusedMoE+SWA CK HTTP](performance/24_fused_moe_swa_http.md)：A64 59.61tokens/s、C20 TTFT p99 3.82秒；相对单项小负载观察改善4.58%/83.17%，非正式ABC。

- [25 Prefix GPU0试验](performance/25_prefix_probe.md)：8请求/128token开关全一致，真实复用46592token；重复/共享请求耗时改善，下一步真实HTTP。

- [26 FusedMoE+原生Attention HTTP](performance/26_native_attention_http.md)：A64 121.50tokens/s，C20 TTFT/TPOT p99 1.726秒/0.13478秒，进入复用HF的完整功能评估。

- [27 FusedMoE+prefix HTTP](performance/27_prefix_http.md)：C20 19/20命中、复用72352token，TTFT p99 19.51秒；有效但弱于原生Attention主线。

- [28 原生Attention+FusedMoE HumanEval](performance/28_native_attention_humaneval.md)：144/164，对HF145/164，差0.6098pp；本地功能门槛通过，作为性能主线继续推进。

- [29 原生prefix组合](performance/29_native_prefix.md)：C20 TTFT p99 1.123秒，19/20命中；组合功能待评。
- [30 原生完整ABC](performance/30_native_full_abc.md)：A512/B50/C200全成功，A121.28tokens/s为官方19.0%，C完整尾延迟仍落后，继续削减批量执行开销。
