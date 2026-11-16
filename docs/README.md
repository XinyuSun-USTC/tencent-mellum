# Mellum on ATOM 文档导航

当前阶段：**Step1基础执行路径恢复与组合收敛。** 最新开发A64为499.64tokens/s；各配置功能与性能证据独立保留，最终配置和完整规模复测待收敛。

## 当前阅读入口

并行CPU准备：[41 prefill元数据复用与完整ABC批次](plan/41_prefill_metadata_and_full_load_campaign.md)：独立campaign分支实现当次forward校验复用及显式profile的完整ABC、失败继续/恢复，GPU待当前两批结果后决定。

当前实验：[40 融合结果与组合调度](plan/40_fusion_results_and_combined_scheduling.md)，scheduling分支ee3e238。QK融合A64为1062.46tokens/s、路由ABBA均值+7.39%；下一批GPU0组合重复、GPU1prefix/chunk性能筛选，仍不逐项跑HF。

并行CPU开发：[39 Graph前缀缓存与chunked prefill](plan/39_graph_prefix_and_chunked_prefill.md)，独立scheduling分支12087e7。40项CPU检查通过，GPU待空闲后筛选，不修改正在运行的fusion目录。

当前实验：[38 Graph路由ABBA与QK/RoPE融合](plan/38_graph_routing_and_qk_fusion.md)，独立fusion分支2dc21b2；上一批Graph A64达到943.26tokens/s，同卡较eager提升51.9%。新批仅性能筛选，不逐项跑HF。

当前实验：[37 连续优化批次](plan/37_optimization_sprint.md)，独立sprint分支3f81b6d：GPU0空观察器移除/decode Graph，GPU1Packed消融/原生路由，集中性能筛选，完整正确性留最终组合。

交付收敛：[36 提交与版本验收](plan/36_submission_and_version_acceptance.md)：核对原题Git历史、单命令server、≤4页报告、消融、AI说明与WORKLOG要求，区分最终提交缺口与内部整理建议。

当前工作：[35 基础快路径优先](plan/35_basic_fastpaths_first.md)。独立fastpaths提交68a4086：GPU0短trace/常量驻留，GPU1融合Norm/packed QKV/组合，90项CPU检查及两track真实数据准备通过，待GPU终端运行。

上一轮：[17 直接索引decode一次D200补证](plan/17_indexed_decode_d200_followup.md)：已完成，见[性能15](../report/performance/15_indexed_decode_d200.md)：总体位置78.23%、完整序列53.5%，当前直接索引版本停止晋级。保留共享路由，下一主线prefill专家GEMM/调度开销。

上一批MoE：[16 MoE路由共享与decode直接索引](plan/16_moe_routing_and_indexed_decode.md)：批次已完成，见[性能14](../report/performance/14_moe_routing_and_indexed_decode.md)：共享路由D200与CPD逐条200/200相同，短/长ABBA加速1.092×/1.047×；直接索引候选局部快，但尚无D200结论。默认未切换。

上一批：[15 一次启动的批量准确性实验批次](plan/15_batch_accuracy_campaign.md)：一次开发观察器、shape控制、DAG调度与分析器，最多14个GPU worker、90分钟总预算，减少逐阶段手工运行。批次已完成，结果见[性能13](../report/performance/13_batch_accuracy_campaign.md)：CPD位置96.81%、完整序列87%，有2.442倍短负载耗时成本，下一主线MoE优化。命令与恢复方法保留在计划末节。

当前准确性评测：[11 代表性开发集与HumanEval](plan/11_representative_correctness_and_humaneval.md)：先跑开发200候选及独立HumanEval工具自检，避免以微观数值差异代替题目门槛。实现及GPU测试待执行边界详见该计划。

扩展基线已完成：[性能02](../report/performance/02_expanded_bs1_profile_study.md)；协议见[09](plan/09_expanded_profile_study.md)。
当前可执行实验：[08 首轮单请求Profile](plan/08_profiling_experiment.md)。首轮三组GPU已完成，实测分析见[性能01](../report/performance/01_bs1_profile_baseline.md)，当前优先评估Attention。

仓库管理：[07 项目Git组织与迁移](plan/07_project_repository_management.md)，总仓库`/workspace/tencent-mellum`，ATOM保留上游历史，旧工作路径通过链接兼容。

Step1规划入口：[06 可选优化路线](plan/06_step1_optimization_routes.md)，包含计分目标、源码开销假设、MoE/Attention主候选、第三点分支及首轮profiling安排；候选路线随性能01/02实测更新，尚无优化后性能成绩。

1. [39 Step0阶段总结](../report/validation/39_step0_summary.md)：成果、实现、问题闭环、最终证据与阶段边界。
2. [05 基线与复现](plan/05_step0_baseline_and_reproduction.md)：真实通过配置、工作树指纹、验证命令与Step1回归方法。
3. [题目要求摘要](requirements/01_assessment_summary.md)与[原始PDF](requirements/2026%20CSIG云一计算加速团队实习校招面试题.pdf)：正式规则；不将本地阶段完成等同出题方正式通过。
4. [项目指南](MELLUM_ATOM_RESEARCH_GUIDE.md)：背景、冻结环境与开发边界。
5. [Step0范围与交接](plan/01_step0.md)、[决策记录](plan/02_decisions.md)、[正确性回归协议](plan/04_correctness_protocol.md)：当前执行约定。
6. [研究与结果索引](../report/README.md)：按稳定编号查找源码研究和逐轮原始证据。
7. [工作日志](../report/worklog.md)：11月10–13日按自然日归纳。

下一主线：验证第一个SWA prefill优化候选，并推进实际负载所需服务/批量能力；保留Step0及各优化开关作消融。

## 历史与管理

[03 早期实施计划](plan/03_atom_step0_implementation_and_validation.md)及validation/01–38保留实验发生时的状态。早期“待验证”“Step0未完成”等叙述应按其运行ID和配置理解；当前阶段看39，具体配置看05。

同类文档按目录内两位编号排序；只在[worklog](../report/worklog.md)按日期记简要日志。遵循[文档管理约定](01_document_management.md)。重构前的追加式导航、协议和完整日志已[保留快照](../report/archive/step0_document_snapshots/README.md)，最初研究原文另见[归档](../report/archive/initial_research/README.md)。

当前待运行：[22 固定MoE回放与Attention分阶段消融](plan/22_bounded_native_phase_diagnostics.md)。120项CPU检查和两份prepare-only通过，独立工作区未改原R推理代码。

当前并行准备：[25 功能评估恢复与交付](plan/25_parallel_preparation_and_delivery.md)。原HumanEval候选164题已完成；HF序列化失败需独立恢复。修复提交df33885，服务配置和C前缀分析提交a2cbcce，均在独立工作区，主仓库此轮仅归档文档与CPU证据。

GPU1待运行：[26 候选HTTP容量诊断](plan/26_candidate_http_capacity.md)，独立next工作区提交9a3b3ca；smoke→A64→C20，35分钟执行预算，24项CPU检查与真实输入prepare/hash检查通过，不作为正式ABC成绩。

当前功能结果：[性能22](../report/performance/22_fused_moe_humaneval.md)，完整HumanEval配对达到本地2pp门槛；下一步按计划26完成候选HTTP容量诊断。

GPU0待运行：[27 前缀缓存兼容与定向试验](plan/27_prefix_cache_gpu0.md)，独立prefix工作区80cd14c，57项相关CPU检查和真实输入prepare-only通过；GPU1的next工作区不修改。

GPU0待运行：[28 FusedMoE+prefix HTTP](plan/28_prefix_http_capacity.md)，独立prefix工作区，HTTP8251/engine8926；首轮结果见性能25。33项相关CPU检查、输入准备和启动配置核验通过。

GPU1待运行：[29 原生Attention组合完整HumanEval](plan/29_native_attention_functional.md)，独立native-functional工作区；复用已完成HF164，不在GPU0启动生成，8项CPU检查及真实引用prepare通过。性能结果见报告26。

当前已完成：[性能28](../report/performance/28_native_attention_humaneval.md)，原生Attention+FusedMoE完整HumanEval达本地2pp门槛；下一步原生路径与prefix整合及真实负载消融。

当前待运行：[30 原生prefix与完整ABC两卡批次](plan/30_native_prefix_and_full_abc.md)。GPU0独立native-prefix提交c34a96a（72项CPU检查），GPU1独立native-abc提交cd2c3e3（22项）；真实输入prepare/指纹核验通过。两项GPU尚未执行。

两卡批次30完成，结果见[性能29](../report/performance/29_native_prefix.md)/[性能30](../report/performance/30_native_full_abc.md)。当前GPU1待运行：[31 恢复整批投影](plan/31_batched_projection_dispatch.md)，独立native-batched提交0555b76，16项CPU检查及真实输入/源码/客户端指纹通过。

GPU1批量投影运行期间，GPU0待运行：[32 原生prefix完整功能评估](plan/32_native_prefix_functional_gpu0.md)，独立prefix-functional提交990f38a，10项CPU检查及真实HF引用/源码hash核验通过；仅GPU0，复用HF。

GPU1下一项：[33 整批投影完整功能评估](plan/33_batched_projection_functional_gpu1.md)，独立batched-functional，复用HF，22项CPU检查和真实引用prepare通过；性能结果见报告31。

新增功能证据：[性能32](../report/performance/32_native_prefix_humaneval.md)，原生prefix组合147/164达到本地2pp门槛；整批投影配置的功能评估按计划33独立执行。

整批投影功能结果：[性能33](../report/performance/33_batched_projection_humaneval.md)。149/164高于HF145/164；因绝对差2.439pp，原2pp脚本False，不能写成严格绝对差已通过。保留高性能候选，下一步prefix整合与最终复测。

两卡批次34已完成，结果见[性能34](../report/performance/34_batched_prefix.md)/[性能35](../report/performance/35_direct_kv.md)：prefix改善C首token尾延迟，单次KV写入将A64提高到499.64tokens/s。下一步整合与整体功能评估，Graph仅有界可行性试验。
