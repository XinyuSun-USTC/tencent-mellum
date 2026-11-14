# Mellum on ATOM 文档导航

当前阶段：**Step0适配开发完成，本地生成一致性验证通过，进入Step1准备。** 200条开发prompt前64token全部一致，16k输入64token回归通过；HumanEval pass@1与出题方公开集口径待正式补证。当前不追加同类数值定位实验。

## 当前阅读入口

当前执行：[12 一轮Triton窗口局部筛选](plan/12_bounded_triton_swa_screening.md)：复用真实QKV，不加载模型；筛选后转向服务/批量能力。

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
