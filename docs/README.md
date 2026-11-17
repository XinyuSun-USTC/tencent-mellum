# 项目文档索引

当前交付采用32槽r32作为Step0，优化①仅增加decode专家group4；当前交付树为 `/workspace/tencent-mellum-delivery-r32`。定义与数据来源见[决议67](plan/67_r32_baseline_revision.md)，[当前矩阵](../artifacts/submission/final_r32_matrix/matrix.md)保留历史基线与本轮各行来源。

## 交付与复现

- [技术报告](../TECHNICAL_REPORT.md)、[AI说明](../AI_USAGE.md)、[工作日志](../WORKLOG.md)
- [优化探索与取舍](submission/12_explored_optimization_routes.md)
- [四版本正确性证据](submission/11_correctness_evidence.md)
- [当前交付核对](submission/02_delivery_audit.md)、[剩余实验](submission/03_completion_experiments.md)
- [累计实验口径](plan/65_final_delivery_matrix.md)、[r32基线定义](plan/67_r32_baseline_revision.md)
- [Profiling证据](submission/06_profiling_evidence.md)、[技术取舍](submission/07_optimization_decisions.md)、[收益与消融](submission/08_estimates_and_ablation.md)
- [AI协作案例](submission/09_ai_collaboration_cases.md)、[最终编辑清单](submission/10_final_editorial_checklist.md)

## 题目、过程与历史

- [题目要求归纳](requirements/01_assessment_summary.md)、[研究指南](MELLUM_ATOM_RESEARCH_GUIDE.md)
- [源码研究](../report/research/)、[正确性记录](../report/validation/)、[性能记录](../report/performance/)
- `plan/`中的早期编号方案及`report/archive/`保留当时决策，不作为当前启动指令；不将历史单槽配置改名为r32。
