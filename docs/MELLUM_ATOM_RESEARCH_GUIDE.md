# Mellum 2.1 on ATOM：项目背景与环境约束

本指南维护稳定背景与工作原则。原只读研究任务书已完整[归档](../report/archive/initial_research/04_research_guide_original.md)，其中Phase/D0/D1及逐轮指令不再作为当前计划。

当前入口：[文档导航](README.md)；阶段成果：[Step0总结](../report/validation/39_step0_summary.md)；已验证配置：[基线复现](plan/05_step0_baseline_and_reproduction.md)；题目规则：[要求摘要](requirements/01_assessment_summary.md)。原题优先于项目转述；当前用户明确决策优先于旧方案。

## 项目目标

冻结的AMD ROCm环境中，为ATOM适配JetBrains Mellum2.1-12B-A2.5B-Thinking。先完成单卡正确适配Step0，再依据实际表现讨论Step1性能优化。当前不刻意复刻出题人的朴素Baseline，不预选优化目标。

硬件既有记录为MI308X×2；模型为28层MoE、S/S/S/F混合Attention、QK-Norm与分层RoPE。详细参数与源码证据见[模型研究](../report/research/03_mellum_model.md)，不在指南重复维护。

## 目录与冻结边界

| 路径 | 用途与边界 |
|---|---|
| `/app/ATOM` | 原始参考，只读 |
| `/workspace/ATOM` | ATOM开发入口，链接至总仓库`/workspace/tencent-mellum/ATOM`，上游历史保留 |
| `/app/aiter-test`、`/app/mori` | 预置依赖，只读参考 |
| `/opt/venv`、`/opt/rocm` | 冻结运行环境，不修改或升级 |
| `/data/models`、`/data/datasets` | 模型、数据只读 |
| `/workspace/scripts/atom` | 工作副本服务入口；默认系统atom指向原始editable安装 |
| `/workspace/docs`、`/workspace/report` | 方案、研究与结果文档 |
| `/workspace/experiments`、`results`、`logs` | 按需保存实验配置、原始结果与日志 |

源码上游基线：`26f23a0bdbf50b9010c07ee3a71171189723e767`；原独立分支`feature/mellum-support`已纳入总仓库main，当前版本与提交见[07仓库管理](plan/07_project_repository_management.md)。既有环境检查记录：Python3.12.3、PyTorch2.10.0+rocm7.2.4.git3d3aa833、Transformers5.2.0、ATOM0.1.4。记录不是重新检测承诺，变动时须查明原因。

不得通过切换ATOM基线、升级Transformers/PyTorch/AITER/ROCm或改评测参数解决适配问题。用户已明确允许工作区独立新版HF环境仅作正确性参考；该环境不用于ATOM适配/性能评测，原始冻结环境仍不修改。具体边界见[决策06](plan/02_decisions.md)。

## 工作原则

- 优先复用兼容组件，差异以源码和官方模型语义为依据；不冒用其他模型身份绕过配置问题。
- 遵循ATOM仓库开发约束；现有compile装饰模型文件的修改限制见其CLAUDE.md，首选新增Mellum模块。
- 模型实现和测试机制同步开发；独立HF参考优先，内部一致性不能代替外部正确性证明。
- 关键结论附文件、类/函数与必要代码依据，区分事实、推断、待验证和未找到。
- 所有改动、运行记录和结果保存于工作区；保留完整提交历史，不覆盖原始结果。
- Step0已完成适配与本地GPU生成一致性验证，进入Step1准备；旧只读研究阶段限制不是当前任务审批规则。当前范围、正式验收边界和配置见阶段总结及基线说明。

## 文档管理

同类文档使用稳定两位数字前缀；数字表示阅读/登记次序，不表示日期。只有[worklog](../report/worklog.md)按自然日期记录。命名、维护、实验记录规则见[文档管理约定](01_document_management.md)。
