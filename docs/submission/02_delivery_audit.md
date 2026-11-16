# 交付材料与复现说明

核对依据：`docs/requirements/01_assessment_summary.md`及题目第7节。根目录三份报告用于直接审阅，内部研究、规划和诊断记录保留于`docs/`与`report/`。

## 1. 材料清单

| 材料 | 仓库位置 | 内容 |
|---|---|---|
| 适配与优化源码 | `ATOM/` | 模型实现、权重映射、原生计算路径及性能机制 |
| 服务启动入口 | `start_server.sh` | 单命令启动，参考/优化配置选择；独立组件开关见 README |
| 技术报告 | `TECHNICAL_REPORT.md` | Profiling、优化依据、收益、正确性及累计消融 |
| AI 使用说明 | `AI_USAGE.md` | 工具分工、错误修正与结果核验 |
| 每日工作日志 | `WORKLOG.md` | D1—D7 的工作、阻碍与决策；详细记录见 `report/worklog.md` |
| 实验依据 | `artifacts/submission/index.json` | 归档结果、SHA256 与原始文件路径 |
| 来源说明 | `docs/submission/01_sources_and_licenses.md` | 上游实现与许可证归属 |

**总体结论：主版本（参考路径、32槽）已通过完整本地正确性验收；同配置ABC已测、C场景仅1/200成功；完整累计消融仍待完成。激进版本单独报告。**

文档以中文 Markdown 为主稿。技术报告按四页内容组织，AI 说明按一页篇幅编写；目前尚未核对排版导出页数，≤4页/1页的版面要求仍需最终确认。本次交付不包含早期英文PDF预览。

## 2. 实现与验证状态

交付入口的两种配置均已通过启动检查，运行编号为 `20261116T152441526820Z`。检查范围包括 5,631 个源权重张量的加载、普通 HTTP 请求及 SSE 流式响应和 usage 统计；测试配置为 512 token 上下文、单调度槽和 8 个输出 token。证据保存在 `artifacts/submission/smoke/`。

| 验收项目 | 当前结果 |
|---|---|
| 本地 200 条前 64 token 一致率 | 主版本96.8125%，通过逐位置门槛；激进版本62.85%，未通过 |
| BF16 HumanEval | 主版本148/164，HF145/164，绝对差1.829个百分点，通过；激进版本150/164，差3.05个百分点，不通过 |
| 16k 输入 | 参考配置 64/64，优化组合 63/64；其余长输入结果见技术报告 |
| MTP、双卡推理 | 未启用，相关附加检查不适用 |
| 事前量化收益预测 | 已有定性假设，缺少统一的事前数值估计；报告据实说明 |
| 报告页数 | Markdown已按四页内容分段；实际导出版面尚未核验 |
| 规定的完整累计消融 | 尚未完成；组件对照和缩减规模测试分别记录 |
| 隐藏集、空闲机器三次中位数 | 待裁判测量 |

主版本完整验收运行为`20261116T160221792307Z_batch32`，20项产物哈希一致，来源见`artifacts/submission/primary_acceptance_batch32/`。主版本自身性能已测：A41.3351 token/s/GPU，B50全部成功；C仅1/200成功，199请求超时，不具备有效C尾延迟成绩。证据见`artifacts/submission/primary_abc/`，不继承附加版本性能。

## 3. 主版本性能与附加实验

主版本通过验收后的同配置完整ABC：

```bash
/opt/venv/bin/python scripts/run_mellum_completion_benchmarks.py --track primary --gpu 0 --acceptance-run /workspace/results/mellum_reference_acceptance/20261116T160221792307Z_batch32 --budget-minutes 180
```

此前激进附加版本完整ABC已完成，其结果独立保留。

缩减规模累计对照：

```bash
/opt/venv/bin/python scripts/run_submission_ablation.py --gpu 1 --budget-minutes 20
```

后者采用 A32/B6/C8，控制行为原生快速配置，与历史 Step0 不同。该对照已完成，12个阶段均成功，运行编号为`20261116T153235801902Z`，结果见`artifacts/submission/delivery_ablation/matrix.md`。其结果用于补充组件收益分析，不替代题目规定的完整累计消融。完整ABC复测已完成，运行编号为`20261116T153025780336Z`，A512/B50/C200全部成功，结果见`artifacts/submission/delivery_abc/`。

## 4. 版本与打包

交付分支为 `release/submission`，当前主版本正确性标签为`primary-correctness-20261116-r32`，此前材料标签为 `submission-20261116-final`，历史 Step0 标签为 `step0-local-validated`。Git 保留提交和合并历史。技术报告、AI 说明、WORKLOG 与证据索引均随代码交付；模型权重、虚拟环境及大体积张量文件保留在外部环境中。

工作树提交完成后，执行：

```bash
/opt/venv/bin/python scripts/verify_submission.py
/opt/venv/bin/python scripts/package_submission.py
```

输出目录为 `/workspace/delivery/<时间>_<提交>/`，包含源码 ZIP、完整本地 Git 历史 bundle、版本清单和 SHA256。ZIP 用于直接查阅，bundle 可用于恢复版本历史：

```bash
git clone tencent-mellum-history.bundle -b release/submission
```

服务运行仍依赖题给模型及冻结的软件环境。
