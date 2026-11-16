# WORKLOG — 2026-11-10 至 2026-11-16

本文件为交付用每日摘要；日期按实际研发记录自然日排列。完整过程、实验路径和补充证据见 [report/worklog.md](report/worklog.md)。Git 保留正常提交与合并历史，未 squash；这里不回填虚构的提交时间。

| 日期 | 工作及产物 | 结果/后续 |
|---|---|---|
| 11-10 | 阅读题目与研究约束，分析 ATOM 调度、模型注册、Attention/MoE、权重加载和 Mellum 配置；形成架构与适配规划 | 确立冻结推理环境、独立 HF 参考环境；先正确适配再优化 |
| 11-11 | 建立 HF5.10.1 参考；实现 Mellum 配置/模型/5631权重加载审计；开发 token、同前缀、逐层和专家回放诊断 | 短输出曾一致，扩展后分叉；定位 MoE grouped_mm/归约与 attention/RoPE 差异 |
| 11-12 | 对齐参考专家、投影、RoPE/SDPA、decode KV、router、生成语义及 tie-break；建立强制历史 decode 和正常生成验证 | 短组与10条正常生成达到 tokens_and_logits_exact；扩大边界覆盖 |
| 11-13 | 增强 Step0：长输入、滑窗边界与developer200精确对比；整理阶段报告、保留Step0标签并建立个人仓库；开发 profiler 与批量实验 | Step0本地验证完成；识别 MoE、attention、投影、调度/launch开销 |
| 11-14 | CK/Triton SWA优化与微基准；200条生成、HF对照；安装独立HumanEval套件并自测；开发batch/HTTP验证 | SWA提速但token分叉；引入功能指标而不替代题面token门槛；服务批处理跑通 |
| 11-15 | batch200与批量campaign；MoE/indexed/native attention参考调研与实验；ABC客户端及基线；两卡独立并行；HumanEval恢复、容量/prefix探测 | 原生路径性能提升，token一致率仍低；HumanEval显示功能可用，未据此豁免token硬门槛 |
| 11-16 | 原生批处理/直接KV、resident/packed、Norm、graph、QK/路由融合、prefix/metadata、调度组合筛选；完整ABC与最终验收；Norm/核心矩阵/阶段隔离恢复诊断；构建交付入口 | 激进A1139.78，C TPOTp99 11.502ms；developer200 62.85%未通过；reference control槽1为100%；提供独立模式开关并如实交付差距 |

最终启动检查结果见 `artifacts/submission/smoke/`（若已返回并归档）；未返回时不得写成已通过。交付配置、源文件及证据版本以发布分支和启动时 `launch.json` 为准。
