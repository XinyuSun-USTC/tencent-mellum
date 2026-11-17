# 实验结论与轻量证据保存

无需保存完整results目录即可保留本项目已使用的主要结论。仓库内有46份性能记录、46份验证记录及优化取舍、工作日志和技术报告；各开发分支另保留当时的实现和文档。现有报告中指向artifacts的相对链接已核验，无缺失。

保存层次如下：

| 层次 | 路径 | 内容 |
|---|---|---|
| 交付结论 | TECHNICAL_REPORT.md、WORKLOG.md | 指标、通过/失败与技术决策 |
| 分阶段分析 | report/performance/、report/validation/ | 各轮发现、负载、限制和转向原因 |
| 既有归档 | artifacts/performance/、artifacts/results/、artifacts/submission/ | 选定原始摘要、日志、输入输出及trace |
| 当前矩阵 | artifacts/submission/final_r32_matrix/ | 四行指标、正确性摘要及测量来源 |
| 补充轻量证据 | artifacts/submission/conclusion_evidence/ | 最新C与profile原摘要、分项分析、最终输出token对照及HumanEval评分 |

补充归档共35个数据文件，约3.34MB，另有哈希索引。token对照保留提示名称、输入身份哈希、HF输出和候选输出，可独立重算四行的200×64位置一致率、完整序列一致数与四条长输入结果，已与矩阵逐项核验。HumanEval保留评分和逐题结果。最新profile保留统计分析、原trace哈希及配置来源，未保存其全部原始trace。

原results中的大规模张量、完整输入重复副本、完整日志和未选trace可不作为远程Git备份内容；省略这些文件会限制以后重新逐层或重新解析trace，但不会删除已归档结论。哈希只能核验身份，不能恢复省略的原文件。

新实验产生后，应继续保存结论、实际参数、通过/失败数及可支撑结论的最小数据。本页不把尚未返回的实验或仅prepare-only目录计为已完成成果。
