# 交付清单与验收边界

## 当前可交付

- `ATOM/`完整适配与优化源码，保留上游许可证、正常Git提交/合并历史与`step0-local-validated`标签。
- 根目录 `start_server.sh`：保守默认/激进显式模式，独立组件关闭开关，固定环境检查及实际配置/源码哈希记录。
- 实际交付入口两种模式均通过 GPU 加载、HTTP非流式、SSE流式与usage冒烟：`20261116T152441526820Z`，证据在`artifacts/submission/smoke/`。范围为512上下文、槽1、8输出token。
- 技术报告 `01_technical_report.pdf`，4页；AI使用说明 `02_ai_usage.pdf`，1页。PDF英文便于无中文字体容器可靠生成；另附中文Markdown详细说明。
- 根目录 `WORKLOG.md`每日摘要；`report/worklog.md`保留详细研发日志。
- `artifacts/submission/index.json`精选原始证据、SHA256及原始路径；大张量/模型/环境不纳入普通Git。
- `SOURCES.md`来源说明；`scripts/verify_submission.py`进行CPU交付完整性检查。

## 正在补测

最终交付工作树的完整ABC：`run_mellum_full_campaign.py --gpu 0 --profiles sprint_graph_qk_router_prefix_metadata --budget-minutes 25`。

短累计矩阵：`run_submission_ablation.py --gpu 1 --budget-minutes 20`，A32/B6/C8，控制行是native fast control，不是历史Step0；不能替代正式全规模累计消融。

结果返回后追加原始数据与短更新，不将正在运行写为已经测得。测量期间不修改被manifest绑定的ATOM/性能入口源码。

## 明确未通过或未完成

| 题目要求 | 状态 |
|---|---|
| 公开200条前64token一致率≥90% | 精确官方集口径待确认；本地conservative为100%，aggressive为62.85%失败 |
| BF16 HumanEval差距≤2pp | 激进150/164、HF145/164，绝对差3.05pp，不按字面宣称通过；保守精确预设未单独测量 |
| 16k单列 | 保守64/64，激进63/64；完整长组另列 |
| MTP、双卡配置额外一致性 | 未启用MTP/双卡推理，不主张支持 |
| Step0及三点累计完整ABC | 尚不齐备；局部和短规模数据单列，不填充缺失正式行 |
| 最终空闲隐藏集3次中位数 | 裁判执行，本地未完成 |

本交付不宣称考核整体通过；不拼接保守正确性与激进速度。独立模式开关用于保留可用实现与研究成果，不是规避题目门槛的办法。

## 提交方式

提交代码仓库的`release/submission`分支及最终标签，附两份PDF、README、WORKLOG与证据索引。源代码压缩包便于离线阅读；Git bundle用于保留版本历史。不要只交一个不含Git历史的源码压缩包，也不要以尚未更新的远端main作为最终版本。
