# 文档管理约定

## 正式材料与内部记录

根目录仅放置主要交付入口：`README.md`、`TECHNICAL_REPORT.md`、`AI_USAGE.md`、`WORKLOG.md`及`start_server.sh`。技术报告和AI说明以中文Markdown为主稿；版面页数在最终导出时确认。

内部资料按用途存放：`docs/requirements/`保存题目来源，`docs/plan/`保存方案和决策，`docs/submission/`保存交付审查及来源说明；`report/research/`、`report/validation/`和`report/performance/`分别保存调研、正确性和性能记录。历史原文归入`report/archive/`。

## 编号与修订

同目录同类资料采用两位递增序号，序号表示文档顺序而非自然日。根目录正式材料、README、原始附件和历史日志保留约定名称。修订已有资料时沿用编号；迁移保持Git历史并更新当前入口链接。历史记录中的路径和阶段结论按当时状态保留。

## 工作日志与实验依据

`WORKLOG.md`是面向交付的D1—D7每日摘要；`report/worklog.md`保留详细研发过程。实验报告记录提交、配置、命令、输入来源、指标及结论。原始结果位于实验目录，精选材料与SHA256索引存入`artifacts/`；不以新报告覆盖原始测量，也不为缺失实验补写结果。

目录导航区分当前交付入口与历史研究记录。正式技术报告由已有证据归纳，内部长篇诊断不直接作为四页报告提交。
