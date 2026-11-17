# 脚本导航

当前服务入口为根目录 `start_server.sh`，调用 `serve_delivery.py` 与 `serve_mellum.py`。优化定义集中在 `mellum_delivery_matrix.py`。

| 工具 | 用途 |
|---|---|
| serve_delivery.py | 累计配置、逐项关闭开关及环境检查 |
| smoke_delivery.py | 真实服务HTTP、SSE及输出计数检查 |
| verify_submission.py | CPU交付完整性、证据哈希及文档链接检查 |
| run_delivery_matrix.py | 完整ABC与正确性批量运行 |
| collect_delivery_matrix.py | 同一源码、同一行定义的分卡结果汇总 |
| assemble_r32_delivery_matrix.py | 校验并汇总当前明确指定的Step0及后三行证据 |
| package_submission.py | 从干净Git提交生成源码ZIP及完整历史bundle |

其他按主题命名的脚本用于研究、诊断及历史结果复核，运行条件见对应docs/plan和report文档。复用参考的工具依赖已归档的HF输入及独立评测环境，不属于服务启动依赖。

旧保守/激进服务入口位于 `internal/start_legacy_server.sh`；旧PDF生成器为 `internal/build_legacy_reports.py`，输出到 `report/archive/legacy_generated/`。当前正式报告是根目录Markdown，不使用旧生成器更新。
