# 45 流式请求提前终止的CPU复现

运行`results/stream_cleanup_cpu/20261116T191201123019Z`，工具`scripts/diagnose_stream_cleanup.py`。该工具从当前`serving_chat.py`提取真实的流式函数及其辅助函数，在无GPU环境下执行；内容解析器为空操作，仅用于隔离回调生命周期。

| 场景 | cleanup回调次数 |
|---|---:|
| 正常收到finished并结束 | 1 |
| 初始role chunk后关闭协程 | 0 |
| 等待队列输出时取消任务 | 0 |

源码中`serving_chat.stream_chat_response`在正常循环结束后调用cleanup，缺少覆盖提前关闭/取消的finally路径。`api_server.cleanup_streaming_request`只删除前端映射与IO记录；`EngineCoreRequestType.ABORT`虽有定义，当前`EngineCore.process_input_sockets`没有处理该类型的分支，也未找到对应manager发送接口。因此，补前端cleanup并不等同于已经停止引擎生成。

这证明存在需要处理的资源生命周期缺口，但不能据此量化线上C超时后的GPU浪费，也不能把C199条失败全部归因于它。实际ASGI断连、引擎资源释放和并发安全仍需独立测试。当前参考路径本身吞吐不足的结论仍成立；取消请求不会使已超时的请求转为成功。

本轮未修改API、scheduler或在跑实验源码。将来若修复，应在独立分支同时覆盖正常完成、等待中取消、运行中取消、重复取消与KV回收，并实际检查其他请求不受影响。该修复作为服务健壮性工作，不用它替代数学路径优化或已失败请求的统计。

归档：`artifacts/submission/stream_cleanup_cpu/summary.json`，含源码SHA256。CPU复现不是HTTP/GPU验证，也不是性能收益数据。
