# 25 Prefix兼容路径首轮GPU试验

GPU0 20261115T225622969152Z prefix_probe_passed。R参考MoE/Attention、BF16/eager、cache off/on两次独立加载；三条真实C输入7889/11039/4294token，每请求16token。8次请求输出全部一致，128/128生成token一致；加载5631项与运行源码hash核验通过。

| 请求组 | cache off秒 | cache on秒 | off/on |
| --- | --- | --- | --- |
| 首次冷请求 | 3.8441 | 3.7731 | 1.019× |
| 相同prompt复用 | 2.1115 | 1.2692 | 1.664× |
| 共享prefix ragged3 | 5.6081 | 3.5713 | 1.570× |
| 逆序复用ragged3 | 5.5709 | 2.1076 | 2.643× |

实际命中事件分别为[7888]、[7888,3808,3808]、[4288,11024,7888]，累计46592token，非虚设开关。共享前缀3808完整块符合CPU分析；全prompt重复还能复用其余完整块。未缓存最终输出。

这是16token、自然顺序显式重复的离线兼容试验，不是HTTP C200、16k验收或端到端稳态收益。逆序复用命中远多于C不同prompt的公共前缀，不能把2.64倍外推C。缓存是新机制，本轮off/on自比较是定向语义检查，没有重跑HF或追逐原生内核浮点差异。

[完整分组结果与命中证据](../../artifacts/performance/25_prefix_probe/summary.json)，同目录保留off/on记录、原始命中与来源hash。

下一步计划28：GPU0用FusedMoE+prefix跑同协议HTTP smoke/A64/C20，C阶段从空引擎自然命中，逐case记录缓存量。先不叠SWA CK/native Attention，独立确认缓存收益；GPU1继续完整原生Attention性能探索。
