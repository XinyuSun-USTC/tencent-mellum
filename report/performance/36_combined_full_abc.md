# 36 Graph/QK/路由/prefix/metadata完整ABC

原始运行：/workspace/results/mellum_full_campaign/20261116T134727613186Z。profile=sprint_graph_qk_router_prefix_metadata，单卡BF16，Graph、原生Attention/MoE路由、Packed、融合QK/RoPE、单次KV写入、prefix及metadata复用。72项产物哈希一致，源码未改变，B50/A512/C200全部成功。

| 指标 | 结果 | 官方baseline | 相对变化 |
| --- | ---: | ---: | ---: |
| A tokens/s/GPU | 1139.7781 | 638.282724 | +78.569% |
| B TPOT p50 ms | 4.86523 | 未提供 | 不虚构 |
| B TTFT p99 ms | 78.3235 | 未提供 | 不虚构 |
| C TPOT p99 ms | 11.50206 | 41.26502 | -72.126% |
| C TTFT p99 s | 1.025998 | 0.993549 | +3.266% |

相对上一轮同GPU无prefix/metadata组合：A1129.3798→1139.7781约+0.92%；C TTFT p99 1.488325→1.025998约下降31.06%。C服务记录199次cached后缀事件，合计757792token。C吞吐76.8348主要受固定到达率限制。

实际负载/客户端/来源参数均保留原manifest；同节点另一卡并行正确性验证，不能等同空闲隐藏集三次中位数。EvalScope版本适配及到达过程差异继续披露。target_g未给，不虚构分数。性能候选不等于整体题目验收通过；同组合正确性见验证40。
