# 35 参考K/V单次写入性能

GPU1 20261116T101221209827Z complete。profile=fused_moe_native_batched_direct_kv，在439.67配置上只启用MELLUM_DIRECT_KV_WRITE=1；保留原参考Q/K公式和BF16缓存布局，去掉重复原生RoPE/cache与二次K覆盖、动态有效slot压缩。

微型真实GPU探针Full/SHUFFLE、SWA/NHD均通过：query、写入K/V、未写槽精确，跨块/负slot与非连续V stride覆盖。实际A/C服务均有prefill、decode的aiter_reference_kv_once记录，未误测回旧路径。

| 指标 | 整批投影旧写入（性能31） | 单次写入 |
| --- | --- | --- |
| A64输出tokens/s/GPU | 439.6727 | 499.6429 |
| A64 TPOT p99秒 | 0.05862 | 0.05012 |
| C20输出tokens/s/GPU | 63.5585 | 65.6675 |
| C20 TTFT p99秒 | 1.58614 | 1.41828 |
| C20 TPOT p99秒 | 0.08037 | 0.07040 |
| B2 smoke TPOT p99秒 | 0.05452 | 0.04664 |

所有请求成功；A吞吐观察+13.64%、TPOT p99下降约14.5%，C TPOT p99下降约12.40%。小B同样改善，符合去除每层重复工作方向。单次小负载且有另一卡活动，不当正式A512/C200或无干扰统计结论。

独立核验全部产物/源码、同一工作负载输入、5631加载和实际写入路径，见[指标与指纹](../../artifacts/performance/35_direct_kv/metrics.json)。该版本prefix关闭，不能直接称完整三机制已组合；局部copy精确也不替代整体功能回归。下一步合并prefix与整批投影，并准备一次有界decode Graph试验，之后再固定版本跑完整ABC。
