# 43 专家/Attention四格短诊断

运行/workspace/results/mellum_core_diagnosis/20261116T144338670827Z完成。52项产物哈希一致、源码未改，5631加载及28层实际MoE类/prefill函数核验通过。20条由原200按来源/长度固定选择，不能当作完整200验收。

| Attention管线 | 专家 | 位置一致率 | 整序列 |
| --- | --- | ---: | ---: |
| 原生varlen/paged/direct KV | AITER FusedMoE | 894/1280=69.84375% | 9/20 |
| 原生varlen/paged/direct KV | 参考Torch | 911/1280=71.171875% | 8/20 |
| 参考SDPA/KV | AITER FusedMoE | 986/1280=77.03125% | 11/20 |
| 参考SDPA/KV | 参考Torch | 1280/1280=100% | 20/20 |

固定参考Norm、分离QKV、Torch路由、batch1/eager、无prefix。只有参考/参考恢复，不能把问题归结为单一整块模块；Attention对照包含缓存外围，不等同只比较FA kernel。与完整R控制200/长组100%共同支持：检查管道可用，需定位原生专家和Attention中具体阶段/数值边界。

下一步按计划49：GPU1拆Attention prefill/decode，GPU0拆专家prefill/decode，各两项同20条，不重跑HF/HE/ABC。选有希望组合后必须补完整200。额外canonical专家副本仅用于诊断布局/阶段，不作为交付性能方案。
