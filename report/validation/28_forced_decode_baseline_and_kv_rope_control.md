# 28 强制历史decode基线与KV旋转单因素对照

## 完整基线结果

原始运行`results/mellum_decode_validation/20261112T134058133231Z`，ATOM`results/atom_mellum_gpu/20261112T134058178343Z`。运行器正常完成，复用已完成的HF，5631/5631加载审计通过；四请求×八位置的实际token/position/逻辑长度均符合强制历史。

四个prefill位置都与HF cached精确一致，28个decode位置全部非精确；首个decode的最大logits绝对差如下：

| 输入 | prompt长度 | decode第1步最大绝对差 |
|---|---:|---:|
| chinese_inventory | 32 | 0.25 |
| raw_16 | 16 | 0.5625 |
| raw_17 | 17 | 0.53125 |
| window_1023 | 1023 | 0.75 |

HF cached相对重新prefill在28个decode位置也均存在数值差异，所以后续继续将HF cached作为逐步参考，内部差异另列，不用uncached替代。强制输出token相同未被用作成功依据。运行器/末尾额外前向修正已使完整基线成功采集，但不代表模型decode正确。

## 当前明确缺口与受控修正

已精确的prefill候选只对prefill Attention计算重算CPU初始化的正确RoPE；实际原rope_cache仍按GPU初始化缓存执行K写入，decode Q/K也沿原系数。因此计算与存储使用不同系数，不能期待prefill精确自动保证decode精确。实际影响需要与decode内核、窗口范围等因素隔离，不提前声称全部原因已确认。

新增`mellum_attention.py:install_reference_kv_rope`，仅绑定Mellum实例的rope_cache：

1. 在原函数原位修改Q/K前，用正确CPU inverse frequency及GPU位置运算计算reference Q/K。
2. 继续执行原rope_cache，保留V、原分配、slot_mapping和缓存元数据路径。
3. `overwrite_reference_keys`按实际slot_mapping把reference K写回现有5维BF16 K缓存，返回reference Q/K给原Attention。

覆盖prefill、decode及末尾未评分前向，使query计算和K存储使用同一系数。保持原V写入、分页布局、滑窗及decode内核；尚不调整滑窗边界或切换decode为SDPA。只支持当前TP1、BF16、非flash-layout诊断路径，检查shape，忽略负padding槽。

`check_atom_mellum.py`及decode运行器显式`--reference-kv-rope`启用，默认关闭，实际flag记录核验。本轮未修改共享AITER或默认数学路径，不作为生产后端直接启用。

## CPU检查及下一GPU对照

模型相关41项、运行器相关5项通过。新增缓存写回测试覆盖15/16/17槽跨block、负槽忽略、未写入槽不变和shape拒绝；语法/diff检查通过。没有运行新GPU候选，物理KV与HF逐元素全量比较仍未完成。

普通终端：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_decode_validation.py --reuse-hf-run /workspace/results/mellum_decode_validation/20261112T123830320864Z --reference-kv-rope
```

只重跑ATOM。与本轮基线比较每个真实logits位置，首先看prefill是否仍精确和首步decode差异是否变化，再看跨1024行为。若仍不同，按同输入Q/K/V分离decode内核与窗口/KV取数；不因单项改善标记Step0完成。运行器继续15秒心跳、默认单阶段300秒超时及worker死亡检测。
