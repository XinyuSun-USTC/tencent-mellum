# 30 短输入decode精确与滑窗首步快照核查

## 实际结果

运行`results/mellum_decode_validation/20261112T142950219118Z`，ATOM`results/atom_mellum_gpu/20261112T142950264193Z`。启用separate QKV、torch MoE、CPU参考RoPE、正确K写回及分页缓存SDPA decode。

- chinese_inventory、raw16、raw17每条8位置（prefill+7decode）全部logits逐元素精确。
- window1023的prefill精确，7个decode仍不同，首次decode最大绝对差0.75。
- 共25/32位置精确；其中新增21个短输入decode位置精确，强制历史/position与加载审计通过。

相同Q/实际缓存的首层原内核与SDPA首decode输出：chinese_inventory精确；raw16/17最大差0.000488281/0.000244141；window1023最大差0.00341797、相对L2约0.00720168。不能由此判断全部层均为内核差异，但现在已排除“所有decode均有差异”的状态，剩余主线集中于窗口样例。

## 边界源码检查

独立HF5.10.1 CPU实测DynamicCache+create_sliding_window_causal_mask：逻辑上下文33/1023返回None；1024/1025/1030返回(1,1,1,1024)全True bool mask，kv_offset分别0/1/6。当前候选的mask/GQA选择与此一致，不能未经比较再改阈值。

需要直接确认相同历史下的Q与实际读取K/V是否等于HF，而不是继续只看最终logits或更换窗口数值。

## 轻量首层重建

新增`scripts/diagnose_mellum_decode_boundary.py`，读取本轮保存的四份first_decode_attention.pt，只重建HF第0层首decode：

1. 从原checkpoint读取embedding和首层input/Q/K norm及Q/K/V投影；embedding留CPU、仅选中token的hidden移GPU，不构造12B完整模型或MoE。
2. 按官方类执行norm、分开投影、CPU初始化RoPE；prefill与单token分别计算，避免把cached单token矩阵形状混为整段计算。
3. 通过官方DynamicCache先写prefix，再按实际mask接口处理当前token，得到真实reference Q、K/V和SDPA context。
4. 分别比较已保存的query、keys、values和sdpa_output，保存reference快照及mask。由首次不同字段区分投影/旋转、缓存写读、context计算。

CPU预检核对四份快照BF16、shape、有限值、可见范围及哈希，产物`results/mellum_decode_boundary/20261112T143959871084Z`。在四份真实快照上CPU执行官方cache/mask/SDPA接口，输出shape及有限值检查通过，CUDA未初始化；语法/CLI通过。GPU重建尚未执行，本轮未改模型后端。

普通终端：

```bash
/workspace/venvs/hf-reference/bin/python /workspace/scripts/diagnose_mellum_decode_boundary.py --run /workspace/results/mellum_decode_validation/20261112T142950219118Z
```

如果Q/K/V已精确而context不同，再查具体mask/布局/SDPA调用；若缓存内容不同，先查写入/物理block映射，不把边界差异一概当成内核误差。短输入decode精确不等于窗口、多请求和正式Step0已完成，默认路径继续不变。

## 轻量首层重建结果

`results/mellum_decode_boundary/20261112T145752408259Z`四例query、keys、values及sdpa_output全部逐元素精确，包括1024逻辑上下文。首层实际K/V写读及该点SDPA可复现HF；剩余最终logits差异不能继续归因于首层。该结论只覆盖首个decode的第0层，不代表全部层/后续步。下一步见[验证31](31_window_decode_layer_observation.md)。
