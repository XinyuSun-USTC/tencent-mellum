# 23 RoPE初始化差异证据与匹配参考的prefill候选

## 实测证据

原始结果`results/mellum_rope_coefficients/20261112T103115794039Z`。CPU/GPU构造的官方inverse frequency两种类型均非精确一致，最大绝对差2.9802322387695312e-8；AITER GPU系数与GPU初始化HF系数在本次172位置全部精确一致。

相对原参考使用的CPU初始化HF：

| 类型 | cos差异 | sin差异 |
|---|---|---|
| sliding/default | 0 | 10元素，位置43/86/129/139/8192，最大0.0009765625 |
| full/YaRN | 0 | 2元素，位置8192，最大0.00048828125 |

这支持初始化来源导致已测系数差异，不再只是猜测。范围为172个位置的BF16系数，不能推广为全部位置或断言已解释全部模型分叉。不能将CPU/GPU浮点差异本身认定为AITER内核bug。

## 显式候选实现

`ATOM/atom/model_config/mellum_attention.py:cpu_inverse_frequency`在显式CPU上下文中调用现有default/YaRN频率公式，复用先前已对照官方的AITER公式；默认类型传base，YaRN传scaling_factor并保留mscale。返回CPU FP32 inv_freq，避免引擎默认GPU设备改变初始化来源。

`reference_position_rope`将inv_freq移到Q/K设备，在autocast关闭时以FP32 matmul计算实际position对应的相位、cos/sin，再按attention_factor缩放并回BF16，最后执行分步旋转。没有预先将逆频率截断为BF16，也不使用旧GPU构造的cos/sin缓存。

`MellumAttention.__init__`新增显式`sdpa_cpu_rope_diagnostic`，与separate QKV和torch MoE组合，仅用于已有受限单请求、窗口内的prefill。实例保存CPU频率元数据，不增加checkpoint参数或改写共享AITER。实际KV写入与decode仍沿原路径，因此不作为完整Attention/cache修复，默认aiter不变。

`run_mellum_prefill_study.py --cpu-rope-study`仅跑这一个组合，复用上一轮相同HF参考；与已完成separate+SDPA/缓存分步旋转组合比较，不重复HF模型加载或自由生成。

## 本轮CPU验证

- 用实际AITER频率类源码的CPU构造与官方HF MellumRotaryEmbedding直接比较，default/YaRN两类型各172位置：CPU inv_freq及BF16旋转Q/K逐元素精确一致。该检查没有导入AITER GPU扩展或初始化CUDA。
- ATOM模型/配置/专家/Attention/投影36项通过；新增测试验证外层默认设备为meta时仍CPU初始化，以及外层autocast开启时结果不变。
- 脚本相关9项通过，语法与diff检查通过。
- prepare-only结果`results/mellum_prefill_study/20261112T104249959042Z`。GPU候选未执行。

普通终端：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_prefill_study.py --cpu-rope-study --reference-run /workspace/results/mellum_prefill_study/20261112T095954484085Z
```

优先检查首层context是否精确、最早出现差异的层和完整prefill logits。若完整prefill仍不一致，固定最早差异层输入继续分离，不能根据四个下一token选择性宣称成功；若精确，再按协议增加独立输入并进入强制同历史decode/KV，尚不开始自由生成扩展验收或性能优化。
