# 22 Prefill组合结果与RoPE系数来源核查

原始产物：`results/mellum_prefill_study/20261112T102117972534Z`，复用HF参考`20261112T095954484085Z/hf`。

## 结果与边界

两个组合的5631加载槽审计、首层QKV及9项选定权重均通过。separate+SDPA下一token4/4，separate+SDPA/分步旋转3/4，均非自由生成测试。

| 输入 | separate+SDPA logits RMS | 再加分步旋转 logits RMS |
|---|---:|---:|
| h_0_prefix_0 | 0.325464 | 0.0981571 |
| h_0_prefix_14 | 0.366507 | 0.513880 |
| h_1_prefix_0 | 0.219722 | 0.103150 |
| h_1_prefix_9 | 0.208116 | 0.408337 |

两组整体数值未闭合，不因token匹配数或部分RMS下降提升默认。首层embedding、input norm、QKV、Q/K norm四例全部精确，最早观察差异在Attention context（约3.59e-5–5.20e-5）；该首层各子算子结果在两个组合间相同。分步旋转主要从后续full层改变结果，但没有消除首层残差。最终norm：SDPA组相对L2约0.0534–0.0752，分步旋转组约0.0284–0.0618，仍不可当作数值通过。

## 下一假设：系数来源，而非再改变乘加顺序

源码证据：官方`MellumRotaryEmbedding.__init__`调用频率初始化，正常参考模型在CPU构造后转GPU；`forward`把inv_freq搬到输入设备，再以FP32计算频率、cos/sin并转输入dtype。独立官方CPU构造再移GPU的RoPE+SDPA已实际四例精确复现保存HF context。

ATOM `ModelRunner`设置默认GPU设备后创建模型，AITER `RotaryEmbedding._compute_inv_freq/_compute_cos_sin_cache`按当时默认设备生成频率和缓存；已有CPU FP32 cache相等不能证明GPU构造路径相等。CPU与GPU频率初始化可能导致细微系数变化，但这目前是假设，不是已确认根因。

新增`scripts/diagnose_mellum_rope_coefficients.py`，不加载模型权重或引擎：

1. 官方RoPE在CPU初始化再转GPU，作为与原参考一致的构造方式。
2. 同一官方类在GPU初始化，唯一改变初始化设备。
3. 实际AITER GPU BF16缓存，对照前两者。
4. 覆盖sliding/default及full/YaRN两种类型，位置0–165及1023/1024/8191/8192/8193/16383，共172位置。

保存两种官方inv_freq、三种cos/sin、不同元素数及位置，先检查inv_freq是否不同，再检查AITER系数是否与GPU初始化的HF一致。后者若成立且相对CPU初始化HF不同，才支持初始化来源假设；若不成立，继续查其他缓存构造差异，不直接修改模型。

CPU接口检查已覆盖两种类型的全部172位置及比较函数，输出shape均(1,172,128)，有限且未初始化GPU；语法检查通过。尚无GPU系数结果，本轮没有修改模型或默认后端。

普通终端：

```bash
/workspace/venvs/hf-reference/bin/python /workspace/scripts/diagnose_mellum_rope_coefficients.py
```

这是轻量系数实验，不重复加载12B模型。根据结果决定是否建立匹配参考初始化和位置计算的Mellum局部旋转实现；不把局部系数差异未经验证当成全部生成问题的根因。

## 系数实验结果

GPU证据已获得：AITER系数与GPU初始化HF一致，与CPU初始化HF在若干sin元素不同；已建立匹配参考初始化的显式prefill候选，详见[验证23](23_rope_initialization_candidate.md)。默认后端不变。
