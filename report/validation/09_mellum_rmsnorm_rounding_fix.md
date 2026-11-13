# RMSNorm首个数值差异：证据与Mellum局部修正

日期：2026-11-11。状态：已修正第一个已确认的计算顺序差异，CPU回归通过；GPU整模型结果待验证，不能宣称解决全部64token分叉。

## 子算子证据

[stages_comparison.json](../../results/mellum_validation/20261111T115534143208Z/stages_comparison.json)

第0层所检查的9项权重（input/post norm、Q/K norm、Q/K/V/O投影及router）全部逐元素、dtype、shape一致；不包含expert权重。四个探针的input norm输入也精确一致，但其输出相对L2约0.4076%–0.4137%，最大绝对差0.0009765625。随后QKV、QK-Norm、Attention及MoE的输入已不同，不能直接将它们的端到端差异判为各自kernel故障。

这将首个有证据的差异定位到input RMSNorm的数值实现，而非输入/该norm权重加载。

## CPU重算

[重算脚本](../../scripts/check_mellum_norm_cpu.py) · [结果](../../results/atom_mellum_norm_fix/cpu_replay.json)

使用保存的真实BF16输入和weight，在CPU FP32中计算归一化。ATOM旧输出与“FP32归一化后乘FP32 weight，最后截断低16位到BF16”的结果：两个探针100%逐元素一致，另两个分别99.997586%和99.998194%一致（各差1个元素）。因此证据支持单次截断/舍入顺序差异；不将这次重算视为所有GPU内部指令的反汇编证明。

HF官方`MellumRMSNorm.forward`顺序明确：先FP32归一化，再转回输入dtype，最后乘weight。CPU按此顺序重算与捕获的HF GPU最大差0.0001220703125，旧ATOM与HF GPU为0.0009765625。CPU/GPU reduction差异尚存在，不能宣称修正后与GPU逐元素一致。

本次另观察到未改数学路径时arithmetic同前缀token由失败变为通过，保留运行间变化证据，不以单次token匹配当作已修复。HF cached/prefill自身差异仍单独追踪。

## 必要适配修正

`ATOM/atom/models/mellum.py:MellumRMSNorm`新增Mellum专用实现，覆盖input/post norm、Q/K norm和final norm：

1. 如有residual，先在输入dtype执行相加并保留结果，匹配HF Decoder显式残差边界。
2. 转FP32、均方、rsqrt(eps)、归一化。
3. 转回输入dtype，再乘同dtype weight。
4. 对有residual的调用返回(normalized, combined)，保持ATOM骨干接口。

不修改共享RMSNorm或AITER/PyTorch，不更换依赖，不调路由/窗口，不以性能为目标。权重名称保持原样，加载审计继续使用原覆盖要求；qk/rope/cache融合仍关闭，否则会绕过该显式路径。当前支持范围仍为单卡BF16 Step0。

采用官方PyTorch顺序属于正确性适配，可能增加kernel调用；Step1如需要优化再验证等价实现，不在当前替换成未经核验的融合kernel。

## CPU验证

配置/模型26项通过，新增BF16/FP32、三维输入、有/无residual、输入不变以及实际官方HF类源码对比。四条真实输入的修正实现与官方CPU逐元素一致，残差接口也一致。没有GPU执行。

## 下一步GPU回归

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite short --dump-layers --dump-stages
```

一次运行同时检查64token及保存新的第0层/层间输出，用相同首版配置256。需要确认修正后的norm是否对齐、下游差异是否降低、64token是否仍分叉。若有剩余差异再依据新的首差位置继续排查，暂不运行window/long。

## 修正后GPU短回归（2026-11-11）

已直接读取[比较结果](../../results/mellum_validation/20261111T120512707283Z/comparison.json)、[子算子结果](../../results/mellum_validation/20261111T120512707283Z/stages_comparison.json)和[ATOM输出](../../results/atom_mellum_gpu/20261111T120535331115Z/results.json)。

- 两条各64个新token全部与HF精确一致，共128token，未发现生成长度差异。
- 第0层input norm和Q norm均逐元素一致；9项选定权重仍全部一致。旧的首差得到实际GPU证据验证。
- 第0层materialized hidden相对L2为0.2914%/0.3274%，低于此前约1.1%；但最终norm相对L2仍约10.49%/7.71%，不能表述成全模型误差已全面消除。
- prefill logits最大绝对差python_add=1.75、arithmetic=2.5，RMS分别0.3999/0.9331，top1一致。零容差比较仍失败，因此汇总为tokens_exact_logits_review_required，不是数值验收通过。
- arithmetic第0层router logits已精确一致，但MoE输出仍有约0.4034%相对L2；其输入也有约0.0495%的差异，尚不能单凭此直接认定FusedMoE有bug。

结论：RMSNorm修正确有作用，短64token回归已恢复；剩余数值差异继续追踪，不采用臆定BF16容差。接下来可运行window开发集检查独立的mask/缓存边界，结果仍须与剩余数值问题分开报告，不宣称Step0已完成。
