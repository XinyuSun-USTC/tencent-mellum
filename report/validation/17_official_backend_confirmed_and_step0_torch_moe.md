# 17 官方专家后端确认与Step0 PyTorch MoE接入

## 已确认的参考行为

用户结果`results/mellum_expert_math/20261111T162402874401Z`：官方grouped_mm在四条保存输入上逐元素精确复现原HF MoE输出；官方eager不能复现。结合装饰器分派源码，原参考的专家数值路径已由实际重放确认。ATOM在固定相同输入及HF路由下仍约0.413%–0.431%相对L2不同。不能把官方不同后端之间的差异直接判为AITER内核bug，但Step0需要有明确可验证的实现基础。

## 本轮适配变更

新增`ATOM/atom/model_config/mellum_experts.py:MellumTorchExperts`及`grouped_linear`，实现TP1、无量化专家计算：FP32 softmax/Top-k/归一化后回BF16；按专家排序token，计算offsets；用冻结PyTorch的`torch.nn.functional.grouped_mm`执行gate/up和down；BF16 SiLU结果再乘up；BF16加权后恢复token/Top-k顺序，sum归约。GPU不导入新Transformers，也未改变冻结软件版本。CPU使用逐组linear供测试，不能视作GPU内核验证。

`MellumTorchExperts.weight_loader`保留`w13_weight`、`w2_weight`名称及w1/w3/w2、expert_id回调协议；gate/up写入canonical上下半区，down独立存储，检查专家范围、shard和shape。该模块无FusedMoE的shuffle后处理。既有5631槽审计仍适用，待真实GPU运行核验；加载回调覆盖不等于全权重数值审计。

`ATOM/atom/models/mellum.py:MellumSparseMoeBlock`保留`gate`/`experts`接口，gate沿用ReplicatedLinear。`MellumDecoderLayer`默认`MELLUM_MOE_BACKEND=torch`，显式`aiter`可保留原Qwen3-MoE/FusedMoE实现进行对比；非法值拒绝。正确性路径拒绝非BF16量化配置，现有模型已有TP1约束。未修改共享AITER、其他模型或compile装饰器。

`scripts/check_atom_mellum.py --moe-backend torch|aiter`默认torch，实际选择写入settings并传入worker。正常生成验证和固定输入重放都使用可审查的同一模型接口。当前重放默认native路由指的是所选后端默认路由：torch后端即本模块HF公式；历史native结果来自AITER，必须同时看结果settings，不能混为同一实现。

选择这个路径是Step0正确适配的实现决策，不是性能优化：在保留参数映射、模型接口及原后端入口的基础上建立可比较的算术基线，后续可替换专家实现。尚不保证其吞吐、图捕获或多卡能力；当前不把这些扩展作为已实现范围。

## 验证与边界

- 配置、模型、专家CPU31项通过：包括真实5631映射审计既有测试、shard写入/错误拒绝、路由排序及逐token独立公式对照。
- 独立HF环境调用官方grouped_mm专家CPU后端，在FP32/BF16、1/7/32 token共6种小规模组合与新模块逐元素精确一致。两者CPU均走逐组矩阵乘，不证明GPUgrouped_mm已通过。
- Python语法和ATOM diff检查通过。未运行GPU；未宣称首层数值或完整模型分叉已修复。Attention及其他剩余差异仍可能导致生成不同。

下一步先固定输入重放，检查输入/router及MoE输出是否精确一致，且loading audit完整：

```bash
/opt/venv/bin/python /workspace/scripts/replay_mellum_moe.py --run /workspace/results/mellum_validation/20261111T140735437405Z
```

这是诊断干预结果，不做生成验收。若MoE对齐，再运行short64及扩展集正常回归；若仍不一致，按router、grouped投影、激活和归约逐项检查，不能跳过诊断直接宣称Step0完成。

## GPU固定输入重放结果

用户产物`results/mellum_moe_replay/20261111T170208537720Z`，实际ATOM运行`results/atom_mellum_gpu/20261111T170208566140Z`。已核验settings中`MELLUM_MOE_BACKEND=torch`，加载审计5631/5631、missing/unexpected为空，选定九项权重精确一致。四条输入的moe_input、router_logits、moe_output共12项全部逐元素精确一致。

这确认了新后端在实际冻结环境、原始checkpoint加载后的首层同输入数值闭环；尚不证明全部层、正常生成、Attention或缓存正确。该轮替换了首层MoE输入，生成结果不能用于验收。

下一执行点恢复无干预short64正常回归，同时保存短输入逐层/第0层子算子结果，以便检查剩余误差而不重复单独GPU取证：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite short --dump-layers --dump-stages
```

正常入口没有`--moe-replay-inputs`，会清除MoE输入与路由干预环境变量。核验两条各64新token、prefill logits及误差随层变化；通过后再运行expanded。即使token一致，非零logits仍保留审查状态，不自动标记Step0完成。

## 新后端正常short64回归

真实套件`results/mellum_validation/20261111T172731825963Z`，ATOM`results/atom_mellum_gpu/20261111T172753495222Z`。两条各64 token共128/128完全匹配，HF元数据已实际记录experts_implementation=grouped_mm。该轮没有MoE输入替换。

| 指标 | 原AITER后端短回归 | 新torch后端短回归 |
|---|---:|---:|
| python_add prefill最大绝对差 | 1.75 | 1.75 |
| python_add prefill RMS | 0.399905 | 0.773096 |
| arithmetic prefill最大绝对差 | 2.5 | 1.25 |
| arithmetic prefill RMS | 0.933083 | 0.265387 |
| python_add 首层输出相对L2 | 0.00291362 | 0.000570821 |
| arithmetic 首层输出相对L2 | 0.00327450 | 0.000322374 |
| python_add final norm相对L2 | 0.104878 | 0.103626 |
| arithmetic final norm相对L2 | 0.0770844 | 0.0554404 |

旧结果为`20261111T120512707283Z`。这是不同运行之间的观测对比，不能把全部数值变化自动归因于单个算子。首层明显更接近，但代码题全词表RMS反而升高，不能笼统宣称全模型数值已改善或闭合。Attention context两例仍不同，相对L2约0.000408/0.000257；自然前向中MoE输入也不同，不能把当前MoE输出差异等同于此前固定同输入实验。

当前状态保持tokens_exact_logits_review_required。下一轮重跑expanded，核验此前8/11分叉是否改变及长输入64token覆盖，不重复已完成路由实验，不保存长输入逐层dump：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite expanded
```

若仍分叉，优先从原始/公共前缀对照定位剩余Attention及逐层误差；即使全部生成匹配，也不据此掩盖logits差异或宣称官方验收完成。
