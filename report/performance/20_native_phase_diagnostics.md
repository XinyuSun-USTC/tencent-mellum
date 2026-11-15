# 20 固定MoE重复差异与Attention分阶段结果

[计划22](../../docs/plan/22_bounded_native_phase_diagnostics.md)完成。MoE诊断累计16.28秒，Attention两个D200累计531.34秒（8.86分钟），均未触及预算。原始结果未改，本轮仅只读核验、源码跟进和归档，无追加GPU或数学修改。

## MoE：排序顺序重复，固定输入仍变化

固定layer0真实权重、输入、路由，Opus/CK排序各独立进程；prefill150、decode1、decode32均12次调用。每个shape在两种排序下，后11次均与第一次不同，所有输出有限。输入/路由未被修改的断言通过，token-id有效前缀排序记录在各shape内相同。

| shape | Opus最大重复差 | CK最大重复差 | Opus/CK相对Torch参考L2 |
| --- | --- | --- | --- |
| prefill150 | 0.015625 | 0.015625 | 0.004065 / 0.004053 |
| decode1 | 0.00390625 | 0.0078125 | 0.005217 / 0.003279 |
| decode32 | 0.0078125 | 0.0078125 | 0.004108 / 0.004569 |

参考是canonical grouped_mm/bmm算术的局部回放，不是全模型D200/HumanEval分数。decode-shaped输入来自已保存prefill末行，不能称为完整真实decode轨迹。

全部实际metadata为run_1stage=false、ksplit=0，stage1=aiter.fused_moe.ck_moe_stage1，stage2=aiter.ops.moe_op.ck_moe_stage2_fwd；block_m分别128/32/128。**此前并未测到one-stage，因此不能把强制关闭one-stage当成本轮修复。** 更换排序不能消除本轮固定回放差异，不追加“换排序再D200”。

### CK原子归并的具体依据

短真实GPU trace中有ck::kernel_moe_gemm_2lds，包含`(ck::InMemoryDataOperationEnum)1`的实例。冻结源码：

- `/app/aiter-test/3rdparty/composable_kernel/include/ck/ck.hpp:278`，enum顺序Set、AtomicAdd、AtomicMax、Add，故1为AtomicAdd。
- `.../include/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp:280`，输入GEMM使用Set，输出GEMM使用AtomicAdd。

这确认当前CK输出归并走原子加路径，且固定输入/排序下仍有变化，与浮点原子归约顺序变化相符。仍未逐stage捕获中间结果，不能证明重复差异全部只来自stage2，也不能把数值非确定性直接称作内核语义bug。它为改进累加/归并策略提供了具体依据，不再是无目标逐层对齐。

下一项有价值的MoE修正应针对现有stage2的归并精度/顺序：例如保留高性能专家GEMM，将专家贡献独立写出再固定top-k归约；需先核对冻结API能否承接，并考虑HF的激活/加权舍入差异。重复稳定不等于与HF一致，不能保证仅修归并就过D200。

## Attention：两个阶段各自替换都未过90%

均保持R专家和参考RoPE/KV处理，正常greedy64、200请求、真实decode32；实际native/reference两阶段审计通过。

| 模式 | HF位置匹配 | HF完整序列 | 短输入位置 | 长输入位置 |
| --- | --- | --- | --- | --- |
| native prefill＋reference decode | 9287/12800=72.5547% | 89/200=44.5% | 74.2560% | 63.6230% |
| reference prefill＋native decode | 9888/12800=77.2500% | 101/200=50.5% | 80.6083% | 59.6191% |

这否定了“只撤掉其中一个阶段，就已有可晋级候选”的设想。二者均低于位置90%，脚本正确跳过256性能，没有不完整性能任务或预算截断。不把77.25%称为接近达标，也不把局部张量差异小当放行依据。

本轮native prefill涉及所有窗口及全局层，**不同于此前仅对激活窗口层替换的SWA CK候选**。因此不能用本轮72.55%否定此前SWA CK开发200的93.11%，也不能把此前93.11%套给本轮配置。原生decode的窗口/全局各层贡献尚未拆解；当前无源码证据证明存在页表或窗口索引错误。

## 决策与边界

- 停止当前整套原生Attention和两项分阶段版本晋级，不再原样重跑；保留已经有独立证据的SWA CK作为后续组合选项。
- MoE排序和one-stage假设已得到有限结论；若继续，优先一个“专家贡献独立输出＋明确归约”的实质修正，先局部回放再正常生成，不扩展无边界配置矩阵。
- 两条高性能方向有性能潜力，但目前尚无同时满足性能和生成门槛的新增最终配置。完整HumanEval、16k、最终ABC仍未补齐，不能在总结中消失。
- 不降低门槛、不缓存输出、不要求HF内部bitwise；下一项修正以题目生成/任务指标决定是否保留。

[MoE轻量证据](../../artifacts/performance/20_native_phase_diagnostics/moe.json)、[Attention证据](../../artifacts/performance/20_native_phase_diagnostics/attention.json)。独立验证14+20=34个产物、两批各240源码指纹，共480项，全部一致。

原始目录：`/workspace/results/mellum_native_moe_diagnosis/20261115T195607968694Z`、`/workspace/results/mellum_native_attention_diagnosis/20261115T195614914099Z`。MoE的diagnostic_complete仅表示诊断执行完成，不是重复性或正确性通过。
