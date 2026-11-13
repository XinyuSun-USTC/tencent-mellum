# Step0：当前适配与正确性验证方案

当前状态：核心Step0适配及开发生成一致性验收完成（200条前64token全部一致，16k长输入64token精确）；题目HumanEval正式门槛待补，下一主线转入性能测量入口与profiling。当前执行协议与实现状态以[04 正确性协议](04_correctness_protocol.md)为准；逐轮实验保留原始结果，不将旧计划中的“尚未实现”误当成现状。

## 目标与约束

最快、最简单地完成正确单卡BF16版本，为Step1保留必要接口；不刻意复刻出题人的朴素Baseline。冻结ATOM/PyTorch/AITER/ROCm及原Transformers；独立HF5.10.1环境仅用于参考。用户已允许开发与验证脚本同步实施。

优先复用ATOM启动、调度、加载和兼容组件。已经证实不匹配参考数值路径时，可以新增Mellum局部正确性实现；复用优先不意味着必须保留FusedMoE。避免修改共享模型与现有compile装饰模块，不预建复杂优化框架。

## 当前里程碑

- 真实checkpoint HF GPU参考与ATOM配置、模型及5631权重加载审计已建立。
- 普通short两例128token和多样十例640token均逐token、逐步完整logits精确，含1023/1024滑窗边界；HF隐式专家切换与greedy并列调度已对齐。
- 当前候选仍是显式受限路径：单请求完整prefill与单请求decode。长prefill band mask已通过1025及8192附近/16k GPU smoke；长输入64token回归与服务负载仍待验证。
- 本地Step0验收审计结论不完整，详见[报告36](../../report/validation/36_step0_acceptance_audit.md)。旧数值定位实验作为历史保留，不再作为当前未解决问题。

## 当前工作顺序

1. 固定已验证候选与配置，结束同类数值定位/集合扩展。
2. 推进性能测量所需最小服务入口与profiling，依据瓶颈选择Step1优化点。
3. 补HF/ATOM配对HumanEval与参考材料核对，可与性能分析交错；未完成前不宣布题目全部正确性门槛通过。
4. 最终配置完成必要请求调度、服务与题目负载验证，不将额外完整服务测试无限扩大为profiling前置门槛。

## 完成标准

能生成文本不是完成标准。需具备可复现的ATOM单卡执行/服务、加载审计、可信参考对照及prefill/decode/KV/长输入结果。原题要求200条前64token匹配≥90%、BF16 HumanEval差距≤2个百分点及16k子集；正式数据/口径缺失必须披露，不能用自建集合冒称通过。不先编造统一数值容差，不因分叉修改评测参数。

当前状态依据[报告38](../../report/validation/38_step0_developer_200_agreement.md)末节：开发200条100%一致；完整logits逐值一致是既有诊断证据，后续验收以题目要求为准，不要求所有优化保持完整logits逐值精确。
