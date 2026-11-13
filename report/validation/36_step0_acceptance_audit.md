# 36 Step0验收审计

## 结论

**当前不能宣称完整Step0通过。** 本轮完成的是基于题目要求的本地证据与能力审计，不是官方evaluator全量运行。已验证短输入单卡适配；16k能力明确不支持，200条与HumanEval尚未评估，服务交付验证缺失。不能把这些缺项记成通过，也不能把未运行的项目记成数值失败。

可复现入口：`scripts/audit_mellum_step0.py`。本轮结果`results/mellum_step0_acceptance/20261112T221537256942Z/results.json`，status=incomplete、claim_step0_passed=false；退出码1表示验收不完整，非脚本崩溃。此脚本只读检查原始材料并写新结果，不启动推理、不执行HumanEval生成代码、不改变原始结果。

## 验收矩阵

| 项目 | 状态 | 本轮证据/原因 |
| --- | --- | --- |
| 单卡短输入正确推理 | 通过当前开发范围 | 重验short两例128token与多样十例640token，所有同历史原始完整logits逐值精确 |
| 权重加载 | 通过当前记录 | 两次均5631/5631，无missing/unexpected |
| 200条前64token ≥90% | 未运行 | 只有12条开发样例；缺正式200条结果，不能以640token一致替代prompt规模 |
| BF16 HumanEval差距≤2个百分点 | 未运行 | 无本版本ATOM/HF配对结果；现有参考值明确Rejected |
| 16k输入单独统计 | 当前候选不支持 | 实际调用候选prefill函数，16384输入被长度guard拒绝，1025同样拒绝 |
| OpenAI兼容服务交付 | 尚未验证 | 有通用服务入口，但无本候选GPU-backed服务验收产物；批量decode被候选限制拒绝 |
| 双卡/MTP | 未启用 | 不是单卡Step0的先决条件；启用后才需对应额外验证 |

服务是最终交付要求，不能把“未做HTTP验证”单独解释为模型数值失败；但当前候选能力与正式负载之间确有缺口。16k不支持本身已足以否定完整验收通过，无需为确认该限制再次启动昂贵GPU任务。

## 本轮执行内容

### 重新验证已有GPU证据

- `results/mellum_validation/20261112T193650834240Z`，2条×64token。
- `results/mellum_validation/20261112T213129468143Z`，10条×64token。

通过`compare_mellum_results.compare`重新检查参考与候选schema、token和prefill；`compare_mellum_generation.compare_generation`逐文件检查SHA256、载入原始张量并比较768个完整向量，全部精确，所选token均为各自argmax。不是仅信任原summary状态。

同时核对实际sampling_mode=normal、ignore_eos=False、无forced计划、实际候选环境以及HF每次前向grouped prefill/batched decode后端记录。使用的独立HF参考环境与ATOM冻结环境保持区分。

这些结果是已保存GPU运行的离线复核，**不是本轮重新执行GPU推理**。审计时未发现`/dev/kfd`，当前代理不能补跑GPU验收；用户终端已保存的GPU证据仍可离线验证。

### 实际能力探测

直接导入`ATOM/atom/model_config/mellum_attention.py`，使用CPU小张量及对应metadata触发相同guard：

- `sdpa_single_prefill`，输入1025、16384，sliding_window=1024，均报`Diagnostic SDPA currently requires input within sliding window`。
- `sdpa_cached_decode`，2请求/2token decode，报`Diagnostic cached SDPA requires single-request single-token decode`。

源码依据：`sdpa_single_prefill`明确只接受一请求完整未缓存prefill，且n>sliding_window直接拒绝；`sdpa_cached_decode`要求q.shape[0]==1且context_lens.numel()==1。当前正确性预设正是选择这些函数，并不是普通优化路径尚未benchmark的情形。探测只确认已知不支持分支，不能作为GPU数值测试。

### 参考材料核验

`/data/datasets/HumanEval.jsonl`含164题；`scenario_a_prompts.jsonl`含512条，`scenario_c_prompts.jsonl`含200条。这些性能场景数据的存在不证明已找到题目200条正确性公开集，不擅自等同。

`baseline.json:humaneval_note`明确将3/164的结果标记为Rejected，原因是raw completions未用chat template/thinking。`humaneval_bf16.json`也是count164、passed3，不能作为有效pass@1门槛参考。需要按正确模板与代码提取口径建立HF/ATOM配对评估，或获得有效发题reference/evaluator；不代入模型卡分数。

### CPU回归

ATOM的Mellum config/model/experts/attention/projection/decode_probe及greedy sampler测试共49项通过（3.48s）。覆盖实现与已有guard，不替代真实checkpoint GPU长输入/服务验证。未修改ATOM功能代码、冻结依赖或GPU计算设置。

## 后续执行优先级

1. **先补能力，不先重复短样例**：把已验证数学路径从受限诊断函数收敛为可用于Step0的实现，支持长prompt；按1025、8192附近、16k递进检查，保持单卡BF16及参考口径不变。
2. **固定可复现启动配置并验证服务**：确保正确后端无需散落手工环境变量，检查实际GPU推理、连续请求、必要批量/调度能力；不能只看health接口。
3. **完成题目规模验收**：200条greedy及16k子集；官方集合不可得时明确开发替代集来源与统计口径，不冒称官方通过。HumanEval同时建立HF/ATOM同模板、同停止和代码提取、执行评估，报告绝对百分点差。
4. 验收满足后固定自己的Step0版本与结果，再进行Step1 profiling和优化选择。对外目前可表述“单卡短输入适配已完成，12条开发样例768token及逐步logits精确，长上下文和正式验收正在补齐”，不能表述“Step0全部通过”。
