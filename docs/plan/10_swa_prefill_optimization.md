# 10 优化点1：原生窗口SWA prefill候选实现与验证

## 决策及预期

扩展实测[性能02](../../report/performance/02_expanded_bs1_profile_study.md)显示16K prefill SDPA约占该阶段kernel累计时间72.2%，SWA每层约122ms、Full每层约36ms。第一个候选只替换达到窗口阈值的SWA prefill，复用冻结AITER local attention，避免reference实现的n×n显式mask和KV头展开。预期主要改善长输入TTFT；长输出总时间仍受decode制约，不预先承诺加速倍数。

这算一个Attention执行路径优化点，不宣称复用既有组件免计优化点。未经GPU数值与性能证据，不提升为默认后端。本轮未安装或升级工具/依赖。

## 源码依据与范围

- 原路径：[sdpa_single_prefill](../../ATOM/atom/model_config/mellum_attention.py)验证单请求、完整未缓存prefill，然后参考RoPE；n≥window时显式band mask、repeat_interleave展开KV后SDPA。
- AITER本地冻结源码：`/app/aiter-test/aiter/ops/mha.py:flash_attn_varlen_func`（2603行起）的接口文档定义位置i可见`[i+lenK-lenQ-left, i+lenK-lenQ+right]`，含两端；因此相同长度全prefill传`(1023,0,0)`表示最多1024个key。该文件`_flash_attn_varlen_forward.can_impl_fmha_v3_fwd`排除SWA，不能把该候选称为fmha_v3加速；具体底层选路及性能待实际执行确认。
- 新函数`aiter_swa_prefill`使用原生32Q/4KV头、BF16、原scale、causal=True、无sink/dropout；Q/K/V contiguous保证调用布局。n、mask阈值与元数据验证保留在调用者。
- [MellumAttention.__init__](../../ATOM/atom/models/mellum.py)读独立开关`MELLUM_SWA_PREFILL_OPT=0/1`，1必须结合CPU逆频率reference prefill；默认0，其他非法值拒绝。没有修改已有compile装饰模型文件。
- 仅n≥window且window>0进入候选；Full attention和短SWA输入保持reference；不是同时恢复旧native整条attention链。保留QK norm、CPU逆频率/参考RoPE、K覆盖写回、实际分页cache、decode SDPA、router、MoE和greedy调度。
- kernel不可用直接报错，不静默回退，避免把回退结果当优化性能。

目前仍是单请求完整prefill，不新增batch/chunk/prefix支持。候选使用同一RoPE输入与缓存语义，但attention归约/舍入可能不同，因此不能假定logits逐位一致或生成不变。

## 已完成的实现与本地检查

2026-11-13：代码、CLI开关及GPU评测脚本已实现。CPU相关54项通过（attention/model/runner/profile/新benchmark摘要测试），包括窗口阈值分流、window−1语义、原生GQA头数、无静默回退、候选CLI继承Step0配置。py_compile及diff空白检查通过。仅有既有Torch profiler提示，不是测试失败。

窗口验证prepare-only：`/workspace/results/mellum_validation/20261113T203336729685Z`；ABBA计划prepare-only：`/workspace/results/mellum_swa_optimization/20261113T203336638798Z`。这些是准备记录，非GPU通过或性能提升。

当前工具进程中`/dev/kfd`不存在，冻结Torch报告cuda_available=False、device_count=0。不能直接运行GPU，且不能通过安装工具解决设备不可见问题；以下命令由用户在已有GPU终端运行。

## GPU实验步骤

### 1. 单层数值与性能诊断

```bash
/opt/venv/bin/python /workspace/scripts/benchmark_mellum_swa.py --micro-only
```

固定seed，Q32/KV4/head128，随机BF16 Q/K/V，长度1023/1024/1025/4096/16384。reference为显式窗口mask+KV展开SDPA，candidate为AITER local；记录输出max_abs/relative_l2、各10次CUDA event计时，三次预热、基线/候选交错。reference计时包含mask/展开创建，与当前代码执行边界一致。1023的candidate只用于检查窗口接口，在整模型1023仍不切换。

CUDA event测GPU区间，不能替代整模型端到端计时。随机QKV不能替代真实模型验证；数值差异只是诊断，不发明新的BF16验收阈值。kernel失败或非有限输出会终止，不继续声称结果有效。

### 2. 真实模型HF对照

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite window --swa-prefill-opt --stage-timeout 900
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite long --swa-prefill-opt --stage-timeout 900
```

window覆盖1023/1024/1025（各8输出），long覆盖8191/8192/8193/16384（各4输出），用于快速边界筛查，不冒充完整64token验收。`--swa-prefill-opt`自动继承已验Step0配置，只打开本候选；实际开关在results中核对。默认捕获完整logits辅助诊断；runner若报comparison_failed，保留token/序列匹配率及差异，不因GPU内核数值不逐位相同就直接认定违反原题。

边界筛查通过或差异评估可接受后，使用现有expanded（包含边界/长输入64token）及200条固定开发集进行正常greedy回归。原题≥90%公开prompt匹配与HumanEval差距口径保持不变；同时报告token位置和整序列匹配率，不通过放宽脚本阈值冒充正式验收。正式公开集身份与HumanEval口径仍沿用既有边界。

### 3. 整模型性能ABBA

```bash
/opt/venv/bin/python /workspace/scripts/benchmark_mellum_swa.py
```

先执行上述micro，再四个独立引擎阶段reference→candidate→candidate→reference，避免把旧运行不同时间的结果当直接基线。每阶段相同TP1/DP1、BF16/eager、maxlen16640、maxseq1、util .35、无prefix/chunk、无profiler和标签，加载审计5631/5631。1K×256、4K×1/64、16K×1/64；每配置一次同预算预热、三次计时，即每种后端每配置6次计时。

记录所有wall/TTFT/TPOT、输出token、环境和源码哈希；summary报告中位数比值、原始分布、同后端重复一致性、基线与候选token位置匹配率。1输出不报告TPOT。固定输出ignore_eos，属于开发性能负载；精度正式检查用正常生成和独立HF对照。

四轮需要反复加载模型，按当前基线估计约10–15分钟，编译/JIT和机器负载可能增加时间；这是预算估计，非实测承诺。每15秒心跳，单阶段1800秒上限、错误清理owned进程组、关闭引擎。micro较短，但首次AITER编译时间未知。结果写`/workspace/results/mellum_swa_optimization/<id>/`，包括micro.json、四阶段worker_results及summary.json；不覆盖旧基线。

## 采纳条件与后续

先核对窗口语义、实际候选配置、加载审计和生成差异，再比较关闭profiler的端到端收益。主要观察4K/16K TTFT，1K及decode TPOT检查回退；保留两轮基线与候选原始样本，时段漂移较大时追加交错重复，不仅展示最快值。

如数值差异造成不可接受的greedy匹配，保留失败证据，研究attention归约或其它窗口实现；不同时改RoPE/MoE来掩盖首差。如收益不显著，检查真实内核选路后决定放弃。正式并发A与到达过程C尚未接入，任何bs1收益均不外推为正式吞吐/尾延迟成绩。GPU结果到来后再新增性能03实测报告，不预写收益。

## 2026-11-14：长生成分叉后的三方对照

ABBA已实测长输入TTFT有收益，但同后端重复稳定、基线与候选分别在第11/20/12个token分叉，见[性能04](../../report/performance/04_swa_prefill_abba.md)。候选默认仍关闭；本轮不改attention实现或准确率阈值。

新增`run_mellum_swa_hf_check.py`：严格读取121938 ABBA四阶段一致的plan，校验输入token SHA、原始源文件SHA与模型config SHA，取原样1K/4K/16K三个token序列；decode文本仅供展示，不重新编码、不重新套chat模板。生成协议统一正常greedy最多64token、正常EOS；不加载ABBA生成结果作为模型输入、不强制历史。

执行顺序为一次HF参考与优化候选，再复用该HF参考运行关闭优化的基线；三组模型顺序加载。每组检查实际后端、正常采样及5631加载审计。分别输出HF-vs-candidate/HF-vs-baseline的token位置/序列匹配与同历史logits；同历史比较包含首差当步，首差之后停止。新增诊断字段：平均logits偏移、去均值RMS、top1-top2间隔、双方所选token在对方logits中的偏好差。它们只解释差异，不新增接受阈值。

另将两种正常生成输出与对应ABBA同后端前缀比较，披露正常EOS可能截断、1K原性能预算256与本次64的差别。这样可以区分HF-vs-Step0差异、候选引入差异及运行协议变化。`diagnostic_complete`只表示证据齐全，候选token不一致也会保留结果，不表示验收通过；执行/配置/缺失产物错误才中止。

本地准备记录：`/workspace/results/mellum_swa_hf_check/20261114T130643862112Z`。输入/哈希验证、分叉前比较及偏移诊断相关10项CPU测试通过。工具环境再次核验cuda_available=False、device_count=0，未启动GPU。用户终端命令：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_swa_hf_check.py
```

预计约3–5分钟（实际随机器状态变化），每15秒心跳，HF+候选阶段外层上限为2×900+120秒，内部各900秒，基线900秒。结果位于`/workspace/results/mellum_swa_hf_check/<id>/summary.json`及对照子目录；原始HF/ATOM张量仍保留各自运行目录，不覆盖旧产物。结果到来后再决定同输入同层attention回放或更换候选，不马上扩大完整200条或新增重复性能任务。
