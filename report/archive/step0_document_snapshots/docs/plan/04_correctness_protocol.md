# 04 Step0正确性检查协议与当前执行状态

当前执行状态（2026-11-13）：核心适配与开发200条验收通过，200/200完整64token序列精确，16k 64token亦通过。停止同类精度扩测；主线推进最小服务测量入口与profiling，HumanEval正式门槛单独补齐。完整logits精确不作为新增验收门槛。详见报告38末节。


本文是当前正确性工作的主要入口，替代此前逐轮追加说明作为执行顺序；原始实验保留在report/validation。目标是先完成可解释的prefill和强制同历史decode闭环，再恢复自由生成，不通过修改阈值或挑选样例消除失败。

## 固定环境与参考语义

| 项目 | 约定 |
|---|---|
| ATOM解释器 | `/opt/venv/bin/python`，Transformers 5.2.0，原始ROCm PyTorch；不升级 |
| HF解释器 | `/workspace/venvs/hf-reference/bin/python`，Transformers 5.10.1，复用原ROCm PyTorch |
| 权重 | `/data/models/Mellum2.1-12B-A2.5B-Thinking`，配置SHA256必须匹配；不改原始文件 |
| HF专家 | `grouped_mm`为当前参考语义；官方四例独立重放逐元素复现原参考。旧运行未记录后端的事实仍保留，不补造元数据 |
| HF Attention | SDPA；每次记录实际实现，不把eager/grouped或cached/prefill混用 |
| 数值设置 | BF16、eval/inference、无采样；固定token IDs、mask、position、输出长度、EOS/pad |
| ATOM工作范围 | 单卡TP1/DP1、eager/level0、BF16 KV；现阶段关闭prefix caching/chunked prefill |
| 原始证据 | 每次新建结果目录，保存命令、环境、后端、代码版本或指纹、参考和张量哈希；不覆盖旧结果 |

独立HF环境只用于参考。默认ATOM仍采用PyTorch MoE+AITER Attention；正确性检查使用GPU普通终端，工具进程目前无GPU。新的后端若被框架自动选择/回退，必须记录实际值；仅记录请求值不充分。

## 当前实现与实验状态

| 路径 | 状态 | 证据边界 |
|---|---|---|
| Mellum配置、结构、权重映射 | 已接入 | 5631回调槽加载通过；不是全部权重逐元素审计 |
| Mellum RMSNorm | 已采用的修正 | 相同输入的舍入顺序有直接证据 |
| `MELLUM_MOE_BACKEND=torch` | 当前默认 | 四份真实固定输入下完整首层MoE与HF精确一致；不等于所有层正确 |
| `MELLUM_MOE_BACKEND=aiter` | 保留原后端 | 完整HF路由下首层同输入仍有约0.41%–0.43%差异，不宣称内核错误 |
| prefill=`aiter` | 当前默认 | 新MoE下short128token匹配，expanded仅4/11全匹配，仍未正确性闭环 |
| prefill=`sdpa_diagnostic` | 显式受限实验 | 只替换短单请求prefill；代码第38新token分叉；KV写入/decode原样，不作为生产后端 |
| prefill=`sdpa_rope_diagnostic` | 显式受限实验 | 额外分步旋转仍复用AITER缓存；未整体改善，不提升默认 |
| prefill=`sdpa_cpu_rope_diagnostic` | 显式候选，10例新集及四探针精确 | 与参考KV旋转/cached SDPA组合后32个强制历史位置精确；普通长prompt未支持 |
| 线性投影对照 | GPU证据已取得 | 分开torch重放精确、相同打包shape内核一致、K/V有打包形状差异；仅覆盖四份输入 |
| QKV=`separate` | 显式候选，默认仍packed | 与其他参考路径组合后完整prefill/cached位置精确；单独启用曾未闭合 |
| router=`torch`、参考KV旋转、cached SDPA | 显式候选，默认不变 | 32/32强制历史logits精确；普通生成待运行 |

较新的正常生成结果：[17 MoE路径](../../report/validation/17_official_backend_confirmed_and_step0_torch_moe.md)、[18 扩展集与Attention](../../report/validation/18_torch_expanded_and_attention_replay.md)。诊断结果：[19 SDPA实验](../../report/validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)、[20 旋转与线性核查](../../report/validation/20_stepwise_rope_result_and_projection_replay.md)。不得把旧AITER的9输入168token匹配计作新torch后端完整覆盖。

## 检查顺序与出口

### A：先验证实验本身

固定参考输入、权重、position及mask，核对shape/dtype/有限值/来源哈希。用明确的参考算术重放保存输出，并重复执行检查稳定性。参考自身重放不成立或重复不稳定时，先查协议、后端及执行方式，暂停内核归因。

### B：线性投影同输入对照及候选prefill（当前执行点）

脚本：`scripts/diagnose_mellum_projection_math.py`。四条固定探针，一次收集：

| 比较 | 能区分什么 |
|---|---|
| torch分开QKV vs保存HF | 参考投影能否复现 |
| torch合并QKV vs torch分开QKV | 固定算子框架下，打包shape的影响 |
| AITER合并QKV vs torch合并QKV | 相同输入/权重shape下内核差异 |
| 同输入O/router：torch与AITER各对保存HF | 非QKV打包层的计算差异 |
| 每种计算重复至少两次 | 本次输入上的重复稳定性 |

脚本自动输出`reference_replay_exact`、`all_repeats_exact`、`packing_effect`和`kernel_effect`。只在前两项成立时解释后两项；非零差异是数值事实，不自动等于实现bug。`diagnostic_complete`只表示采集完成。

决策分支：

- 参考失败/不稳定：先调查参考和执行，暂不改ATOM数学路径。
- 打包有差异：考虑保持打包参数加载、执行时分开投影的Mellum局部候选。
- 相同shape的AITER/torch有差异：考虑Mellum局部PyTorch线性正确性候选；保留加载接口，不能改共享线性模块。
- 两者都不同：分开处理两个因素，不只替换内核后声称对齐HF。
- 全部一致：这些输入上的线性假设未获支持，返回旋转/缓存及最早出现差异的层，不继续盲改线性。

当前投影结果已满足参考/重复门槛；只观察到打包形状差异，未观察到相同shape内核差异。成对prefill已完成，首层QKV精确但整体未闭合；下一轮复用同一HF参考，补separate+SDPA及separate+SDPA/分步旋转两组固定prefill。结果与代码边界见[验证21](../../report/validation/21_qkv_packing_and_paired_prefill.md)。

局部候选必须经过相同输入重放及独立输入的prefill检查后再决定是否作为默认。输出头需要时单独补查，不以QKV结果代替lm_head证据。

### C：完整prefill

少量固定输入只做一次前向，不自由生成。比较各层输入/输出、最后hidden states与完整logits，优先定位最早差异，再固定相同层输入重放。自然输入已经不同时，不能从输出差异推断本层错误。完整Attention语义必须包含位置、旋转、mask、窗口及计算；诊断中只替换其中一部分的结果不当作完整正确性后端。

出口：前向路径与剩余差异能够解释。局部精确匹配用于定位，不擅自要求所有跨内核BF16算术都逐元素一致；也不把任何非零差异一概视为可接受。当前没有充分依据设定新数值容差。

### D：强制同历史decode与KV

prefill基础明确后，实现两端逐步输入同一条固定token序列，记录每步position、上下文长度、KV写读范围、各层/输出logits。参考内部区分cached执行与重新prefill，避免将自身路径差异当作ATOM错误。先短单请求，再跨block和滑窗边界；不允许自由生成分叉污染后续定位。

强制同历史机制已完成GPU验证：当前候选32/32原始logits精确，历史/position通过，HF cached/uncached差异独立记录。强制token一致不计正确性，物理KV槽全量逐元素审计尚未实现。普通生成预设与后续边界见[验证34](../../report/validation/34_cached_decode_exact_and_normal_generation.md)。

### E：恢复端到端验收

完成前两类闭环后，依次short64、现有expanded、窗口/长输入、多请求与服务。自由生成用于整体回归，不替代算子定位。正式200条与HumanEval口径缺项如实披露，开发集通过不等于官方验收。Step1仍后置。

## 下一条GPU命令

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite short --correctness-candidate
```

32/32强制历史位置已精确。下一步关闭强制token/EOS干预，运行两条各最多64token的正常greedy生成，并被动记录同历史逐步logits，避免只依据输出相同。CPU46+9项及独立HF3项通过，新正常生成GPU待执行。预设显式、默认不变，长prompt/多请求/服务及正式验收未完成。见[验证34](../../report/validation/34_cached_decode_exact_and_normal_generation.md)。

最新正常生成回归173200失败，固定32位置精确不推广为正常生成通过。下一步为python_add两步逐层与正常generate原始logits交叉检查，命令及证据见[报告34](../../report/validation/34_cached_decode_exact_and_normal_generation.md)。

HF5.10.1 generate的GPU decode会临时将grouped_mm专家切至batched_mm。普通候选已显式对齐这一序列并校验运行时记录；手动cached诊断仍使用原grouped_mm。下一步重跑short --correctness-candidate，详情见报告34末节；尚未宣布正常生成通过。

最新里程碑：193650普通short候选128/128 token及完整逐步logits精确，实际HF专家切换记录通过。下一步执行10条多样/1023与1024边界正常生成回归，固定输入与命令见报告34末节；完整Step0与长prompt/并发/服务仍未完成。

205013多样生成：630个同历史logits全精确，2条序列因最大值并列采样规则分叉；Sampler.forward已将纯greedy在过滤分流前统一argmax，CPU3项通过。下一步原样重跑10例普通生成，证据与命令见[报告35](../../report/validation/35_greedy_tie_break_and_sampling_dispatch.md)。

最新验证213129：普通10例共640token及640个采样前完整logits全部精确，采样并列问题闭环。停止重复当前定位组；下一优先补输入>1024的长prompt能力，分1025、8192附近及16k验证，再补并发/服务与正式指标；当前受限候选不宣称Step0全部完成。证据见报告35末节。

当前正式状态以[报告36 Step0验收审计](../../report/validation/36_step0_acceptance_audit.md)为准：短输入证据重验通过，但16k能力不支持，200条/HumanEval/服务缺项；本轮不宣称Step0通过，后续先补长prompt能力。

Step0增强已实现长prefill band mask，旧>1024拒绝已移除，但GPU长输入尚未验证；验收状态仍不完整。下一命令为`run_mellum_validation.py --suite window --correctness-candidate`，通过后再到8192附近及16k。详见[报告37](../../report/validation/37_step0_long_prefill_extension.md)。

2026-11-13更新：085534 window组3×8token及逐步logits精确，1025越窗prefill已验证。下一组`--suite long --correctness-candidate`覆盖8191/8192/8193/16384各4token；仍为能力smoke，正式Step0未完成。见报告37末节。

最新085844：8191/8192/8193/16384各4token及完整logits精确通过，16k已支持且有GPU能力证据。下一步同4输入各64token，输入与命令见报告37末节。报告36“不支持16k”仅为增强前历史结论；正式验收仍缺200条/HumanEval/服务。

103154长输入64token回归256位置全精确。下一步执行固定200条开发对照，每阶段超时3600秒，输入来源/局限/命令见[报告38](../../report/validation/38_step0_developer_200_agreement.md)；HumanEval pass@1与服务仍待单独验收。
