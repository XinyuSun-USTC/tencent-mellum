# ATOM Step 0 适配实施与验证计划

历史状态：以下是2026-11-11实施前方案及后续追加，保留当时“尚未实现/待验证”的上下文，不作为当前入口。Step0现已完成本地开发验证，当前结果见[报告39](../../report/validation/39_step0_summary.md)，配置见[05基线复现](05_step0_baseline_and_reproduction.md)。

上位方案：[01 Step0](01_step0.md)；决策：[02 decisions](02_decisions.md)。本方案细化文件、批次和放行条件，不预定Step1优化项目。

## 1. 目标、基线与已具备条件

目标：冻结ATOM/PyTorch/AITER/ROCm/Transformers环境中，以最快简单的方式实现Mellum单卡BF16正确推理与ATOM OpenAI兼容服务。优先复用兼容组件，包括FusedMoE；不复刻出题人torch-loop/SDPA Baseline。不通过关闭SWA、改YaRN或冒用Qwen身份换取跑通。

- ATOM基线`26f23a0bdbf50b9010c07ee3a71171189723e767`，工作副本`/workspace/ATOM`，原始`/app/ATOM`只读。
- ATOM解释器`/opt/venv/bin/python`，Transformers5.2.0；启动`/workspace/scripts/atom`。
- 参考解释器`/workspace/venvs/hf-reference/bin/python`，官方Transformers5.10.1；原始依赖保持冻结。
- [CPU小模型24项通过](../../report/validation/02_hf_mellum_cpu.md)；[真实12B HF BF16短输入通过](../../report/validation/03_hf_mellum_gpu.md)：12,149,923,072参数，加载无缺项，两条16token cache一致。
- 主模型5631个张量的header已核对；当前checkpoint没有额外MTP权重。完整模型与算子依据见[模型研究](../../report/research/03_mellum_model.md)。

首版不做MTP、量化、并行、compile/graph调优、SWA缓存截断或调度优化。保留现有逻辑块与KV分配，先验证基础路径，之后再验证可选缓存/调度设置。不得为测试通过降低模型真实数学要求。

## 2. 文件与职责规划

以下新增文件均为拟议位置，实施时遵循实际工程约束；不可把表内符号当作已存在实现。

| 文件 | 操作/拟议职责 | 已核验依据 |
|---|---|---|
| `ATOM/atom/model_config/mellum.py` | 新增MellumConfig，适配5.2配置生命周期，保留原身份/字段/分层RoPE | `atom/config.py:get_hf_config():592+`；本地PreTrainedConfig有显式__init__，无新版__post_init__ |
| `ATOM/atom/config.py` | 增加窄范围Mellum配置分支，直接加载自定义类 | `_CONFIG_REGISTRY`当前为别名映射并依赖AutoConfig；仅向其添加mellum字符串不足以接入自定义类 |
| `ATOM/atom/models/mellum.py` | 新增MellumAttention/Decoder/Model/ForCausalLM；必要时局部RoPE参数适配辅助函数 | `qwen3_moe.py:142–262,265–352,355–431,444+` |
| `ATOM/atom/model_engine/model_runner.py` | 增加MellumForCausalLM→新类的注册，不改执行主流程 | `support_model_arch_dict:58+`、Runner根据architectures[0]构造并调用load_model:619+ |
| `ATOM/tests/test_mellum_config.py` | CPU配置、非法输入、字段/序列化回归 | 配置属于低成本、核心接入契约，值得独立测试 |
| `ATOM/tests/test_mellum_weights.py` | QKV/expert合成权重映射及覆盖审计 | 通用`model_loader/loader.py:load_model`与参数weight_loader |
| `scripts/export_hf_mellum_reference.py` | 参考进程：固定输入、教师强制、选层输出、生成参考 | 复用已跑通HF脚本的加载/模板流程，不改官方模型源码 |
| `scripts/check_atom_mellum.py` | ATOM进程：配置/模型诊断、真实加载审计、固定输入forward与生成 | 实施时核对Runner初始化、forward context、KV绑定；不在普通Python中孤立构造GPU模型 |
| `scripts/compare_mellum_results.py` | CPU离线比较两端artifact，报告首个差异与数值统计 | 两套解释器隔离运行，比较器不依赖两套模型类 |
| `scripts/run_atom_mellum_step0.sh` | 固定首版参数与显式环境，保证启动开发副本 | 已有`/workspace/scripts/atom` |

不修改原有`@support_torch_compile`模型文件；Qwen3-MoE backbone的layer_type参数没有实际替换其内部硬编码Decoder，不能把它当作无改动继承入口。新模块首版不加compile装饰器，沿用embedding/linear/norm/FusedMoE/Attention等组件契约。

`loader.py`、通用Attention后端、debug helper默认不在必改清单。只有发现并定位实际接入阻碍才作最小修正，另附原有路径回归；不顺手重构全局模块。

## 3. 模型与配置实现约束

### 配置

自定义配置保留`model_type=mellum`、`architectures=[MellumForCausalLM]`、28层layer_types/全sparse MLP、explicit head_dim、dtype/BOS/EOS、untied embedding、MoE与嵌套RoPE。先设置必要结构字段，再调用5.2基类，验证嵌套RoPE的初始化顺序；不能原样复制新版strict dataclass配置。

当前目标仅当前结构：非法层数/类型、head分组不整除、top-k超出expert数、未知rope类型明确报错。HF默认全Full/默认theta不能覆盖checkpoint。对于未实现的dense/MTP等变体明确拒绝或说明范围，不静默按当前模型解释。

不改第三方site-packages，不把mellum改名为qwen3_moe。CPU检查不实例化会触发GPU与warmup的ModelRunner。

### 模型结构与参数

独立骨干保持ATOM的hidden_states/residual契约，语义等价于HF的pre-norm residual。最终norm、LM head独立，forward返回hidden states，compute_logits负责输出投影；遵循现有wrapper接口。

Attention复用QKVParallelLinear、Q/K per-head RMSNorm、RowParallelLinear与PagedAttention；D128、Q32/KV4保持原配置，不能由2304/32错误推导D72。按照layer_types选择RoPE与逐层SWA，不依赖Qwen的decoder_sparse_step。

MoE优先复用Qwen3MoeSparseMoeBlock/FusedMoE现成路径；仅在检查确认router和权重契约兼容时直接引用。目标为FP32 softmax→top8→renormalize，expert SiLU(gate)*up→down→加权求和，无shared expert/bias。不改路由数学以追求token更接近。

### RoPE与Attention的优先验证

- `PagedAttention(...per_layer_sliding_window=...)`：接口在`atom/model_ops/paged_attention.py:23–56,174–189`；SWA1024/Full无窗口必须逐层传递。
- `AITER get_rope()`在`/app/aiter-test/aiter/rotary_embedding.py:1705+`。YaRN读取`attn_factor`而不是HF`attention_factor`，默认mscale已包含`1+0.1*log(factor)`。拟用显式参数适配，使最终cos/sin幅度等于HF，不直接改名；用频率/cos/sin/QK输出验证后才定案。缓存dtype和FP32计算顺序也要检查。
- HF可见位置是`k<=q && k>q-W`，W1024包括当前token和1023历史位置。先对prefill、单token decode与跨窗口分块逐项验证。
- `atom/model_ops/attention_mha.py:PagedAttentionImpl.rope_cache():125+`按窗口/head选择路径；不能为绕开差1问题盲切统一后端。若复现窗口bug，按backend的参数语义局部修正，不把模型全局W改成1023导致其他路径错误。
- 初始关闭`ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION`，保留显式QK-Norm路径；现有融合不会默认变成Step1目标。其他融合设置显式记录，必要时为定位关闭allreduce RMSNorm融合，但之后需验证选定首版配置。

## 4. 分批实施与放行条件

每批形成可审阅diff和验证记录；完成后保留独立提交，不混入后续优化。序号是实施顺序，不是自然日工期承诺。

| 批次 | 实现 | 验证与放行 |
|---|---|---|
| A 配置入口 | MellumConfig和get_hf_config分支 | 在冻结解释器解析真实config；与HF5.10.1导出的字段逐项一致，roundtrip保留语义，非法输入报错；其他既有配置分支无回归 |
| B 参考artifact契约 | 两端输入/输出规范、比较器，扩展HF导出脚本 | 使用现有两条真实样例重导出；验证token与模板一致、错误集合可序列化、比较器能定位人为注入差异 |
| C 模型接入和加载 | 独立Mellum模块、模型类注册、packed/expert mappings | CPU合成权重映射；GPU首轮完整加载，逐分片/expert覆盖审计；短prefill finite并能生成，尚不宣称正确 |
| D 算子/层对齐 | 在新Mellum调用边界增加可关闭诊断；必要的语义修正 | 相同权重/输入比较RoPE、Norm、Attention、router/expert和Decoder输出，定位首个差异；Teacher forcing对齐 |
| E 端到端一致性 | KV decode与server路径打通 | 固定token输入greedy、cache、窗口、多请求与短chat模板一致；HTTP能返回不够，必须核对GPU执行和原始token |
| F 长输入与交付 | 16k、缓存/调度开关回归，一键启动文档 | 参考对比、稳定多请求、正式门槛缺项披露；满足Step0完成表后才讨论Step1 |

A/B可交错推进，不等待所有参考样例预生成；C/D以短样例尽早闭环，不能堆完所有测试后才第一次启动。实际GPU不可从当前工具访问时，交付一条普通终端命令，读取用户生成的artifact后继续，不混淆CPU检查和GPU执行。

## 5. 权重加载审计

首次`ATOM_LOADER_USE_THREADPOOL=0`，使实际加载异常直接传播；源代码默认线程池wait没有future.result（`loader.py:368–379,597–600`），不能单靠“Model load done”判断成功。后续若恢复线程池，需单独确认异常传播，不能默认同样可靠。

审计至少包括：

1. 5631个源张量逐一有明确目标或经论证的忽略项；当前主模型不预期忽略项。分片、shape、dtype匹配。
2. Q/K/V各分片均完成，28层×64expert的gate/up/down全部覆盖，不只统计目标参数名出现一次。
3. 合成权重使用可辨识模式验证q/k/v拼接、gate/up顺序、expert编号与down方向，逐元素精确比较；新增非法shape/漏expert负例。
4. 真实参数对齐至少每个源张量的确定性样本及norm/router等完整小张量；必要时按块完整核对。记录抽样范围，不能将样本相等表述成全量元素一致。
5. FusedMoE post-load shuffle前检查canonical权重；shuffle后只用对应反变换或算子输出验证，不能将重排张量直接当普通矩阵乘。
6. 实际总参数/有效存储布局需说明packed/shuffle/padding差异，不能只以numel相同代替审计。HF已加载正确不代表ATOM loader正确。

尽量在新模型的参数加载回调或测试工具中记录源键/专家/shard信息；若必须在通用loader加诊断，保持默认无副作用并检查其他映射路径。

## 6. 参考与比较协议

### 进程隔离、输入固定

HF和ATOM分别用自己的解释器和模型类读取同一只读checkpoint。先生成一份JSON输入清单，记录rendered prompt、原始input_ids、attention_mask、position_ids（如显式）、模板hash、thinking模式、EOS/pad和生成设置。初始比较raw token及logits，不以文本重新tokenize作为主要依据，避免特殊token/空白处理歧义。

两端BF16权重/KV、greedy、相同停止规则；分别声明Attention后端、融合与cache配置。参考无需与ATOM采用相同内核，只需相同模型语义。默认不跳过thinking token。短预算只生成思考开头仍可比较数值，不能当HumanEval有效代码。

ATOM离线工具从Runner/Engine现有入口获取原始token与logits，不改生产API以方便测试。`/v1/completions`的协议当前prompt是str（protocol.py:103+），不能假设接受token列表。服务验证使用两端同一渲染字符串并审计服务tokenizer；内部API若不足，才增加最小诊断出口。HTTP输出内部已有token_ids组织（api_server.py:290+），需核对实际response是否保留该字段，不能仅凭内部字典断言公网API已暴露。

### 比较层级与失败定位

| 层级 | 输入对齐与输出 | 判据 |
|---|---|---|
| 配置/映射 | 双端字段、合成权重、加载coverage | 语义字段精确；合成重排逐元素精确，禁止missing/unexpected |
| RoPE/Norm | 相同head_dim/position/输入，FP32参考及BF16输出 | 频率/幅度/旋转配对正确，max abs/rel/RMS、finite；不只看余弦相似度 |
| Router/MoE | 相同hidden states、canonical专家权重 | logits、top-k集合/权重、聚合输出；相近routing分数记录边界间隔，不能只比较最终文本 |
| Attention | QK-Norm前后、RoPE后Q/K、mask、KV位置、输出 | mask精确，有限值；prefill/decode分别比较数值 |
| Decoder/整模型 | 同一teacher-forced前缀，hidden+residual的语义状态、norm、logits | 找首个差异层，比较HF实际hidden与ATOM materialized hidden+residual，不比较不等价的存储中间态 |
| 自回归 | 同一prompt与停止设置，保存所有新token | 短固定回归集要求逐token完全一致；分叉后改用HF前缀teacher forcing排查，不把不同前缀的后续logits当单算子误差 |

BF16数值误差没有先验通用阈值。本轮CPU FP32的2e-5/2e-6容差不能移用到GPU。开发中先依据同HF后端/cache差异和独立算子比较建立、记录容差，再使用；不在失败后反复放宽到“通过”。关键语义、mask、加载必须精确，非有限值直接失败。logits报告top1/top2间隔及差异位置，greedy不一致不能只凭“BF16正常”关闭问题。

诊断优先使用现有`atom/utils/debug_helper`的block hooks/compare；HF侧使用测试进程hooks，不编辑site-packages。若现有hook只捕获层边界，细分调用点仅加在新Mellum模块，默认关闭、限制层/位置/输出量。无需每次dump全部28层和整段logits。

## 7. 验证样例矩阵

| 范围 | 最小开发样例 | 目的 |
|---|---|---|
| 初始真实模型 | 已有python_add/arithmetic；随后64token | 保留最初基准，暴露更晚分叉 |
| 混合层 | 第0–3层重点，第27层/最终norm/logits | 同时覆盖3个SWA与1个Full及残差传递 |
| 窗口 | 实际token长度1023/1024/1025；1024附近prefill→decode跨越 | 边界差1、KV偏移；不能用字符长度替代token长度 |
| 分块/块边界 | 前缀在KV block_size前后；分块跨SWA窗口 | 槽位、逻辑块、绝对位置与分块可见范围 |
| RoPE上下文 | 8191/8192/8193、16384输入 | YaRN与长位置；输入精确token构造、保存，不靠字数估算 |
| 批处理 | 两种长度、先后到达、不同结束时间、重复/共享前缀 | 请求隔离、padding、已结束请求不污染KV；比较独立运行 |
| 可选运行功能 | 基础prefix/chunked关闭；之后逐项打开并组合回归 | 确认选定服务配置，非预选Step1优化 |
| 服务 | raw completions和真实chat模板、thinking开/关 | BOS/EOS、模板、文本与token输出口径 |

先短后长；16k参考使用SDPA，控制只保留必要logits/少量层输出，避免显式全量分数矩阵和无界dump。HF物理SWA cache W-1与ATOM full KV布局不同允许存在，只要求数学可见位置与输出一致。

正式200prompt及HF HumanEval参考仍缺。开发可自建明确标记的固定集，但不能冒充题目公开集合。HumanEval后续固定模板、thinking预算、答案代码提取和执行环境，在两端同一协议下运行；旧3/164与模型卡HumanEval+91.5不作为阈值参考。正式匹配率统计分母不明时同时报告token级与整序列级，EOS细则明确披露。

## 8. 首版启动与资源安排

启动脚本显式设置单卡、BF16权重/KV、level0、enforce_eager、MTP关闭、基础缓存/调度开关、context上限和诊断设置；实现前核对CLI实际参数名称，不在计划里给未经验证的完整命令。短context用于开发，长输入阶段恢复所需上限，不更改模型自身最大长度或正式负载。

HF与ATOM优先顺序运行，先导出artifact再比较，减少显存和进程干扰；如同时使用两张卡，明确一张HF一张ATOM，不算ATOM双卡推理实现。ATOM worker遵循spawn；不对GPU错误盲目重试或杀不明进程。

每轮结果记录UTC运行ID、commit/diff、脚本hash、配置/模板/输入hash、版本、GPU、flags、加载审计、逐样例输出、比较统计和失败原因。旧结果只读，不覆盖。日志中的tensor/set需可序列化，先原子保存再汇报成功，保留之前已修复的日志回归。

## 9. Step0完成核对表与下一次实施范围

- [ ] 自定义配置在冻结5.2.0通过真实配置、非法输入与roundtrip验证。
- [ ] Mellum类注册和单卡加载通过；源张量/shard/expert审计闭合，无静默漏载。
- [ ] 混合Attention、QK-Norm、YaRN、MoE语义经HF分层对比。
- [ ] 短greedy与teacher-forcing、cache/窗口/块边界通过。
- [ ] 多请求、16k、选定缓存/调度配置回归通过。
- [ ] ATOM OpenAI服务可一键启动，真实GPU执行与模板行为可复现。
- [ ] 正式正确性门槛和可获得的验收均有结果；缺失资产/口径单列，不虚报通过。
- [ ] 文档、测试、结果和完整提交历史齐全；冻结环境/原始仓库不变。

下一次实施以A+B为第一批：配置入口及其CPU验证、统一参考artifact/比较器；随后进入独立模型与首次GPU加载。GPU运行仍可能需要用户普通终端执行，但脚本、预检和CPU测试应先由工具完成。每批先说明已完成/未完成，不将“能构造”“能生成”“HF一致”“正式验收”混为一谈。

## 执行进展：第一批A+B（2026-11-11）

A配置接入已完成，12项测试与真实32字段HF对比通过。B初版artifact/比较器已完成，14项测试通过；已有两条HF参考转换完成，尚无ATOM结果。教师强制/逐层导出后续D补充。既有量化46项通过；Kimi测试mock错误在原始基线复现。详见[第一批验证](../../report/validation/04_atom_step0_batch_ab.md)。下一批为C模型接入与加载，不视为Step0已完成。

## 批次C初版（2026-11-11）

独立Mellum模型/注册、分层RoPE/SWA与加载回调覆盖审计已实现，21项配置/模型CPU测试及46项既有配置回归通过。RoPE实际源码CPU对比为0误差。尚未完成真实GPU加载、forward或HF token对比，不能标记C已全部放行。下一步普通终端执行check_atom_mellum.py，见[模型接入记录](../../report/validation/05_atom_mellum_model_wiring.md)。

## 首个GPU闭环（2026-11-11）

普通终端运行已通过真实5631项加载覆盖、warmup、两条各16token的HF greedy对比。C的首次运行闭环已具备；数值packing全量检查仍未完成。下一步D/E的teacher-forcing/分层与真实SWA边界检查，不能据短样例直接勾选Step0完成。见[验证证据](../../report/validation/05_atom_mellum_model_wiring.md)。

## D/E边界验证准备（2026-11-11）

64token/1024窗口/8192与16k三套固定开发输入和双端顺序运行入口已建立；CPU专项22项及脚本20项通过。新增可关闭的单请求prefill末位置logits诊断，尚未完成GPU套件及全部teacher-forcing/分层导出。下一步普通终端先运行short，见[套件说明](../../report/validation/06_mellum_boundary_suite.md)。

## 64token回归失败与下一执行点

2026-11-11：python_add第38、arithmetic第7个新token分叉，prefill logits已不同。HF新旧参考一致，原因未定。暂停边界/长输入，改为4条同前缀一token探针和双端分层dump；CPU配置/模型23项、运行器/诊断5项通过，GPU诊断未执行。详见[分叉诊断](../../report/validation/07_mellum_short_mismatch_diagnosis.md)。

## 第0层子算子定位

同前缀GPU结果已分析，embedding一致、第0层开始累积。另有HF cached/prefill内部差异须单独审查。下一次仍执行4个短探针，新增第0层子算子与9项实际权重dump，见[诊断08](../../report/validation/08_mellum_layer0_diagnosis.md)。不先改数学路径，长输入继续暂停。

## RMSNorm必要适配修正

2026-11-11：第0层9项实际权重与norm输入一致，首差定位到RMSNorm。新增Mellum专用FP32归一化→回输入dtype→乘weight及显式残差舍入接口，CPU26项通过。下次短64token带dump GPU回归，尚未宣称整体修复；见[证据09](../../report/validation/09_mellum_rmsnorm_rounding_fix.md)。

## 修正后短回归与后续

2026-11-11：两条各64token匹配，首层input/Q norm一致；整体logits/末层差异仍未解决，不标记数值门槛通过。下一组window用于独立边界验证，保留数值诊断问题；见[回归09](../../report/validation/09_mellum_rmsnorm_rounding_fix.md)。

## 当前执行点：扩展回归与路由隔离

short/window/long已有9条168新token一致，logits问题仍保留。下一轮11条代码、数学及1025/8192/16384边界输入各最多64新token，同时对保存的同一份router logits重放HF与AITER路由。先收集专家集合、权重精度及重复性证据，不提前改路由。见[验证11](../../report/validation/11_mellum_expanded_and_router_replay.md)。GPU待执行。

- 当前执行点更新：扩展回归8/11分叉，暂停扩大覆盖；先对h_0/h_1的原始及首差前公共前缀做逐层/子算子观察，同时区分HF cached与prefill自身差异。路由权重精度差仍是假设，不提前改计算路径。见[验证12](../../report/validation/12_expanded_mismatch_and_router_findings.md)。

- 当前执行点：四条短探针已完成；先以保存HF输入重放首层完整MoE，分离上游Attention误差与MoE自身误差，明确诊断干预不计生成验收。CPU27项通过，GPU待执行；见[验证13](../../report/validation/13_short_probes_and_moe_input_replay.md)。

- 当前执行点：同输入MoE重放确认首层内部计算路径仍有约0.45%输出差异；通过既有custom_routing_function诊断入口隔离权重舍入及HF完整路由，正常模型路径保持原样。见[验证14](../../report/validation/14_moe_same_input_and_routing_study.md)。

- 当前执行点：路由舍入/完整HF路由均只小幅改善，转为原始第0层专家权重的激活及累加舍入隔离实验；先核验HF公式重放，再解释候选变体。见[验证15](../../report/validation/15_moe_routing_results_and_expert_arithmetic.md)。

- 当前执行点：六组合实验指向HF累加路径；先在独立HF环境直接调用官方grouped_mm/eager，核验参考实际语义，补充后端元数据，避免只根据MellumExperts正文继续推断。见[验证16](../../report/validation/16_expert_reduction_and_hf_dispatch.md)。

- 当前实现决策：官方grouped_mm四例精确复现原HF后，接入Mellum专用PyTorch grouped-mm正确性后端（默认torch，原aiter显式保留）。先GPU固定输入重放，确认后恢复short/expanded正常生成；CPU31项和官方CPU六组合通过不代替GPU。见[验证17](../../report/validation/17_official_backend_confirmed_and_step0_torch_moe.md)。

- 当前执行点更新：新torch MoE四例GPU固定输入重放精确一致、5631加载审计完整；下一步无干预short64带逐层/子算子dump，之后再expanded。首层闭环不等于Step0完成；见[验证17更新](../../report/validation/17_official_backend_confirmed_and_step0_torch_moe.md)。

- 当前执行点更新：新torch后端无干预short64正常生成128/128匹配，首层误差降低但整体logits并未全面改善；下一步expanded正常回归，保留剩余Attention/逐层误差问题。见[验证17更新](../../report/validation/17_official_backend_confirmed_and_step0_torch_moe.md)。

- 当前执行点：新torch expanded仍7/11分叉；MoE同输入已对齐，下一步固定HF Q/K/V对照HF/AITER RoPE并使用同一SDPA计算，分离剩余首层Attention误差。见[验证18](../../report/validation/18_torch_expanded_and_attention_replay.md)。

- 当前执行点：HF RoPE+SDPA四例精确复现；独立AITER HIP不同不等于首层实际分支。按源码补测Triton fused_qk_rope_reshape_and_cache同QKV，再决定剩余误差归因，模型未改。见[验证18](../../report/validation/18_torch_expanded_and_attention_replay.md)。

- 当前执行点：同输入已确认prefill内核数值差异，显式sdpa_diagnostic仅替换Mellum短单请求prefill，KV写入/decode保留原实现；CPU32+3项通过，先short64带dump核验，不运行长输入。见[验证19](../../report/validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)。

- 当前执行点：SDPA短实验代码第38新token分叉，算术首层精确但layer9误差放大；新增双端stage-layer并观察第9层，暂不将SDPA实验提升默认。见[验证19更新](../../report/validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)。

- 当前执行点：第9层自然输入已不同，不判作该层bug；sdpa_rope_diagnostic只对prefill使用旋转前Q/K及分步RoPE，与既有SDPA实验比较，KV/decode原样。CPU33+3项通过，GPU待执行。见[验证19](../../report/validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)。

- 当前执行点：sdpa_rope实验未整体改善，不提升默认；读取固定输入/权重，比较torch分开/合并QKV及AITER tgemm，另查O/router，先补源码语义差异证据再改线性实现。见[验证20](../../report/validation/20_stepwise_rope_result_and_projection_replay.md)。
