# 05 优化实现源码对照：冻结ATOM、上游ATOM、vLLM与SGLang

## 结论与范围

下一步应优先恢复ATOM已有的**整批FusedMoE和分页Attention执行能力**，保留当前参考路径作回退/消融，而非继续围绕参考实现减少几个小操作。这里的“恢复”是针对Mellum接入和验证，不是直接打开开关就宣称已正确或已提速。

最强的外部依据是题目指定的[vLLM v0.23.0 Mellum实现](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/model_executor/models/mellum.py)：`MellumAttention`扩展Qwen3 MoE的Attention，`MellumDecoderLayer`按层选择窗口和RoPE，稀疏MLP直接使用`Qwen3MoeSparseMoeBlock`；模型和权重加载也复用Qwen3体系。这支持以相似模型的成熟执行路径为基础，而不是从零写专家内核。

本轮只读源码/公开资料，写报告和来源索引，不修改推理代码、不安装引擎、不运行GPU。[来源清单](../../artifacts/research/05_optimization_sources/manifest.json)保存16份本地文件SHA和符号行号、外部链接与获取限制。冻结基线`26f23a0b...`，本地开发版本包含Step0/R及已归档实验。

## 1. 必须保持的Mellum语义

直接依据本地`/data/models/Mellum2.1-12B-A2.5B-Thinking/config.json`：

| 项 | 实际值 | 对复用的约束 |
| --- | --- | --- |
| hidden / head_dim / Q heads / KV heads | 2304 / 128 / 32 / 4 | Q输出4096，不能用2304/32推导head_dim=72 |
| 专家 / top-k / expert intermediate | 64 / 8 / 896 | 权重、路由和内核shape必须按此核验 |
| 层数和类型 | 28，S/S/S/F重复 | 21个窗口层、7个全局层，不能统一窗口或统一RoPE |
| 激活/路由 | SiLU，softmax top-k，norm_topk_prob=true | 无共享专家/分组top-k；路由权重施加位置必须一致 |
| 窗口 | 1024 | 语义为包含当前位置的1024个位置，核对内核left-window约定 |
| RoPE | sliding默认；full YaRN factor16，原长度8192，attention_factor=1.2772588722239782 | 不直接复制其他模型的默认theta/scale或query缩放 |
| 精度 | BF16 | 首轮不引入FP8/FP4验收及量化误差变量 |

`ATOM/atom/models/mellum.py:MellumDecoderLayer`已有分层设置；本轮不重做Step0模型结构。

## 2. 冻结ATOM中最值得复用的实现

### 2.1 Qwen3 MoE：首选专家执行模板

`ATOM/atom/models/qwen3_moe.py:85–140`，`Qwen3MoeSparseMoeBlock.__init__/forward`：

- 以config传入num_experts、top_k、hidden_size、moe_intermediate_size、renormalize，关闭grouped_topk/bias。
- gate输出router_logits后，整批hidden_states交给`FusedMoE`；没有逐请求Python专家循环。
- TP归并仅在需要时进行；本轮TP1无需移植多卡通信。

`ATOM/atom/model_ops/moe.py:507–616`，`UnquantizedFusedMoEMethod`：

- `create_weights`创建[E,2I,H]、[E,H,I]，与Mellum gate/up/down逻辑布局吻合。
- `process_weights_after_loading`做shuffle；`apply`调用`FusedMoE.select_experts`，再进入modular kernel或AITER fused_moe。
- `FusedMoE.__init__`约1960行提供`custom_routing_function`、`scoring_func`、`activation`等参数；可先保留已验Torch路由语义，同时替换专家计算，不必把所有差异同时打开。

`ATOM/atom/model_ops/utils.py:124`的`shuffle_weights`与`ATOM/atom/model_loader/loader.py:692–699`构成加载后处理链。**不能把canonical Torch专家权重直接视为shuffled权重，也不能重复shuffle。** `FusedMoE.make_expert_params_mapping`约3106行将gate/down/up映射到w13/w2，Mellum已有同类映射；应沿用加载链和5631源张量审计，再做输出探针。

`/app/aiter-test/aiter/fused_moe.py:fused_moe_1stage_dict`约667行确实有gfx942、BF16、SiLU、QuantType.No入口，`fused_moe_1stage`约486行调用aiter.fmoe。这是**当前依赖具备相关路径的源码证据**，不是证明Mellum所有token数/shape均能选到最快内核或已通过生成验收。

补充模板`ATOM/atom/models/mixtral.py:MixtralMoE`也复用FusedMoE，但Qwen3的GQA/QK norm结构更接近Mellum；DeepSeek MLA和共享专家配置不应成为首选移植模板。

### 2.2 分页Attention：另一条需要实测的主路径

`ATOM/atom/model_ops/attention_mha.py:PagedAttentionImpl`：

| 函数 | 已有能力 | 与当前参考桥的区别 |
| --- | --- | --- |
| `rope_cache`，约125行 | 按窗口/head维度选路，写现有分页KV | 标准执行流程与缓存布局保持联动 |
| `paged_attention_triton`，394行 | context_lens、block_tables、cu_seqlens驱动多请求decode | 直接访问分页KV，避免每请求先收集完整KV |
| `prefill_attention`，555行 | flash_attn_varlen_func，按cu_seqlens执行ragged prefill | 避免逐请求SDPA和显式窗口mask |
| `prefill_attention_triton`，581行 | unified attention、flash布局等路径 | 必须匹配实际metadata和KV布局 |
| `dispatch_backend`，644行 | prefill/decode、Triton/ASM按条件选择 | 不能对所有层盲目强制同一backend |

`ATOM/atom/models/gpt_oss.py:OAIAttention`约125行按层传入sliding_window，是冻结版混合窗口接入的参考；GPT-OSS的attention sinks、bias和专家激活不等于Mellum，不照搬其整体模块。

## 3. 当前Mellum为何没有充分使用这些能力

| 当前路径及证据 | 已知代价/限制 | 具体改造落点 |
| --- | --- | --- |
| `mellum_experts.py:MellumTorchExperts.forward_routed` | prefill sort/histc+grouped_mm；decode先索引复制选中专家权重再bmm | 用FusedMoE执行整批专家计算 |
| `mellum_campaign.py:Campaign`与`mellum_moe_optimization.py:optimized_experts` | R仍逐请求专家GEMM，L仍逐请求线性；这是主动数值控制 | Fused专家不再受E拆分；L保留/撤销分别消融 |
| `mellum_batch.py:ragged_prefill` | 循环每个请求调用参考SDPA | 接入原生varlen Attention和批量metadata |
| `mellum_batch.py:ragged_decode` | Python逐请求循环，每次调用sdpa_cached_decode | 原生paged decode整批执行 |
| `mellum_attention.py:gather_reference_kv/sdpa_cached_decode` | 根据block_table gather KV；窗口条件下将4个KV头repeat至32个Q头 | 由原生GQA分页内核直接处理 |
| `mellum_attention.py:sdpa_single_prefill` | 窗口阈值后构造n×n mask并扩展KV头 | 复用已实现SWA CK/原生窗口Attention |
| `mellum_projection.py:separate_qkv_forward` | packed权重拆成三次F.linear再cat | packed QKV可作为后续独立消融 |
| `scripts/mellum_runtime_config.py:CANDIDATE` | eager、融合关闭、prefix/chunk关闭 | 先恢复算子主路径，再按证据启用Graph/缓存 |

这些是源码差异，不等同于已量化的端到端占比。性能17的A仅官方6.51%、C199超时，加上性能14专家profile，支持优先替换主执行路径；不能据此断言单独换MoE一定追回15倍差距。

**已有开关≠直接可测组合：** `mellum.py:MellumDecoderLayer`292行已有`MELLUM_MOE_BACKEND=aiter`，会选择Qwen3MoeSparseMoeBlock。但R的campaign安装器假定Torch专家具有`route/forward_routed/batched`接口；直接套用R配置会不兼容。首个Fused候选需取消E控制/旧MoE wrapper，必要时仅保留L；若需要Torch gate，应在Mellum模块明确接入，不假定Qwen3模块自动读取Mellum router开关。

同理，`MELLUM_REQUEST_BATCHING=1`在`mellum.py`约209行强制要求参考CPU RoPE/SDPA decode；它表示“参考ragged桥”，并非引擎原生batch总开关。恢复原生Attention时需另设清晰配置和batch审计，不能只关闭该变量后就宣称批量已验证。

## 4. 上游新版ATOM：有用，但目前不能宣称获得完整最新快照

已读取的上游[registry.py](https://github.com/ROCm/ATOM/blob/main/atom/models/registry.py)未列Mellum，常规`atom/models/mellum.py`返回404。PR页面抓取不可用、搜索未找到确认的Mellum PR；因此结论限于“未找到可确认的现成实现”，**不扩张为所有分支/PR均不存在**。

上游[Qwen3MoE](https://github.com/ROCm/ATOM/blob/main/atom/models/qwen3_moe.py)仍采用FusedMoE；增加了MoRI输出归并判断，主要涉及分布式，本轮单卡优先级低。新版[moe.py](https://github.com/ROCm/ATOM/blob/main/atom/model_ops/moe.py)已把路由/放置/专家执行边界展开，并引入moe_shuffle_weight、GateMode等接口；冻结版仍用shuffle_weights/shuffle_weight。**不可将新版整文件覆盖冻结版**。

新版[attention_mha.py](https://github.com/ROCm/ATOM/blob/main/atom/model_ops/attention_mha.py)有更多硬件专用选路；源码中的某些ASM分支限定gfx1250，不能当作MI308X/gfx942方案。主线原生分页/varlen能力在冻结版已存在，优先复用冻结版，避免额外依赖迁移。

限制：shell的git ls-remote因本地代理127.0.0.1:11080不可达失败，未获得远端HEAD SHA或完整checkout；网页main来源缓存时间不同，不能宣称它们构成同commit的最新源码快照。本轮报告链接和获取状态已保存。后续确需回移某个新补丁时，应先取得其具体commit/diff和依赖，再移植最小范围；不为等待全面抓取阻塞现有能力接入。

## 5. vLLM和SGLang给出的直接参考

### vLLM：优先使用题目指定的版本源码

[v0.23.0 Mellum](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/model_executor/models/mellum.py)：`MellumAttention`26行、`MellumDecoderLayer`109行、`MellumModel`167行、`MellumForCausalLM`185行。保留显式head_dim，依据layer_types选窗口和rope_parameters，稀疏层复用Qwen3MoE。

其[v0.23.0 Qwen3MoE](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/model_executor/models/qwen3_moe.py)的`Qwen3MoeSparseMoeBlock`130行、FusedMoE构造约198行，传入专家规模/renormalize，router可在FusedMoE内部执行。可借鉴“模型语义薄封装＋共享高性能算子”的结构，不能直接复制新版gate返回值/内部router协议到ATOM。

这一引用确认的是模型实现的版本位置，不是我们已经在本机运行vLLM，更不是所有CUDA kernel可直接在ROCm运行。最新main的文档/注册表检索存在变动，报告不以搜索摘要替代v0.23.0固定源码。

### SGLang：窗口与分层YaRN处理有直接价值

[models/mellum.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/mellum.py)：

- `get_attention_sliding_window_size`115行返回W-1，说明其后端参数是“左侧历史长度”，不能机械把ATOM或CK的W原样替换。
- `MellumDecoderLayer`294行验证layer_types，按类型取得rope_parameters，稀疏层复用Qwen3MoeSparseMoeBlock。
- `MellumAttention`127行使用RadixAttention；`_compute_yarn_from_rope_params`70行、`apply_qk_norm_rope`242行维护分层YaRN和可选融合，融合有硬件/类型/配置条件。

当前模型131072/8192=16，与本地factor吻合；不是由此推断任意配置都能重算覆盖原factor。SGLang条件分支和runtime接口与ATOM不同，取语义与融合参数设计，不整文件搬运，不安装另一套引擎替代交付。

## 6. 建议优先级与实验边界

| 优先级 | 方案 | 当前依据 | 首轮验证 |
| --- | --- | --- | --- |
| P0 | 整批BF16 FusedMoE，移除E逐请求拆分 | 冻结Qwen3/AITER gfx942已有路径；vLLM同结构复用 | 加载/路由审计、正常生成200、batch1/32的真实计时 |
| P0并行 | 原生paged decode＋varlen/SWA prefill | 冻结ATOM已有；参考桥存在gather/repeat/循环 | 不同长度/结束时刻batch、窗口边界/16k、真实batch与KV审计 |
| P1 | packed QKV和QK norm/RoPE/cache融合 | 冻结Qwen3/SGLang给出接口依据 | 按层YaRN/窗口保真；与前两项分别开关 |
| P1条件 | prefix caching / chunked prefill | C共享prefix且当前关闭；参考桥明确不支持 | 原生Attention稳定后，缓存命中与chunk边界验证；不能先盲开 |
| P2条件 | compile/CUDA Graph | 框架机制存在，但当前tolist/item/循环/动态控制不利 | 动态路径移除后才做capture和padding验证 |
| 暂缓 | 双卡推理、MTP、低比特、上游Mono整套回移 | 增加计点、依赖与正确性成本 | 时间紧张，不抢占首轮主路径接入 |

不是要求一次打开全部选项。建议两个主候选独立实现、并行筛选，再只测试一个组合：R / +FusedMoE / +原生Attention / 两者组合，保留开关作消融；SWA CK归入Attention优化点。最终按题目最多三个点归纳，不把每个小开关拆作一个点。

筛选接受合理数值差异：先确认实际执行且有明显速度收益，再以正常greedy200和完整HumanEval判断；不用中间张量bitwise或自设相对R下降2pp否决。200条同时报告位置与完整序列，不隐瞒题目口径未确认；BF16 HumanEval按同协议完整生成/执行、差距2个百分点。16k和最终组合单列。

首轮性能样本必须含256输出、batch1/32、长输入，避免16token测量偏向prefill；profiling与无profile计时分开。候选未显著改善前不再跑完整86分钟ABC。通过后才按题目ABC复测，失败/超时不剔除，最终正式性能采用空闲环境。

## 7. 尚未解决的问题

1. FusedMoE实际shape(64,2304,896,top8)在gfx942的选路/速度和完整生成指标尚未测；旧层级重放误差不代替全模型结论。
2. 原生Attention同参考RoPE/cache写入组合的完整batch、窗口、prefix/chunk支持需实现和验证，不能由函数存在推断通过。
3. 哪项贡献最大需要新主路径短实验，而非旧参考路径profile机械外推；尤其不能保证单点追回15倍差距。
4. 尚无完整上游HEAD/所有PR核验；新实现若依赖更高AITER版本，仅能最小回移或放弃，冻结环境不升级。
5. 完整HumanEval164、题目公开集身份/计数口径和最终组合验收仍需完成。现有4题pilot和前64token不能代替功能pass@1。

下一轮建议阅读/修改入口：`mellum.py:MellumSparseMoeBlock/MellumDecoderLayer`、`moe.py:UnquantizedFusedMoEMethod/FusedMoE`、`loader.py:process_weights_after_loading`调用处、`attention_mha.py:PagedAttentionImpl`、`mellum_batch.py:install_ragged_attention`。只在Mellum模块/安装器扩展，遵守现有compile装饰模型文件不改的约束。
