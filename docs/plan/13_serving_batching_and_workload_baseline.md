# 13 服务与多请求接入：建立A/B/C可测基准

## 下一阶段目标与边界

结束本轮Attention替代实现筛选（[性能09](../../report/performance/09_triton_activation32.md)）。下一阶段首要目标是：以冻结ATOM提供可复现的一条命令启动服务，Mellum真正承载多请求，先完成小规模正确性，再建立场景A并发32及B/C的可测基准。第二个算子优化点由批量profile决定，当前不预选MoE/Graph。

本文件为详细实施计划，不代表服务/批量已经实现或GPU已经通过。计划制定轮只读源码和撰写计划；后续实现状态见文末，GPU结果仍待验证。保持单卡TP1/DP1、BF16、eager，不同时引入prefix cache、chunked prefill、Graph、量化、MTP或双卡；C虽有共享prefix，先允许不复用prefix运行，再依据实际瓶颈决定是否作为独立优化。

## 源码确认：已有能力与真实缺口

| 源码类/函数 | 事实 | 计划含义 |
| --- | --- | --- |
| `atom/entrypoints/openai_server.py` | 兼容入口转调`openai.api_server.main` | 复用既有服务器 |
| `openai/api_server.py:main` | EngineArgs CLI、加载tokenizer、create_engine，支持default_chat_template_kwargs | 新增固定Mellum配置的启动包装，不重写API |
| `api_server.py:completions/chat_completions`、`setup_streaming_request` | 已有两类端点、stream回调与core_mgr.add_request | 可复用HTTP并发提交及SSE，需实测token/usage/结束语义 |
| `api_server.py:health` | 静态返回status=ok | 健康检查不能代替实际模型生成或GPU就绪证明 |
| `model_engine/llm_engine.py:LLMEngine.generate/add_request` | 一次提交多prompt，再消费已完成输出 | 用现有offline多请求入口先验证，再加HTTP |
| `model_engine/scheduler.py:Scheduler.schedule` | 优先prefill；存在prefill时返回prefill批，否则decode；源码TODO混合batch | 初版不需要实现prefill/decode混合在同次forward，保留原调度 |
| `model_runner.py:prepare_inputs` | 生成cu_seqlens_q、attn_metadata、positions及context.batch_size；prefill按请求token拼接 | 模型应支持不同长度请求的ragged布局 |
| `model_ops/attentions/aiter_attention.py:AiterAttentionMetadataBuilder` | 准备slot_mapping/context_lens/block_tables/cu_seqlens等现有缓冲 | 复用实际分页KV，不另建旁路缓存 |
| `model_config/mellum_attention.py:sdpa_single_prefill` | cu_seqlens q/k元素数必须2，完整未缓存prefill，拒绝chunk/prefix | 本阶段明确要修改的核心限制 |
| `sdpa_cached_decode/gather_reference_kv` | q.shape[0]及context_lens元素数必须1；实际block table收集K/V | 需按每条请求的上下文、block table及绝对位置处理 |
| `models/mellum.py:MellumSparseMoeBlock.forward` | prefill grouped experts，decode batched expert分支由context.is_prefill选择 | 多token行已有表达，但批量shape的数学/内存行为未验证 |
| `mellum_experts.py:MellumTorchExperts.forward` | decode索引物化专家权重再bmm | bs32可能放大临时显存/搬运，不能假定无风险或提前宣称MoE优化收益 |

以上路径均相对`/workspace/tencent-mellum/ATOM/atom/`。共用配置当前在`scripts/mellum_runtime_config.py:CANDIDATE/engine_options`；现有`scripts/atom`只设置PYTHONPATH，不会自动启用完整Step0参考组合。冻结环境中evalscope当前未安装，正式客户端需另行固定版本与参数兼容，不能假称已经可跑。

## 基线与优化的记录方式

| 名称 | 用途 | Attention选择 |
| --- | --- | --- |
| S0 | 已验单请求Step0回归 | 原reference SDPA |
| B0 | 新服务/多请求参考接入后的开发基线 | 逐请求reference SDPA |
| B0+点1 CK | 点1在批量/服务下的增量 | 可选CK窗口prefill |
| B0+点1 Triton | 同一点的替代实现 | 可选Triton窗口prefill |

先使B0可运行再比较点1，不把S0不能执行的并发A记0、不给不存在的加速比。B0是本项目内部增量基线，不替代题目给定官方Baseline。批量接入若改了执行策略、调度、GEMM shape或并行方式，单独披露其贡献和计点边界；不能作为“免费优化”隐藏。

开发首先使用B0，避免把已有CK/Triton数值差异与新批量问题混在一起。参考bf16输出不要求逐位一致作为新的验收门槛；相同小样本优先观察单请求是否不退化，多请求按token位置/完整序列均报告，正式门槛仍是题目规定。

## 阶段1：固定配置的bs1服务（不修改模型数学）

交付一个Mellum专用启动入口及配置清单：固定工作副本PYTHONPATH、冻结解释器/模型、Step0后端、TP/DP1、BF16 KV、eager、正常greedy；显式清除继承的forcing/replay/dump/profile开关，记录所有实际选项。外部HTTP端口与引擎内部通信端口分开，关闭默认重请求日志和张量dump；独立开关选择reference/CK/Triton，三者共享其它配置。

首轮max_num_seqs=1验证服务与offline一致性，不能以此声称支持并发32。分别检查raw-token completion（若现有协议支持，避免重编码）与chat template路径、enable_thinking、正常EOS及性能请求ignore_eos=true。复用已有短输入及1K/长输入中少量固定样本，不重做200/HumanEval。

验证：真实一次成功生成、后端/加载审计、进程正常关闭，SSE与非stream输出在相同输入及参数下对齐；记录role/空chunk、reasoning_content、最终usage/DONE，防止Thinking token被客户端忽略或重复计数。`/health`仅用于连接，不能当模型就绪证据。断开连接/超时先检查现有清理行为，不预承诺引擎支持abort，发现实际问题才做必要修复。

## 阶段2：最小ragged多请求参考桥接

### Prefill

按cu_seqlens切分各请求Q/K/V与绝对positions；逐请求执行同一reference RoPE/SDPA语义，拼回原始token行顺序。21层SWA窗口与7层Full保持，阈值1024及mask规则不变。沿用原slot_mapping写入真实分页cache；不能让一个请求的key参与另一个请求的attention。

以新独立Mellum批量模块/显式开关接入，保留S0单请求函数可回退；一个请求时可直接走原路径，避免无意改变bs1输出。原参考prefill涉及raw Q/K重建RoPE，需确认是整批还是请求切片的一致索引；不能只删掉numel!=2检查。

### Decode

按真实请求行，读取对应context_lens、block_tables和positions；各自取Full完整历史或SWA最后window，再调用原cached SDPA并拼回Q行顺序。不同长度、已结束请求和调度重排行都必须正确映射，不能按永久固定slot索引猜请求ID。KV仍为现有block管理器拥有的缓存，禁止另存每请求完整历史替代真实分页实现。

初版允许attention内逐请求循环，模型投影/MoE仍接收真实批量行；这是一条正确性桥接路径，不是高效batch attention内核。客户端必须真并发提交、调度器应实际形成batch>1，不能逐条LLM.generate伪造并发。prefix/chunked/MTP不支持仍显式拒绝，保留原Scheduler，不新增调度策略。

### 投影与MoE

批量改变linear/router/grouped_mm/bmm输入shape，即使attention逐请求也可能出现新数值差异。先报告，必要时用少量首差前同历史探针定位，不能把整个模型改成逐请求串行后仍宣称已实现批量吞吐优化。decode专家权重物化随batch增加，先测bs2/4/8及allocator峰值，必要时调整安全运行预算或披露处理策略；不同时悄悄替换MoE后端。

元数据若可从现有CPU缓冲获得，集中准备请求边界/长度，避免每层逐元素GPU→Python读取；这属于接入实现选择，若带来性能变化记入B0，不单独包装成已完成优化点。GPU/cache布局依据源码与测试，不凭推测索引。

## 阶段3：小规模正确性与容量阶梯

| 检查 | 最小覆盖 | 解决的问题 |
| --- | --- | --- |
| bs1回归 | 短、1024边界、1条长输入 | 新开关与服务不退化 |
| ragged prefill | 两条不同长度，同批；含1023/1025 | 请求边界、SWA阈值、输出重排 |
| ragged decode | 两条不同上下文，跨16-token KV块 | block table与绝对位置、窗口裁剪 |
| 生命周期 | 不同max_tokens导致错峰结束；运行中加入短请求；完成后复用块 | 删除/重排/新请求不会串缓存 |
| 容量阶梯 | bs2→4→8→16→32 | 实际调度batch、临时显存、无错误/超时 |

先用每请求相同输入的HF bs1正常greedy64作为对照，区分prefill首token与后续decode。记录所有请求token位置与序列匹配、输入/输出长度及结束原因。小集用于发现串请求等逻辑错误，不把其≥90%冒称公开集验收；不新加全logits逐位精确硬门槛。

CPU测试重点为ragged切分、重排、分页读取/写入、请求释放与循环回退；GPU先少量bs1/2，再扩大容量。每阶段失败先处理阻塞，不盲目跑32。BS32成功不等于优于baseline；显存峰值分模型、KV池与临时算子，内存预算根据实际观测记录。

第一轮实施交付到此：专用启动配置、reference多请求桥接、offline/HTTP小集GPU验证与容量结果。完成后才让用户运行完整正式负载；当前不提供尚未实现的启动命令。

## 阶段4：建立实际负载基准与批量profile

性能客户端放独立环境，不升级冻结推理依赖。核对题目evalscope perf脚本需要的版本与选项，固定版本、请求文件hash、实际过滤/取样数量、chat/completion模板及ignore_eos/temperature/stream；若不能复现题目工具，开发测量必须另标，不悄悄替代正式成绩。

1. 开发小负载：先8/32条请求验证并发链路、token预算、实际batch、无请求丢失与超时。客户端并发与GPU batch是两种指标，二者都记录。
2. B：单卡并发1、1024输入、256输出、50请求；先B0，再选择点1实现，同一协议、关闭profiler。报告TTFT/TPOT p50/p99及全部样本；50条经验p99不夸大稳定性。
3. A：给定512请求、512–4K、输出256、并发32、ignore_eos。报告输出tokens/s/GPU、实际成功数、总token数、wall、错误及峰值batch/显存；不得减少number/并发却仍称正式A。
4. C：给定200请求、4K–16K、共享prefix、Poisson rate0.30 req/s、输出256。完整固定负载的到达跨度量级可能超过十分钟，需先澄清原题total-timeout=180的工具语义，不擅改rate/timeout或截去未完成请求。报告排队与服务影响、TTFT/TPOT尾延迟和失败，不把无prefix复用包装成已优化prefix。
5. 正式计时关闭profiler。只从A稳态、C长prefill/等待窗口抽短trace，记录实际batch与请求长度；不按bs1间隙直接认定CPU瓶颈。B0与点1候选交错/多轮运行，保留原始样本，区分时段漂移。

A/C先完成B0有效基准，再测试点1增量；不要求为进入批量研究先跑全HumanEval。最终方案确定后补最终准确性和正式消融，包括必要16K及HumanEval协议。

## 阶段5：决定第二个算子优化点

只在批量数据后选：若MoE权重gather/执行占主导，优先专家路径；若分页KV整理/窗口decode占主导，优先paged decode；若CPU launch/同步支配且能消除动态操作，再评估Graph。最多三个优化点，不为凑数引入新点。服务/批量接入涉及何种策略和计点，在结果报告中显式说明。

## 交付物、记录与节奏

- 代码：Mellum启动包装、多请求模块、CPU小集测试、offline/HTTP验证入口；尽量不改既有API/Scheduler核心，只有实际阻塞才小范围修复。
- 实验：S0/B0/B0+点1实际配置、源码与输入hash、每请求输出、阶段日志、batch与内存统计、失败记录；原始数据不覆盖。
- 文档：本计划13维护方案，实际GPU结果才新增性能10及后续；worklog继续按自然日记录。维持现有有序编号。
- 当前不报“几小时即可全完成”的承诺：服务bs1接入、模型批量桥接、实际负载三个可审查里程碑逐个交付。每次用户运行命令前，先完成脚本、CPU检查及prepare-only，再说明GPU预算/超时与产物位置。

初次计划时只完成源码核查；以下更新记录第一轮代码实施状态。evalscope仍未安装，没有新的GPU服务/批量成绩。

## 第一轮实现交付：待GPU小集核验

已新增`atom/model_config/mellum_batch.py`，由`MellumAttention.__init__`在`MELLUM_REQUEST_BATCHING=1`时安装。单请求直接调用原函数；多请求prefill按cu边界切分raw Q/K与各自positions，沿用原CPU RoPE/窗口SDPA后拼回。multi-decode按context_lens/block_tables/positions逐行调用原cached SDPA。实际K/V仍由原rope_cache与slot_mapping管理；无旁路历史cache，无Scheduler/API改动。prefix/chunk、graph padding/MTP等不兼容元数据显式拒绝；batch与forcing/replay/旧逐样本logits探针不混用。

`apply_serving`复用CANDIDATE参考数学，清除继承的MELLUM诊断环境并关闭profile标签/control；`serve_mellum.py`提供固定CLI、source/config记录、独立HTTP/内部端口、reference/CK/Triton选择。默认reference、max_num_seqs=1，multi需显式`--batching`；优化默认仍关闭。没有改原S0默认配置。

### 第一步：offline ragged smoke

```bash
/opt/venv/bin/python /workspace/scripts/validate_mellum_batch.py
```

一次加载max_num_seqs=4，maxlen1152、prefill token预算4096、TP/DP1/eager/BF16、util.35、prefix/chunk关闭。复用现有HF短输入两例及window1023/1025，输出预算16/32/8/8；先4次bs1，再pair、ragged4、反序/块复用及短请求结束后加入新请求。保留实际token与HF前缀比较、结束原因、提交ID映射、pending状态及layer0 batch审计。审计只验证实际cardinality，不把异步末尾额外forward计成额外生成token，也不当性能trace。`batches.jsonl`有CPU/GPU读取及落盘开销，只用于验证，正式性能默认关闭。

输出`results/mellum_batch_validation/<id>/results.json`及阶段audit/log。强制检查至少出现真实multi-request decode，避免只测排队序列化。`batch_smoke_completed`表示执行完成，accuracy_review_required如实标记token差异，不自动宣称题目通过。预计1–3分钟，15秒心跳、1200秒总阶段上限；启动前要求至少80GiB可用。

如基础bs4正常，再按顺序用`--max-seqs 8/16/32`扩大；这些是小输入容量检查，不替代正式A。当前不先让用户跑32，避免用尚未核验的GPU路径扩大成本。

### 第二步：HTTP bs1 smoke

```bash
/opt/venv/bin/python /workspace/scripts/validate_mellum_http.py
```

脚本拥有并自动关闭server进程组，默认127.0.0.1:8051/内部8026。端口已占用则拒绝，不复用未知服务；请与offline验证顺序运行。启动后先health连通，再真实生成验证，不把health等同模型已通过。

当前CompletionRequest.prompt是str，故先用本地tokenizer验证rendered_prompt重新encode与HF输入相同；HTTP比对skip_special_tokens=True的参考输出文本、完成token usage、SSE DONE及并发两个HTTP请求。另检查chat普通/stream内容与reasoning_content一致性。此处是文本/协议对照，精确token级检查在offline完成，不声称通过重编码HTTP输出证明全部token相同。

输出`results/mellum_http_validation/<id>/summary.json`、server.log与server/launch.json。`http_smoke_completed`为执行完整，review_required标记文本/usage/stream差异；异常保留日志并清理owned进程。预计1–3分钟，启动等待300秒、请求超时180秒、退出先SIGINT再有限升级清理。HTTP bs1完成后可用`--max-seqs 4`检查桥接服务并发；当前尚未执行GPU，也未验证断连/abort或所有长输入服务边界，不能承诺阶段1–3全部验收通过。

独立长驻服务命令（小集通过后使用）：

```bash
/opt/venv/bin/python /workspace/scripts/serve_mellum.py --batching --max-num-seqs 4
```

### 本地完成的检查

相关59项CPU通过（ragged隔离、RoPE raw切片与无self状态修改、两种cache格式/窗口、跨块、重排、原模型与配置、SSE解析及profile回归），仅既有Torch profiler提示。compile/diff检查通过。prepare-only：offline212248、server212249/212719、HTTP212419/212720。fastapi/uvicorn/zmq/httpx已存在，无新依赖安装；工具中cuda_available=False，未启动GPU或服务。当前先交付上述两条短验证命令，取得真实结果后再处理必要问题和扩大容量。

## 首轮GPU结果与容量阶梯

214224 offline实际prefill/decode batch4、迟到decode3完成；bs1四例64/64位置精确，多请求234/240=97.5%（少量批量/排列差异待观察，不新增bitwise门槛）。214721 HTTP bs1 completion/SSE/usage及双客户端通过；chat仅reasoning前导换行差异，另发现非流完成请求前端记账未释放，已覆盖三种路径修复并加CPU回归。见[性能10](../../report/performance/10_batch_and_http_smoke.md)，32项相关CPU通过。下一步`validate_mellum_batch.py --max-seqs 32`一次加载依次8/16/32，再`validate_mellum_http.py --max-seqs 4`复核服务桥接和记账；修复后尚无GPU结果，不宣称正式A通过。
