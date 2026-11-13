# 06 Step1优化路线：候选、证据门槛与实施顺序

实测更新：首轮bs1 trace已完成，见[性能报告01](../../report/performance/01_bs1_profile_baseline.md)。16k prefill的Attention路径占该阶段kernel时间78.6%，decode占51–66%；当前将Attention排在MoE前实施，但A并发32尚未测，最终排序仍需批量数据。下文保留候选规划与条件。

## 状态、目标与证据等级

本文是Step0收尾后的优化规划，不是profiling或性能结果。用户要求结合原题、早期方案及近期实现讨论可选路线；本轮只读源码并撰写方案，未启用新优化、未启动GPU测量。已有基线见[05](05_step0_baseline_and_reproduction.md)，正确性成果见[39](../../report/validation/39_step0_summary.md)，题目口径见[要求摘要](../requirements/01_assessment_summary.md)。

建议主方向：**单卡BF16，MoE与Attention为优先评估的两个候选，第三点依据优化后profiling在执行开销或场景C的调度/前缀复用中选择。** 这是有源码依据的优先假设，不是已证明的瓶颈排序；允许最终只做1–2点。Step0虽有200条与16k开发验证，HumanEval正式补证仍待完成，但不继续以全logits精确作为进入性能研究的额外条件。

全文区分：事实=当前源码/原始结果直接支持；推断=可能的开销及收益方向，需测量；设计=拟实施边界与开关，不是现有功能。

## 1. 优化围绕什么指标

| 场景 | 原题单卡负载 | 关注指标 | 对策略的约束 |
| --- | --- | --- | --- |
| A | 512请求，输入512–4k，输出256，ignore_eos，并发32 | 总输出tokens/s/GPU，性能30分 | 不能只证明bs1 TPOT下降；要验证真实动态批量和GPU利用率 |
| B | 50请求，并发1，输入1024，输出256 | TTFT/TPOT p50/p99，须报告，不直接计分 | 最适合诊断单步decode；不能只为B选择高成本低计分收益方案 |
| C | 200请求，4k–16k、共享prefix，Poisson rate=0.30请求/s，输出256 | TTFT p99、TPOT p99，性能10分 | 需要区分计算延迟和队列/长prefill阻塞；不能仅看平均吞吐 |

实际复测使用题目evalscope命令，不修改parallel/rate/长度/输出预算来提分。双卡N=2时A并发/请求量和C到达率/请求量随N扩大，A吞吐再除N；C延迟不除N。最终卡数是统一申报配置，不能按场景挑选最有利卡数冒充同一方案。

`/data/datasets/baseline.json`提供单卡eager/BF16 KV/MTP off/SWA on的出题方记录：A约638.282724 tokens/s/GPU；C TTFT p99约993.549ms、TPOT p99约41.265ms。其target_g_a/target_g_c为空，不自行补造目标或评分公式。A的正式相对值为`实测tokens/s/GPU / 638.282724`；C分别报告两种尾延迟与参考比值。B须有本配置实测，不拿A/C的p50替代B。

自己的Step0基线用于增量归因，正式加速比相对出题方Baseline。我们的reference实现更慢，优化回原生算子可能带来很大内部提升，但这不保证超过出题方数值，必须同时报告两组比较。

## 2. 早期方案哪些保留，哪些更新

早期[原题分析16.5–16.7](../../report/archive/initial_research/02_atom_research_original.md)提出MoE、Attention及profiling决定第三点，仍有合理性；其中“纯expert-loop Step0”已不符合现状。当前实际是grouped_mm prefill/batched_mm decode、CPU逆频率、参考SDPA和真实分页KV。不能以旧假设预测新基线耗时。

Step0提供的新约束：

- 正确性参考不只取决于公式，HF generate还会改变专家后端；开关、负载shape和实际分派必须记录。
- packed QKV、AITER router、fused专家、RoPE/prefill内核都曾有数值差异；这些不自动禁止优化，但过去11条扩展集的大量分叉表明直接恢复旧组合风险较高。
- 部分旧失败是RMSNorm、分派和采样规则问题；不能把旧8/11失败直接归因于某一个内核，也不能因后来精确就推断其所有差异已可忽略。
- 全logits逐值精确是诊断工具。候选以题目greedy/任务指标判断，局部误差用于解释；不重新陷入追求每个BF16向量bitwise一致的循环。
- 当前正确性后端有单请求硬限制，必须先形成可运行真实负载的可比较服务基础，而非直接跑并发32。

## 3. 当前源码揭示的机会与前置工作

### 3.1 MoE：显式选中权重复制

事实：`ATOM/atom/model_config/mellum_experts.py:MellumTorchExperts.forward(batched=True)`执行`w13_weight[expert_ids]`、`w2_weight[expert_ids]`，再两次torch.bmm。28层全部sparse，64专家/top8、H2304/I896，canonical w13=[64,1792,2304]、w2=[64,2304,896]。

按这些shape，单token单层索引生成的BF16选中权重元素合计约94.5MiB；28层合计约2.58GiB。若32token同批，显式临时权重元素量线性扩大至单层约2.95GiB，尚未计GEMM读原权重、路由及其他中间量。**这是按shape计算的逻辑materialization量，不是已测HBM流量、峰值常驻显存或延迟。** 同一专家可能被多个token重复选择，融合/分组有复用机会。

推断：MoE可能是A与decode的高收益点，但要测index/gather/bmm/sort/activation/sum的GPU时间与kernel间隙，不能只根据张量体积宣称首要瓶颈。

### 3.2 Attention：参考式整理和冗余处理

事实：`mellum_attention.py:sdpa_single_prefill`在窗口阈值后创建n×n bool mask并展开KV heads；16k单个bool mask约256MiB，不代表SDPA必然显式计算完整score矩阵。`sdpa_cached_decode/gather_reference_kv`每步按block table收集连续K/V，再调用SDPA；21层滑窗只需要窗口内key，7层Full需要全部合法历史。

`install_reference_kv_rope`先计算参考Q/K，又调用原rope_cache并覆盖K；prefill参考路径还可从保存raw QK重算旋转。此控制式实现刻意保留旧缓存行为，存在重复计算、clone和写回，需profile确认其份额。不能把所有时间统称SDPA kernel。

推断：直接读取分页KV、真实滑窗kernel及一次旋转/写cache可减少整理开销，改善长输入TTFT和decode；也是支持批量、prefix/chunked和Graph的基础之一。

### 3.3 CPU dispatch与Graph阻碍

事实：`sdpa_single_prefill`读取`.tolist()`；`sdpa_cached_decode`读context_lens/positions的`.item()`、构造arange、动态gather。GPU tensor到Python值会要求同步，实际次数和耗时需trace。Torch专家还包含sort/histc/cumsum、动态索引/分配。`ModelRunner.run_model/capture_model`已有eager与graph框架，但模型目前未按编译模型接入，单纯取消enforce_eager不等于可捕获。

推断：bs1下CPU dispatch/launch和同步可能显著；bs32下也可能被MoE/Attention GPU耗时掩盖。因此Graph不直接预定为第三点，先测前两点之后的占比。

### 3.4 服务与批量是测量前置，不能偷偷计为内核收益

当前single-prefill检查cu_seqlens仅一请求；cached decode要求q行数/context_lens元素数均1。`scripts/check_atom_mellum.py`固定max_num_seqs=1。普通`/workspace/scripts/atom`并不自动设置正确性组合。

拟先实现统一候选启动配置、关闭诊断dump的正常服务，并用每请求相同reference数学支持ragged多请求。第一版可按请求切片、逐请求计算SDPA再拼回，复用同一分页cache和绝对positions；不为此改Scheduler策略、不启用prefix/chunked/graph。逐请求循环是最简单适配桥接，不是批量高效Attention；后续paged内核与之形成可比对照。

保留S0（已验单请求）与B0（服务/多请求接入后reference）两份记录，先验证B0单请求不退化及双请求/错峰结束/不同长度正确，再扩大到实际并发。A的基线必须用能承载原题负载的B0，不把S0不能运行的A填0，不用串行客户端冒充并发32。批量桥接也可能改变GEMM shape和输出，不能视为零风险改动。若桥接本身引入性能策略或并行化，需独立披露计点边界，不能“免费”藏进B0。

## 4. 候选M：MoE执行优化（优先候选1）

**目标瓶颈**：选中专家权重materialization、重复读、分散kernel和低效专家小GEMM。A吞吐为主，B/C TPOT为辅。

### M1：接入现有AITER BF16 FusedMoE

入口`atom/model_ops/moe.py:FusedMoE`及unquantized method、`process_weights_after_loading/weight_loader`；Mellum目前有`MELLUM_MOE_BACKEND=aiter`入口，但它实例化`qwen3_moe.py:Qwen3MoeSparseMoeBlock`，**同时使用该类的ReplicatedLinear gate，不能假定当前Torch router仍生效**。

设计：新增或调整Mellum局部wrapper，使gate、topk与experts选择解耦；先保留已验证Torch gate和规范路由，只替换专家执行。canonical与shuffled布局分开加载、各自审计；不能拿已shuffle权重给普通Torch，也不为热切换同时常驻两套全模型。

如现有FusedMoE支持custom routing，核查参数与BF16权重舍入语义后使用；不能通过修改待选专家、logit margin或采样规则提升匹配率。对照单层固定输入、不同token数/专家分布，再做普通生成。已知fused激活/专家贡献累加顺序可能与HF不同，是主要风险。

优点：复用被冻结且已有gfx942运行证据的库，实施成本相对低；prefill/decode/batch均可能受益。限制：曾测AITER完整MoE输出与HF不同，因此不是“打开已有开关必过”。gfx950-only opus警告不能推广为所有AITER MoE不可用；必须选实际gfx942支持路径。

### M2：保留数学边界的索引专家GEMV/GEMM

若M1无法在合理试验内满足token门槛，替代方案是定制BF16专家执行：直接依据expert_id访问常驻权重，避免先复制selected weights；分离gate/up、BF16激活中间量、down与最终加权归约，尽量保留Step0舍入边界。可以先专注小batch decode，prefill继续grouped_mm；大batch再使用按专家分组的GEMM。

实现位置独立Mellum专家模块或工作区内核，不修改冻结AITER安装。Triton或已有库接口可选，先做最小shape kernel可行性/数值/速度验证；“定制kernel”本身不保证比现有BMM快。不能把每个子kernel拆成三个优化点。

**计点**：M1和M2是同一MoE瓶颈的替代路线，最终选一个自包含执行方案计1点；若最终再叠加权重量化，它是额外候选，不能默认混入。建议沿用现有MoE选择轴，新增模式须清楚注明是拟议而非已实现。

**继续条件**：路由语义不变、加载完整，目标开销实降；代表性短/窗口/长输入通过token检查，再看A/B/C。若短集已经系统性大量分叉，先排查布局与执行公式，不立即回到全层bitwise追踪；M1不成立则评估M2或停止本点。

## 5. 候选A：分页/滑窗Attention执行优化（优先候选2）

**目标瓶颈**：显式mask与KV展开、逐步gather/复制、参考式重复旋转/写回、无效窗口工作。主要改善C TTFT/TPOT，也支持A批量吞吐。

### A1：复用ATOM AITER paged/varlen路径

主要源码`atom/model_ops/attention_mha.py:PagedAttentionImpl.rope_cache/prefill_attention/dispatch_backend`、`attentions/aiter_attention.py:AiterBackend/AiterAttentionMetadataBuilder`。先保留正确Q/K Norm和CPU来源RoPE语义，仅换Attention kernel与必要缓存接入；不一次恢复旧QK-Norm/RoPE融合、packed QKV、router等所有开关。

Prefill按cu_seqlens处理变长请求，SWA以kernel窗口参数表示，Full仍完整causal；decode直接使用实际block table，不先gather成连续历史。去除重复旋转/K覆盖须确保最终K与absolute positions正确，是该Attention执行替换的必要接口工作；不借机把独立RMSNorm/QKV融合捆绑为一点。

### A2：现有Triton paged/unified attention

主要源码`attentions/triton_mha.py:TritonMHABackend/TritonMHAMetadataBuilder`和`attention_mha.py`导入的AITER unified_attention。该builder使用flash layout并对融合QK/RoPE路径有不兼容检查；当前参考KV覆写显式拒绝flash layout。必须整体核对builder、cache binding、slot写入与kernel期待布局，不能只改一个ATOM_USE_UNIFIED_ATTN环境变量。

A1和A2可先在同一层比较性能/数值，按可行性选一个最终后端，不默认同时保留两条复杂优化。布局切换只在重新启动/加载时进行，不承诺热切换。

**计点与风险**：必要分页布局/窗口兼容属于一个自包含Attention替换；prefix cache、chunked调度、KV量化、物理环形窗口回收仍是独立机制。Step0已证明不同内核数值有差异；以题目token/任务门槛和保留的严格采样规则验收，查语义错误时再启用固定输入诊断。

**验证重点**：Full与SWA、1023/1024/1025、8191/8192/8193、16k；非零位置的q_len!=kv_len、分页跨块、不同长度多请求、错误复用旧请求cache。不能只验证一条query和连续物理block。

## 6. 第三点候选：互斥预算，profiling后决定

### G：CUDA Graph，必要时接入torch.compile

适合证据：M/A之后trace中decode CPU dispatch/launch间隙、细碎kernel仍占显著比例。先做不改变数学的decode graph可行性，再看是否需要torch.compile；不是无证据同时增加编译和一堆局部融合。

利用`ModelRunner.capture_model/run_model`的固定batch bucket和静态输入buffer，去除capture内GPU→Python取值、动态metadata/分配、Python状态修改；正常性能路径不安装保存张量探针。必要操作封装按框架编译/graph接口接入，在新增Mellum文件或调用处完成，不改既有`@support_torch_compile`模型文件。

风险：capture时长、显存、不同context长度、padding及dummy输入；不能为了静态图把Full历史截短。prefill可先保持eager，decode graph能否实际replay必须在trace验证。计1点的前提是针对同一launch-overhead瓶颈的捕获兼容链，不夹带额外无关融合收益。

### C1：Chunked prefill / 一个明确调度策略

适合证据：场景C到达长prompt时已有decode停顿、queue delay或TPOT p99尖峰，而不是单个kernel吞吐不足。沿用Scheduler的token budget机制，选择一个chunk策略/参数自包含优化，固定外部到达率、并发和token参数。

当前reference不支持cached/chunked prefill，A后端也须实证q_len!=kv_len的绝对causal/窗口语义和position/slot一致。chunk过小可能增加kernel launch和TTFT；过大仍阻塞decode，因此观察C TTFT/TPOT同时变化，不只报告有利一列。

### C2：Prefix caching

适合证据：场景C实际tokenized公共前缀形成block级cache hit，重复prefill工作显著；先统计token级公共长度/命中率。文件有shared_prefix_group不保证服务模板和tokenizer后仍有足够完整公共block。

必须验证cached-prefix续算、prefix之后绝对position、block refcount/生命周期和混合SWA/Full层；HF模板/thinking不改变。只缓存KV合法，不缓存最终答案。收益由命中率×被省prefill占比决定，可能主要改善TTFT而非TPOT。固定冷启动/预热政策、按真实场景请求自然命中，不对评测集特殊预热。

**预算规则**：G/C1/C2三选一，不能全打开后叫“系统优化”一点。已有默认开关启用同样要记录，不视作自动免费收益。若C收益已经由A显著解决且CPU仍明显，选G；若C长prefill阻塞主导，选C1；若重复prefill主导且命中稳定，选C2。若收益不足或时间有限，第三点可以不做。

## 7. 其他可替换路线

| 候选 | 可能价值 | 当前主要代价/风险 | 建议 |
| --- | --- | --- | --- |
| 单卡局部融合/投影 | 降低Norm/RoPE/残差/采样或QKV小kernel开销 | QKV打包及BF16融合曾改变结果；未有占比数据 | 若profile指向这些算子，可替换第三点，不能随意塞进Graph或Attention计点 |
| BF16→FP8专家权重 | 降低读带宽和权重空间，A可能受益 | 当前Mellum Torch路径显式仅BF16；scale、量化误差、校准与gfx942支持需新验证；不能偷混KV量化 | 仅在BF16融合仍受权重带宽限制且有时间时替换一个候选；不默认追加第四点 |
| FP8 KV | 降低长历史读和cache容量压力 | 是Attention之外的量化机制，RoPE写回/布局/scale均需验证 | 容量或KV带宽有证据时考虑，单独计点 |
| SWA物理cache窗口回收 | 21/28层的历史cache容量可减少，利于并发 | Scheduler/block allocator跨层共享策略与prefix生命周期复杂；7个Full层不能截断 | 先判定显存/分配是否真限制吞吐，不将逻辑SWA已启用误称物理回收已做 |
| 双卡TP/EP/DP | 单请求时延或并发吞吐可能改善 | Mellum显式TP1限制；collective/专家通信、归一化评分、双卡正确性；DP也算双卡方案，不能只看总吞吐 | 默认单卡。必须证明A每卡效率或C尾延迟获益，作为一个预算候选替换已有点 |
| MTP/投机 | 减少昂贵target decode次数 | 已核对checkpoint未见MTP head，当前注册/接受路径还需严格一致性处理；代码存在relaxed rejection路径 | 当前不推荐。没有适配权重/有效draft前提，不把“开关存在”当可用方案 |

若选择题目定义的FP8精度方案，题目给出5个百分点的HumanEval门槛，但具体权重/KV混合精度如何归类需按出题方口径确认，不能仅因启用FP8 KV就自动套用更宽阈值。仍需正确生成对照与真实评估，不能以更宽门槛作跳过验证的理由。多卡、MTP是条件要求，不为凑优化点强上。

## 8. 可选择的组合包

| 方案 | 最多三点 | 适用profile | 主要优点/代价 |
| --- | --- | --- | --- |
| 推荐：单卡均衡 | M + A + G | 专家复制/Attention先主导，之后launch开销明显 | 可归因，覆盖A/B/C；Graph取决于前两点落地 |
| C尾延迟优先 | M + A + C1 | 优化kernel后长prefill仍阻塞decode | 面向C TPOT p99；chunk策略需压住TTFT回退 |
| 前缀复用优先 | M + A + C2 | 场景C真实公共block命中率高 | 复用引擎能力；cached prefix语义与初态规范需补 |
| 工期保守 | M + A，或只做收益最大的1点 | 前两点可带来主要增益，第三点不明确 | 题目最多3点，不要求做满；更有时间做正确性/交付 |
| 条件替换 | A + G + FP8-MoE，或M + A +双卡 | 相应量化/并行证据比原第三点更强 | 不预定，重新核对实现时间、计点和归一化收益 |

M和A实施先后依首轮profile：A显著主导时先A；否则M作为初始第一候选。M1/M2、A1/A2是各自候选内部替代，不是承诺都做。原三点早期方案保留为候选池，而非在没有测量时固定项目。

## 9. 第一轮profiling与测量计划

### P0：形成可测B0

1. 固定Step0文件/配置与git工作树，保留现有S0证据；不能把原HEAD当已验候选commit。
2. 正常服务显式选择正确性组合，关闭被动logits/层dump/强制计划及调试重放。推理日志保持必要程度，不在测量热路径写每token张量。
3. 单请求256输出token能力检查；再做2请求、不同长度/结束时刻、多请求基础验证。逐请求reference桥接可作为B0，记录其局限与启动配置。
4. 核对评测客户端：本轮在`/opt/venv`未检出evalscope包，不代表容器其他位置完全不存在。优先核对发题入口/现有客户端；不为跑工具升级冻结推理依赖。自写开发负载可辅助诊断，不代替最终evalscope。

### P1：小范围但有代表性的trace

- Decode：batch1、8、32；context1024、4096、16384中选择代表组合；区分21个SWA与7个Full层，不能只测首层SWA。
- Prefill：单请求1024/4096/16384，以及若干短请求的总token预算对照，区分full prefill与后续cached/chunked能力。
- 端到端：B型单流，A型批量，C型长短请求混合/真实到达过程；开发trace可缩小请求数，但明确非正式分数。
- 保存CPU/GPU时间线、kernel调用数、间隙、显式index/copy/分配、MoE与Attention/投影/采样份额、峰值显存；需要时加路由分布和prefix命中。区间重叠不能把CPU时间与GPU累计时间简单相加。

现有接口：`model_runner.py:ModelRunner.start_profiler/stop_profiler`，`llm_engine.py:start_profile/stop_profile`，`api_server.py:/start_profile,/stop_profile`，`ModelRunner.run_model`已有prefill/eager_decode区间。优先Torch profiler，必要时使用现有`/opt/rocm/bin/rocprofv3`等工具读取硬件数据；工具存在不等于权限和所有counter已验证。

`atom/examples/profile_offline.py`默认temperature=0.6，随机词串可能重编码改变长度；不能不改参数就视为正式greedy精确长度评测。新测量入口用固定token、temperature=0，正式性能负载按题目ignore_eos输出256。完整profiler仅采样少量迭代，正式计时关闭profiler，避免trace开销混入成绩。

### P2：决定一个优化并验证增益

用热点占比f与局部实测加速s做Amdahl估计`1 / ((1-f)+f/s)`，只作为适用固定负载的上限近似，不代替服务队列模型和C p99。无profile不承诺2×/5×。每个候选先做单层/代表性shape，确认目标分量下降，再做一个集成版本；若目标并非热点或收益只转移到另一处，停止或换候选。

### P3：累计与单点关闭消融

至少记录B0、B0+O1、B0+O1+O2、B0+O1+O2+O3；最终还能分别关闭每一点。每行记录卡数、precision、服务/负载参数、A tokens/s/GPU、B TTFT/TPOT p50/p99、C TTFT/TPOT p99、正确性指标、源码与配置指纹。某组合不支持/失败用N/A及原因，不填0、不隐藏。

公平基线要求：同一批量桥接、采样规则、引擎参数、数据、GPU数与cache初态；除当前优化所需变化外固定配置。若内核优化必须改变max_num_seqs/内存预算才能运行，额外记录这项资源策略对结果的影响。已有Step0 util=0.35用于保守验证，不应无记录同时调成0.9并归因给某kernel；资源调参若构成独立容量瓶颈优化应单独披露/计点讨论。

A/B/C正式计时与HF参考/profiler隔离，另一张卡也不并行跑可能影响共享CPU/I/O的参考任务。开发先少量重复确认信号，最终按题目3次取中位数。C保留请求失败/超时、实际完成数、排队及长尾，不只统计成功的快请求；不能把cold cache与预热命中结果混成同一消融。

## 10. 正确性、停止规则与交付

- 保留既有严格argmax并列规则、RMSNorm语义、原SWA/YaRN和原权重；每项优化关闭可回到对应基线，按启动时切换并重载，非带cache热切换。
- 每点先跑小集、窗口/分页边界及受影响长输入。达到候选后再跑固定200条和16k；记录完整序列率与位置率。FP32局部对照仅用于诊断，不替代外部HF。
- 单项若持续不满足题目门槛、缺少目标开销下降或让C尾延迟明显恶化，停止并保留放弃理由；不为凑3点叠加风险。外部HumanEval参考/口径待补，不将当前缺失变成无限bitwise实验。
- 阶段材料：当前06只维护候选与决策；真正开始profile后才建立report/performance/01…，保存实测而非占位成绩。最终≤4页报告突出瓶颈、为什么选/弃、预估vs实测和消融；1页AI说明引用真实失败发现与工具用途。

**下一项建议实施**：P0最小可测服务与reference批量桥接，加P1一个bs1 decode/长prefill trace；据此确认M或A谁先做。暂不直接改MoE内核、不启用三项开关同时跑分。
