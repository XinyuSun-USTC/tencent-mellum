# Mellum–ATOM 适配与优化工作日志

作者：XinyuSun-USTC

本文按2026年11月10日至16日的自然日对应D1—D7，记录任务推进、主要证据、阻碍及技术决策；11月17日的工作单独列为补充记录，不并入前七日。日期依据既有研究记录、实验目录的UTC运行编号及Git历史归纳，不据此推算每日工时。跨日实验按实现与结果分别记录，后续结论不倒写为当日已知事实。

本文为交付用逐日记录，原始实验细节保留于 `report/research/`、`report/validation/` 和 `report/performance/`。Git保留原有提交及探索分支，不通过squash重写过程；早期集中纳入版本管理的修改，也不追溯补造为逐日开发提交。

## D1｜11月10日：题意分析、源码研究与参考方案确定

**目标与工作。** 完整阅读研究指南及题目说明，区分Step0模型适配、Step1优化、正确性门槛和ABC性能场景。以只读分析为起点，检查ATOM的配置解析、模型注册、权重加载、调度、注意力和MoE调用链，并核对目标模型的配置与checkpoint布局。整理冻结环境、模型及数据资产，确认后续适配必须在题给推理依赖上开展。

**主要问题。** 容器内Transformers版本不能直接提供所需的原生Mellum参考路径；已有数据及结果文件也不足以构成可直接复用的完整正确性评测流程。仅有checkpoint能够加载，不能证明网络计算、tokenizer模板和生成行为正确。

**决策。** 首先实现自己的最小正确适配，不以复刻出题方baseline源码为目标。推理环境保持冻结，独立新版HF环境仅作为正确性参考。适配开发与验证脚本同步推进，避免功能完成后再临时补建测试。文档按研究主题组织，逐日记录统一保留在工作日志中。

**当日状态与后续。** 完成架构与环境研究，准备参考环境安装和支持探针。该日不宣称已完成真实checkpoint的GPU推理或Step0验收。下一步优先验证HF参考可运行，再接入ATOM模型。

证据：[模型研究](report/research/03_mellum_model.md)、[正确性参考调查](report/research/04_correctness_reference.md)、[HF环境验证记录](report/validation/01_hf_reference_environment.md)。

## D2｜11月11日：建立HF参考并完成首次ATOM推理

**目标与工作。** 独立参考环境安装Transformers 5.10.1，在共享冻结Torch的条件下确认Mellum配置、模型导入和真实BF16 GPU生成。实现ATOM侧模型配置、注册、网络与权重映射，增加加载完整性审计，核对5631个源张量。首次ATOM两条提示、各16token生成与HF一致，形成从checkpoint到推理输出的基本闭环。

**主要问题。** 测试扩展至64token及更多输入后出现生成分叉，说明短输出通过不足以证明完整适配正确。同前缀重放进一步显示，在MoE输入与router logits相同的情况下，专家输出仍可能不同，不能简单归因为输入历史或路由选择。

**定位与处理。** 建立相同前缀、逐层边界和固定HF输入重放工具，分别检查RMSNorm舍入、专家激活与归约、RoPE及注意力路径。官方HF专家后端对照表明，实际使用的grouped_mm路径与eager实现存在数值差异；调整后的专家重放达到精确一致，但端到端生成尚未闭环。同步修复加载结果中set无法JSON序列化、诊断脚本导入路径和接口版本差异等工具问题。

**决策与后续。** 将诊断范围收缩至已定位边界，以固定输入排除自回归历史放大，逐项核验计算路径；不因部分短测试通过而宣布Step0完成。下一步检查投影打包、旋转系数与prefill/decode差异。GPU任务通过用户终端运行，工具侧无法访问设备的限制与模型运行失败分别记录。

证据：[模型接线验证](report/validation/05_atom_mellum_model_wiring.md)、[RMSNorm定位](report/validation/09_mellum_rmsnorm_rounding_fix.md)、[官方专家后端确认](report/validation/17_official_backend_confirmed_and_step0_torch_moe.md)。代表运行：HF `20261111T044314965290Z`、ATOM首次短生成 `20261111T085551407745Z`、专家精确重放 `20261111T170208537720Z`。

## D3｜11月12日：完成参考生成路径的系统性对齐

**目标与工作。** 在固定输入条件下对照独立与打包QKV投影，核验RoPE逆频率的初始化来源、旋转与舍入行为，并检查滑窗边界的显式mask和KV展开。保留ATOM实际分页KV管理，建立同一历史下的强制decode测试，以区分prefill误差、缓存读写和新增token计算的影响。

**主要发现。** 投影计算形状能够影响输出；参考RoPE初始化及窗口mask也需与HF实际执行路径一致。prefill达到精确后，decode仍存在局部差异，需要继续沿层边界排查。对HF普通生成的同实例诊断还发现，生成阶段的专家执行后端与独立forward不完全相同。之后出现的并列最大logits选择差异，则需从采样分流处理，而不能继续归因为模型权重或注意力错误。

**处理与验证。** 逐步对齐参考投影、RoPE、SDPA、router及HF生成专家行为，并修正无过滤greedy路径的分流。固定历史decode达到精确一致；短生成闭环后，十例普通生成共640个token及对应logits一致。为防止诊断过程本身产生误导，补充进程心跳、死亡检测、阶段超时及清理，并处理异步末尾额外forward与可变元数据采集问题。

**决策与后续。** 将“局部精确”明确限定为定位工具，不以其替代端到端验证。当天已有短组闭环，随后扩展长prefill和SWA因果带状mask；新增长输入能力的GPU验收留待次日，不将实现完成等同于测试通过。

证据：[prefill与强制decode基线](report/validation/26_prefill_passed_and_forced_decode_baseline.md)、[router投影定位](report/validation/33_router_projection_root_and_candidate.md)、[普通生成闭环](report/validation/34_cached_decode_exact_and_normal_generation.md)、[greedy分流修复](report/validation/35_greedy_tie_break_and_sampling_dispatch.md)。代表运行：`20261112T164434545261Z`、`20261112T193650834240Z`、`20261112T213129468143Z`。

## D4｜11月13日：扩大Step0验证，建立版本与性能基线

**目标与工作。** 先完成1023/1024/1025的滑窗边界测试，再扩展至8191/8192/8193/16384输入。长输入四例各64token，共256个token及logits一致；随后固定164条HumanEval提示和36条场景A提示，完成200条各64token的开发集对照，12800/12800位置及完整logits一致，加载审计通过。

**验收判断。** 这些结果支持Step0适配和本地生成一致性成立，但64token可能仍为思考内容，不能直接替代完整HumanEval代码执行评分。核对题目后，不再把逐层或完整logits精确作为后续优化的额外硬门槛；公开集身份及功能评分口径继续单独记录。

**工程整理。** 整理阶段报告、脚本、环境与输入指纹。协助完成个人仓库认证及首次推送，将ATOM源码、项目脚本与轻量证据纳入总仓库，保留上游历史和原始结果清单；历史Step0标签用于保存已验证基线，而非后续默认入口。早期未逐次提交的修改以真实快照纳入，不补造开发历史。

**性能研究与决策。** 开发torch profiler采集及扩展实验工具，先观察单请求1K—16K负载，再确定优化方向。参考profile显示短输入专家执行较重，而16K prefill主要受SDPA影响。决定以MoE执行粒度、注意力和重复数据处理为主要候选，并同步考虑从单请求到服务批处理的能力缺口。

证据：[长输入验证](report/validation/37_step0_long_prefill_extension.md)、[200条一致性](report/validation/38_step0_developer_200_agreement.md)、[Step0总结](report/validation/39_step0_summary.md)、[初始profile](report/performance/01_bs1_profile_baseline.md)、[扩展profile](report/performance/02_expanded_bs1_profile_study.md)。代表运行：`20261113T103154043872Z`、`20261113T104458011376Z`、`20261113T190525160560Z`；历史标签：`step0-local-validated`。

## D5｜11月14日：窗口注意力优化与功能评估体系建设

**目标与工作。** 根据长输入SDPA占比较高的证据，首先实现并评估AITER CK窗口prefill路径，开展微基准、同输入重放及端到端ABBA对照。窗口和长组短生成通过后，继续扩展生成长度和提示覆盖，避免仅凭短smoke判断候选可用。

**结果与阻碍。** CK候选在16K输入、64输出的离线实验中，将TTFT约由3996ms降至1426ms，但长生成出现分叉。实际Q/K/V重放表明，不同注意力实现即使输入相同也存在输出差异。随后筛查Triton候选，仍未形成可直接纳入最终正确组合的完整证据。由此确认：局部提速与局部误差大小都不能直接推导最终生成或功能指标。

**评价方法调整。** 在保留题目逐token一致率门槛的同时，引入官方HumanEval固定版本套件，开发代码提取、执行评分、自检和HF配对评估，区分生成一致性与功能正确性。当天进行的配对试验不写成后续完整164题验收结果，也不将功能评分通过视为免除逐token检查。

**服务能力建设。** 实现ragged参考注意力桥接，扩展多请求生成、HTTP非流式/流式响应和并发smoke，修复完成请求的前端状态清理。批处理改变算子形状，后续必须验证其对数值及服务容量的影响。

**决策与后续。** 保留CK/Triton的性能与失败证据，不直接升级默认方案；将下一阶段重点转向批处理兼容性、专家执行及完整服务评测。

证据：[SWA ABBA](report/performance/04_swa_prefill_abba.md)、[HF三方对照](report/performance/05_swa_hf_triangulation.md)、[真实QKV诊断](report/performance/06_swa_real_qkv_diagnosis.md)、[评估试验](report/performance/07_representative_correctness.md)、[批处理与HTTP](report/performance/10_batch_and_http_smoke.md)。代表提交：`142a929`、`3e48282`。

## D6｜11月15日：批量实验、上游调研与ABC服务评测

**目标与工作。** 将逐项手工运行扩展为有预算、可批量采集和分析的实验流程，检查批处理投影、共享routing及专家执行策略。索引式专家GEMV获得约4.892倍局部加速，但扩展至200条前64token后，位置一致率仅78.23%，低于90%门槛；prefill逐元素合并没有获得实质端到端收益。两类结果分别说明数值约束和收益不足的限制。

**外部实现研究。** 调查题给ATOM中其他模型的优化方式，并参考上游ATOM、vLLM等实现，重新评估原生专家、分页注意力、缓存与融合路径。相关实现作为技术参考，未升级冻结推理依赖。原生路径带来性能潜力，但同输入路由、专家归约和注意力阶段仍需独立诊断。

**评测与服务。** 建立独立EvalScope客户端和ABC输入适配，核对实际命令、流式参数、token计数及场景规模。首次完整参考服务测量暴露场景C严重超时，表明单请求正确性不能覆盖持续到达负载。并行开展原生计算、HTTP容量和前缀缓存实验；两张卡分别运行独立服务，不作为双卡模型并行。

**功能评估与方法修订。** 按讨论确定关闭thinking、最多2048token的完整HumanEval协议。候选生成完成后，HF参考因工具问题需要单独恢复，保留已完成候选结果，避免重复全部生成。完整功能评估表明，token分叉不能直接推出功能评分不合格；但题目的两类门槛仍需同时满足。

**决策与后续。** 减少对局部逐位差异的无界追查，用完整功能评估与开发集一致率筛选有价值候选；同时开始重视C的持续容量，而非只比较短负载或局部kernel。对尚未覆盖全部门槛的原生组合，不提前认定为最终方案。

证据：[索引专家筛查](report/performance/15_indexed_decode_d200.md)、[客户端口径](report/performance/16_abc_protocol_and_client.md)、[参考ABC](report/performance/17_r_abc_baseline.md)、[原生阶段诊断](report/performance/20_native_phase_diagnostics.md)、[完整功能评分](report/performance/22_fused_moe_humaneval.md)、[前缀实验](report/performance/25_prefix_probe.md)。代表提交：`f52d196`、`64362b7`、`7d6a4ce`。

## D7｜11月16日：优化路线收敛与场景C专项处理

**并行优化与资格复核。** 围绕原生计算、批量投影、graph、融合、数据通路及调度开展两卡独立实验，并补充HumanEval和ABC。部分激进组合具有较高性能，但一次完整开发集复核仅62.85%位置一致率。此前以功能评分或短测试推断整体可接受的判断因而需要修正：必须同时核对题目逐token门槛，不能将不同配置的性能和正确性拼接。

**回到可验证的增量路径。** 从参考兼容实现重新逐项接入专家分组与KV单次写入。专家group4在短A负载相对参考批处理提高13.74%；加入KV单次写入后，配对短A吞吐提高3.81%，短B TPOT p50降低14.32%。两项组合完整本地验收达到96.8125%位置一致率，长输入256/256一致，HumanEval为148/164，HF为145/164，绝对差1.8293个百分点。局部收益不连乘为正式ABC收益。

**C容量问题与转向。** 上述组合C20虽能20/20完成，但C200仍仅1/200，证明短负载不足以覆盖积压。随后将工作重点转向分段prefill、decode配额和公共前缀复用。C60中四轮配额的512/1024/2048分段分别完成21/24/22请求，对照为1；较小decode配额与缓存组合仍未解决容量。真实公共前缀复用约35.47%的输入token，但收益受准入、计算和管理开销共同影响。

**继续定位而非仅调参数。** 隔离decode测量显示原参考批32交付速率约43.68token/s；断连诊断发现客户端退出后引擎仍可能继续生成。分别实现安全取消回收与SWA解码分组、每forward元数据复用。取消交错测试验证存活请求正常完成、取消ID移除后不再调度，并能分配缓存给后续请求。SWA group4批32隔离decode达到60.32token/s，相对同轮首尾对照均值提高40.74%；开发集与长组输出序列仍与已验收参考组合一致，未据此免除集成后的完整验收。

**交付决策。** 优先单卡方案，暂缓P/D、FP8、MTP等新增工作。将三项最终方向收敛为：①参考兼容批处理与专家分组；②参考RoPE/KV单次写入；③SWA解码分组与混合负载执行。当天末准备将SWA、取消、前缀和分段机制组合比较，最终候选与累计消融最后一行须采用同一入口及配置。由于C仍存在失败，不宣称整题已达标。

证据：[激进组合ABC](report/performance/36_combined_full_abc.md)、[参考组合完整验收](report/performance/37_reference_kv_and_expert_groups.md)、[分段筛查](report/performance/38_c_chunked_prefill_screen.md)、[缓存与配额](report/performance/40_c_prefix_and_quota_screen.md)、[断连诊断](report/performance/41_isolated_decode_and_live_disconnect.md)、[取消与SWA分组](report/performance/43_swa_decode_groups_and_abort_interleaving.md)。代表提交：`177f7d8`、`d6edc3d`、`6ddec3c`、`6730362`、`e4b4909`、`a939efe`、`386d124`。

## 补充记录｜11月17日：最终候选与交付实验收敛

**接收跨日结果。** 前一日启动的集成C60实验完成：SWA group4与取消回收为11/60，增加prefix为28/60，再加chunk1024/d4为43/60。所有小规模生成检查均为72/72位置一致，取消后的已移除请求没有再次调度；但仍有17个失败请求，不能据此宣称完整C通过。

**补充瓶颈证据。** 对最佳组合采集稳态decode与受控混合负载trace。稳态decode阶段，全局层SDPA、专家和参考KV读取分别占kernel时间30.19%、25.54%和12.01%；混合负载约74.04%的kernel时间落在prefill。该结果支持全局注意力和长输入处理仍有优化空间，也说明仅调整调度参数不足以解释全部性能缺口。鉴于交付时间约束，停止扩展新的优化路线，转入完整测量与材料收敛。

**统一交付与消融。** 建立Step0、Step0＋①、Step0＋①②、Step0＋①②③四行累计矩阵；默认启动入口等于最后一行。Step0是在最终源码上关闭优化后的单槽参考路径重跑，不冒充历史提交逐字节复现；①开启32槽，其余行保持32槽，客户端负载不变。批量流程覆盖实际入口HTTP/SSE检查、完整ABC、200条前64token、长上下文及HumanEval，并支持冻结校验与性能续跑。

**工程问题及修复。** 首轮入口smoke的Step0已通过，后续行因复用尚未释放的端口而被占用保护拒绝；主流程又未处理缺失摘要。将其定位为测试编排缺陷，在新工作树中为每行分配独立端口，改为直接替换启动进程，检查监听端口释放并保存异常摘要。保留旧失败目录，不覆盖为成功。修复提交为 `6f02551`、`4fa8ddb`；完整实验从新目录重新执行。

**统计口径澄清。** 核对EvalScope源码后确认，其分位数仅覆盖成功请求；题面命令的180秒为单请求超时，并非单列的TTFT指标阈值。报告将成功样本分位数与完成数、失败数一并解释，不用超时样本被排除后的较低p99证明全部负载达标，也不自行补造题目未给出的失败率评分规则。

**文档与当前状态。** 重构技术报告，按适配方法、profiling、三项优化与取舍、最终消融及正确性组织，最终数据留空。完整实验已交由两张卡独立运行，截至本日志修订时尚未收齐结果；不预填通过结论、最终加速比或裁判成绩。本次日志亦仅修改文档，不改变正在测量的功能代码。

证据：[集成C60](report/performance/44_integrated_single_gpu_c_screen.md)、[最新profile](report/performance/45_integrated_c_bottleneck_profile.md)、[最终实验方案](docs/plan/65_final_delivery_matrix.md)。代表提交：`e1f3e22`、`753bbe8`、`6f02551`、`4fa8ddb`、`fb2971a`。当前候选分支：`release/final-matrix-smoke-fix`。
