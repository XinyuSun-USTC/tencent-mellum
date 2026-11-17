# Mellum–ATOM 适配与优化工作日志

作者：XinyuSun-USTC

本日志按2026年11月10日至17日记录项目进展，对应DAY0—DAY7。内容涵盖模型适配、正确性验证、性能分析、优化取舍及交付整理。相关源码、实验结果和提交历史保留于仓库，各日条目附主要证据入口。

## DAY0｜2026年11月10日：需求分析与架构研究

连接分配的服务器并进入项目容器，检查工作区目录、Python虚拟环境、ROCm与GPU设备，以及模型权重和数据集的存放路径。确认远程终端、文件访问和脚本执行方式，完成后续源码分析与实验所需的基础环境准备。

阅读研究指南和题目说明，梳理模型适配、优化目标、正确性门槛及ABC场景的负载特征。以只读源码分析为起点，检查ATOM配置解析、模型注册、权重加载、调度、注意力和专家计算的调用关系，并核对Mellum配置及checkpoint布局。整理容器内的模型、数据集与冻结依赖，明确后续开发范围。

参考环境是当日的主要问题。容器内Transformers不能直接提供所需的Mellum参考实现，现有结果文件也不足以支持完整验证。决定保持ATOM推理环境不变，单独建立新版HF参考环境；适配代码与测试工具同步开发。目标是尽快得到自己的正确实现，并为后续优化保留扩展性。

完成架构与环境研究后，准备模型支持探针、参考环境安装及输入协议检查，作为首次真实权重推理的前置条件。

相关记录：[模型研究](report/research/03_mellum_model.md)、[正确性参考调查](report/research/04_correctness_reference.md)、[HF环境验证](report/validation/01_hf_reference_environment.md)。

## DAY1｜2026年11月11日：建立参考并打通首次推理

在独立环境中安装Transformers 5.10.1，完成Mellum配置导入、真实BF16权重加载与GPU生成。实现ATOM侧配置、注册、网络和权重映射，增加加载审计，覆盖5631个源张量。首次两条提示各生成16个token，与HF参考一致，打通了从checkpoint到输出的基本链路。

扩展至64个token后出现生成分叉，因此建立同前缀、逐层边界和固定输入重放工具，分别检查RMSNorm、路由、专家、RoPE和注意力。在专家输入与路由logits相同的条件下，输出仍存在差异。进一步对照HF官方后端，确认分组矩阵乘法与eager实现的计算路径不同；调整后的专家重放达到一致，端到端验证则继续推进。

根据这些结果，采用固定输入逐项定位，避免将自回归历史变化与当前算子误差混在一起。同步修复set序列化、导入路径及接口版本差异等工具问题，使后续实验能够持续保存结果。

相关记录：[模型接入](report/validation/05_atom_mellum_model_wiring.md)、[RMSNorm定位](report/validation/09_mellum_rmsnorm_rounding_fix.md)、[专家后端确认](report/validation/17_official_backend_confirmed_and_step0_torch_moe.md)。代表运行：`20261111T085551407745Z`、`20261111T170208537720Z`。

## DAY2｜2026年11月12日：对齐参考生成路径

对比独立与打包QKV投影，核验RoPE逆频率初始化、旋转和舍入行为，并检查滑窗边界的显式mask及KV展开。保留ATOM分页缓存管理，增加固定历史下的decode测试，区分prefill结果、缓存读写与新增token计算对输出的影响。

定位过程中发现，投影形状、RoPE初始化以及HF生成阶段的专家执行方式均会影响数值路径。对齐这些因素后，继续处理并列最大logits的greedy选择差异，修正采样分流。十例普通生成共640个token及相应logits达到一致，形成参考生成闭环。

同时完善实验运行器的心跳、进程死亡检测、阶段超时和清理机制，处理异步末尾额外forward及可变元数据采集问题。随后扩展长prefill与SWA因果带状mask，为边界和长上下文验证做准备。局部精确比较作为定位方法，最终判断回到自然生成结果。

相关记录：[prefill与decode基线](report/validation/26_prefill_passed_and_forced_decode_baseline.md)、[router投影定位](report/validation/33_router_projection_root_and_candidate.md)、[普通生成闭环](report/validation/34_cached_decode_exact_and_normal_generation.md)、[greedy分流](report/validation/35_greedy_tie_break_and_sampling_dispatch.md)。代表运行：`20261112T213129468143Z`。

## DAY3｜2026年11月13日：扩大验证并建立性能研究流程

完成1023、1024、1025输入的滑窗边界检查，并扩展到8191、8192、8193、16384输入。四条长输入各生成64个token，共256个位置及logits一致。固定164条HumanEval提示和36条场景A提示构成开发集，初始单请求适配版本在200条各64个token的对照中达到12800/12800位置一致，加载审计通过。

据此完成适配阶段总结，并将生成一致性与完整代码功能评分分开处理。前64个token可能仍包含思考过程，后续需使用HumanEval完整生成与执行评估。停止将中间张量及完整logits逐位一致作为所有优化的额外目标，把时间转向性能分析。

整理源码、脚本、环境及输入指纹，完成个人仓库认证、代码迁移与首次推送，保存上游历史、轻量证据和原始结果清单。初始适配版本通过 `step0-local-validated` 标签保留。

开发torch profiler采集与扩展测量工具，开展1K至16K输入的单请求研究。结果显示，短输入专家执行占比较高，16K prefill则主要受SDPA影响。因此，后续优先研究专家执行粒度、注意力及重复缓存处理，并补齐多请求服务能力。

相关记录：[长输入验证](report/validation/37_step0_long_prefill_extension.md)、[开发集验证](report/validation/38_step0_developer_200_agreement.md)、[适配总结](report/validation/39_step0_summary.md)、[扩展profiling](report/performance/02_expanded_bs1_profile_study.md)。代表运行：`20261113T104458011376Z`、`20261113T190525160560Z`。

## DAY4｜2026年11月14日：窗口注意力探索与服务能力建设

根据长输入SDPA开销，接入AITER CK窗口prefill，完成微基准、真实Q/K/V重放及端到端ABBA实验。在16K输入、64token输出的离线负载中，TTFT约由3996ms降至1426ms，但较长生成发生分叉。随后测试Triton替代实现，局部误差有所降低，实际激活32条提示的一致率仍仅61.62%，因此暂停继续扩大该路线的配置搜索。

为更完整地评价候选，引入官方HumanEval固定版本套件，开发代码提取、功能评分、自检与HF配对评估。明确token一致性与代码功能评分分别报告，局部算子误差和功能得分均不能单独解释全部生成行为。

实现ragged参考注意力桥接，增加多请求生成、HTTP非流式与流式响应、并发服务检查，并修复已完成请求的前端状态清理。工作重点由单请求算子比较转向真实批处理能力，为后续A并发及C持续到达测试建立基础。

相关记录：[SWA配对测量](report/performance/04_swa_prefill_abba.md)、[真实QKV诊断](report/performance/06_swa_real_qkv_diagnosis.md)、[Triton生成检查](report/performance/09_triton_activation32.md)、[批处理与HTTP](report/performance/10_batch_and_http_smoke.md)。代表提交：`142a929`、`3e48282`。

## DAY5｜2026年11月15日：专家优化、上游调研与完整服务测量

建立可批量执行的实验流程，检查逐请求投影、共享路由及专家执行策略。索引式专家GEMV获得约4.892倍局部加速，但完整开发集前64token位置一致率仅78.23%，停止该版本晋级。prefill逐元素合并保持了对照输出，短、长输入配对测量却没有实质收益，也未继续投入。

调研ATOM中其他模型及上游ATOM、vLLM等实现，扩展原生专家、分页注意力、融合和缓存候选。原生路径有性能潜力，但生成一致率不足。通过固定输入重复、排序对照和prefill/decode拆分，进一步检查专家归并与注意力路径，而非仅凭一次端到端失败作判断。

建立独立EvalScope客户端，核对ABC命令、输入格式、流式参数和token计数。首次完整参考服务测量中A、B完成，C出现大量超时，说明单请求计算正确与持续服务容量是两个不同问题。两张卡分别承担独立实验，提高候选筛选效率。

完整HumanEval采用关闭thinking、最多2048token的协议。候选结果完成后，单独恢复受工具问题影响的HF参考，避免重复生成。功能评分表明部分token分叉并不导致功能得分下降，但逐token门槛仍需独立核验。决定结合两类正确性结果筛选候选，并将C的持续容量纳入优化优先级。

相关记录：[索引专家复核](report/performance/15_indexed_decode_d200.md)、[评测客户端](report/performance/16_abc_protocol_and_client.md)、[参考ABC](report/performance/17_r_abc_baseline.md)、[原生阶段诊断](report/performance/20_native_phase_diagnostics.md)、[完整HumanEval](report/performance/22_fused_moe_humaneval.md)。代表提交：`f52d196`、`64362b7`、`7d6a4ce`。

## DAY6｜2026年11月16日：优化路线筛选与场景C专项分析

继续测试整批投影、原生算子、Graph、融合和数据通路组合。部分配置完成完整ABC并取得较高吞吐，但对应开发集一致率仅62.85%。因此，将其保留为性能探索结果，回到参考兼容路径逐项接入优化，确保生成与性能来自同一配置。

专家group4在短A配对负载中提高吞吐13.74%。在此基础上，KV单次写入使短A吞吐进一步提高3.81%，短B TPOT p50降低14.32%。两项组合完整检查达到96.8125%位置一致率，四条长输入256/256一致；HumanEval为148/164，HF为145/164，绝对差1.8293个百分点。

C容量仍是主要阻碍。组合在C20完成20/20，但C200仅1/200。随后测试分段prefill、decode配额与公共前缀缓存：四轮decode配额下，512、1024、2048分段的C60分别完成21、24、22条，对照为1条。缓存实际复用约35.47%的输入token，仍未消除超时。实验显示分段与配额还会影响新请求等待，不能只按缓存命中率预测服务收益。

进一步测量隔离decode，并检查客户端断连后的资源占用。发现请求断连后引擎可能继续计算，遂实现同步后取消与KV回收，验证存活请求及后续分配正常。SWA group4和每次前向的元数据复用将批32隔离decode提高至60.32token/s，相对同轮对照均值提高40.74%。据此组合SWA、缓存、分段及取消机制，优先延续单卡路线，暂缓P/D、FP8与MTP。

相关记录：[探索组合ABC](report/performance/36_combined_full_abc.md)、[专家与KV组合](report/performance/37_reference_kv_and_expert_groups.md)、[分段测试](report/performance/38_c_chunked_prefill_screen.md)、[缓存与配额](report/performance/40_c_prefix_and_quota_screen.md)、[SWA与取消验证](report/performance/43_swa_decode_groups_and_abort_interleaving.md)。代表提交：`177f7d8`、`d6edc3d`、`6ddec3c`、`6730362`、`a939efe`、`386d124`。

## DAY7｜2026年11月17日：双路线验证、累计消融与交付集成

首先复核前一阶段的C60组合结果：SWA分组与取消回收完成11/60，增加前缀缓存后为28/60，再加入1024分段和四轮decode配额后为43/60。补充profiling发现，稳态decode中的全局SDPA、专家及KV读取分别占kernel时间30.19%、25.54%和12.01%；受控混合负载中约74.04%的kernel时间位于prefill。由此确认，C的积压同时涉及长输入处理与持续decode容量，局部算子加速尚不足以保证完整负载完成。

首次累计验收将SWA分组、元数据复用、前缀、分段及取消机制合并为第三项。该组合使A吞吐达到71.2464token/s/GPU、C完成48/200，但开发200位置一致率降至75.9453%，长组仅43/256匹配。复核发现，部分长输入从第9个输出token开始分叉，早期8token入口检查未能覆盖。决定将入口可运行检查与完整正确性验收分开，并重新拆解第三项，避免用短检查代替候选晋级依据。此组合随后被新的精度优先第三项取代。

同时统一消融基线：将32槽、参考兼容批处理、逐请求投影与共享路由纳入Step0，①仅计新增的专家group4。复用已有完整Step0测量，并保存配置与来源，不再为初始单槽实现追加长时间性能测试。检查正确性数据构成后，明确开发200为164条HumanEval与36条A提示，前64token位置一致率按12800个位置统计；该固定集合用于各版本对照，HumanEval完整功能评分另行执行。

在原生算子、投影、Graph与融合探索的基础上，完成性能优先路线的累计消融。①采用原生Attention/FusedMoE、AITER归一化及整批packed投影；②增加原生路由、驻留RoPE、KV单次写入、QK融合与decode Graph；③增加原生前缀兼容、前缀缓存及prefill元数据复用。无分段组合在C200全部完成，尾延迟优于同轮chunk1024候选，因此最终未保留分段及取消机制。完整四行A吞吐依次为41.3351、492.9431、1128.2608、1135.6885token/s/GPU；B TPOT p50依次为61.2527、47.1220、4.9184、4.8985ms。三个优化累计行均完成C200，最终C TPOT/TTFT p99为11.3619/1026.0455ms。

性能优先路线同时测量生成一致性和完整HumanEval。最终组合位置一致率为66.8438%，HumanEval为147/164，HF参考为145/164；①②的HumanEval达到152/164，但按采用的绝对差规则，其正向差值4.2683个百分点仍单独列为超出2个百分点范围。功能评分接近或高于参考，并不意味着token门槛通过。决定保留性能路线的完整成果与消融证据，同时继续寻找保持参考一致性的第三项，而不将两类正确性指标相互替代。

精度优先路线以已通过检查的专家group4与参考RoPE/KV单次写入为对照，分别验证decode元数据复用，以及元数据复用与参考SWA四请求分组的组合。两项候选均通过完整开发200、长输入与HumanEval检查。性能筛选按“对照—元数据—SWA分组—对照”执行，前后对照A吞吐为52.0415和52.0798token/s；组合候选短A吞吐提高42.76%，batch32隔离decode提高38.15%。据此选定元数据复用与SWA分组为精度路线新③，不再混入前缀、分段、Graph或原生专家。

随后对精度最终组合完成A512/B50测量，全部请求成功；A为74.7620token/s/GPU，B TPOT p50为51.2821ms。新③相对①②使A吞吐提高40.02%，B延迟增加1.38%；累计相对Step0，A提高80.87%，B延迟降低16.28%。完整正确性结果为12392/12800位置一致（96.8125%）、174/200整序列一致、长组256/256及16K组64/64匹配，HumanEval为148/164，正差1.8293个百分点，均通过本地对应检查。

为补齐精度路线最终组合的C指标，按题给0.3请求/秒、256输出和180秒请求超时，独立启动服务测量三轮。三轮均仅1/200成功，其余各199条均记录TimeoutError，失败耗时约180–181秒。保留逐轮完成数和成功请求延迟，不将单个成功请求的p99作为完整负载成绩，也不汇总有效三轮中位数。至此，两条路线的差异得到完整测量支持：精度路线保持本地一致性，但C容量不足；性能路线完成ABC并显著加速，但未达到90%的token位置一致率。决定以双模式交付这一结果，不再扩大优化范围。

建立独立交付分支`release/dual-mode`，将最新两条路线统一接入`start_server.sh`，默认performance，通过`--mode precision`切换；两种模式均支持Step0至①②③累计行及逐项关闭。性能②③依赖①，入口明确拒绝不兼容组合；精度路线支持全部八种子集。配置、环境与源码哈希写入启动记录，历史profile保留原义，防止新入口与旧矩阵混用。CPU验证覆盖13种合法组合，112项相关测试通过。随后两卡分别执行四行真实GPU入口烟测，八个配置均通过5631张量加载、HTTP/SSE、usage、模式开关核对及端口回收检查。

交付整理同步保留全部开发分支与提交历史，并归档支撑结论的生成token、功能判定、性能摘要、配置与SHA索引；模型、虚拟环境及完整结果目录不纳入源码仓库。技术报告重构为双路线设计，前置说明参考一致性与C容量难以兼顾的实验依据，分别列出优化机制、未采用路线、性能消融和正确性矩阵。技术报告与启动入口统一使用最新精度第三项与性能组合，旧版数据作为探索记录保留。

主要证据：[瓶颈profile](report/performance/45_integrated_c_bottleneck_profile.md)、[性能优先累计矩阵](artifacts/experiments/performance_cumulative/collected/matrix.md)、[精度第三项筛选](docs/experiments/02_precision_third_results.md)、[完整A/B](docs/experiments/03_precision_third_full_performance.md)、[C三轮补测](docs/experiments/05_precision_c_completion_results.md)、[双模式入口集成](docs/submission/15_dual_mode_entrypoint.md)、[技术报告](TECHNICAL_REPORT.md)。

代表提交：`e2d388f`（精度第三项实验工具）、`a0bbe6f`（正确性与筛选结果）、`18769c7`（完整A/B）、`b01696c`（C三轮结果）、`9a194c9`（双模式集成）、`edaa455`（八行GPU入口验证）、`e555e74`与`79c448b`（技术报告重构与润色）。
