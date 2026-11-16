# 工作日志

按自然日归纳，每天一个简要条目。日期依据既有日志、对话和结果目录的UTC运行ID；D0/D1是旧研究批次，不自动等于不同工作日。重构前全部追加记录已[保存快照](archive/step0_document_snapshots/report/worklog.md)，SHA256见[清单](archive/step0_document_snapshots/manifest.json)。当前有据可补日期为2026-11-10至16，不补造更早日期、时刻或提交；D1–D7映射待考核起始日确认。

## 2026-11-10

- 完成：容器/冻结环境准备记录，ATOM配置、模型、加载和调度链研究；Mellum模型/5631权重布局调查；阅读题目PDF并审计正确性资产；完成最初主题文档拆分与历史归档。
- 发现与决策：原Transformers5.2.0无原生Mellum支持；现有HumanEval结果Rejected，缺完整正式评测资产。用户确认最快正确适配、不刻意复刻出题方Baseline，测试与开发同步；独立新版HF只用于参考。
- 验证状态：配置、tokenizer/template及本地支持探针完成；该日记录不声称已完成真实checkpoint GPU推理。参考venv与安装脚本建立，包下载受网络限制，等待普通终端安装。
- 问题与下一步：建立可信HF参考后再适配；冻结原依赖。详见[环境/模型研究](research/03_mellum_model.md)、[参考调查](research/04_correctness_reference.md)、[验证01](validation/01_hf_reference_environment.md)。

## 2026-11-11

- 完成：用户终端安装HF5.10.1；小模型CPU24项与真实BF16 HF GPU参考通过。新增ATOM Mellum配置、模型、注册、权重映射/回调审计，首次5631张量加载与两条16token对照通过；建立short/window/long及扩展集。
- 诊断与修复：64token与扩展集暴露分叉；同输入逐层定位RMSNorm舍入顺序并修正。固定HF输入的MoE重放排除路由集合为主因，确认专家装饰器实际grouped_mm后端；接入Torch专家并精确重放。继续分离实际融合旋转、prefill内核、投影打包因素，尚未达到最终生成闭环。
- 工具问题：修复HF加载报告set序列化、诊断AITER导入路径及RoPE构造接口。工具进程无GPU，实际GPU任务由用户普通终端执行，不将预检阻塞当作推理失败。
- 验证/阻碍：初始短/边界token结果有通过，但扩展集仍分叉；当日不宣布Step0完成。部分文件已暂存，Git作者身份缺失导致提交未完成，未伪造身份。
- 下一步与证据：固定投影形状、完整prefill及同历史decode验证。见验证[05](validation/05_atom_mellum_model_wiring.md)、[09](validation/09_mellum_rmsnorm_rounding_fix.md)、[17](validation/17_official_backend_confirmed_and_step0_torch_moe.md)、[18–20入口](validation/20_stepwise_rope_result_and_projection_replay.md)。

## 2026-11-12

- 完成：同shape投影与QKV打包对照，separate QKV候选；CPU/GPU RoPE逆频率来源核验，采用CPU初始化，修正1024处显式mask/展开KV后完整prefill精确。保留实际分页KV，修正参考Q/K与K写回、cached SDPA；窗口首decode逐层追至索引8层router投影并局部修复，固定4例32位置精确。
- 普通生成闭环：HF同实例对照揭示generate在GPU decode临时grouped_mm→batched_mm，候选对齐后short128位置精确。十例中两处相同logits并列最大值选择不同，修复Sampler无过滤时绕过all_greedy的分流；`213129468143Z`十例640token及完整logits精确。
- 工具与验证：修正探针异步末尾额外forward、HF trace可变列表元数据，加入15秒心跳、死亡检测与超时清理。`221537256942Z`本地审计重验旧12例768位置、CPU49项通过，但当时>1024 prefill和多请求decode不支持；该结论是历史快照。
- 能力增强：随后移除越窗拒绝，实现HF语义causal band mask；Attention CPU8项、独立HF5项、运行器5项通过，window prepare-only完成。当天尚未取得新长输入GPU结果。
- 下一步与证据：验证1025，再到8192/16k。见验证[23–26入口](validation/26_prefill_passed_and_forced_decode_baseline.md)、[33](validation/33_router_projection_root_and_candidate.md)、[34](validation/34_cached_decode_exact_and_normal_generation.md)、[35](validation/35_greedy_tie_break_and_sampling_dispatch.md)、[36](validation/36_step0_acceptance_audit.md)、[37](validation/37_step0_long_prefill_extension.md)。

## 2026-11-13

- 长输入验证：`085534271528Z`的1023/1024/1025各8token，`085844629113Z`的8191/8192/8193/16384各4token全部精确；`103154043872Z`同四条长输入各64token，共256token及完整logits精确。16k已有实际GPU证据，不再受旧审计“不支持”结论约束。
- 规模验证：固定164条HumanEval prompt加36条场景A prompt，输入63–4113token；`104458011376Z`的200条各64token，200/200完整序列、12800/12800 token及逐步完整logits精确，200个prefill精确，5631加载审计通过。新增可配置阶段超时，运行器CPU5项通过。
- 阶段决策：核对题目≥90%门槛，承认完整logits/层状态精确是诊断工具，停止同类精度扩测。Step0适配开发完成、本地生成一致性通过，进入Step1；公开集身份/HumanEval pass@1待正式补证，不能将缺失材料断言为隐藏测试。64token含思考内容，不等于完整代码执行结果。
- 文档收尾：形成[39阶段总结](validation/39_step0_summary.md)、[05基线复现](../docs/plan/05_step0_baseline_and_reproduction.md)和当前工作树/产物指纹；清理过时导航与执行点、保存原文快照；将此前混在11月12日下的13日记录归入独立日期。只整理文档与离线证据，不启动新GPU任务或修改功能代码。
- 遗留与下一步：源码仍有未提交改动，HEAD不是已验证候选完整版本；固定版本后推进最小服务测量入口、所需批量能力与profiling，依据瓶颈选择优化点。HumanEval完整评估、官方口径、最终服务与性能交付仍需补齐。详见[37](validation/37_step0_long_prefill_extension.md)、[38](validation/38_step0_developer_200_agreement.md)、[39](validation/39_step0_summary.md)。

- Step1只读规划：重读原题/早期16.5方案并核对当前MoE、Attention、Graph、调度及profiler源码。形成[06候选路线](../docs/plan/06_step1_optimization_routes.md)：M/A优先评估，第三点由profile在Graph/chunked/prefix等候选中选择；明确单请求→服务批量桥接的B0对照、独立计点与正确性回归。只写方案，未改功能或启动GPU；冻结解释器未检出evalscope包，需核对发题客户端入口。

- Git落地：用户完成个人空仓库SSH认证并确认提交署名；保留上游769个可达提交，以非squash subtree建立总仓库，Step0源码/脚本真实快照提交00a75c9。备份旧Git、未提交文件与完整results；轻量509文件纳入Git，51501原始文件建立哈希库存。旧路径兼容迁移、当前文档/标签与网络限制见[07](../docs/plan/07_project_repository_management.md)。工具侧DNS/agent受限，远端首次push待用户终端执行；不补造历史开发提交。

- Git迁移验证：旧路径已链接至总仓库，65源文件和509轻量证据校验通过，181链接有效，Mellum/sampler CPU50项通过；源码数学未改、未启动GPU。完整results同机归档及清单已保存，远端push仍待已认证用户终端执行。

- Step1实验开发：分支step1/profiling-baseline，新增候选共享配置、受监控单请求profile运行器、GPU trace离线归因与Markdown摘要；worker增加opt-in启停回执和仅采集期间安装的标签，避免异步空trace与计时hook开销。固定1024/4096/16384输入，预热8、无profile两次64、trace16；CPU15项通过，兼容路径指纹问题修正后prepare-only172408完成，未启动GPU/未改数学。归档验证改为默认核对Step0标签。见[08实验协议](../docs/plan/08_profiling_experiment.md)。

- 首轮性能采集：180921三组bs1实际GPU profile通过；无profile TTFT均值217/663/3994ms，TPOT56/56/61ms。发现ROCm gpu_user_annotation镜像导致CPU区间重复，离线修正parser并保留原始结果，相关CPU7项通过；派生分析181952，不重跑GPU。16k prefill Attention约78.6%、MoE15.3%，decode Attention约51–66%，据此先评估Attention，未外推并发32。完整证据见[性能01](performance/01_bs1_profile_baseline.md)。

- 扩展性能评测开发：按用户希望增加深度与覆盖的要求，准备五长度×三输出档及真实A/C样本，默认19配置132次计时/22617输出token，另9短trace+1详细trace。新增逐层/子阶段/CPU self/语义op/拷贝归因、同预算profiling扰动对照、allocator峰值和只读SMI遥测；24项CPU通过，既有GPU trace离线检查与prepare-only190127通过，无新GPU任务或模型数学优化。命令、半小时量级预算与bs1边界见[09](../docs/plan/09_expanded_profile_study.md)。

- 扩展性能实测：190525三阶段完成，19配置132次无profile计时/22617输出token、9短trace+1详细trace，10份trace哈希及配对token核验通过，轻量证据归档performance/02。16K prefill SDPA约72.2%，SWA每层约122ms、Full约36ms；1K×256的50次TPOT均值55.006ms，后段由约54升至56ms，后续需交错计时。未重跑GPU或改模型数学；见[性能02](performance/02_expanded_bs1_profile_study.md)。

- 优化点1实现：根据扩展profile，新增opt-in AITER local SWA prefill，仅替换n≥window的窗口attention，保留reference RoPE/KV/decode/MoE及默认基线。新增HF回归开关、单层诊断及整模型ABBA计时工具；相关54项CPU通过，窗口和性能计划prepare-only通过。工具进程无/dev/kfd、cuda设备数0，未运行GPU、不宣称加速；命令与验收步骤见[10](../docs/plan/10_swa_prefill_optimization.md)。

## 2026-11-14

- 优化点1 GPU单层初测：用户执行013307 micro-only，五长度均完成。4K/16K窗口attention局部加速16.78×/40.07×，>=1024输出relative L2约0.57%–0.62%，未逐位精确；1023候选在整模型并不启用。归档原始JSON/log与源码指纹，见[性能03](performance/03_swa_prefill_micro.md)。下一步HF真实模型窗口/长输入对照，尚无整模型收益或准确率结论；本轮未追加GPU。

- 优化点1真实模型短回归：112559 window 3例×8及112653 long 4例×4，共7/7序列、40/40 token与HF相同；1023 logits精确，启用候选的其余32位置非精确，prefill max_abs 1–5.75。实际开关/正常采样/5631加载审计核对通过，追加[性能03](performance/03_swa_prefill_micro.md)及轻量证据。可进入ABBA性能，但不宣称64token/200条或HumanEval验收通过；未追加GPU和功能修改。

- 优化点1整模型ABBA：121938四阶段60条计时完成，16K TTFT约4.00→1.43秒（2.8×），4K约0.66→0.41秒（1.6×）；1K无明确收益，decode未见加速。1K/4K/16K分别第11/20/12个token首差，重复稳定；前64位置仅10/19/12个相同，故候选暂不采纳，默认Step0不变。详见[性能04](performance/04_swa_prefill_abba.md)及哈希归档；下一步同输入64token HF对照，未追加GPU。

- 优化点1分叉核验准备：新增同ABBA三条输入的HF/候选/Step0三方正常64token对照，严格校验来源并复用同一HF参考，保留与ABBA前缀的比较；同历史logits增加整体偏移、去均值RMS与top1间隔。相关10项CPU通过、prepare-only130643通过；GPU仍不可见，未改模型数学和验收门槛，待用户执行新命令。详见[计划10](../docs/plan/10_swa_prefill_optimization.md)。

- 优化点1三方HF对照：134033完成。Step0与HF三条×64共192token及完整生成logits精确；候选41/192token、0/3序列一致，首差为11/20/12，与ABBA前缀完全复现。首差双方均选自身非并列argmax，去均值RMS仍0.33–0.45，不能用统一偏移或采样器修复。详见[性能05](performance/05_swa_hf_triangulation.md)；候选不采纳，下一步少量真实Q/K/V单层oracle/窗口诊断，暂不扩大200条或重复性能，未启动新GPU。

- 优化点1单层诊断开发：新增显式首层prefill observer及1K/4K真实QKV回放，要求原SDPA输出逐位重现；对照AITER窗口±1、FP32 oracle/最终BF16舍入，并采短trace确认选路。35项CPU通过、prepare-only134928通过；模型默认不安装observer，未启动GPU，用户命令见[计划10](../docs/plan/10_swa_prefill_optimization.md)。

- 优化点1单层GPU证据：145253真实1K/4K QKV均精确回放原基线，正确窗口实际trace确认CK Tile。CK vs reference relative L2约0.00569，vs FP32约0.00558；reference vs FP32约0.00167。1K同causal可见集合仍有差异，4K窗口±1不消除，支持数值实现方向但未定位具体舍入步骤。详见[性能06](performance/06_swa_real_qkv_diagnosis.md)及哈希归档；候选继续关闭，下一步复用QKV筛替代实现，无新GPU或数学修改。

- 用户要求以题目门槛而非张量精确决定候选：准备复用HF参考的200条候选对照，拆分168短/32激活长输入（全部来自scenario A）；新HumanEval164提示词73–456，全不激活该优化。开发独立官方工具获取、4题完整生成与固定代码提取、成对pass@1/逐题变化及隔离评分自检，冻结推理环境未改。工具无法连接GitHub、Landlock/chroot能力缺失，未宣称工具安装/隔离已通过；提取5项通过、隔离1项跳过，200/pilot prepare-only通过。当前不启动全164长生成，命令/预算见[计划11](../docs/plan/11_representative_correctness_and_humaneval.md)。

- 代表性准确性实测：165437开发200位置11918/12800=93.1094%，完整序列175/200=87.5%；32条激活子集位置56.9336%，168条不激活全同。官方HumanEval commit b9453e8及本地164数据逐条一致已核验，181019 chroot/uid65534自检得到0.5。181033四题pilot双方5660token全同、3/4通过、差0pp，第10题均2048截断思考未结束。见[报告07](performance/07_representative_correctness.md)。据此保留CK实验候选，不用张量bitwise或三例分叉增加门槛；尚非正式公开集/全HumanEval通过，暂不启动全量长任务或新GPU。

- 一轮有界替代实现筛选准备：核对冻结AITER Triton default local源码，发现right必须-1，按causal+left1023映射1024窗口；新增筛选适配器（未接模型）、首key脉冲/GQA探针、1K/4K真实QKV的SDPA/CK/Triton轮转计时与短trace。相关22项CPU通过、prepare-only193745通过，未加载模型/执行GPU/改生产路径。只考察这一条替代路线，有优势才测32条激活生成；计划与停止条件见[12](../docs/plan/12_bounded_triton_swa_screening.md)。

- Triton有界筛选200217完成：边界/GQA探针、参考精确回放和实际Triton kernel trace通过。1K/4K relative L2为0.000576/0.0000161（CK约0.00569）；Triton耗时0.333/1.085ms，局部对SDPA1.62×/12.06×，慢于CK。接入独立impl=ck/triton选择，默认仍关闭优化；相关44项CPU通过，32条prepare-only204321通过，待用户GPU正常生成验证。见[性能08](performance/08_triton_swa_screen.md)，未执行新的GPU或切默认。

- Triton32条生成204611完成：实际impl=triton、正常64token、5631加载审计通过；HF匹配1262/2048=61.6211%，完整10/32；CK为1166/2048=56.9336%、7/32。逐题13改善/8下降/11相同，完整新增5条但回退2条。局部数值显著改善未转化同幅生成收益，且较CK慢，按时间边界结束本轮筛选，保留两实现不切默认；下一主线服务/批量，未追加完整200/ABBA/HumanEval。详见[性能09](performance/09_triton_activation32.md)。

- 下一阶段方案核查：已有OpenAI服务、异步提交与prefill/decode分批Scheduler可复用；Mellum参考prefill/decode显式单请求检查为主要接入缺口，批量GEMM/MoE shape和临时权重物化需验证。形成[计划13](../docs/plan/13_serving_batching_and_workload_baseline.md)：bs1服务→ragged桥接→2/4/8/16/32小集/容量→正式A/B/C及批量profile→第二算子优化。区分S0/B0/点1、正常服务配置、原始分页KV及真实并发；evalscope未安装，当前只读核查/文档，无新功能/GPU。

- 服务/多请求第一轮代码：新增opt-in ragged reference attention（bs1原函数回退、按请求raw RoPE切片、实际分页cache逐行decode），保留S0且拒绝不兼容缓存/强制探针；新增apply_serving、固定配置server入口、offline 1/2/4+反序复用/迟到请求验证、自动启动/关闭的HTTP completion/SSE/chat/并发smoke。59项CPU通过，prepare-only多组通过，依赖已在环境，GPU仍不可见，未实际运行/宣称通过。当前先跑offline bs4再HTTP bs1，命令与剩余边界见[计划13](../docs/plan/13_serving_batching_and_workload_baseline.md)。

- B0首轮GPU：214224实际prefill/decode batch4及迟到batch3完成，bs1四例全同、多请求234/240位置=97.5%，reverse全同；214721 HTTP bs1 completion/usage/SSE与两个客户端正常，chat只有reasoning开头换行差异。按题目指标保留批量差异，不新增bitwise阻塞。核查发现非流完成请求前端登记漏释放，修复文本/多模态/fanout成功完成路径，已有stream清理保留；32项CPU通过，HTTP脚本保留raw差异并分类reasoning空白、观察pending上界。详见[性能10](performance/10_batch_and_http_smoke.md)，下一步容量8→16→32及HTTP batch4，未运行新GPU。

- B0容量阶梯235336：真实decode8/16/32完成，无本轮执行错误；小集位置128/128、254/256、458/512（89.45%），单请求四例精确、迟到decode3及反序复用完成。容量成功不等于batch准确性验收，bs32同输入副本分叉待代表性检查。见[性能11](performance/11_batch32_and_http_batch.md)。

## 2026-11-15

- HTTP多请求032334：maxseq4配置实测decode2，顺序completion/SSE/usage均匹配；并发humaneval_0文本分叉、另一条一致，review仍保留。chat仅reasoning前导换行，归一分类通过；pending最大2支持正常完成登记修复。服务正常退出。下一步固定developer200真实batch对照再决定性能基准/有界定位，不直接宣称B0通过或启动完整A/C；本轮只分析/归档，无新GPU或数学修改。

- B0代表性200对照开发：新增固定200原token/HF缓存校验、bounded outstanding32补充请求、显式seq-ID/输入映射与乱序完成记录；B0 reference、正常64token/EOS，审计真实decode32，报告总体/长短/HumanEval/场景A双一致率，提前EOS保留分母。不重跑HF/不存全词表logits，15项相关CPU通过、prepare-only033338通过，未执行GPU/改数学。命令/决策边界见[计划14](../docs/plan/14_b0_batch200_representative_check.md)。

- B0 batch200 034222：200提交/完成、最多outstanding32、真实decode32，5631加载审计通过；未启用点1，位置9579/12800=74.8359%、序列88/200=44%，短输入77.56%、长输入60.55%。代表性开发门槛未达，暂不进正式性能基准。归档[性能12](performance/12_b0_batch200.md)，下一轮只做前32中少量最早首差、最多8输出的真实prefill/decode对照，区分逻辑与shape数值原因；未新增GPU或改数学，不追加bitwise门槛。

- 用户要求减少介入频次：形成[计划15](../docs/plan/15_batch_accuracy_campaign.md)，一次开发batch-aware探针、通用phase/linear/experts shape控制、自动DAG/分析/断点续跑；4规模+最多6控制+2代表性200+2回归/profile，最多14GPU worker，预算30–60分钟/硬上限90分钟。复用HF，按同历史首差分析，不改代码刷数据、不新加bitwise门槛、默认不跑完整A/C或HumanEval。当前仅方案，runner尚未开发，无新GPU。

- 计划15开发交付：新增默认关闭的真实请求shape控制与batch-aware观察器、异步dispatch前ID发布、缓存HF前8步logits核验、4规模+条件矩阵+最多2项D200+候选计时/真实子进程trace/HTTP的统一入口；累计90分钟与14worker预算、hash恢复、失败隔离和自有进程组清理。GPU未运行，默认后端/冻结库未改。CPU假worker覆盖完整14分支、参考复现失败止步、预算/恢复/损坏产物/失败重试、同历史过滤与head/cu语义；28项脚本CPU测试、58项ATOM Mellum回归（合计86项）及prepare-only通过，详见计划15交付记录。下一步用户只需运行总入口并回传根RESULT。

- 批次043721完整执行14个GPU worker、累计1888.67秒；独立核验585产物/232源码指纹及21组重复输出一致，D200每候选仅一轮。CPD两阶段L/E按请求控制达到developer200位置12392/12800=96.8125%、174/200=87%完整序列，长输入93.457%；CE小集改善未泛化（D20072.156%），弃用。K4与CPD HTTP小规模回归通过，默认未切换。独立无probe短负载B0/CPD均值5.489/13.405秒，慢2.442倍；真实子进程trace experts占累计kernel77.33%，下一步优先MoE发射/索引开销，不继续无界bitwise定位。轻量证据约4.4MB，见[性能13](performance/13_batch_accuracy_campaign.md)。本轮只分析归档，无新GPU或数学修改。

- 依据性能13开展MoE候选开发：拆分专家route/forward_routed并保留默认数学，新增批内共享路由、decode直接索引canonical权重的Triton GEMV两项opt-in候选，L和prefill专家请求shape仍按CPD。增加单入口微基准/小集/D200/短长ABBA/真实trace/HTTP条件批次，最多12worker/90分钟，复用hash恢复与自有进程组清理。102项相关CPU测试通过；CPU解释器6组地址/stride/repeat检查、gfx942三组离线编译通过（private segment0），尚无GPU数值/性能结果。详见[计划16](../docs/plan/16_moe_routing_and_indexed_decode.md)。默认服务和冻结库未改。

- MoE批次121305实际10worker/1739.80秒完成；独立核验169产物与236源码指纹、重复小集及ABBA稳定，CPD重构后D200与历史200/200相同，R共享路由与当前CPD也200/200相同，维持位置96.8125%/完整序列87%。R短/长ABBA加速1.09154×/1.04675×、几何均值1.06891×，实际子进程trace与HTTP小规模回归通过。I直接索引decode32局部4.892×、额外分配64.20→3.71MiB，但小集94.34%/93.95%相对CPD下降超过预设2个百分点，未跑D200；不等同题目90%失败。约3.6MB证据归档见[性能14](performance/14_moe_routing_and_indexed_decode.md)。默认未切换，本轮仅分析归档；下一步建议I一次固定D200，不重复整批或无限浮点定位。

- 按性能14建议准备indexed_decode一次D200补证：新增独立入口，核对与上批相同ATOM数学源码/配置、200输入hash及HF/CPD/R缓存；仅一个GPU job、正常greedy64与实际indexed_triton/batch32审计。分支只看总体位置≥90%，序列/长短/来源单列，不使用原CPD相对下降筛选，不自动追加性能或服务任务。CPU入口/恢复/判定测试及prepare-only通过；候选数学未改，GPU尚未运行。详见[计划17](../docs/plan/17_indexed_decode_d200_followup.md)。

- indexed D200补证133034：一个GPU worker/227.38秒完成，200条全部64token；10产物/237源码及226条与上批一致的推理指纹核验通过，实际indexed_triton decode32/8正常，5631加载审计通过。总体10013/12800=78.2266%、完整107/200=53.5%，短81.2593%/长62.3047%；当前版本明确未过本地90%位置门槛。相同200输出前16/32/64token匹配95.34%/89.16%/78.23%，说明短生成不能外推完整64。按计划停止I版本晋级，不追加ABBA/HTTP/功能长测，保留R与CPD，下一主线prefill专家GEMM/调度开销。见[性能15](performance/15_indexed_decode_d200.md)。本轮只分析归档，无新GPU或数学修改。

- 按题目转入A/B/C基线接入：重新提取PDF第4–6页命令，核对官方EvalScope源码发现A的dataset路径/注册插件及JSONL请求语义需适配，C180秒为每请求超时。原始A512/C200 prompt均含换行；A裸507/chat505条满足512–4096，不能静默丢弃或改长度。开发独立客户端安装器（evalscope1.15.0、客户端TF5.10.1）、实际安装源码/请求/B随机长度/SSE reasoning计时审计、CPU协议manifest及稳定R配置。工具网络Operation not permitted且本地未安装，真实安装需用户终端；GPU runner须待真实发布包审计后完成，未冒称已可正式测量。见[计划18](../docs/plan/18_evalscope_client_and_abc_baseline.md)。本轮无GPU或推理数学修改；12项协议/服务CPU回归及prepare-only通过，实际发布包探针待用户终端安装后运行。

- 用户安装客户端第二次成功144518，冻结推理依赖清单前后不变；实际reasoning SSE/usage/TPOT探针通过。核对发布版源码后新增独立prompt-field dataset插件，保留原生过滤/包装/循环及调度/指标：A505有效循环512、C完整200，原生请求序列CPU核验一致；B原生50条均1024token。开发smoke→独立服务B50/A512/C200单入口、SQLite请求/长度/预算核验、nearest-rank分位、失败保留、阶段hash恢复/累计3小时预算；25项相关CPU测试通过。C指数间隔归一化及客户端版本差异明确披露，不冒称裁判同版。见[计划18](../docs/plan/18_evalscope_client_and_abc_baseline.md)实施更新/[性能16](performance/16_abc_protocol_and_client.md)。工具无GPU且loopback禁止，真实HTTP/GPU待用户一次启动，未修改推理数学或默认配置。

- 首轮ABC smoke 152546：5631张量加载成功，客户端解析阶段因1.15不接受`--stream True`退出，未发起测量请求；修正为等价`--stream`，新增真实已安装解析器A/B/C回归，相关26项CPU测试通过。下一次新批次运行，保留旧失败证据。

- GPU0 R配置ABC `20261115T154407018497Z`流程结束，累计85.9分钟：B50/50、A512/512成功，A41.5797tokens/s/GPU为官方6.5143%；C仅1/200成功，其余客户端180秒超时，服务无日志崩溃/OOM证据。状态complete_with_failures不代表性能通过，C成功子集p99不可用；独立94产物/231源码hash通过，归档性能17。未追加GPU或修改数学，不原样重跑ABC，优先显著性能瓶颈。
- GPU1计划19批次`20261115T170157802373Z`完成，4worker累计16.57分钟：候选D200与R200/200输出完全相同（HF位置96.8125%）；配对ABBA短1.00666×、长0.98934×、综合0.99796×，无实质收益，停止候选，不追加ABC/HumanEval。独立90产物/237源码指纹通过，归档性能18；实现保留独立分支c22198d，原配置未切换。

- 按用户要求转向成熟实现复用调研：冻结Qwen3MoE/FusedMoE/gfx942 BF16与分页Attention，公开ATOM main、vLLM v0.23.0 Mellum和SGLang Mellum均做源码对照；发现当前已有aiter MoE开关但与R Torch包装器不兼容，参考ragged桥阻止原生batch/缓存路径。输出研究05、来源清单和计划20，优先整批FusedMoE+原生Attention独立候选，不继续外围小优化；明确未取得上游HEAD/穷尽PR，不能冒称最新完整快照。本轮无推理代码修改、安装或GPU运行。

- 原生主路径代码在独立commit cb09aed交付：GPU0工作区tencent-mellum-native、GPU1同快照tencent-mellum-native-attention；整批FusedMoE保留Torch gate/路由（BF16舍入后FP32传AITER），原生Attention保留参考RoPE/KV写入、替换varlen/paged计算。117项相关CPU检查通过，两份prepare-only通过，计划21给出命令/60分钟预算；主R工作区只同步文档，冻结库和主R推理源码未改，GPU尚未执行。
- 原生两卡批次完成：MoE184205累计26.82分钟，批量1.4663×、D200位置72.36%；Attention184214累计26.45分钟，批量1.4507×、D200位置71.38%。均低于90%不晋级。MoE256重复S1/S32/L32分别1/1、31/32、29/32输出变化，Attention全相同；尚不能据此断言具体kernel根因。98产物/476源码hash通过，性能19归档；无追加GPU/数学修改。

- 计划22有界诊断入口交付：独立GPU0固定layer0/路由12次Opus/CK回放、stage metadata和短trace（15分钟）；GPU1仅prefill/仅decode各D200，达到90%者才256性能（45分钟）。120项CPU检查及195403/195402两份prepare-only通过，53b6833及排序前缀补丁1cac5e2/c300a25；主R源码未改，无新GPU实测。
- 计划22完成：MoE195607累计16.28秒，Opus/CK固定排序下三shape的12次输出均出现重复差异；metadata确认CK two-stage，trace和冻结CK源码确认输出AtomicAdd路径，尚未单独证明全部差异由stage2造成。Attention195614累计8.86分钟，仅prefill位置72.55%、仅decode77.25%，均不达90%，按计划跳过性能。34产物/480源码指纹通过，归档性能20；不原样重跑或追加排序D200，后续若修MoE聚焦贡献输出/归并，不默认继续数值搜索。

- 计划23交付：独立分支step1/moe-isolated-reduction、commit ee1215c，top8贡献展开为native top1独立输出，外部BF16加权/求和；不复制权重，增加按N*8增长的临时存储。126项CPU检查、py_compile/diff和202806 prepare-only通过；入口GPU0/8526、最多4worker/45分钟，固定回放稳定/finite才D200、达到90%才R/候选256性能。主R仅同步文档，未执行新GPU。
- 计划23运行203133完成，累计3.84分钟：独立贡献归约三组固定回放各12次精确，消除所测重复变化；D200位置74.18%、完整序列45.5%，相比原native MoE总体+1.82pp但长输入降至52.98%，仍低90%。加载/路径/真实batch32通过，按门槛未跑性能，不套用旧候选加速比；17产物/241源码hash通过，归档性能21。无追加GPU/数学修改。

- 用户确认HumanEval关闭thinking/最多2048token优先控时；交付独立分支step1/humaneval-functional-evaluation、7d6a4ce，一条命令HF/GPU0与FusedMoE/GPU1真实batch32并行164题、统一提取/官方隔离评分，90分钟总预算，保留截断/提取失败和逐题功能变化，不以token位置低于90%阻挡评分。10项CPU通过、1 Landlock检查工具受限跳过（终端仍强制self-test），164输入/模板和205642 prepare-only通过；未运行新GPU，冻结库/原R源码未改。

- HumanEval原批次候选164题完成，HF因loading_info的set序列化失败，完整配对尚未得出。在独立delivery提交df33885修复，6项CPU检查及恢复prepare-only通过，HF-only recovery可复用原候选。独立next提交a2cbcce增加显式服务profiles与C前缀CPU分析，15项CPU检查通过；200条C共有3820token，按16token块可复用757792/2039880=37.15%，非实际命中率或加速承诺。参考K覆盖晚于prefix gather，暂不启用缓存；计划25/研究06归档，无额外GPU任务。

- HF恢复运行期间准备GPU1候选HTTP批次：next提交9a3b3ca，复用已审计EvalScope与原ABC worker，显式fused_moe/HTTP8151/engine8726，smoke→A64并发32→长度分层C20率0.3，35分钟执行预算。24项CPU测试及实际输入prepare/current_matches通过，GPU未运行；计划26已归档。保持HF运行工作区不变。

- HF恢复20261115T213706098181Z完成：HF145/164=88.4146%，FusedMoE147/164=89.6341%，绝对差1.2195pp，本地2pp检查通过；增5失3，预算截断4/5，提取失败5/6，均保留164分母。独立核验逐题结果与评分，归档性能22及来源hash，不等同整体题目验收；继续GPU1计划26性能主线。

- GPU1容量20261115T220256351771Z完成：smoke/A64/C20均全成功，A57.0039tokens/s、实际batch32；C TTFT p99 22.6893秒、TPOT p99 0.44394秒、batch19。独立核验产物与源码、5631加载、native调用；归档性能23。用户要求以更多实现/实测效果为主，不以计点讨论阻塞开发；下一步现成Fused+SWA组合与prefix兼容开发，最终只提交最多三个独立点。

- GPU1组合测试期间在独立prefix分支80cd14c开发cached-prefill：reference K先写后gather、共享metadata副本、bottom-right矩形causal/SWA mask、ragged后缀支持；NHD按实际逻辑token读取避免整个缓存池连续化。GPU0 off/on小试验单命令、25分钟预算、三条真实C、每请求16输出，不重跑HF。57项CPU检查与实际输入prepare-only通过，GPU待终端执行；未修改GPU1源码。

- GPU1组合20261115T223459375865Z完成，smoke/A64/C20全成功；A59.612tokens/s较单项+4.58%，C TTFT p99 22.689→3.819秒、TPOT p99 0.44394→0.36259。核验同输入、产物与源码hash，归档性能24。GPU0 prefix继续运行不改；下一步GPU1筛选FusedMoE+完整原生Attention，性能优先，有收益后复用HF作候选功能评估。

- GPU0 prefix_probe 20261115T225622969152Z通过，8次/128token off/on全部一致、实际命中46592；共享ragged5.608→3.571秒、逆序复用5.571→2.108秒（非HTTP收益）。核验源码/加载/命中，归档性能25。服务新增显式prefix开关及逐case命中审计，GPU0独立8251/8926跑FusedMoE+prefix容量计划28，33项CPU检查及实际prepare/hash通过；GPU1 next原生Attention继续运行未改。

- GPU1原生组合20261115T231450301731Z完成，smoke/A64/C20全成功；A121.4966tokens/s，约SWA组合2.04倍；C TTFT/TPOT p99 1.7263/0.13478秒。审计aiter_varlen/paged_attention_triton与FusedMoE、同输入/产物/源码，归档性能26。独立native-functional工作区开发仅GPU1复用HF的完整HumanEval入口，8项CPU检查；GPU0 prefix继续，未改其源码。

- GPU0 prefix HTTP 20261115T231925241791Z完成：A64/C20全成功，C19/20命中72352token，TTFT p99 19.5069秒、TPOT p99 0.41837，A57.0946tokens/s。核验输入、源码、产物与真实命中，归档性能27；用户同时回传的231450原生Attention是已分析批次，不算重复实验或HumanEval完成。下一主线仍原生组合功能评估，后续prefix接原生路径。

## 2026-11-16

- 原生Attention+FusedMoE完整HumanEval 20261116T010456387933Z完成：候选144/164=87.8049%，复用HF145/164=88.4146%，绝对差0.60976pp，本地2pp检查通过；增2失3，预算截断4/5、提取失败5/6，均保留164分母。核验双方逐题/评分、原参考相同、源码和5631加载；124次原生prefill/2926次paged decode确认路径。性能28归档，结合性能26保留高速主线，下一步prefix与原生Attention整合，不追加无界token对齐诊断。

- 按计划交付两项并行实验：独立native-prefix c34a96a接入原生cached prefill，验证Q/K独立边界/真实后缀位置，保留K写回/gather，自动GPU0结构内核探针→off/on→HTTP A64/C20，45分钟预算，token分叉只报告；独立native-abc cd2c3e3支持显式高速profile与隔离端口的完整B50/A512/C200、90分钟预算和恢复。72/22项CPU检查、py_compile/diff、真实数据prepare-only及最终源码/输入/客户端指纹通过；未运行GPU、未改冻结库或既有已验工作区。计划30与交付清单归档，最终正式复测仍需空闲机器。

- 两卡批次完成：native-prefix 20261116T025408685476Z，104产物/源码核验，kernel结构全精确，off/on4/8序列同，C20命中19/20、72352token，TTFT/TPOT p99 1.12319/0.12185秒，归档性能29；native-ABC 20261116T025416247034Z，98产物/源码核验，47.37分钟，B50/A512/C200全成功，A121.2755tokens/s为官方19.0%，C TTFT/TPOT p99 2.54046/0.21940秒，归档性能30。未把相对R提升约2.92倍当官方加速。下一步根据campaign L仍逐请求拆分QKV/O/router的源码证据，独立native-batched profile恢复整批投影，16项CPU检查及数据prepare，性能优先、有收益再复用HF评估。

- GPU1 native-batched运行期间，独立prefix-functional提交990f38a补原生prefix组合的完整HumanEval入口：候选固定GPU0/9126、164题、关闭thinking/2048、复用HF，记录实际缓存命中；10项CPU检查和真实引用/源码hash prepare核验通过，GPU待终端执行。并行只读研究07定位Graph/CPU后续边界：空campaign触发eager guard、参考K动态布尔索引、重复RoPE/KV写入、审计tolist先求值；未改GPU1运行源码，不预先宣称新收益。

- GPU1整批投影20261116T043117581045Z完成，84产物/源码核验，与性能26模型源码内容及输入完全相同；A64全成功439.6727tokens/s、TPOT p99 0.05862秒，约原生L控制组合3.62倍；C20全成功、TPOT p99 0.08037秒。性能31归档。独立batched-functional扩展功能入口--batched-projections，GPU1/8626复用HF164，22项CPU检查与真实输入/引用hash通过，GPU待终端；GPU0 prefix-functional保持不变。

- GPU0 prefix HumanEval 20261116T044507881892Z完成：147/164=89.6341%，对HF145/164差1.2195pp，本地功能门槛通过。预算截断/提取失败双方4/5，均保留164分母；独立核验官方逐题/score、HF复用、源码/5631加载及128次native cached事件、132请求/2128token命中。性能32归档。此次仍groups=L，不能赋给整批投影候选；用户同时回传043117是已归档性能31，不作为新一轮实验。下一步仍计划33整批投影功能结果。

- 整批投影HumanEval 20261116T085201556563Z完成：149/164=90.8537%，HF145/164=88.4146%，高2.4390pp，增6失2；预算/提取失败候选3/3、HF4/5，164分母完整。核验源码、原参考、官方逐题/评分、加载与groups为空。原脚本按绝对差给False，保留原判定并补方向解释（observed drop=0），不改门槛、不为降低分数调实现；性能33归档，继续整批主线和prefix整合，正式口径待裁判确认。

- 按计划交付两卡新批次：GPU0 batched-prefix e4cfb91整合整批投影与原生缓存，74项CPU检查/真实数据prepare/源码输入客户端指纹通过，45分钟自动结构探针→off/on→HTTP；GPU1 direct-kv b4e77d9新增参考K/V单次AITER写入，保留BF16 NHD/SHUFFLE与QK公式，拒绝缓存/融合norm等未支持配置，59项CPU检查（含跨块/负slot/非连续V/非法shape）及真实prepare/指纹通过，40分钟自动布局探针→HTTP。未开Graph、未运行GPU或修改冻结库；计划34与交付指纹归档，导航顶部更新到当前阶段。

- 两卡批次34完成：GPU0 batched-prefix 20261116T101213244110Z，A64 438.8336tokens/s近似持平、C20 TTFT/TPOT p99 1.01427/0.07378秒、19/20命中72352；结构探针精确，off/on3/8序列同如实记录，功能待整体评估。GPU1 direct-kv 20261116T101221209827Z，布局/跨块/负slot/strided V GPU探针全部精确，A64 499.6429tokens/s（较439.67约+13.64%）、C20 TPOT p99 0.07040秒；实际prefill/decode单次写入审计通过。独立核验产物、源码、输入与5631加载，归档性能34/35。下一步合并有效机制，补整体功能；有界试decode Graph，再固定配置完整ABC。

- 根据用户对朴素Baseline差距的质疑，只读审计当前执行路径并归档研究08：L拆分和重复KV已有3.62倍/+13.64%实证；仍有113处手写Norm、3路F.linear+cat、每层CPU inverse-frequency上传及审计tolist。确认官方eager，不能把缺Graph当全部差距理由；建议下一批优先短trace和基础快路径恢复，保留协议/硬件干扰核对，不猜测官方源码或宣称已达硬件上限。无新GPU或代码修改。

- 根据研究08重排优先级，交付独立fastpaths分支68a4086/计划35：默认保留reference，新增113处Norm工厂切换原ATOM RMSNorm、既有packed QKV、GPU inverse-frequency驻留及组合profile；实际构造类/投影属性/驻留device单独审计。GPU0短bs1/32 trace及基线/驻留HTTP，GPU1基线/Norm/packed/组合四项HTTP；一次准备复用只读输入、独立失败保留，组件失败跳过组合，不重跑HF/完整ABC。90项相关CPU检查、py_compile/diff、真实两track prepare及全部job源码/输入/客户端指纹通过。未运行GPU、未升级依赖/修改既有装饰模型；Graph后移至基础路径实测后再决定。

- 核对原始题目第2、6–8页与当前仓库，形成计划36：保留真实非squash历史，后续独立集成交付版本；当前main/缓存远端仍为Step0、实验分散分支，缺统一启动入口、最终消融及4页报告/1页AI说明。明确完整ABC与开发缩减负载、最终组合与组件证据的边界，更新陈旧首页状态。本轮仅文档审计，未改运行代码、合并分支或推送。

- 基础快路径两批结果核验532项哈希/5631加载完整，GPU1组合A64达到625.616tokens/s，较同卡507.7752提升23.2%；C20 TPOT/TTFT p99为57.912ms/1.02807s。按用户要求暂停逐项完整正确性，独立sprint提交3f81b6d实现空观察器移除、既有decode Graph接入与实际replay回执、原生路由及Packed消融；27项CPU检查通过，准备两卡A64/C20批量实验。未运行GPU或预先宣称新收益。

- 优化sprint两批完成，266项产物哈希一致且全部请求成功。GPU0clean620.9951→Graph943.2600tokens/s（+51.9%），C20 TPOT p99降至22.841ms，真实回放证据完整；GPU1关闭Packed退化、原生路由约+2.8%。独立fusion分支2dc21b2实现Graph路由ABBA和Triton仅RoPE/QK-Norm+RoPE两级候选，37项CPU检查、两专门化gfx942离线编译通过。局部GPU数学冒烟后直接A64/C20筛选，完整模型正确性留最终组合，未运行GPU或宣称新融合收益。

- 当前fusion两卡运行期间，独立scheduling分支12087e7完成Graph+direct-KV前缀兼容桥接、prefix开关及2048/4096 chunk内部调度候选。复用此前矩形Attention与缓存逻辑，SHUFFLE调用原生gather、NHD只读本批逻辑token，避免整个KV池转换；40项CPU检查通过，含跨块、写后gather及调度配置，准备CPU数据。未占GPU、未改fusion运行工作区，完整正确性留最终组合。

- fusion两批完成并核验322项哈希/5631加载/Graph回放及后端：GPU0路由ABBA均值939.49→1008.94tokens/s（+7.39%），GPU1基线941.63→RoPE1032.72→QK融合1062.46（+12.83%），C20 TPOT p99至18.639ms，局部数学冒烟通过。scheduling分支ee3e238准备QK+路由组合ABBA及QK基线上的prefix/chunk2048/4096四项筛选，41项CPU检查通过。保持完整正确性集中到最终组合，不按单项收益叠乘或宣称正式ABC通过。

- 组合/调度两卡运行期间，独立campaign工作区准备prefill元数据复用（当次ForwardContext、默认关闭，处理inference tensor版本缺失）、实际复用回执及完整ABC多profile批次。复用cd2c3e3的profile/端口/resume CLI，完整批次独立stage失败继续、成功产物hash校验后跳过、失败attempt保留；40项相关CPU检查联合ATOM测试夹具通过（含新增恢复及元数据检查）；单独收集scripts测试缺少ATOM轻量导入夹具，改用完整相关检查组复验通过。未占GPU或修改scheduling运行目录；等当前结果选择候选，不默认扩大量化/MTP实验。

- combine/latency两批核验369项哈希且全部请求成功。QK+原生路由ABBA吞吐1124.91/1127.32，相对QK均值约+8.26%；prefix C20 TTFT/TPOT p99为0.75063s/15.9079ms，有19次/72352token复用；chunk2048/4096吞吐与尾延迟均退化，停止晋级。campaign提交22597b2交付GPU0最快组合完整B50/A512/C200，GPU1最快组合/prefix/prefix+metadata三项最后筛选；40项CPU检查通过，等待GPU执行，不逐项跑HF。

- 完整ABC/finish运行期间，独立acceptance分支25350e6准备最终profile的一次加载验证入口：复用已核验HF200/长组/HumanEval164，保留Graph、路由、融合及缓存开关，原始IDs避免HTTP再分词；按题目位置率和HE功能分差独立报告，无tensor精确门槛。44项CPU检查及实际参考输入准备/指纹通过，冻结候选环境、评分沙箱和HE164分母不变。未占GPU或改campaign运行目录，待最终候选确定后运行一次。
