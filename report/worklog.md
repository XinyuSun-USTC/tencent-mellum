# 工作日志

按自然日归纳，每天一个简要条目。日期依据既有日志、对话和结果目录的UTC运行ID；D0/D1是旧研究批次，不自动等于不同工作日。重构前全部追加记录已[保存快照](archive/step0_document_snapshots/report/worklog.md)，SHA256见[清单](archive/step0_document_snapshots/manifest.json)。当前有据可补日期为2026-11-10至13，不补造更早日期、时刻或提交；D1–D7映射待考核起始日确认。

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
