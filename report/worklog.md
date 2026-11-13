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
