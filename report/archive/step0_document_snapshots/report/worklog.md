# 工作日志

按自然日期记录，每天一个简要条目。D0/D1旧文件是研究批次，不对应不同日期。下列首次条目依据已有记录在文档重构时汇总，不补造精确时刻或提交；最终若需D1–D7标签，再以实际考核起始日映射。

## 2026-11-10

- 完成：环境准备记录；ATOM架构、配置/加载链、参考模型及Mellum/Baseline调查；五分片header与5631个张量布局核对；原题PDF阅读；容器正确性资产审计；本轮文档拆分、排序、归档和导航重构。
- 发现与决策：未找到完整正式正确性套件，旧HumanEval结果已rejected；最快正确复用完成Step0，不刻意复刻官方Baseline；适配与验证同步，HF为主参考，Step1后置。
- 验证状态：既有Tokenizer/template与配置检查有记录；尚未完成Mellum功能适配、完整权重加载、GPU推理或HF对齐。本次仅文档操作及迁移/链接检查。
- 问题与下一步：本地复核和官方来源调查现已完成，接下来固定官方源码并实施工作区最小参考桥接及CPU验证；随后推进必要适配与分层验证。
- 详细记录：[环境](research/01_environment.md)、[ATOM](research/02_atom_architecture.md)、[模型](research/03_mellum_model.md)、[参考调查](research/04_correctness_reference.md)、[当前方案](../docs/plan/01_step0.md)、[决策](../docs/plan/02_decisions.md)。

- 当日后续：重新执行离线CPU探针，确认实际Transformers5.2.0无Mellum注册/类、AutoConfig失败；读取官方v5.10.1实现和发布说明，形成配置生命周期、mask cache_position与expert转换兼容清单。保存[脚本](../scripts/probe_transformers_mellum.py)和[原始结果](../results/transformers_mellum_support/local_probe.json)。下一步转为官方参考最小桥接CPU验证；尚未实施或启动GPU。

- 当日后续：用户确认独立新版HF参考环境；创建venv与安装/检查脚本，确认复用原ROCm PyTorch、CUDA未初始化。PyPI包下载受网络限制，尚未安装5.10.1、未运行目标模型；见[环境建立记录](validation/01_hf_reference_environment.md)。下一步完成包获取及参考环境验证，优先路线不再是源码回移。

## 2026-11-11

- 完成：用户手动执行参考环境安装；复核独立HF5.10.1配置解析、模型类导入和必需依赖，检查原环境仍为5.2.0。
- 验证状态：配置/导入通过，无必需依赖缺失或版本冲突，未执行模型forward、加载权重或GPU任务。
- 下一步：小模型CPU forward与KV cache检查，随后真实checkpoint参考推理。
- 记录：[HF参考环境](validation/01_hf_reference_environment.md)。

- 当日后续：新增独立HF CPU回归脚本，43,936参数随机Mellum模型24/24通过，覆盖cache/分块、SWA边界、padding、eager/SDPA、greedy和逐expert加载转换。未加载真实checkpoint，CUDA未初始化，ATOM代码不变；见[CPU验证](validation/02_hf_mellum_cpu.md)。下一步准备真实权重单卡BF16参考推理。

- 当日后续：准备真实权重单卡BF16/SDPA加载审计及短greedy cache对比脚本；工具进程无/dev/kfd或/dev/dri、设备数0，预检以blocked_no_gpu退出，未读权重payload。需普通容器终端执行；见[GPU参考验证](validation/03_hf_mellum_gpu.md)。

- 当日后续：用户首轮真实权重加载在HF报告JSON序列化处失败，尚未进入GPU推理；修复set序列化、原子保存与异常兜底。真实小模型CPU加载报告及非空集合/保存失败回归通过，等待普通终端重跑。

- 当日后续：用户普通终端真实12B HF GPU smoke通过，已复核原始JSON：参数12,149,923,072、加载审计无缺项，两条各16token greedy cache开/关完全一致。参考执行路径已建立，尚无ATOM对比和长上下文/正式验收；见[GPU验证](validation/03_hf_mellum_gpu.md)。

- 当日后续：复核配置、Qwen3-MoE骨干、逐层Attention窗口、AITER YaRN及服务token入口，形成[详细适配与验证计划](../docs/plan/03_atom_step0_implementation_and_validation.md)。明确A–F批次、源权重审计、HF/ATOM隔离执行与teacher-forcing比较；本轮仅文档，未实施功能或GPU任务。

- 当日后续：实施A+B，新增Mellum配置接入，12项配置测试、32字段官方HF比对、14项比较器测试及46项既有量化测试通过。转换两条HF真实参考为统一artifact。Kimi既有mock错误在原始/app基线复现；本批未运行GPU，ATOM执行模型未实现。见[第一批验证](validation/04_atom_step0_batch_ab.md)。

- 提交状态：A+B的ATOM三个文件已暂存，Git作者身份缺失导致提交未完成，已询问用户姓名/邮箱；未伪造身份。

- 当日后续：新增Mellum Attention/Decoder/骨干/wrapper、执行注册、RoPE比例桥接与逐分片/专家加载回调审计。配置/模型21项、既有量化46项CPU测试通过；AITER实际cache源码对HF边界位置FP32误差0。GPU预检因工具无设备退出，准备check_atom_mellum.py供普通终端执行；尚无ATOM真实加载/推理结果。见[模型接入](validation/05_atom_mellum_model_wiring.md)。

- 当日后续：用户普通终端ATOM真实运行成功，复核5631/5631加载回调槽、无缺项，两条各16token与HF精确一致；引擎正常关闭。首个短输入闭环成立，SWA边界、长输入、多请求及正式验收仍待验证；见[GPU闭环](validation/05_atom_mellum_model_wiring.md)。

- 当日后续：建立short64、window1023/1024/1025、long8191/8192/8193/16384固定token开发集，准备HF→ATOM顺序运行与prefill logits比较；配置/模型22项、脚本20项CPU测试通过。GPU未执行，下一步普通终端先运行short；见[边界套件](validation/06_mellum_boundary_suite.md)。

- 当日后续：64token回归失败（代码第38、算术第7新token），prefill logits已不同；复核新旧HF参考逐元素一致，暂停长输入。准备4条同前缀探针与双端层输出诊断；配置/模型23项、运行器/诊断5项CPU测试通过，尚未GPU执行/修复。见[分叉诊断](validation/07_mellum_short_mismatch_diagnosis.md)。

- 当日后续：分析同前缀分层结果，四例embedding精确一致，第0层起误差累积，final norm相对L2约9%–13%。发现代码探针HF cached/prefill下一token自身不同，算术HF内部一致。准备第0层norm/QKV/Attention/router/MoE与9项权重观察；24+5项CPU测试通过，根因未定、未修改数学路径。见[子算子定位](validation/08_mellum_layer0_diagnosis.md)。

- 当日后续：第0层选定9项权重精确一致，input norm输入一致但输出不同；旧ATOM输出99.9976%–100%符合FP32乘权重后截断模型。按官方顺序新增Mellum专用RMSNorm和残差接口，CPU26项通过；真实输入对官方CPU精确一致。GPU回归未执行，不宣称全部64token问题解决；见[RMSNorm修正](validation/09_mellum_rmsnorm_rounding_fix.md)。

- 当日后续：RMSNorm修正后GPU短64回归128token全部匹配，首层input/Q norm精确一致。prefill logits仍有1.75/2.5最大绝对差、末norm约7.7%–10.5%相对L2，未宣称数值验收通过；下一步window边界，继续追踪剩余差异。见[GPU回归09](validation/09_mellum_rmsnorm_rounding_fix.md)。

- 当日后续：window真实输入1023/1024/1025各8token，共24token与HF一致；prefill最大差2.25–2.5仍需审查。不能据此排除mask差1；下一步long8191/8192/8193/16384。见[边界套件06](validation/06_mellum_boundary_suite.md)。

- 当日后续：long8191/8192/8193/16384各4新token均与HF一致；结合short/window共9个开发输入168新token匹配。logits最大差2.75–6.0未闭合，长输入输出仍短；整理[验证覆盖总表](validation/10_step0_validation_status.md)，未宣称Step0完成或进入优化。

- 当日后续：准备11条扩展开发输入（代码/数学及1025/8192/16384边界，各最多64新token），增加同输入HF/AITER路由重放与token统计。保存的ATOM路由Top-8边界并列为9/24、6/18行，尚未确认根因；CPU24项通过，未改模型数学路径，GPU待普通终端执行。见[扩展验证11](validation/11_mellum_expanded_and_router_replay.md)。

- 扩展11条704新token仅3条完全一致，8条分叉；四份同输入路由专家集合一致且重复稳定，权重精度差约0.000976但根因未定。新增按名称选择短探针，CPU3项通过；下一步h_0/h_1原始及公共前缀子算子诊断，见[验证12](validation/12_expanded_mismatch_and_router_findings.md)。

- 四条短探针两条整段prefill生成不同；h_0存在HF cached/prefill自身差异，h_1 ATOM最高logits并列。首层norm/Q精确、Attention及MoE已有差异。准备固定HF输入的首层完整MoE重放，诊断状态与正确性状态分离；CPU27项通过，GPU待执行。见[验证13](validation/13_short_probes_and_moe_input_replay.md)。

- 同输入完整MoE重放四例输入/router精确一致，输出相对L2约0.446%–0.450%。准备原生路由权重BF16舍入及HF路由两组诊断；CPU28项通过，正常模型路径未改，GPU待执行。见[验证14](validation/14_moe_same_input_and_routing_study.md)。

- 两组路由重放完成，仅小幅降低完整MoE误差，HF完整路由仍约0.41%–0.43%。准备独立第0层专家算术六组合诊断，语法/CLI检查通过；未改变正常ATOM路径，GPU待执行。见[验证15](validation/15_moe_routing_results_and_expert_arithmetic.md)。

- 专家六组合显示HF激活+FP32累加最接近保存HF（相对L2约4e-5）。发现此前遗漏MellumExperts装饰器：默认尝试grouped_mm，正文eager不一定执行。补充实际后端元数据与官方grouped_mm/eager直接重放，CPU24项通过，GPU待执行。见[验证16](validation/16_expert_reduction_and_hf_dispatch.md)。

- 官方grouped_mm四例逐元素复现保存HF，eager不能复现。接入Mellum专用PyTorch grouped-mm正确性路径，保留w13/w2加载及原AITER入口；CPU31项、官方CPU六组合精确通过。默认切换torch，GPU固定输入重放待执行，未宣称生成修复。见[验证17](validation/17_official_backend_confirmed_and_step0_torch_moe.md)。

- 新PyTorch MoE GPU固定输入重放通过：四例输入/router/完整MoE输出全部精确一致，torch后端和5631/5631加载审计已核验。仅首层诊断闭环，下一步无干预short64带层/子算子正常回归。见[验证17更新](validation/17_official_backend_confirmed_and_step0_torch_moe.md)。

- 新torch后端无干预short64正常回归128/128新token匹配。首层误差下降，整体仍非全面改善（代码题logits RMS升高，算术下降），状态仍需数值审查。下一步重跑expanded，不重复路由和长输入dump。见[验证17更新](validation/17_official_backend_confirmed_and_step0_torch_moe.md)。

- 新torch expanded仍7/11分叉、4条全匹配，部分样例改善但binary_search/long_8192出现新分叉，Step0未完成。准备四个短探针同QKV RoPE/SDPA诊断，语法/CLI通过，GPU待执行。见[验证18](validation/18_torch_expanded_and_attention_replay.md)。

- 当日后续：Attention诊断首次因HF独立环境未识别AITER editable源码失败。补充冻结源码和已有扩展搜索路径，路径解析/语法通过；工具无GPU导致架构检测阻塞，待普通终端重试。未升级包或改模型。见[验证18](validation/18_torch_expanded_and_attention_replay.md)。

- 当日后续：修正Attention诊断对官方RoPE构造签名的错误调用；独立HF环境四份真实QKV的CPU构造/旋转/mask/SDPA接口检查通过，未初始化GPU。普通终端待重试原诊断命令。

- 同QKV实验HF旋转+SDPA四例精确复现参考，独立AITER HIP旋转不同但不能复现旧ATOM。源码确认首层滑窗实际走Triton融合旋转/缓存，已补该分支诊断、签名/布局/语法检查通过，GPU待执行；未改模型。见[验证18](validation/18_torch_expanded_and_attention_replay.md)。

- 当日后续：实际融合旋转同QKV结果差异很小，融合旋转+SDPA context误差约3.6e-5–5.2e-5，不能解释旧ATOM约6e-4–7.6e-4。补实际AITER prefill内核的HF/融合旋转交叉对照，参数/语法核验通过，GPU待执行，模型未改。见[验证18](validation/18_torch_expanded_and_attention_replay.md)。

- 同QKV交叉结果确认AITER prefill与HF存在约6e-4–7.6e-4 context差异，实际旋转+AITER可精确复现一例旧ATOM。新增显式、受限的Mellum SDPA prefill诊断入口，KV写入/decode沿用原实现、默认不变；CPU32+3项通过，GPU短回归待执行。见[验证19](validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)。

- SDPA prefill短回归算术64token一致、代码第38新token分叉；算术首层完整输出精确，但layer8→9相对L2从0.00106升至0.00938。增加双端stage-layer观察选择，运行器3项/语法通过；下一步同实验观察第9层，默认后端不变。见[验证19更新](validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)。

- 第9层9项权重精确一致，输入已有差异；CPU复核两例各一行Top-8集合改变且靠近边界，未证实路由错误。准备sdpa_rope_diagnostic对齐prefill分步旋转（仍复用缓存、KV/decode原路径），CPU33+3项通过，GPU待执行。见[验证19更新](validation/19_prefill_kernel_evidence_and_sdpa_intervention.md)。

- 分步旋转+SDPA实验仍代码第38新token分叉，代码prefill误差增大，不提升默认。源码确认无量化线性及输出头采用tgemm、HF采用torch linear；准备固定张量的QKV/O/router投影重放，语法/CLI通过，GPU待执行。见[验证20](validation/20_stepwise_rope_result_and_projection_replay.md)。


## 2026-11-12

- 完成：建立[04统一正确性协议与当前状态](../docs/plan/04_correctness_protocol.md)，更新Step0当前计划，区分默认实现/显式实验/历史覆盖；执行顺序调整为固定投影→完整prefill→强制同历史decode/KV→端到端回归。
- 验证：投影脚本补齐CPU预检、37文件来源哈希、打包/内核独立对照、重复稳定性及自动判读；4项CPU测试、语法检查通过，prepare-only未初始化GPU。
- 边界：本轮未改模型功能或默认后端、未运行GPU、不调整数值阈值；Step0仍未完成。
- 下一步：普通终端运行轻量GPU投影实验，先核验参考重放与稳定性，再决定是否需要Mellum局部PyTorch线性候选。详见[验证20更新](validation/20_stepwise_rope_result_and_projection_replay.md)。

- 当日后续：投影GPU参考/重复均精确，相同打包shape的AITER/torch一致，差异仅见K/V打包形状；O/router同输入精确。接入显式分开QKV候选（保留参数/加载接口、默认packed不变），准备一次HF+两组ATOM一步prefill控制。CPU35+8项及prepare-only通过；新GPU控制待执行。见[验证21](validation/21_qkv_packing_and_paired_prefill.md)。

- 当日后续：成对prefill显示separate首层QKV四例精确、下一token4/4，但logits三例RMS变大，完整prefill未闭合，默认packed不变。准备只读复用已完成HF产物，补separate+SDPA、separate+SDPA/分步旋转两个组合；复用协议/后端/输入检查与脚本CPU9项通过，GPU待执行。见[验证21更新](validation/21_qkv_packing_and_paired_prefill.md)。

- 当日后续：两个separate+SDPA组合分别一步token4/4、3/4，logits未闭合；首层input norm/QKV/QK norm全精确，首差仍在context且两组合首层结果相同。准备轻量CPU/GPU逆频率初始化与AITER系数对照，覆盖default/YaRN及172位置；CPU接口/语法通过，未启动GPU或改默认。见[验证22](validation/22_prefill_combination_and_rope_coefficients.md)。

- 当日后续：系数GPU证据确认AITER与GPU初始化HF一致、与CPU初始化HF在default若干短位置及8192的sin不同；CPU/GPU inv_freq最大差2.98e-8。新增显式CPU逆频率+GPU位置计算prefill候选，CPU default/YaRN共172位置inv/Q/K精确对照，36+9项测试通过；GPU候选待执行，默认不变。见[验证23](validation/23_rope_initialization_candidate.md)。

- 当日后续：CPU逆频率候选四例完整prefill闭环成立：logits全精确，120份层状态、56份首层子算子及加载审计通过。新增六条新提示与16/17/1023/1024边界（共10例）一步prefill集，真实tokenizer准备完成、CPU8项通过；新集GPU待执行，默认不切换，decode/KV尚未验证。见[验证24](validation/24_fixed_prefill_exact_and_new_inputs.md)。

- 当日后续：新prefill推理完成但生成器漏enable_thinking使比较失败；保留原件、派生补齐元数据并离线比较，无GPU重跑。9/10完整logits/层精确，仅1024边界不同。CPU核验HF在1024切显式mask+展开KV，已对齐候选边界调用；CPU37+11项通过，GPU边界修正待验证。见[验证25](validation/25_prefill_schema_recovery_and_window_threshold.md)。

- 当日后续：10例新/边界完整prefill通过，logits及300层状态精确，加载审计完整。建立强制历史decode基线：四条输入×8个logits位置，保存ATOM真实logits后才强制后续token，核验实际输入/position，独立对照HF cached/uncached；CPU39+12项及HF小模型1项通过，GPU待执行。KV/decode原实现未修正、默认不变。见[验证26](validation/26_prefill_passed_and_forced_decode_baseline.md)。

- 当日后续：检查decode久等确认worker12:39:14已因探针计划外末尾前向退出，上层未返回，已通知Ctrl+C。保留首请求8步有效数据（prefill精确、decode第1步不同）；修正有界terminal观测与HF首步trace列表快照，加入worker死亡检测/15秒心跳/300秒阶段超时。探针3、运行器5、HF小模型1项通过；复用64份已校验HF logits，恢复GPU待执行。见[验证27](validation/27_decode_probe_failure_and_watchdog.md)。

- 当日后续：完整强制历史基线正常返回，32次实际历史/position核验通过，4prefill精确、28decode不同；HF cached/uncached也均存在decode差异。新增显式Q/K正确旋转+原slot K写回单因素对照，V/分页/窗口/decode内核保留；CPU41+5项通过，新GPU待执行，默认不变。见[验证28](validation/28_forced_decode_baseline_and_kv_rope_control.md)。

- 当日后续：正确Q/K+K写回对照仍四prefill精确、28decode不同，改善不全面。准备同实际分页KV的单tokenSDPA候选，区分NHD/SHUFFLE并对齐cached窗口/mask；每请求首decode额外保存同Q/缓存的原内核输出。CPU43+5项及Attention7项复查通过，GPU待执行，默认不变。见[验证29](validation/29_kv_rope_result_and_cached_sdpa.md)。

- 当日后续：同分页KV SDPA候选使三短输入prefill+7decode全精确，共25/32位置精确，剩余window1023七次decode不同。HF cached mask阈值CPU核验一致；准备轻量首层HF Q/K/V重建对照保存快照，四份CPU预检及官方cache/mask/SDPA接口通过，GPU待执行，模型未改。见[验证30](validation/30_short_decode_exact_and_window_snapshot.md)。

- 当日后续：轻量HF重建确认四例首层Q/K/V/context全精确（含窗口1024）。准备单窗口两位置的全层只读观察，自动比较层状态及各层Attention真实输入/输出；CPU44+5项及独立HF2项通过，prepare-only完成。新HF后层快照需采集，GPU待执行，默认不变。见[验证31](validation/31_window_decode_layer_observation.md)。

- 当日后续：单窗口全层观察定位首decode索引8层输出首差；前0–7层状态及0–8层Q/K/V/context全精确。准备该层后半段canonical权重轻量重放，逐项区分O/norm/router/MoE，要求两路真实输出均精确重建；CPU源输入/Attention预检和norm接口通过，GPU待执行，模型未改。见[验证32](validation/32_layer8_tail_replay.md)。

- 当日后续：索引8层后半段两路重建精确，首差确认router expert44一个logit（0.015625），ID不变、归一化权重改变；O/残差/norm及同输入MoE全精确。新增仅gate的显式F.linear候选，参数/加载接口不变；CPU45+5项通过，原32位置cached GPU验证待执行，默认不变。见[验证33](validation/33_router_projection_root_and_candidate.md)。

- 当日后续：torch router候选使4prefill+28decode共32位置原始logits全部精确，历史/配置/加载审计通过。整理显式correctness-candidate正常生成预设，关闭强制token并正常EOS；被动采集HF/ATOM每步logits及同历史比较，常规运行器加入心跳/失败/超时。CPU46+9项及独立HF3项通过，short64正常GPU回归待执行，默认不变。见[验证34](validation/34_cached_decode_exact_and_normal_generation.md)。

- 2026-11-12：正常short候选173200运行失败：prefill两条精确，生成1/2序列精确，最早cached logits差异第1步。准备python_add两步逐层诊断，CPU3项通过；详见validation/34_cached_decode_exact_and_normal_generation.md。

- 2026-11-12：181028两步python_add逐层精确；交叉核验ATOM正常/强制/HF手动cached一致，HF generate第1步差0.75。准备同实例HF-only两步调用取证，未修改ATOM数学实现。见报告34。

- 2026-11-12：184743同实例复现HF generate差异；源码确认generate在GPU decode临时grouped_mm→batched_mm。显式普通候选加入同序batched专家路径及实际后端取证，CPU官方BF16对照通过，GPU普通回归待执行。见报告34。

- 2026-11-12：193650普通short候选通过：2×64 token及128个采样前完整logits精确，实际HF grouped→batched序列确认，5631加载审计通过。准备10条多样/滑窗边界普通生成回归，每条最多64token，prepare-only通过；Step0尚未完整验收。见报告34。

- 2026-11-12：205013多样生成8/10完整序列一致，630个同历史完整logits全部精确；两首分叉为精确并列最大值选择不同。修复Sampler.forward无过滤时绕过all_greedy判断的问题，CPU3项通过，原10例GPU回归待执行；详见validation/35_greedy_tie_break_and_sampling_dispatch.md。

- 2026-11-12：213129采样修复后普通10例回归通过：640/640 token及640个完整逐步logits向量精确，10个prefill精确、5631加载审计通过，含1023/1024滑窗边界。当前受限候选闭环完成，下一优先补长prompt能力；完整Step0尚未结束。见报告35。

- 2026-11-12：执行Step0本地验收审计221537：原12例768个完整logits重验精确、加载与实际后端核对通过，CPU49项通过；实际guard探测确认1025/16k prefill及双请求decode不支持，200条/HumanEval/服务验收缺项。结论Step0不完整，不宣布通过；见报告36。未启动GPU或更改功能代码。

- 2026-11-12：Step0增强长prefill：移除越窗拒绝并实现HF语义causal band mask，候选验证上限16384；Attention CPU8项、独立HF5项、运行器5项通过，window prepare-only222827通过。GPU先验1023/1024/1025，16k与服务/正式指标仍未完成。见报告37。

- 2026-11-13：085534窗口GPU回归通过，1023/1024/1025共24token及24完整logits精确，5631加载审计通过。准备long suite（8191/8192/8193/16384，每条4token），属于长输入能力smoke，非完整正式验收；见报告37。

- 2026-11-13：085844长输入GPU smoke通过：8191/8192/8193/16384各4token、完整逐步与prefill logits全精确，5631加载审计通过。16k已具能力证据，准备同4输入各64token回归，无新模型数学改动；见报告37。

- 2026-11-13：103154长输入64token回归通过：8191/8192/8193/16384共256token和完整logits精确。准备固定200条开发对照（164 HumanEval prompt+36场景A），长度63–4113，非官方集合且非pass@1；运行器超时可配置，CPU5项通过。详见报告37/38。

- 2026-11-13：104458开发200条对照通过：200/200完整64token序列、12800/12800 token及逐步完整logits精确，200个prefill精确、5631加载审计通过。结束同类数值扩测，核心适配闭环；正式HumanEval待补，工作重心转服务测量入口/profiling。依据题目≥90%而非自加logits精确门槛，详见报告38。
