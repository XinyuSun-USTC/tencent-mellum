# 15 B0批量准确性：一次启动的开发、实验与自动分析批次

## 目标及交付方式

用户要求扩大一次开发和实验的覆盖，减少逐轮手工启动。下一轮一次开发齐全：batch-aware observer、通用shape控制开关、实验调度器、离线分析器及回归检查，再交付一个启动入口。每个阶段不再等待用户逐个确认；调度器依据预先定义的规则运行后续分支，最后统一报告。

当前事实：[性能12](../../report/performance/12_b0_batch200.md)中B0 reference真实batch32下developer200位置74.8359%、序列44%，点1关闭，S0单请求仍有精确参考。目标是定位并改善批量化造成的生成不一致，达到任务层面门槛，而不是把全logits/中间张量bitwise一致变成验收要求。

本文件是计划，以下控制项和总runner尚未实现；本轮只做源码核查、方案与日志更新，无新GPU。计划不是承诺所有问题都能自动修复；预置分支不足时仍完成可独立进行的实验，给出未解决清单，不擅自扩为无限诊断。

## 一次开发的组件

1. **输入/配置冻结器**：源HF与现有batch200输入不变，保存源/配置/代码/权重索引hash；同批次所有比较使用同一组数据。参考输出只在分析器使用，不作为自然生成的推理输入。
2. **batch-aware观察器**：在可取得真实ScheduledBatch/forward context的调用点与新Mellum模块安装，明确请求ID、实际行顺序、cu边界与positions。旧`MellumModel.forward`的layer dump显式只接受batch1，不能删除断言后冒当批量探针。
3. **通用shape控制器**：按请求元数据切分指定运算、再拼回，保持权重、dtype、norm顺序、RoPE、采样、真实KV与调度不变。开关与prompt内容/题目ID无关，不能针对已知输出做特殊处理。
4. **调度器/监控器**：一个入口生成完整DAG，在一张分配的GPU上顺序启动独立worker；复用已有HF，不占第二张GPU重新生成参考。
5. **离线分析与决策器**：每个阶段完成立即生成机器可读结论，按下面规则调度分支，结束生成统一summary和Markdown报告。它按既定规则选择实验，不在运行途中自动重写源码。

不修改冻结ATOM/AITER/Torch/Transformers，不升级工具，不启Graph/量化/双卡/MTP，不混入CK/Triton点1，除非将来明确安排单独增量。所有新数学控制默认关闭，先只作诊断/候选；不自动切换正式服务默认后端。

## 固定实验数据

| 集合 | 选择 | 预算 | 用途 |
| --- | --- | --- | --- |
| S32 | developer200原顺序前32条HumanEval | 自然greedy16 | 复现短输入batch敏感性；不改prompt |
| L32 | developer200原顺序全部32条≥1024输入 | 自然greedy16 | 覆盖ragged长prefill与真实窗口/缓存 |
| K4 | 既有4条小集，含1023/1025 | 沿用16/32/8/8 | 反序、错峰结束、页复用、迟到请求逻辑 |
| D200 | 全部固定developer200 | 自然greedy64 | 最多两个候选的代表性检查 |

HF参考均优先取已保存结果，按输入和实际输出预算取前缀，不重新跑HF。S32/L32选择与初始失败分布对应，不能将小集筛选分数当公开集验收。保留D200原始分母，不删除失败、EOS或难样本。两种一致率及输入长度/来源分层均报告。

## A阶段：批量规模与首差定位（4个GPU worker）

同一B0代码、点1关闭，依次bs1/2/8/32；每个worker均运行S32和L32。原顺序、相同上下文预算/最大batched tokens；每个配置相同输入重复两次，先确认同配置是否稳定。记录实际GPU batch，不以客户端window替代。

只在bs1/32 worker打开有限观察：最多4条目标（原始早期分叉2/13/21，加一条长输入早期分叉），最多前8次采样位置。所有请求仍正常生成，不强制HF历史。分析只比较目标请求历史相同的前缀和首差当步，分叉后的状态不作为“同输入数值对照”。

观察内容：

- 实际请求/行/cu/slot/block table映射与位置，完成、重排和缓存页归属；先排除明显跨请求逻辑错误。
- prefill最后位置、decode各步的目标logits；原HF目标前8步向量可读缓存并验hash。保存top候选、margin和偏好翻转，区别全局偏移、并列与真实排序变化。
- 两个极端batch下各层目标位置的materialized residual/hidden边界，必要的input norm、QKV、attention输出、post-norm、router/experts边界。只保存少量目标，不保存200×64×28全量张量或所有专家权重。
- 若首差指向cache，自动追加**目标层**实际paged K/V与bs1对照；不直接用重新prefill长前缀代替原始decode。

分析产物明确区分：首个非bitwise差异、首个明显数值差异、首个top1翻转。前两者只是定位，不自动判不合格。若S0在当前环境不能复现既有HF目标前缀，暂停依赖该归因的控制实验并报告参考复现问题；不能带着参考漂移得出B0原因。

## B阶段：预置控制矩阵（最多6个额外GPU worker）

先验证“运算shape是否是主因”，避免每发现一个算子就要用户再运行一次。控制均保留真实batch提交、真实分页cache和正常greedy，只有内部指定运算按请求切分；它们不是高效实现，也不把客户端串行化伪装为批量吞吐提升。

| 控制 | Prefill | Decode | 说明 |
| --- | --- | --- | --- |
| C0（复用A的bs32） | 当前共享shape | 当前共享shape | 对照，不重复加载 |
| CP | 指定L/E按请求shape | 当前共享shape | 识别prefill贡献 |
| CD | 当前共享shape | 指定L/E按请求shape | 识别decode贡献 |
| CPD | 指定L/E按请求shape | 指定L/E按请求shape | 联合上界控制 |
| CL | 仅L按请求shape | 仅L按选定phase控制 | 分离线性投影贡献 |
| CE | 仅E按请求shape | 仅E按选定phase控制 | 分离专家贡献 |

L为QKV、O与router线性运算；E为experts，仍保持prefill grouped语义/decode batched语义，按请求调用相同公式，不替换为另一后端。CL/CE具体phase由CP/CD/CPD分析确定并在运行前记录配置，表中“选定”不能在报告中省略。

CP/CD/CPD最多3个worker先跑，两组各16token；CL/CE最多再2个。若A已明确逻辑索引错误，不用shape矩阵掩盖错误，自动完成独立可用检查并输出需修逻辑的位置。若首差明确在未覆盖的norm或lm_head，允许**最多1个**事先实现并通过CPU检查的对应控制；否则标记unresolved，不临时追加任意层/flag搜索。

每个控制优先报告自然生成一致率/稳定性与明确的运行代价。全请求shape控制可能接近串行算子成本；即使恢复准确率，也不能自动称性能优化，或未经披露用它做低速分母放大后续收益。最终若采用，属于B0/B1实现策略变化，单独记录。

## C阶段：自动短名单与D200（最多2个GPU worker）

根据S32+L32两组生成结果挑最多两个互补候选：优先稳定、逻辑检查完整且一致率改善的方案，再考虑粗略代价。中间张量误差更小但生成不改善，不自动晋级；不要求bitwise精确。

候选冻结其通用配置后，正常greedy64跑D200，复用同HF，启用真实有界32窗口。统计总体及来源/长度、token位置及完整序列比例、EOS/失败/完成数量、实际batch直方图。D200只对最多两个候选运行一次；不一边看200结果一边修改公式并反复刷同集合。

位置口径≥90%才可进入“本地开发准确性具备条件”的后续分支；同时记录序列口径与公开集身份歧义。若没有候选达到，统一报告未解决，不继续跑正式A/C，不静默降到batch1充作batch32。

## D阶段：通过后的代价与服务检查（最多2个GPU worker）

仅对C中最合适的一个方案执行：

1. **关闭探针的短执行代价对照+抽样batch trace**：固定一组实际batch32负载，保留相同输入/输出、预热、少量重复。计时与trace分开；标明仅开发负载，不冒称正式A/B/C。观察是否用大量逐请求调用换取准确性，为下一优化点确定依据。
2. **已有HTTP多请求与生命周期回归**：completion/SSE/chat展示、usage、前端pending、错峰/复用；backend开关和操作shape均核对实际值。不得因offline通过省略服务路径。

完整A512、C200到达过程与HumanEval164长生成不在本批默认范围内。批量化影响短HumanEval生成，因此此前“点1对HumanEval不激活”的推断不能作为B0功能通过证据；最终方案仍需正式功能门槛补证。

## 批次规模、预算与存储

默认4个规模worker + 最多6个控制worker + 最多2个D200 + 最多2个回归/profile，**最多14个GPU worker**，而不是无边界的全参数组合。一个worker可加载一次模型连续完成两组，减少反复初始化；不同后端/控制用独立worker隔离状态。

按现有加载约数十秒、B0 D200约147秒、逐请求控制可能明显变慢估算，完整批次先按**30–60分钟**规划，默认总预算90分钟；这是开发预算不是已测承诺。每job也有上限；达到总预算后不新启动job，已完成数据保留、标记partial_budget_limited，不将跳过的验收当通过。第一次实际耗时会用于修正后续预算。

探针默认只保留最多4目标×8位置的有限状态，原始张量总上限预设约3GiB；超过则停止额外capture并标记诊断覆盖不足，不影响已完成普通生成数据。权重从冻结checkpoint/在用模块按需读取，不复制整个模型到结果目录。无probe配置不保存每步全词表logits。

单卡顺序运行，避免GPU任务争用造成无法解释的结果；“批量运行”指自动调度完整矩阵，不要求所有GPU job同时执行。已有第二张卡不为重跑缓存参考而占用。

## 自动监控、失败与断点续跑

- 每15秒统一打印job/配置/进度/已用时间；阶段manifest原子写入。长生成日志暂时不增长不能直接判挂，结合worker存活、进度文件和审计文件活动判断，最终仍受硬超时约束。
- 每job记录实际环境、配置、代码/输入hash、GPU逻辑设备及可可靠取得的硬件标识。SMI映射无法确认时只保留原始值，不猜物理卡号。启动前核对显存，清理只作用于本runner拥有的进程组。
- 配置不支持/编译失败：该候选失败，其独立对照继续；不得静默换后端。OOM/driver故障：停止依赖该环境的后续任务，保留现场，不强行无限重试或杀其它进程。
- 源码/输入/冻结环境不一致：停止混合版本实验。预定义控制开关只切配置，不在GPU批次运行中自动改代码。
- `--resume <run>`校验输入、数学源码、实际配置与完成产物hash，只跳过完整匹配的成功job；失败尝试保留，单独重跑所需job。允许文档变更，但需记录HEAD并确认相关源码未改；不能把不同控制版本输出拼成一轮。
- 最终状态区分complete、complete_with_unresolved、partial_budget_limited、blocked_environment。完成实验不等于修复/正式验收完成。

## 一次分析与用户交互

最终根目录输出：

- `summary.json`：各job状态、采用/跳过原因、匹配率、代价、问题定位与未解项。
- `report.md`：可直接阅读的结论、对照表和下一步。
- `matrix.csv`：配置×集合×批量×匹配率×时间，方便人工核验。
- `first_difference.json`：同历史首差证据、所在phase/层/运算、margin；不能证明的原因明确标推断。
- `jobs/<id>/`：原始配置/日志/输出/必要张量和指纹；`runner.log`用于排查单次运行。

用户正常情况下只执行一个未来的campaign命令，完成后发回一个根RESULT路径；我再一次性核对、归档、撰写阶段结论。中间不需要手工逐个启停job，也不用分别发送十几个日志路径。只有外部设备/端口/环境阻塞或超出预置假设时才可能需要额外处理。

## 实施顺序与交付前检查

下一开发轮先把以下内容全部实现，再交付运行命令：

1. 新batch-aware观察器与target/step边界；CPU测试真实请求映射、反序/结束、raw/residual定义及probe不改输出。
2. CP/CD/CPD/CL/CE等通用shape控制与默认关闭；验证dtype、拼回顺序、prefill/decode区分、权重加载布局不变。
3. DAG调度、自动分析、硬预算、逐job失败隔离、恢复及manifest；用假worker测试分支选择和失败/断点场景，不能到GPU运行时才发现调度器问题。
4. 所有输入/参考hash和资源预算prepare-only；完整列出将跑/条件跑/跳过任务及上限。
5. 只有以上检查完成，才提供真实入口（拟名`run_mellum_batch_campaign.py`，**当前尚不存在、不可执行**）。不让用户在每个小组件完成后就手工试跑。

本轮无新的GPU命令；完整批次开发交付后再一次启动。

## 开发交付（2026-11-15）

以上“当前尚不存在”描述保留为立项时状态。现已实现总入口：[run_mellum_batch_campaign.py](../../scripts/run_mellum_batch_campaign.py)。本轮未运行GPU；首次GPU结果仍可能暴露环境或数值问题，实验完成不保证自动修复。

一次启动：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_batch_campaign.py
```

默认使用逻辑GPU0，顺序运行，累计GPU worker预算90分钟，最多14次启动；先按30–60分钟估算。每15秒输出当前worker耗时、日志大小。默认每job上限1200秒，受剩余总预算约束。正常结束只需回传根`RESULT ... output ...`路径，报告、矩阵、首差及各job原始结果均在该目录。

只准备输入、来源和调度说明（不导入ATOM/启动GPU）：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_batch_campaign.py --prepare-only
```

中断后恢复：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_batch_campaign.py --resume /workspace/results/mellum_batch_campaign/实际批次目录
```

恢复验证输入、HF目标logits、相关源码、模型配置与索引、GPU逻辑选择及成功产物hash；已完成job不会重跑。累计已记录耗时/worker次数不会因resume重置；失败尝试移动到`attempt_N/`。源码修改后应创建新批次，不跨数学版本恢复。硬中断发生在计时落盘前时，该次未落盘尾段无法精确计入累计时长，这一边界需人工留意。

### 已落地的控制与观察

- [mellum_campaign.py](../../ATOM/atom/model_config/mellum_campaign.py)：仅显式`MELLUM_BATCH_CAMPAIGN_CONFIG`安装，默认不存在任何hook或shape控制。CP/CD/CPD、CL/CE按真实请求边界切分L/E；可选norm/head均事先实现。head先按全局cu选择末行，再按请求调用原TP1 tuned GEMM，避免原head重复选行。
- [ModelRunner.run_model](../../ATOM/atom/model_engine/model_runner.py)：在真实ScheduledBatch和forward context均可取得的位置开始/结束观察。旧单请求layer dump的保护保持原样。所有控制要求eager，无Graph/TP扩展。
- 观察bs1/32的原样自然生成，目标固定HumanEval 2/13/21与L32第一条；每组重复两次、每目标前8步。保存last-row hidden/norm/QKV、attention context与O、router/experts、residual两分量与materialized sum、全词表目标logits、实际req/row/slot/block元数据。这里的“首差”是**所观察末行边界的首差**，不宣称遍历全部prefill token后找到全网络最早变化。
- 映射在`core_mgr.add_request`实际dispatch前原子发布，避免异步EngineCore抢先prefill。只在探针worker临时包装该次提交函数，不改变默认提交方式。
- 原HF四目标前8步logits文件在prepare时核验sha，分析时再核验；同历史才计算bs1/bs32/HF误差、均值漂移、中心化RMS、top2及margin。relative-L2超过1e-3仅作定位标注，**不是正确性验收条件**。
- 若首个观察差异为attention context，使用唯一可选CX配额做目标层真实paged KV跟进：同worker依次bs1/32、相同S32/L32自然生成，保留同历史的目标层窗口KV；该分支不同时加norm/head扩展。每探针worker最多768MiB张量，三个观察worker合计至多2.25GiB，为trace留空间；截断明确标记覆盖不足。数值差异不能单独证明缓存映射错误。

### 自动决策与输出边界

bs1需复现S32/L32全部参考前16token且重复稳定，bs32需重复稳定且实测decode32，才进入依赖它们的shape归因。该条件仅决定**诊断能否作因果对照**，不是把题目90%门槛升级为100%。独立A规模对照仍完成。

CP/CD/CPD跑完，取两组较差一致率更高的phase方案决定CL/CE隔离phase。候选需两组均不回退、至少一组改善、重复稳定和实际batch核验通过，最多两项进入D200。若原B0两组已≥90%，允许其作为D200对照。D200位置≥90%只表示本地开发口径可进入后续分支，完整序列比例仍单列。

D阶段一个worker内先固定K4，再分别预热并计时原B0/候选S32两次；所有计时均关闭张量probe与活动profiler。为同模型切换控制，两个配置均承担相同的控制文件读取/边界管理开销，因此仅作开发代价对照，不冒充无插桩正式吞吐。随后用ATOM原生start/stop profile及rank0回执，在**ModelRunner子进程**采集另一段batch32、2输出token trace；检查真实GPU kernel和模块标记后才接受trace。HTTP worker携带同候选配置，复用已有顺序/SSE/并发/chat/pending回归。

每job结果记录逐请求输出、完成顺序、实际batch、两次稳定性、原始TTFT/TPOT/latency及墙钟；根目录输出`manifest.json`、`summary.json`、`report.md`、`matrix.csv`、`first_difference.json`和`runner.log`。失败编译隔离，OOM/HIP/无GPU等环境问题停止依赖任务；进程清理只针对本runner新建的进程组，包括父进程已退出的遗留子进程。

本批不自动实施性能后端替换、不跑正式A/C或完整HumanEval、不修改冻结库，也不自动切换服务默认控制。若预置矩阵不能恢复准确性，根状态为`complete_with_unresolved`；预算耗尽、环境阻塞分别报告，不能将跳过检查当通过。

交付CPU验证：28项相关测试通过（campaign 16项、batch200/serving/process runner回归12项）；最终prepare-only核验S32=32、L32=32、K4=4、D200=200及四目标32份缓存logits，未启动GPU。另有58项ATOM Mellum模型/config/attention/batch/experts/projection回归通过，共86项。最终prepare目录为`/workspace/results/mellum_batch_campaign/20261115T042308376455Z`。额外检查`py_compile`与`git diff --check`通过。实际GPU执行与控制是否改善一致率均待本批验证。
