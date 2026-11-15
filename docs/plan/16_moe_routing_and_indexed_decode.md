# 16 MoE路由共享与decode直接索引：实现和一次启动实验

## 目标与依据

承接[性能13](../../report/performance/13_batch_accuracy_campaign.md)：CPD在真实batch32的developer200位置96.8125%、完整序列87%，但S32耗时是原B0的2.442倍。短trace中experts占累计GPU kernel时间77.33%，prefill83.11%、decode59.93%；prefill大量小GEMM发射、decode专家权重索引/BMM和逐请求辅助操作值得优先处理。

本轮目标是在CPD可选回退和真实batch调度下，减少MoE重复路由、Python调用及decode权重物化。**不是**默认更换整个MoE后端，也不继续展开全层bitwise定位。冻结库和默认服务路径保持；候选均需显式配置，正常greedy验证后再讨论采用。

本计划与代码已落地，GPU结果待运行。本轮工具环境无`/dev/kfd`，不能替用户执行真实GPU测试。

## 源码核查与实现选择

原[MellumTorchExperts.forward](../../ATOM/atom/model_config/mellum_experts.py)在每次调用内执行FP32 softmax、topk、renormalize再转BF16。CPD又在每层按请求调用该函数，decode32意味着每层重复32次路由辅助计算。

同文件decode分支先`hidden_states.repeat_interleave(top_k)`，再用`w13_weight[expert_ids]`与`w2_weight[expert_ids]`物化选中权重后BMM。以H2304/I896/top8/BF16计，每个token两组权重约94.5MiB；批内32token共约3024MiB选中权重数据。CPD按请求调用降低同时在用的峰值，但并未消除这些索引复制。该数字是索引张量元素量，不等于实测显存峰值或HBM流量，重复expert可能命中cache。

prefill中每个请求仍单独执行两次`grouped_linear`/`F.grouped_mm`，本轮**不替换**这些GEMM，不假定共享路由就能消除prefill的主要GEMM成本。

### 公共整理：路由与专家算术分离

`MellumTorchExperts`新增`route`及`forward_routed`；默认`forward`依次调用两者，原BF16激活/乘法/求和、prefill grouped_mm及decode BMM表达式保留。参数、checkpoint布局和加载函数不改。CPU回归之外，GPU批次会重跑当前CPD的S32/L32和完整D200，与历史CPD输出单独核对，避免把重构漂移当优化效果。

### 候选R：shared_routing

在真实批次上统一计算一次route，再按真实请求边界切出weights/ids，调用同一个`forward_routed`。

- L（QKV/O/router线性）仍按CPD请求shape调用；共享的是router logits之后的路由辅助计算，不是撤销L控制。
- prefill/decode专家GEMM请求shape、权重、激活与累加顺序保持。
- 批量softmax/topk实现仍可能有数值变化；不预设GPU逐位一致，以正常生成结果评估。

### 候选I：indexed_decode

在R基础上，仅decode改用[indexed expert GEMV](../../ATOM/atom/model_config/mellum_indexed_gemv.py)：

1. gate/up直接根据每个选中expert ID读取原canonical BF16权重，对应token输入由selected-row/top_k定位，不生成选中权重副本，也不复制repeat_interleave后的hidden。
2. 保留`F.silu(gate)*up`的两个BF16舍入点。
3. down直接索引原权重；BF16路由权重乘法和top-k维求和继续使用原Torch表达式。

内核每program处理4个输出坐标，K补齐到2的幂、掩码处理K/N尾部，FP32乘积归约后存BF16；禁止FP fusion以减少隐式公式变更。FP32归约次序与原BMM并非相同，**不能宣称数值安全或准确性通过**。仅GPU BF16、单token/request、K≤4096、canonical最后维连续权重；不支持自定义路由、量化、Graph/MTP/并行拓展，不静默退回其它实现。

[Campaign](../../ATOM/atom/model_config/mellum_campaign.py)显式配置`moe_optimization=cpd|shared_routing|indexed_decode`。新[优化适配器](../../ATOM/atom/model_config/mellum_moe_optimization.py)保留CPD原始调用作为回退。默认不安装这些wrapper；实验的phase切换按真实ScheduledBatch边界，不读期望输出决定推理计算。

## 一次启动的实验批次

入口：[run_mellum_moe_optimization.py](../../scripts/run_mellum_moe_optimization.py)。复用计划15已有进程组生命周期、累计预算、原子manifest、指纹和产物校验、resume和15秒心跳；总预算90分钟、每job默认1200秒、最多12次GPU worker启动。估算30–60分钟，首次实测后修正，不保证每个条件分支都运行。

| 阶段 | worker上限 | 内容 |
| --- | --- | --- |
| M | 1 | 真实layer0权重微基准、语义边界探针、时延/峰值分配、indexed实际kernel trace |
| S | 4 | 原B0、CPD、R、满足微基准条件的I；K4/S32/L32正常生成，分别重复两次 |
| C | 3 | 当前CPD完整D200一次；最多两个候选各D200一次 |
| T | 2 | 最多两个具备准确性条件的候选，对CPD做S32/L32各ABBAABBA计时 |
| P/H | 2 | 有实际收益的最佳候选：一个真实子进程短trace worker、一个HTTP回归worker |

### M：微基准，不代替生成验收

[M脚本](../../scripts/mellum_moe_micro.py)只加载原checkpoint layer0的全部专家权重，复用已保存四条HF layer0输入/路由logits，校验原文件hash。

- prefill用8条不同长度切片构成ragged请求；decode-shaped回放取保存prefill末行形成1/8/32token组。后者是**decode形状的MoE算术回放，不是原真实KV decode轨迹**。
- 对cpd/R/I轮换顺序预热和CUDA event计时，各6次；记录相对CPD的数值误差、额外分配峰值，不据局部误差阈值宣布生成不合格。
- indexed语义探针覆盖重复expert、非整齐K/N、两个repeat映射；单非零输入坐标避免把浮点容差当验收标准。检查真实GPU trace出现`_indexed_expert_gemv`。
- I只有在语义/有限值检查完成，且decode32局部速度≥1.05倍或额外分配峰值低于CPD一半时进入S阶段；没有这两类收益就停止该候选。
- I编译或局部算术检查失败，不阻止独立B0/CPD/R；OOM/HIP等环境问题按既有规则停止后续GPU任务。

### S/C：自然生成与预先固定的晋级口径

数据仍为相同K4、S32、L32和D200缓存HF；不重新生成HF、不强制历史、不删除失败/EOS位置。小集两次稳定与真实batch32核验是实验可解释性的条件，非新官方bitwise门槛。

晋级规则在GPU运行前冻结：

- 小集：S32/L32每组位置≥90%，且相对本批CPD每组不下降超过2个百分点；K4位置≥90%。
- D200：总体位置≥90%，且不低于本批CPD超过0.5个百分点；长/短输入各≥90%。最多两个候选进入，不根据200结果在线重写kernel。
- 完整序列比例始终单列，公开集身份/计数歧义不隐去；这些保留CPD水平的实验筛选规则不是原题新增门槛。
- 当前CPD也重新跑完整D200，并报告相对历史CPD同输入输出有多少序列相同。一次D200的`stable=true`不解释成重复验证。

### T：先无probe计时，再决定是否profile/HTTP

每个候选独立worker加载一次模型。保持L与prefill专家请求shape，runtime控制文件只在CPD与该候选间切换；所有配置承担相同文件读取/边界管理开销。

S32/L32交替执行两个ABBA周期，每种输入每个配置各4次。每段先跑2token预热，再计时自然greedy16；无张量probe，无活动profiler，也关闭逐步MoE audit写盘。保留layer0的已有batch元数据审计，以核验实际batch。明确这是带相同少量控制/审计开销的开发对照，不是无插桩官方A/C。

离线按配置核对4次输出相同，取wall中位数；两组speedup几何均值≥1.05、每组speedup≥0.95才视为具备初步性能收益。该规则用于控制实验规模，不是统计显著性结论。不能拿CPD较慢的耗时替代题目官方Baseline放大最终成绩。

没有收益则完成报告、保留回退，不继续profile/HTTP或开放无界参数搜索。最佳候选另跑短真实ModelRunner trace（GPU kernel与模块标记均需存在），以及同配置HTTP completion/SSE/并发/chat/usage/pending回归。本批不会把长输入计时误称为已采长输入优化trace。

## 运行、恢复与输出

用户执行一次：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_moe_optimization.py
```

完成后只需返回根`RESULT ... output ...`。根目录包含`summary.json`、`report.md`、`matrix.csv`、`timing_*.json`、`manifest.json`及`jobs/`；微基准数值/时延/分配数据、逐请求实际输出和trace分别在对应worker目录。没有收益也会明确报告，不自动切默认。

只准备、无GPU：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_moe_optimization.py --prepare-only
```

恢复：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_moe_optimization.py --resume /workspace/results/mellum_moe_optimization/实际批次目录
```

要求源码、输入、GPU选择、来源和成功产物hash保持一致；累计预算/启动次数不因resume重置。失败尝试保留。和计划15相同，硬中断落盘前的尾段耗时无法精确补计；修改数学源码后创建新批次，不能混跑。

## 交付前验证及边界

- 新候选的CPU检查覆盖BF16/FP32、renormalize开关、ragged prefill、decode、共享路由只调用一次、expert请求shape不变、动态CPD回退、拒绝不支持的输入；原有模型和批量回归同时运行。
- 假worker覆盖完整12-worker分支、indexed编译失败后R独立继续、准确性淘汰、ABBA四次稳定性、收益筛选；复用原有预算/resume/失败隔离测试。
- [check_mellum_indexed_cpu.py](../../scripts/check_mellum_indexed_cpu.py)用CPU Triton解释器验证repeat=1/2、K=11/896/2304、stride与掩码，且不执行GPU；解释器用FP32，不证明AMD BF16数值。
- gfx942离线编译gate/up K2304/N1792、down K896/N2304、边界K11/N7通过；编译报告VGPR49/24/8，private segment均0。只表示编译成功及资源报告，不代表运行速度或无运行期问题。

本批不换冻结库、不启CK/Triton SWA、不改attention/cache/sampler公式，不跑正式A/C或完整HumanEval。两候选只作用通用请求/专家计算，禁止根据题目ID或期望输出作推理分支。若均无收益或准确性不足，停止这轮路线，保留CPD并据完整结果决定下一优化方向。

交付核验：70项ATOM模型/专家/批量等测试及32项脚本/调度回归通过，共102项；`py_compile`和`git diff --check`通过。CPU解释器/离线编译的[轻量证据](../../artifacts/results/mellum_moe_cpu_checks/manifest.json)已保存。最终prepare-only目录：`/workspace/results/mellum_moe_optimization/20261115T120347030797Z`，实际GPU正确性与性能仍待一次批次验证。

## 执行结果

批次`20261115T121305387577Z`实际10worker、约29分钟完成。R与当前CPD的D200逐条200/200相同，位置96.8125%/完整序列87%保持；ABBA短/长1.0915×/1.0468×，HTTP小规模回归通过。I的decode32局部4.892×、额外分配64.20→3.71MiB，但小集相对CPD下降超过预设2个百分点，未跑D200；这不是题目90%门槛失败的证明。详见[性能14](../../report/performance/14_moe_routing_and_indexed_decode.md)。本计划已执行，默认不重复整批。
