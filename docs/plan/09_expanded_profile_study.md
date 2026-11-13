# 09 Step0性能扩展研究：计时曲面、真实prompt与细粒度trace

## 目的与边界

用户明确要求加深/扩大评测后再开始优化。首轮[性能01](../../report/performance/01_bs1_profile_baseline.md)仅三种长度、两次64token计时，Attention优先是初步假设；本轮先补更具体证据，不实现加速kernel或修改模型数学。

本轮仍是**单卡、顺序bs1离线研究**。当前Step0 reference后端不能直接接收batch8/32，不能把更多bs1样例当作已完成场景A/C。批量桥接后还必须补批量/服务和真实到达负载；正式evalscope客户端问题、HumanEval补证不在本命令内解决。

## 三阶段实验矩阵

| 阶段 | 配置 | 数量/预算 | 回答的问题 |
| --- | --- | --- | --- |
| timing | 合成输入512/1024/4096/8192/16384 × 输出1/64/256 | 通常每配置5次；1024×256为50次 | 首token随长度的扩展、decode输出长度影响、重复稳定性 |
| timing | 场景A/C各取中位长度及90%位置两个真实prompt，完整保留并固定chat/thinking | 每个256输出，3次 | 合成输入结论是否对真实文本/路由同样成立 |
| trace | 上述5合成+4真实输入，各16输出 | 9份短trace，每份配同预算无profile请求 | 层/子算子瓶颈与profiler扰动 |
| detailed | 1024输入，4输出 | 1份shapes/stack/memory trace，加同预算无profile对照 | 看实际调用shape、调用栈和内存事件，避免对所有请求开重采集 |

默认共19个计时配置、132次计时请求、22617个计时输出token；不含各配置一次同预算预热和trace阶段的请求。计时配置顺序以固定seed=20261113打散并保存，减少固定短→长顺序与热状态的混杂，不宣称消除了所有漂移。

输出1token主要用于prefill观察，TPOT标为N/A，不把引擎返回的0当decode性能。引擎仍可能异步多执行末尾一步，因此不是承诺完全没有decode工作的纯kernel微基准。普通配置报告n、mean/median/p90、min/max、CV；只有n>=50才输出经验p99，仍不能把50次样本当充分稳定尾延迟。1024×256×50仅为B形状离线检查，使用固定prompt、没有HTTP，也不是原题random输入正式B。

实际真实样本（本次准备）：A-0338/2303token、A-0002/3857token，C-0071/10448token、C-0152/15007token。按所有512条A、200条C的模板后token长度排序选取，原prompt不截断；当前全部在16384以内，剔除数0。脚本仍记录未来源数据的长度剔除数，不隐瞒采样分布变化。来源与文本/ID哈希保存在plan，不把shared_prefix字段存在等同于已测试prefix cache。

## 干扰控制与采集深度

三个阶段用独立worker顺序执行，分别加载/关闭引擎；最大上下文预算均为16640，TP1/DP1、eager、BF16、max_num_seqs=1、util=.35、prefix/chunked关闭，实际设置保存。计时worker的torch_profiler_dir=None且不安装hook；trace和detailed的profiler只在短窗口开启。温度0、ignore_eos=True，严格核对实际输出预算。

每份trace前后使用相同输入、相同输出预算的无profile/有profile请求，要求greedy token相同，并记录wall ratio。这样可以估计该窗口的观察扰动，不混用初轮64token基线与16token trace推导CPU成本。该配对是一组样例，不能宣称消除所有噪声。

新增标签仅在采集窗口包装原函数并返回原值：reference_position_rope、overwrite_reference_keys、gather_reference_kv和Torch SDPA；停止时恢复原函数引用，不修改冻结依赖的磁盘文件。标签结合外层layer_XX，细分21个SWA层/7个Full层，并区分旋转、K覆盖、分页gather和SDPA。保留总kernel与父子inclusive/exclusive口径，shared函数标签不能丢掉其父层身份。

解析增加每层/阶段模块kernel时间、按CPU语义op而非仅hipLaunchKernel的归因、每个decode上下文区间、CPU self事件时间以及GPU拷贝方向统计。self CPU通过减去嵌套CPU-op区间得到，仍可能包含等待，不等于CPU计算；GPU镜像annotation排除规则保留。

worker在profile开始前reset allocator peak，started/stopped回执记录allocated/reserved/peak；这些是**trace窗口的PyTorch allocator统计**，不包含所有driver/AITER外部分配，也不等于无profile所有计时请求的峰值。detailed的内存观察开销单独看待。

默认后台每15秒只读调用现有rocm-smi，记录利用率、显存、频率、温度、功耗、调用耗时与错误；不设置功耗或锁频。记录所有SMI设备，不能把其编号直接等同HIP可见编号。请求epoch时间戳可与SMI记录对应，粗采样不能推断单kernel行为。可用`--no-telemetry`关闭以检查采样本身的扰动。

## 运行方式

已完成开发/CPU检查和固定数据准备，本轮尚未启动GPU。准备产物：`/workspace/results/mellum_profile_study/20261113T190127027409Z/`。

完整默认实验：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_profile_study.py
```

按第一轮约56–61ms TPOT估计，默认实验为**半小时量级，可能更久**，不是几分钟smoke。每阶段5400秒为保护上限，不是预计耗时；保留15秒心跳，任何阶段失败后停止，已有结果和progress保留。每组计时前预热；运行时避免同GPU其他推理/profiler及资源竞争。

仅检查配置：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_profile_study.py --prepare-only
```

如后续需缩减重复或暂不详细采集，可显式使用`--repeats`、`--b-repeats`、`--no-detailed`、`--no-real-prompts`；不同预算不能混在同一统计表冒称默认完整实验。默认这轮不缩减，也不同时启用Torch profiler与rocprof硬件采集。

## 中断与阶段恢复

长实验支持按阶段恢复，源码指纹、每阶段plan字节哈希与模型配置必须一致；完成阶段核对真实配置/样本数量/输出长度，trace还核对原文件哈希。只复用完整成功阶段，不将部分请求拼成成功。失败阶段重跑前整体移动至previous_attempts，保留日志/trace/进度，避免旧回执干扰新采集。

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_profile_study.py --resume /workspace/results/mellum_profile_study/实际运行ID
```

恢复采用保存的GPU/输入/预算，不以新默认参数重建数据。若仅detailed阶段不支持，可显式加`--no-detailed`复用其他阶段；此时最终状态为study_partially_collected，不能冒称默认全套完成。源码或计划变化会拒绝复用，需检查差异，不强行混入不同实现结果。

## 结果结构与成功条件

根目录`results/mellum_profile_study/<run>/`含summary、三份stage plan、阶段日志、SMI telemetry.jsonl及最终study_summary.md。timing/worker_results保存全部请求和统计，progress保存当前未完成配置的已完成请求；trace/detailed保存配对结果、回执、原trace/哈希与逐例analysis.json/.md。

成功标签`study_collected`表示计划中的三个阶段全部完成，不表示正式A/B/C或优化验收。timing严格检查输出数，trace必须有GPU kernel、prefill/decode与标签并通过配对token检查；导出/采集CPU-only失败不能记完成。原始logits/tensor不额外保存，不重跑200条正确性集。

额外trace仍可能包含引擎异步末尾forward，按实际phase区间计数，不用输出token数强行设定forward次数。SMI失败单独记录，不把它当作推理数学失败。

## 开始优化前应得到的结论

1. 长输入TTFT与短/长decode开销的曲线，是否输出256后排序改变。
2. 五档长度及真实prompt下，SWA/Full、各层、旋转/写回/gather/SDPA和MoE各占比；不以单个kernel名猜后端。
3. CPU self、runtime、拷贝、kernel空隙与profiler扰动之间的区别，Graph是否仍值得优先。
4. 显存峰值/温度/频率有无明显状态变化；是否有离群配置需要定向复核。
5. 当前只能确定bs1候选优先级。之后补最小批量桥接及batch1/8/32，才最终选择面向A的优化顺序；不宣称已经完成吞吐/队列/缓存命中研究。

本轮不自动做ROCm counter roofline，也不依据SMI利用率推导带宽瓶颈。需要时只对已选少数热点单独做rocprof采样，不继续无边界扩大所有组合。

## 检查记录

24项CPU测试通过：矩阵预算、重复统计/单token TPOT、输出截短拒绝、冻结配置一致、真实prompt不截断及剔除计数、CPU self/语义关联、细粒度函数数值/恢复、worker回执及allocator reset生命周期、遥测错误记录、兼容路径等。既有1024 GPU trace离线解析再检查通过，kernel计数53897与原数据一致，阶段计数保持1/16，无新GPU运行。语法、差异和prepare-only检查通过。模型和MoE/Attention数学执行未改变，新增逻辑仅用于测量和采集观察。
