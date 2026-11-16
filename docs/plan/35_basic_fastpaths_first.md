# 35 调整优先级：恢复基础快路径，再决定Graph

研究08确认当前还有参考式Norm、分离QKV和重复常量上传。以性能35的单次KV写入版本（开发A64约499.64tokens/s）作为统一基线；先短trace和独立组件筛选，暂缓复杂Graph/量化/MTP/双卡推理改造。

## 已实现的独立开关

| profile | Norm | QKV | inverse frequency |
| --- | --- | --- | --- |
| fused_moe_native_batched_direct_kv | reference | separate | 每次to GPU |
| fast_resident | reference | separate | GPU一次驻留 |
| fast_norm | ATOM RMSNorm | separate | 原路径 |
| fast_packed | reference | 框架packed | 原路径 |
| fast_combined | ATOM RMSNorm | 框架packed | GPU驻留 |

所有配置均保留BF16权重/KV、FusedMoE、原生varlen/paged、整批投影和单次KV写入，eager/level0、TP/DP1、prefix/chunked/MTP关闭。router仍为原Torch参考路径，未混入其它改动。默认旧reference不变；新Norm通过工厂返回原框架模块，参数名字/shape保持，不改带support_torch_compile的既有模型文件或冻结依赖。

MELLUM_NORM_BACKEND=reference/aiter控制113处Norm（28×input/post/Q/K+final）；MELLUM_QKV_BACKEND=separate/packed复用既有接口；MELLUM_RESIDENT_ROPE=1在首次forward把常量存GPU并持有，后续不再从CPU上传。同一公式/FP32逆频率不变；Norm/packed可能改变数值，不承诺逐位生成一致，入选后按任务指标评估。

## GPU0：短trace＋常量驻留对照

```bash
/opt/venv/bin/python /workspace/tencent-mellum-fastpaths/scripts/run_mellum_fastpaths.py --track resident --budget-minutes 45
```

自动在基线采batch1/32、固定1024输入、每次8输出的短trace；先预热及16输出非profile计时，启停有rank0回执，必须真实GPU kernel和模块标签，导出CPU self/GPU kernel/拷贝归因。此固定prompt重复trace只用于结构诊断，不代表A/C路由分布或正式吞吐；不把GPU空隙全部解释为CPU瓶颈。

然后顺序运行基线与fast_resident的HTTP smoke B2/A8/C2→A64并发32→长度分层C20/rate0.3。GPU0 trace engine9226，HTTP8251/engine8926，45分钟执行总预算（CPU准备另计）。trace失败保留信息并继续容量任务，最终状态明确含失败。

## GPU1：融合Norm、packed QKV与组合批次

```bash
/opt/venv/bin/python /workspace/tencent-mellum-fastpaths/scripts/run_mellum_fastpaths.py --track kernels --budget-minutes 75
```

顺序四个配置：基线、fast_norm、fast_packed、fast_combined；每个配置相同HTTP smoke/A64/C20，不重跑完整ABC/HF。某独立候选失败仍保留其它候选；组件失败则跳过相应组合，避免重复崩溃。GPU1 HTTP8151/engine8726，75分钟执行总预算，最多允许90分钟。

## 自动核验与数据管理

两条命令共享同一已提交只读代码，可同时运行，端口/GPU隔离；运行期间不修改该工作区。每track只准备一次原始数据，配置通过只读inputs链接复用，实际输入、到达率、stream/256输出和客户端版本不变。结果/workspace/results/mellum_fastpaths/<UTC>_resident或_kernels；每profile的summary.json包含原始指标，根summary.json包含批次状态。

新增fastpaths.json检查实际113个Norm类、28个QKV及separate属性，常量驻留还要在真实KV写入事件中确认inverse_device=cuda。权重5631加载审计、真实batch32、输入/输出长度、失败率、源码与客户端指纹继续保留。不把配置flag存在当已启用，不在计时期间开profiler。

90项相关CPU检查通过，含原模型/权重接口/缓存/旧路径、Norm参数与残差API、GPU常量复用、新profile隔离、自动失败停步；py_compile/diff和两track真实prepare-only/最终指纹通过。工具无GPU，未执行新GPU/宣称收益，无需安装工具。

## 收敛规则

1. 以同card基线及组件/组合A/C收益选择，不预先承诺某配置一定更快；并行另一卡的共享资源影响与顺序偏差在最终空闲复测控制。
2. 选择最快的有意义候选，复用HF164只补候选功能；保留原绝对差判定和方向性分差，不以token相同前置阻断。
3. 当前最高功能149/164比HF145/164高2.439pp，原abs2pp=False保留；新候选不自动继承旧分数。
4. 基础路径若已解决主要差距，优先最终组合/完整ABC；若短trace仍显示明显发射成本，再一次有界decode Graph试验。prefix在最快合格配置上整合。
5. 研发候选可多于三个，最终按题目选最多三个独立可消融优化点，详细披露复用/必要适配边界，不以计点讨论阻塞筛选。
