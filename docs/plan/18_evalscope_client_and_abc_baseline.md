# 18 题目A/B/C基线：独立客户端与协议接入

## 本轮目标和实际交付边界

按用户要求将主线转向题目计分负载：先核对原题命令及实际客户端，再建立共享路由R的A/B/C服务基线。已有开发微基准、D200和ABBA不改称正式场景成绩。

本轮已开发**独立客户端安装器、实际版本源码审计、请求/流式计时探针、原始数据与命令CPU准备、稳定R配置**。工具进程访问PyPI/GitHub返回`Operation not permitted`，本地也没有evalscope；无法代为联网安装。实际安装/客户端审计须由用户终端执行后，才能据真实发布版本完成GPU批次命令映射。**完整A/B/C GPU runner尚未交付，当前命令不会启动推理或GPU benchmark。**

这不是额外权限审批，也不等待用户提供题目未给出的信息；是依赖尚未进入本地、工具网络受限的实际边界。先完成可独立完成的开发与检查，避免没有客户端就直接猜测命令兼容性并消耗GPU。

## 题目原文复核

本轮重新从本地PDF第4–6页的内容流及Unicode映射提取ASCII命令，核对关键参数。PDF SHA仍为`16b292e791d6d782d9f5cafe79bf2073cbd4f24f32b255d711194170513a7536`；未将旧研发摘要当成唯一依据。

| 场景 | 原题单卡参数 | 需要报告 |
| --- | --- | --- |
| A | number512、parallel32、输入512–4096、输出256、stream、ignore_eos、temperature0 | tokens/s/GPU，性能30分 |
| B | number50、parallel1、random、输入1024、输出256，同样的生成参数 | TTFT/TPOT p50/p99，必须报告 |
| C | number200、open-loop、rate0.30、line_by_line指定C文件、输出256、total-timeout180 | TTFT/TPOT p99，性能10分 |

保留题目字面参数，不静默调整并发、输出长度、请求数、C到达率或超时。正式相对比较使用`/data/datasets/baseline.json`，不是用较慢CPD当官方Baseline。A记录638.282724 tokens/s/GPU，C记录TTFT p99=0.993549秒、TPOT p99=0.041265秒；其中目标评分参数为空，不编造评分函数。

## 已识别的兼容性问题

### A：dataset路径、输入长度与数量

原题A使用`--dataset /data/datasets/scenario_a_prompts.jsonl`，而当前上游的benchmark通过`DatasetRegistry`选择插件。该路径是否可直接作dataset，必须在安装版本核验；不能擅自宣称题目字面命令已经可运行。

本地A实际512条，C实际200条，JSONL内均有`id/prompt/intended_words`，C另有共享前缀组。所有prompt都包含换行。把原JSONL整行当文本、把元数据当请求体、或直接展开成纯文本多行，都可能改变实际负载。

当前冻结tokenizer按完整prompt计算：

| 数据 | 裸prompt token范围 | 含当前chat模板/生成标记范围 | A的512–4096过滤有效数 |
| --- | --- | --- | --- |
| A | 540–4113 | 548–4121 | 裸507条 / chat505条 |
| C | 4060–16019 | 4068–16027 | 不在C中套用A过滤 |

`intended_words`不是实际token数。C题目命令没有A的min/max过滤，不能因为2条chat长度小于4096就擅自删除。A是否按裸/chat过滤，以及不足512后是否循环取样，需要核对该客户端原生实现并保留实际ID顺序，不静默截断/补齐。

### C：每请求超时和真实开放到达

当前上游`Arguments`将`total_timeout`定义为每个请求的超时；180秒不等于整个200条任务只能持续180秒。题目C基线记录的总时间约671.77秒也与此区分一致。

必须核对安装版本实际使用的HTTP timeout，以及open-loop调度是否按Poisson间隔独立发出请求；不能把封闭并发+sleep充当开放到达。后续正式运行保留全部失败/超时及尾延迟，不删除未完成请求改善统计。

### 流式计时与thinking

本模型会输出`reasoning_content`。若评测客户端只识别最终`content`，TTFT和TPOT可能被错误统计。当前上游parser已有reasoning处理，但不能假定PyPI选定版本与main完全一致；必须审计实际发布包并用合成SSE检查。

## 已开发组件

- [setup_mellum_perf_client.py](../../scripts/setup_mellum_perf_client.py)：创建`/workspace/venvs/evalscope-perf`，固定`evalscope[perf]==1.15.0`及客户端`transformers==5.10.1`。这是本地可复现的客户端版本选择，不声称题目指定了该版本。
- 新环境通过只读`.pth`桥接既有依赖，子环境包优先；pip在子环境prefix执行，不在`/opt/venv`升级包。前后对冻结环境distribution清单比较；保存pip来源/包hash报告及审计日志。可用`--wheelhouse`进行离线安装，也可`--skip-install`只审计已装版本。
- [audit_mellum_perf_client.py](../../scripts/audit_mellum_perf_client.py)：在隐藏所有GPU的子进程核验真实包版本、CLI help、源码hash并复制关键源码到results供下一轮核查；不根据浏览到的main假造安装源码。
- 审计直接调用安装版本的dataset/API插件：测试A字面dataset路径、A/C原始JSONL和保留prompt的JSON messages候选格式、B random前三条实际请求/token长度；保存请求样例。
- 合成SSE探针只启动本地CPU HTTP mock：初始role元数据、先reasoning后content、最终usage；检查reasoning保留、计数、可观察首输出TTFT。它使用实际客户端parser，不能冒称模型或GPU验证。若发布版接口不同，保存具体错误/源码，明确阻塞，而不是静默换工具。
- [mellum_perf_protocol.py](../../scripts/mellum_perf_protocol.py)、[prepare_mellum_perf_baseline.py](../../scripts/prepare_mellum_perf_baseline.py)：原题A/B/C字面命令、输入/hash/真实长度、官方Baseline及服务启动参数。候选messages文件每行用JSON数组保留完整换行/Unicode与原始次序；**只是兼容性候选，不自动替换正式命令、不施加私自过滤**。
- [稳定R配置](../../configs/mellum/shared_routing.json)：从实验开关整理出来，不依赖某次results中的控制文件；通过现有`serve_mellum.py --campaign-config`显式使用，不改变全局默认。

## 现在需要执行的无GPU步骤

```bash
/opt/venv/bin/python /workspace/scripts/setup_mellum_perf_client.py
```

该命令联网安装**仅客户端环境**，随后自动运行实际发布版本的CPU审计；不启动ATOM server，不加载模型权重，不跑GPU推理。完成后回传根`RESULT ... output ...`，我再读取已安装源码与审计数据，收敛命令映射。

安装器固定包版本并记录依赖解析，首次安装可能需要几分钟，网络速度影响较大。安装日志、pip report、冻结环境前后清单均保留在`results/mellum_perf_client/<run>/`。`client_audit_complete`表示源码/探针审计已完成，不表示正式协议已确认；审计发现的问题仍由`issues`与`protocol_ready_for_full_gpu=false`明确保留。

可先只准备安装计划、无网络/安装：

```bash
/opt/venv/bin/python /workspace/scripts/setup_mellum_perf_client.py --prepare-only
```

## 依赖就绪后安排的GPU批次

1. 按实际客户端源码确定A数据适配/过滤与C输入构造，保存题目字面命令、执行命令、差异说明及具体请求manifest；仍有不能确定的口径则明确标为本地复现，不冒称裁判完全同版。
2. 自动启动R配置服务，小规模连通检查stream、usage、256token、maxseq与真实并发；这些是smoke，不当正式成绩。
3. 通过后按原题完整规模运行B、A、C，每项完整保存原始输出、失败/超时、客户端与服务日志、实际配置/数据/源码hash。先一套R基线，不一次重跑所有失败候选或三遍全矩阵。
4. 正式负载预计明显长于此前16token开发实验；运行前据smoke给出预算和心跳/恢复方案。不能把短trace的prefill比例直接当作256输出A负载的主要占比，后续优化按真实A/C瓶颈确定。
5. 获得数据后同时报告相对官方Baseline及内部基线的含义；再安排有界优化、最终正确性组合回归和交付消融。完整HumanEval独立预算，不与每次性能测试捆绑。

## 来源

- [本地原题PDF](../requirements/2026%20CSIG云一计算加速团队实习校招面试题.pdf)、`/data/datasets/baseline.json`及两个原始场景文件。
- EvalScope官方[Arguments](https://github.com/modelscope/evalscope/blob/main/evalscope/perf/arguments.py)、[benchmark数据生成](https://github.com/modelscope/evalscope/blob/main/evalscope/perf/benchmark.py)、[line_by_line插件](https://github.com/modelscope/evalscope/blob/main/evalscope/perf/plugin/datasets/line_by_line.py)、[默认API流式解析](https://github.com/modelscope/evalscope/blob/main/evalscope/perf/plugin/api/default_api.py)、[包信息](https://pypi.org/project/evalscope/)。这些main源码只作为安装前研究依据，最终以安装包源码/hash审计为准。

本轮无GPU性能结果，不升级冻结推理依赖。交付CPU验证：12项协议/服务回归通过，`py_compile`和`git diff --check`通过；R配置与已验证D200控制逐项一致。CPU协议prepare目录`/workspace/results/mellum_perf_protocol/20261115T143111840944Z`，安装计划目录`/workspace/results/mellum_perf_client/20261115T142520310848Z`；[轻量CPU证据](../../artifacts/results/mellum_perf_protocol/20261115T143111840944Z/provenance.json)。实际EvalScope导入、请求探针和SSE客户端测试均须安装后执行，未冒称通过。

## 安装完成后的实施更新：单入口基线已准备

用户终端第二次安装成功：`results/mellum_perf_client/20261115T144518825402Z`，EvalScope1.15.0/客户端Transformers5.10.1，冻结`/opt/venv`distribution清单前后相同。第一次失败是下载超时/代理拒绝连接，不是推理环境被改坏。实际reasoning SSE探针TTFT约0.052秒、总延迟约0.352秒、usage7输入/3输出及TPOT约0.150秒，均符合预设合成流；torch CUDA未初始化。

新增[evalscope_mellum_adapter.py](../../scripts/evalscope_mellum_adapter.py)与[run_mellum_abc_baseline.py](../../scripts/run_mellum_abc_baseline.py)。前文“GPU runner尚未交付”为安装前边界；**现已实现并通过CPU准备/假worker测试，真实GPU/HTTP运行待用户执行。**

### 适配是如何确定的

只增加命名空间dataset插件，不改EvalScope安装源码：从原JSONL读取完整`prompt`，交给其`DatasetPluginBase.prepare_messages`做原生chat包装和长度过滤。A/C执行命令改为`--dataset mellum_assessment_jsonl --dataset-path 原始文件`；B仍为原生random。model、tokenizer、stream、ignore_eos、temperature、输出预算、并发、请求数、C rate与timeout不改。增加seed42与结果目录以便复现，题目未指定seed。

实际发行版原生过滤与`benchmark.get_requests`均已在CPU执行核验：

- A：505条有效，按原生循环取样达到512；排除ID为A-0015/0105/0176/0298/0382/0383/0403；循环前7条有效记录。输入未裁剪/补长，完整请求ID/hash/token清单保存。
- C：200条全部保留、不套A过滤、不截断内容；原生生成200个请求与清单一致。
- B：seed42，原生生成50条请求，逐条chat token长度均1024。

运行后从官方SQLite `result`表独立核验实际请求多重集（允许乱序完成）、服务端usage输入长度、256输出、stream/greedy设置与记录数，避免命令参数看似正确但实际负载不同。

**C发行版行为需披露**：1.15.0使用指数间隔后，将累计间隔整体缩放至`n/r`；请求无信号量背压，HTTP每请求timeout180秒。它不等于严格无条件独立Poisson间隔。本轮保留该官方发行版行为，既不私改调度器，也不声称与未指定版本的裁判完全同版。输出定位为“注明版本/适配差异的本地题目规模复现”。

### 一次启动

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_abc_baseline.py
```

默认顺序：

1. 一个smoke服务，依次B2条、A8条、C2条（含最大16k输入）；仍输出256token，C保持rate0.30。smoke不作正式成绩，出现参数/计数/实际并发问题就停止完整负载。
2. smoke通过后，分别启动独立R服务完成B50、A512、C200；不同场景重启，避免超时请求遗留到下一场景。
3. 每项保留客户端数据库/summary/percentile、服务/客户端日志、实际batch、加载审计、源码与数据hash。失败/超时保留在总数；有失败的阶段不生成有效完整官方Baseline比值，成功子集p99不掩盖失败率。

这是首次完整基线，不默认把所有历史候选或每阶段三遍一起跑。预计约1–2小时，实际由当前参考式后端决定；总预算3小时，smoke/B/A/C单阶段上限分别15/30/90/30分钟。C调度本身约11分钟再等待在途请求，不能因为日志短期不增长就认为挂死。每15秒有runner及client心跳；只清理本次拥有的进程组，不杀其它GPU任务。

没有活动profiler、张量probe或MoE逐步audit。保留少量layer0 batch元数据审计用于证明真实并发，其开销与配置如实记录；此处不会宣称完全零插桩。

如仅想跑连通检查：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_abc_baseline.py --smoke-only
```

完整模式中断恢复：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_abc_baseline.py --resume /workspace/results/mellum_abc_baseline/实际运行目录
```

完成且hash有效的阶段跳过；失败尝试归档再重跑，源码、客户端、原数据、输入清单和选项需相同。累计已记录预算不重置；硬中断落盘前尾段仍有既有计时边界。`--smoke-only`批次与完整批次不混合恢复。其他安装记录可通过`--client-audit`显式指定。

结果状态区分complete、complete_with_failures、smoke_failed、stage_failed、partial_budget_limited、smoke_passed；根`summary.json/report.md`含A吞吐、B TTFT/TPOT p50/p99、C尾延迟及失败统计。百分位按该发行版nearest-rank，原始官方JSON与数据库保留，不自行换算法。官方target_g字段为空，不编造计分函数。

本工具环境仍无GPU且禁止loopback socket，无法代跑实际HTTP/GPU；已完成真实客户端CPU请求生成和CLI help、假SQLite/调度/恢复/预算/失败分支检查，真实连通由命令的smoke阶段验证。

最终交付核验：25项相关CPU测试通过，完整模式prepare-only成功，目录`/workspace/results/mellum_abc_baseline/20261115T151514998080Z`；尚无本轮GPU性能结果。
