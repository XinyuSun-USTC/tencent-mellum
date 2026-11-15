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
