# Mellum 2.1 在 AMD ATOM 上的适配与推理优化

本项目为AMD ATOM推理框架增加Mellum2.1-12B-A2.5B-Thinking支持，并研究其在BF16单卡环境下的推理效率。实现覆盖模型注册、网络结构、权重加载、分页KV缓存和OpenAI兼容服务；仓库包含完整ATOM源码、优化开关、评测工具与实验记录。

项目最终提供**性能优先**与**精度优先**两种运行模式。两者共享同一适配基线，分别采用三项累计优化，并以相同的数据与协议报告性能和正确性。默认启动性能优先模式。

## 1. 为什么保留两种模式

模型适配首先以Hugging Face Transformers（HF）输出为参考，建立正确性对照。随后实验发现，保留参考计算路径能够维持较高的生成一致率，但逐请求注意力、专家计算和缓存处理开销限制了服务吞吐；采用ATOM/AITER原生算子、批量投影、融合与Graph执行后，性能显著提升，生成序列却与HF出现差异。

这一取舍在场景C中最为明显。该场景以泊松到达的4K–16K共享前缀请求，同时产生长输入处理与持续生成负载：输入的首次计算称为prefill，后续逐token生成称为decode，长prefill会干扰正在进行的decode。参考路径即使经过局部优化，仍有大量请求触及180秒超时；原生高效组合则能够完成全部200条请求，但未达到90%的token位置一致率。

因此，交付保留两条路线，分别呈现参考一致性约束下的优化收益，以及原生高效路径的完整服务性能：

| 模式 | 主要实现思路 | 当前实测边界 |
|---|---|---|
| **性能优先** `performance`，默认 | 原生计算、批量投影、融合与decode Graph，结合前缀复用 | 完成ABC全部请求；未达到90%的token位置一致率 |
| **精度优先** `precision` | 保留参考计算方法，优化专家分组、KV写入及SWA分组 | 通过本地生成一致率与HumanEval差距检查；C负载容量不足 |

“精度优先”指与HF生成行为的一致程度，两种模式均使用BF16。HumanEval检验生成代码的功能正确性，token一致率检验与参考生成序列的接近程度，两项结果分别报告。

## 2. 最终配置的实测结果

性能测试包含：A，512请求、并发32；B，50请求、并发1；C，200请求、0.3请求/秒。三场景均固定输出256token，单卡32个调度槽。TPOT为平均每个输出token的生成耗时，TTFT为首token延迟，p50/p99分别表示相应分位数。

| 最终配置 | A tokens/s/GPU ↑ | B TPOT p50 ms ↓ | C TPOT p99 ms ↓ | C TTFT p99 ms ↓ | C完成数 |
|---|---:|---:|---:|---:|---|
| 性能优先 | **1135.6885** | **4.8985** | **11.3619** | **1026.0455** | **200/200** |
| 精度优先 | 74.7620 | 51.2821 | — | — | 三轮均1/200 |

两种最终配置的A、B均全部完成。精度模式C每轮其余199条均超时；延迟分位数仅来自唯一成功请求，故上表不将其列为完整负载成绩。两条路线尚未在同一配置下兼顾90%的位置一致率与C200全完成。

| 最终配置 | 前64token位置一致率，要求≥90% | 长输入匹配位置 | 其中16K组 | HumanEval pass@1 | 相对HF差值 |
|---|---|---:|---:|---:|---:|
| 性能优先 | 66.8438%，未通过 | 138/256 | 63/64 | 147/164（89.6341%） | +1.2195个百分点 |
| 精度优先 | **96.8125%，通过** | **256/256** | **64/64** | **148/164（90.2439%）** | +1.8293个百分点 |

位置一致率使用固定开发集：164条HumanEval提示与36条A提示，每条比较前64token，分母12800。长组另测8191、8192、8193及16384token四个输入。HumanEval独立生成完整答案，关闭thinking、最多2048token，HF参考为145/164（88.4146%）；两种最终组合与参考的绝对差均在2个百分点内。

完整累计消融、profiling证据和优化取舍见[技术报告](TECHNICAL_REPORT.md)。上述结果为本地完整规模测量；裁判隐藏集的三次中位数属于最终复测。

## 3. 运行环境与启动方法

启动面向题给容器，使用`/opt/venv/bin/python`，保留冻结的Torch、Transformers、Triton及safetensors版本；入口会检查版本。模型默认路径为：

```text
/data/models/Mellum2.1-12B-A2.5B-Thinking
```

在仓库根目录执行一条命令即可启动默认服务：

```bash
./start_server.sh --gpu 0
```

显式选择两种模式：

```bash
# 性能优先，全部三项优化
./start_server.sh --mode performance --gpu 0

# 精度优先，全部三项优化
./start_server.sh --mode precision --gpu 0
```

服务默认监听HTTP端口8000，引擎端口8026；使用单卡、BF16、32调度槽，最大模型长度16640。性能模式完整组合启用decode Graph，精度模式使用eager执行；MTP和双卡模型并行均关闭。推理不需要独立HF参考环境或HumanEval工具，也不依赖其他实验工作树。

健康检查与请求示例：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/models/Mellum2.1-12B-A2.5B-Thinking","messages":[{"role":"user","content":"Write a Python function add(a, b)."}],"temperature":0,"max_tokens":64,"stream":false}'
```

## 4. 三项优化与消融开关

**模式与优化点是两个独立选择。** `--mode`选择技术路线，`--row`选择该路线内的累计配置；①②③在两种路线中的具体机制不同。

共享基线Step0已包含模型适配、32槽服务、参考兼容批处理、逐请求投影与共享路由。性能分析表明，短输入专家执行、长输入注意力以及decode中的重复处理是主要开销，因此分别构建以下优化：

| 优化点 | 性能优先 | 精度优先 |
|---|---|---|
| ① 计算效率 | 原生Attention/FusedMoE、AITER归一化与整批packed QKV投影 | 将相邻四个decode请求组织为参考专家计算组 |
| ② 数据通路与逐步执行 | 原生路由、驻留RoPE、KV单次写入、QK归一化与RoPE融合、decode Graph | 参考旋转位置编码（RoPE）结果直接接入缓存，K/V仅写入一次 |
| ③ 剩余重复工作 | 前缀KV兼容与缓存、prefill元数据复用，减少共享输入的重复计算 | 每次前向复用decode元数据，将符合条件的滑动窗口注意力（SWA）按四请求分组执行 |

两条路线的第三项均采用最新已测组合：性能模式不使用chunked prefill；精度模式不启用前缀缓存、Graph或原生专家。

### 累计配置

| `--row` | 启用内容 |
|---|---|
| `step0` | 共享基线，无新增优化 |
| `step01` | Step0＋① |
| `step012` | Step0＋①② |
| `step0123`，默认 | Step0＋①②③ |

例如：

```bash
./start_server.sh --mode performance --row step0 --gpu 0
./start_server.sh --mode performance --row step012 --gpu 0
./start_server.sh --mode precision --row step0123 --gpu 0
```

### 逐项关闭

`--disable-opt`作用于所选累计行，可重复指定：

```bash
# 性能模式关闭③，保留①②
./start_server.sh --mode performance --disable-opt 3

# 精度模式关闭①③，仅保留②
./start_server.sh --mode precision --disable-opt 1 --disable-opt 3

# 关闭全部新增优化，回到Step0
./start_server.sh --disable-opt 1 --disable-opt 2 --disable-opt 3
```

性能模式②③依赖①的原生计算路径，入口会拒绝保留②或③而关闭①的组合。精度模式支持全部八种子集。两种路线的四级累计行具有既有完整测量及GPU入口验证；其他合法子集可用于扩展实验，其结果需单独测量。

配置统一定义于[scripts/mellum_submission_modes.py](scripts/mellum_submission_modes.py)。

## 5. 日志、验证与复现

常用运行参数：

```bash
# 修改设备与端口
./start_server.sh --mode precision --gpu 1 --port 8001 --engine-port 8027

# 只检查环境并生成配置，不加载模型或启动GPU推理
./start_server.sh --mode performance --prepare-only

# 指定日志目录
./start_server.sh --mode precision --output /workspace/results/mellum_server

./start_server.sh --help
```

默认输出目录为`results/server/<UTC时间>/`。其中`launch.json`记录模式、实际启用优化点、环境、引擎命令与源码哈希，加载审计记录5631个源权重张量的映射结果。

提交文件与CPU配置检查：

```bash
/opt/venv/bin/python scripts/verify_submission.py
/opt/venv/bin/python -m pytest -q scripts/tests/test_dual_submission.py scripts/tests/test_delivery_matrix.py scripts/tests/test_mellum_execution_profiles.py
```

实际入口GPU烟测覆盖两种模式各四行累计配置，检查权重加载、HTTP/SSE、usage、实际模式及端口回收：

```bash
/opt/venv/bin/python scripts/smoke_dual_delivery.py --gpu 0 --all-rows
```

省略`--all-rows`时，只检查两种最终组合。两卡独立执行时，可分别指定`--mode performance --gpu 0`与`--mode precision --gpu 1`；这不构成双卡模型并行。八行烟测结果均已通过，[验证证据](artifacts/submission/dual_mode/gpu_smoke/index.json)随仓库保存。

## 6. 交付文档与仓库结构

| 文档 | 内容 |
|---|---|
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | 适配方法、profiling、双路线选择与取舍、完整性能消融及正确性矩阵 |
| [AI_USAGE.md](AI_USAGE.md) | AI协作分工、建议纠正与人工决策 |
| [WORKLOG.md](WORKLOG.md) | DAY0—DAY7的开发、实验与交付记录 |

| 路径 | 用途 |
|---|---|
| `ATOM/` | 完整框架源码与Mellum适配实现 |
| `scripts/` | 服务入口、配置、验证与实验工具 |
| `configs/` | 冻结环境及运行配置 |
| `artifacts/` | 原始统计摘要、必要trace、功能判定与SHA索引 |
| `docs/`、`report/` | 需求、研究计划、历史实验分析与补充说明 |

最新实测可从[性能累计矩阵](artifacts/experiments/performance_cumulative/collected/matrix.md)、[精度完整A/B](docs/experiments/03_precision_third_full_performance.md)及[精度C三轮补测](docs/experiments/05_precision_c_completion_results.md)追溯。模型权重、虚拟环境与完整运行输出不纳入源码仓库；历史分支和完整提交记录保留。旧版实验定义在相应历史分支复现，当前服务与消融统一使用双模式入口。
