# Mellum 2.1 在 AMD ATOM 上的适配与推理优化

BF16单卡Mellum2.1-12B-A2.5B-Thinking适配，提供性能优先与精度优先两条路线。完整ATOM源码位于`ATOM/`。推理沿用题给冻结环境，不安装或更新依赖。

## 一条命令启动

```bash
./start_server.sh --gpu 0
```

默认使用**性能优先、①②③全部启用**。切换为精度优先：

```bash
./start_server.sh --mode precision --gpu 0
```

两种模式均为单卡、32调度槽、BF16，最大模型长度16640，MTP关闭。HTTP默认8000、引擎端口8026。性能优先①②及①②③启用decode Graph；精度优先使用eager执行。模型目录：`/data/models/Mellum2.1-12B-A2.5B-Thinking`。

| 最终组合 | A tokens/s/GPU | B TPOT p50 ms | C完成数 | 开发200位置一致率 | HumanEval（HF145/164） |
|---|---:|---:|---|---:|---:|
| 性能优先（默认） | 1135.6885 | 4.8985 | 200/200 | 66.8438% | 147/164 |
| 精度优先 | 74.7620 | 51.2821 | 三轮均1/200 | 96.8125% | 148/164 |

性能优先未达到90%位置一致率；精度优先通过本地正确性检查，但C200大量请求触及180秒超时。模式开关独立于优化点编号。开发200为164条HumanEval及36条A场景输入；两条路线使用相同本地验证口径。

## 累计消融与逐项关闭

```bash
./start_server.sh --mode performance --row step0
./start_server.sh --mode performance --row step01
./start_server.sh --mode performance --row step012
./start_server.sh --mode performance --row step0123
```

替换为`--mode precision`即可启动精度路线的四级累计配置。两条路线Step0相同，①②③含义分别如下：

| 优化点 | 性能优先 | 精度优先 |
|---|---|---|
| ① | 原生Attention/FusedMoE、AITER Norm、整批packed投影 | Torch参考专家decode group4 |
| ② | 原生路由、驻留RoPE、KV单次写入、QK融合、decode Graph | 参考RoPE/KV单次写入 |
| ③ | 前缀兼容与缓存、prefill元数据复用 | decode元数据复用、参考SWA四请求分组 |

关闭参数作用于选定行，可以重复指定：

```bash
# 两条路线均可关闭③，得到①②
./start_server.sh --mode performance --disable-opt 3
./start_server.sh --mode precision --disable-opt 3

# 精度路线仅保留②
./start_server.sh --mode precision --disable-opt 1 --disable-opt 3

# 关闭全部优化
./start_server.sh --disable-opt 1 --disable-opt 2 --disable-opt 3
```

性能路线②③依赖①的原生计算路径，入口拒绝保留②或③却关闭①的组合。其余支持组合可以启动，只有四级累计配置具有完整的既有测量证据。精度路线支持全部八种子集，非累计子集尚未经过完整GPU验收。定义集中于`scripts/mellum_submission_modes.py`；历史`delivery_step*`配置保留供旧实验溯源，不用于当前入口。

## 参数、日志与检查

```bash
./start_server.sh --mode precision --port 8001 --engine-port 8027 --gpu 1
./start_server.sh --mode performance --prepare-only
./start_server.sh --mode precision --output /workspace/results/mellum_server
./start_server.sh --help
```

默认日志目录为`results/server/<UTC时间>/`。`launch.json`保存模式、实际启用优化点、环境、引擎命令与源码哈希。`--prepare-only`检查冻结版本并生成配置，不加载模型或启动GPU推理。

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/models/Mellum2.1-12B-A2.5B-Thinking","messages":[{"role":"user","content":"Write a Python function add(a, b)."}],"temperature":0,"max_tokens":64,"stream":false}'
```

CPU配置检查：

```bash
/opt/venv/bin/python -m pytest -q scripts/tests/test_dual_submission.py scripts/tests/test_delivery_matrix.py scripts/tests/test_mellum_execution_profiles.py
```

实际入口GPU烟测，覆盖两种模式的全部累计行，检查权重加载、HTTP、SSE、usage及模式配置：

```bash
/opt/venv/bin/python scripts/smoke_dual_delivery.py --gpu 0 --all-rows
```

仅测两种最终组合时省略`--all-rows`。分卡并行时分别指定`--mode performance --gpu 0`与`--mode precision --gpu 1`。

## 文档与证据

[技术报告](TECHNICAL_REPORT.md)已更新双模式设计、性能消融及正确性矩阵；[AI使用说明](AI_USAGE.md)与[工作日志](WORKLOG.md)保留此前稿件，后续分别修订。最新精度优先数据见[完整A/B测量](docs/experiments/03_precision_third_full_performance.md)与[C三轮补测](docs/experiments/05_precision_c_completion_results.md)。性能优先证据位于`artifacts/experiments/performance_cumulative/`。

模型、虚拟环境与完整实验结果不纳入仓库；结论摘要、必要原始统计和SHA索引随代码保留。历史分支与提交继续保存，旧版矩阵脚本需在相应历史分支复现，不能与当前模式定义混用。
