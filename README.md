# Mellum 2.1 在 AMD ATOM 上的适配与推理优化

本仓库提供Mellum2.1-12B-A2.5B-Thinking的BF16单卡实现、三项优化开关、验证工具和实验记录。推理使用题给冻结环境，完整ATOM源码位于 `ATOM/`。

| 交付材料 | 内容 |
|---|---|
| [技术报告](TECHNICAL_REPORT.md) | Profiling、优化设计、技术取舍、消融及正确性结果 |
| [AI使用说明](AI_USAGE.md) | 协作分工、人工审查与纠正案例 |
| [工作日志](WORKLOG.md) | DAY0—DAY7的实施、实验与决策记录 |
| [测量证据](artifacts/submission/final_r32_matrix/matrix.md) | 累计结果、配置来源与原始摘要哈希 |

## 运行环境

在题给容器中使用 `/opt/venv/bin/python`。启动脚本检查Torch、Transformers、Triton和safetensors版本，配置保存在 `configs/submission/environment.json`。模型默认位于：

```text
/data/models/Mellum2.1-12B-A2.5B-Thinking
```

推理不需要HF参考虚拟环境，也不依赖其他实验工作树。模型权重、独立评测环境及完整运行输出不随源码打包。

## 一条命令启动服务

在仓库根目录执行：

```bash
./start_server.sh --gpu 0
```

默认启用三个优化点，对应消融末行 `step0123`。服务使用单卡、32调度槽、BF16、eager模式，HTTP端口为8000，引擎端口为8026；MTP关闭。健康检查地址为 `http://127.0.0.1:8000/health`，API前缀为 `/v1`。

可显式选择累计配置：

```bash
./start_server.sh --row step0 --gpu 0
./start_server.sh --row step01 --gpu 0
./start_server.sh --row step012 --gpu 0
./start_server.sh --row step0123 --gpu 0
```

| 配置 | 优化① | 优化② | 优化③ | 开发200位置一致率 | HumanEval |
|---|---|---|---|---:|---:|
| Step0 | 关 | 关 | 关 | 96.8125% | 148/164 |
| Step0＋① | 开 | 关 | 关 | 96.8125% | 148/164 |
| Step0＋①② | 开 | 开 | 关 | 96.8125% | 148/164 |
| Step0＋①②③ | 开 | 开 | 开 | 75.9453% | 147/164 |

HF HumanEval参考为145/164。前三行通过本地正确性检查；第三项组合未达到90%的token位置一致率。C场景完成数依次为1/200、1/200、1/200和48/200。需要使用已通过本地正确性检查的组合时，选择 `--row step012`。

### 优化开关

Step0包含32槽服务、参考兼容批处理、逐请求投影与共享路由。三个优化点在该基线上增加：

| 优化点 | 作用 | 关闭参数 |
|---|---|---|
| ① 解码专家分组 | 将相邻四个decode请求组成专家计算组 | `--disable-opt 1` |
| ② RoPE与KV去冗余 | 参考旋转结果直接接入缓存，K/V只写入一次 | `--disable-opt 2` |
| ③ SWA与混合负载执行 | SWA group4、元数据复用、前缀缓存、chunk1024/d4、断连回收 | `--disable-opt 3` |

关闭参数作用于所选累计配置，可以重复指定。例如：

```bash
# 关闭③，实际启用①②
./start_server.sh --gpu 0 --disable-opt 3

# 仅启用②
./start_server.sh --gpu 0 --disable-opt 1 --disable-opt 3

# 关闭全部优化，使用Step0
./start_server.sh --gpu 0 --disable-opt 1 --disable-opt 2 --disable-opt 3
```

累计四行的实测成绩见上表；其他开关组合用于独立实验，不能直接引用累计行的成绩。代码定义位于 `scripts/mellum_delivery_matrix.py`，实际开关、环境和引擎命令写入每次启动目录的 `launch.json`。

### 常用参数与请求示例

```bash
# 修改端口
./start_server.sh --row step012 --gpu 0 --port 8001 --engine-port 8027

# 只检查环境与生成启动配置，不运行GPU推理
./start_server.sh --row step012 --prepare-only

# 指定日志目录
./start_server.sh --row step012 --output /workspace/results/mellum_server
```

不指定输出目录时，每次启动创建独立的 `results/server/<UTC时间>/`。完整参数可通过 `./start_server.sh --help` 查看。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/models/Mellum2.1-12B-A2.5B-Thinking","messages":[{"role":"user","content":"Write a Python function add(a, b)."}],"temperature":0,"max_tokens":64,"stream":false}'
```

## 验证与复现

以下命令使用CPU检查交付文件和开关配置：

```bash
/opt/venv/bin/python scripts/verify_submission.py
/opt/venv/bin/python -m pytest -q scripts/tests/test_delivery_matrix.py
```

实际GPU入口检查：

```bash
/opt/venv/bin/python scripts/smoke_delivery.py --gpu 0 --modes step0 step01 step012 step0123
```

完整性能与正确性流程由 `scripts/run_delivery_matrix.py` 管理。题给数据位于 `/data/datasets/`；复用HF参考的验证工具还需要既有参考结果和独立评测环境，相关路径及协议见[实验说明](docs/plan/65_final_delivery_matrix.md)。已完成结果见[性能矩阵](artifacts/submission/final_r32_matrix/matrix.md)和[正确性证据](docs/submission/11_correctness_evidence.md)。

## 仓库结构

```text
start_server.sh       统一启动入口
TECHNICAL_REPORT.md   技术报告
AI_USAGE.md          AI使用说明
WORKLOG.md           每日工作日志
ATOM/                模型与引擎完整源码、测试及上游文档
scripts/             启动、实验、验证和打包工具
scripts/internal/    旧配置入口，不用于当前累计消融
configs/            配置文件
artifacts/           已归档的结果、原始trace和哈希索引
docs/                题目来源、复现说明与内部技术材料
report/              按主题编号的研究、验证和性能记录
manifests/           原始结果库存与项目快照清单
```

源码、trace和必要实验记录纳入Git；临时结果、缓存和虚拟环境由 `.gitignore` 排除。ATOM的上游许可证保留于原目录，复用说明见[来源与许可证](docs/submission/01_sources_and_licenses.md)。需要离线交付时，`scripts/package_submission.py` 可从干净提交生成源码ZIP和保留完整提交历史的Git bundle。
