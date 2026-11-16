# Mellum 2.1 在 ATOM 上的适配与推理优化

Mellum2.1-12B-A2.5B-Thinking 的 BF16 单卡适配与性能实验，基于题目提供的冻结 ATOM、AITER、ROCm 和 PyTorch 环境。交付分支：`release/submission`；最新发布标签：`submission-20261116-final`。

## 交付材料

| 题目要求 | 文件 | 内容 |
|---|---|---|
| 技术报告，≤4页 | **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** | Profiling、选点依据、预期与实测收益、正确性结果、消融表 |
| AI 使用说明，1页 | **[AI_USAGE.md](AI_USAGE.md)** | 工具用途、人工分工、错误与发现过程 |
| D1—D7 工作日志 | **[WORKLOG.md](WORKLOG.md)** | 每天的工作、阻碍和决策 |
| 一条命令启动及优化开关 | **[start_server.sh](start_server.sh)**、本页下文 | 默认参考配置、显式优化配置及组件开关 |

文档均为可编辑中文 Markdown。技术报告按四页内容分段，AI 说明按一页篇幅组织；实际导出页数仍需排版核验。

**验收状态：尚未完全满足题目要求。** 参考计算配置在本地200条前64 token及长输入组上完全一致；优化组合的一致率为62.85%，未达到90%。正式Step0累计消融仍不完整。详细核对见[交付审查](docs/submission/02_delivery_audit.md)。

## 启动服务

在题目提供的容器中，从仓库根目录执行：

```bash
./start_server.sh --mode conservative --gpu 0
```

服务地址：`http://127.0.0.1:8000/v1`；健康检查：`/health`。默认使用 `/opt/venv/bin/python` 与 `/data/models/Mellum2.1-12B-A2.5B-Thinking`，可通过 `--model PATH` 指定兼容的 BF16 检查点。`ATOM/` 包含完整源码；启动不依赖其他实验工作树或 HF 参考虚拟环境。

```bash
# 性能实验配置：尚未通过本地 token 一致率门槛
./start_server.sh --mode aggressive --gpu 0 --audit
# 更换 HTTP 和引擎端口
./start_server.sh --mode conservative --port 8001 --engine-port 8027
# 检查配置与冻结版本，不启动 GPU
./start_server.sh --mode conservative --dry-run
```

两种配置的上下文上限均为16640，默认调度槽分别为1和32。模式选择是独立的交付配置开关，不计作Step0或一个优化点。启动器记录实际配置、软件版本与源码哈希；`--audit`增加后端、图回放、缓存和元数据执行记录。

## 优化组件开关

在 `--mode aggressive` 后追加下列参数。配置改变后应按对应组合重新验证；已有结果只适用于其记录的配置。

| 优化方向 | 组件关闭参数 | 作用 |
|---|---|---|
| 原生计算与数据通路 | `--disable-native-moe` | 专家计算改用Torch后端 |
| 同上 | `--disable-native-attention` | 注意力改用参考路径，同时关闭KV直接写入 |
| 同上 | `--disable-direct-kv` | 单独关闭分页KV直接写入 |
| 同上 | `--separate-qkv`、`--reference-norm` | 分别恢复分离投影与参考归一化 |
| 解码执行与融合 | `--disable-graph` | 关闭decode graph，使用eager执行 |
| 同上 | `--disable-qk-fusion`、`--disable-native-routing` | 分别关闭QK Norm/RoPE融合与原生路由 |
| 同上 | `--disable-resident-rope` | 关闭常驻RoPE系数路径 |
| 前缀与元数据复用 | `--disable-prefix`、`--disable-metadata-reuse` | 分别关闭KV前缀缓存与prefill元数据复用 |

容量参数为`--max-num-seqs`、`--max-model-len`、`--max-batched-tokens`，其中batched tokens须不小于model len。关闭组件仅改变相应执行路径，不自动切换参考/优化模式。

## 最终版本测量

| 指标 | 优化配置实测 | 题给baseline |
|---|---:|---:|
| A 吞吐（token/s/GPU） | 1137.0824 | 638.282724 |
| B TTFT p50 / p99（ms） | 69.442 / 89.525 | 未提供 |
| B TPOT p50 / p99（ms） | 4.941 / 5.610 | 未提供 |
| C TPOT p99（ms） | 11.396 | 41.265 |
| C TTFT p99（ms） | 1024.007 | 993.549 |

最终源码的A512/B50/C200全部成功，运行编号为`20261116T153025780336Z`；四组A32/B6/C8缩减对照也已完成，运行编号为`20261116T153235801902Z`。前者见[完整ABC结果](artifacts/submission/delivery_abc/summary.json)，后者见[缩减对照矩阵](artifacts/submission/delivery_ablation/matrix.md)。缩减对照的控制配置已包含原生优化，与历史Step0不同。

上述性能配置未达到token一致率门槛。HumanEval为150/164，HF为145/164；按绝对差≤2个百分点的口径不通过。参考配置的精确预设尚未单独完成HumanEval和完整ABC。两张卡用于独立并行实验，交付推理为单卡，MTP关闭。

## 检查与复现

```bash
# 文件、链接及证据完整性检查
/opt/venv/bin/python scripts/verify_submission.py
# 实际启动两种配置，检查加载、普通请求与流式响应
/opt/venv/bin/python scripts/smoke_submission.py --gpu 0
# 完整 ABC；评测客户端使用已有独立环境
/opt/venv/bin/python scripts/run_mellum_full_campaign.py --gpu 0 --profiles sprint_graph_qk_router_prefix_metadata --budget-minutes 90
# 缩减规模累计对照
/opt/venv/bin/python scripts/run_submission_ablation.py --gpu 1 --budget-minutes 20
```

两种配置均已通过真实交付入口的启动检查。该检查采用512上下文、单槽和8个输出token，覆盖权重加载、HTTP与SSE/usage；不替代完整验收。冻结推理版本见`configs/submission/environment.json`。

性能复现采用EvalScope1.15的兼容适配。实际命令、输入和到达过程差异均保留在运行记录及技术报告中；本地结果不等同于裁判的空闲机器隐藏集三次中位数。

## 仓库结构与补充资料

| 路径 | 内容 |
|---|---|
| `ATOM/`、`configs/` | 实现源码及运行配置 |
| `scripts/` | 启动、验证、性能测试和归档工具 |
| `artifacts/submission/` | 交付证据、实际配置和SHA256索引 |
| `docs/submission/` | 交付核对及上游来源说明 |
| `docs/requirements/` | 题目原件与要求摘要 |
| `docs/plan/`、`report/` | 内部计划、研究、诊断及历史实验记录 |
| `manifests/` | 外部资产与大体积产物索引 |

[来源与许可证](docs/submission/01_sources_and_licenses.md) · [内部文档索引](docs/README.md) · [完整研发日志](report/worklog.md) · [证据索引](artifacts/submission/index.json)

模型权重、虚拟环境和大体积张量不纳入Git。历史Step0保留于`step0-local-validated`，研发提交和合并历史未squash。离线材料可通过`scripts/package_submission.py`生成源码ZIP与完整历史bundle，具体说明见交付审查。

## 补充资料

[交付审查与复现清单](docs/submission/02_delivery_audit.md) · [实现来源与许可证](docs/submission/01_sources_and_licenses.md) · [实验依据索引](artifacts/submission/index.json) · [内部研发文档](docs/README.md)

`docs/`与`report/`中的方案、研究和逐轮诊断用于追溯开发过程，不属于正式报告正文。模型权重、虚拟环境及大型张量输出不纳入普通Git。
