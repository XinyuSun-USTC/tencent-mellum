# Mellum 2.1 在 ATOM 上的适配与推理优化

在题目提供的冻结环境中，实现 Mellum2.1-12B-A2.5B-Thinking 的 BF16 单卡推理，并研究批处理、KV 写入和长输入混合负载优化。

| 交付材料 | 文件 |
|---|---|
| 技术报告，目标不超过四页 | **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** |
| AI 使用说明，目标一页 | **[AI_USAGE.md](AI_USAGE.md)** |
| D1—D7 每日工作日志 | **[WORKLOG.md](WORKLOG.md)** |
| 启动与验收状态 | **本页**、[交付核对表](docs/submission/02_delivery_audit.md) |

文档为可编辑 Markdown，导出页数由作者最终确认。未取得的实验数据以“—”表示，不代表零或通过。

## 当前状态

**尚未确定同时满足正确性与完整 C 容量要求的最终交付配置。** 当前参考组合为专家分组 4 + KV 单次写入，已通过完整本地正确性检查，但 C200 仅成功 1/200。分段调度把 C60 的成功数提高到 21–24/60，仍有超时；一轮解码配额和前缀缓存实验正在进行，结果待填。

| 已验证项目 | 参考组合结果 |
|---|---|
| developer200，前 64 token 逐位置一致率 | 12392/12800，**96.8125%** |
| 完整 64-token 序列一致数 | 174/200，另行保留，不与位置比例混用 |
| 长输入组 / 单独 16k | **256/256 / 64/64** |
| HumanEval pass@1 | 组合 **148/164**，HF **145/164**，绝对差 **1.8293 个百分点** |

以上为本地口径：developer200 为 164 条 HumanEval 提示与 36 条场景 A 提示，其与裁判公开集的精确对应关系未确认。推理为单卡、MTP 关闭。两张卡运行独立实验，不构成双卡模型推理。详见[组合证据](artifacts/submission/reference_combination_acceptance/summary.json)。

## 启动入口与代码位置

本工作树 `release/submission` 的现有入口仍启动旧参考配置 `r`，**不是当前参考组合或新分段方案**：

```bash
./start_server.sh --mode conservative --gpu 0
# 仅检查参数与冻结版本，不启动 GPU
./start_server.sh --mode conservative --dry-run
```

默认模型位于 `/data/models/Mellum2.1-12B-A2.5B-Thinking`，解释器为 `/opt/venv/bin/python`，HTTP 地址为 `http://127.0.0.1:8000/v1`。冻结版本见 [environment.json](configs/submission/environment.json)。旧 `--mode aggressive` 未通过本地正确性门槛，仅保留为历史实验入口，不推荐作为正式验收配置。

新候选的一键启动与独立开关已在分支 `release/qualified-entrypoint` 完成 CPU 集成准备，尚未合并为本工作树默认入口，也未完成该入口的 GPU 启动检查。当前实验分布如下；每个分支保存完整源码，交付时需选定并集成一个自包含版本。

| 分支 / 本地工作树 | 用途 |
|---|---|
| `step1/qualified-combination` · `/workspace/tencent-mellum-qualified-combination` | 已通过本地正确性的参考组合；C200 失败 |
| `step1/reference-chunked-prefill` · `/workspace/tencent-mellum-qualified-chunked` | 四轮解码配额分段实验，C60 已测 |
| `step1/c-prefix-scheduling` · `/workspace/tencent-mellum-qualified-c-prefix` | 一轮配额与前缀缓存实验 |
| `release/qualified-entrypoint` · `/workspace/tencent-mellum-qualified-entrypoint` | 最终入口集成准备，尚未晋级 |

上述绝对路径仅是本容器研发位置，不是最终交付运行依赖。**现阶段不存在已完成验收的新方案默认启动命令。** 选定方案后，须统一源码、预设、开关、启动检查和消融最后一行，不能只修改文档中的版本名称。

## 当前优化方案

1. **参考兼容的批处理与专家执行**：增加多请求执行能力，解码专家按四请求分组；保持参考路由和计算语义。
2. **RoPE／KV 写入去冗余**：参考旋转结果直接写入分页缓存一次，删除重复处理。
3. **C 场景干扰治理（评估中）**：分段预填充与解码服务配额；前缀缓存作为独立增量实验。实际交付选点以测量和最多三个优化点的题目限制为准，不预先合并名称代替消融。

短负载增量收益与 C60 结果见技术报告；它们不能替代完整 ABC。原生内核、graph、融合等历史实验的性能与未通过结果保留在 [内部实验记录](report/performance/36_combined_full_abc.md)，不赋予当前候选。

## 后续收敛与资料导航

待当前实验结束后：选择容量有改善的候选 → 同配置 C200 与完整正确性 → 最终入口检查 → 完整 A/B 和累计消融 → 更新正式报告及版本标签。失败场景不以成功请求尾延迟报告为通过。

- [当前补测与集成计划](docs/submission/03_completion_experiments.md)
- [实验与证据索引](artifacts/submission/index.json)
- [当前内部文档导航](docs/README.md)
- [来源与许可证](docs/submission/01_sources_and_licenses.md)
- [题目原件](docs/requirements/2026%20CSIG云一计算加速团队实习校招面试题.pdf)

历史 Step0 保留于 `step0-local-validated`。其他旧发布标签只对应当时快照，不表示新方案通过；Git 历史不 squash。模型权重、虚拟环境与大型张量不纳入普通 Git。最终打包与验证流程见交付核对表。
