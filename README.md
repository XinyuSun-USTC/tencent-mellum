# Mellum 2.1 on ATOM — delivery

冻结 AMD ROCm 环境中的 Mellum2.1-12B-A2.5B-Thinking BF16 适配与性能实验。交付分支：`release/submission`。单卡运行，MTP 关闭。

**默认保守模式已在本地 200×64 token 和长输入组逐 token 完全一致。激进模式有性能收益，但未通过本地 90% token 一致率门槛，因此不能宣称整体考核通过。** 两种运行策略的选择是独立交付开关，不算作 Step0 或一个优化点。

## 一条命令启动

在题目给定容器中，从本仓库根目录执行：

```bash
./start_server.sh --mode conservative --gpu 0
```

OpenAI 接口 `http://127.0.0.1:8000/v1`，健康检查 `/health`。默认模型 `/data/models/Mellum2.1-12B-A2.5B-Thinking`；可用 `--model PATH` 替换同架构 BF16 checkpoint。推理解释器固定 `/opt/venv/bin/python`；无需安装新版 Transformers、无需 HF 参考虚拟环境、无需其他实验工作树。`ATOM/` 是完整源码而非 submodule。启动器清理继承的 Mellum 实验变量，写入实际配置、源文件哈希和版本记录。

```bash
# 激进实验配置：显式选择；未通过 token 门槛
./start_server.sh --mode aggressive --gpu 0 --audit
# 换 HTTP/引擎端口；两者必须不同
./start_server.sh --mode conservative --port 8001 --engine-port 8027
# 只检查配置与版本，不启动 GPU
./start_server.sh --mode conservative --dry-run
# 实际启动两种模式，验证权重加载、HTTP、流式接口并关闭自有进程
/opt/venv/bin/python scripts/smoke_submission.py --gpu 0
```

生产默认上下文上限 16640；保守模式调度槽 1，激进模式 32。冒烟检查为节省时间使用 512 上下文、槽 1、生成 8 token，不能替代生产规模验收。输出位于 `results/submission_server/` 与 `results/submission_smoke/`，不纳入普通 Git。

## 模式与验收边界

| 配置 | 本地 200 条前64 token | 长输入组 | HumanEval | 完整 ABC |
|---|---:|---:|---|---|
| conservative，profile r，槽1 | 12800/12800，100% | 256/256，100%；16k 64/64 | 此精确预设未单独测量 | 此精确预设未测量 |
| aggressive，最终组合，槽32 | 8045/12800，62.85%，不通过 | 138/256；16k 63/64 | 150/164；HF145/164 | A1139.78 token/s/GPU；B TPOTp50 4.865ms；C TPOTp99 11.502ms、TTFTp99 1026.0ms |

HumanEval 关闭 thinking、上限2048、官方 openai/human-eval 评测套件。候选较 HF 高5题/+3.05个百分点；按题面“差距”绝对值≤2pp的字面口径，仍不宣称通过。公开200条的官方精确口径未完全确认，本地集称 developer200。最终裁判的隐藏集、空闲机三次中位数未由本项目完成。两卡用于并行独立实验，交付推理为单卡，不主张双卡并行或 MTP 支持。

**不得把保守模式的正确性与激进模式的速度拼接成同一版本通过证据。** 保守模式是后期 reference control，不等于历史 `step0-local-validated` 标签的原始 Step0。

## 独立组件开关

在 `--mode aggressive` 后追加下列开关。它们用于关闭组件、复现和消融；任意新组合均不自动继承已有正确性结论。

| 开关 | 关闭/替代内容 |
|---|---|
| `--disable-graph` | 关闭 decode graph，使用 eager |
| `--disable-qk-fusion` | 关闭融合 Q/K Norm+RoPE |
| `--disable-native-routing` | 关闭原生 AITER routing |
| `--disable-prefix` | 关闭 KV prefix caching |
| `--disable-metadata-reuse` | 关闭 prefill 元数据复用 |
| `--separate-qkv` | packed QKV 改为分离投影 |
| `--disable-native-attention` | 使用参考 attention，关闭 direct KV 写入 |
| `--disable-native-moe` | 使用 Torch 专家后端 |
| `--reference-norm` | 使用参考 Norm |
| `--disable-direct-kv` | 关闭直接写入分页 KV |
| `--disable-resident-rope` | 关闭常驻 RoPE 系数路径 |

`--max-num-seqs`、`--max-model-len`、`--max-batched-tokens` 可调整容量；后两者须满足 batched tokens≥model len。`--audit` 保存实际执行后端、graph replay、prefix/metadata 事件；默认不记录逐层张量。

## 交付材料与复现

- [技术报告 Markdown 主稿（按四页内容组织）](report/submission/01_technical_report.md)。
- [AI 使用说明 Markdown 主稿（约一页）](report/submission/02_ai_usage.md)。
- [交付清单与缺口](report/submission/03_delivery_checklist.md)。
- [每日简报 WORKLOG](WORKLOG.md)；[完整研发日志](report/worklog.md)。
- [提交依据与哈希](artifacts/submission/index.json)，[来源与许可证](SOURCES.md)。
- [Step0总结](report/validation/39_step0_summary.md)、[完整ABC](report/performance/36_combined_full_abc.md)、[最终候选验收](report/validation/40_optimized_candidate_acceptance.md)。
- [题目原件](docs/requirements/2026%20CSIG云一计算加速团队实习校招面试题.pdf)、[历史文档索引](docs/README.md)。历史计划中的“下一步”以交付说明为准。

正式文档以中文 Markdown 为准，便于修改润色。技术报告按四页内容分段，AI说明按一页篇幅编写；最终导出页数取决于排版。现有英文PDF仅为可选历史预览，Markdown修改后须重新导出核对，不能假定二者自动同步。

冻结推理版本见 `configs/submission/environment.json`。CPU 交付检查：

```bash
/opt/venv/bin/python -m pytest -q scripts/tests/test_submission_entrypoint.py
/opt/venv/bin/python scripts/verify_submission.py
```

完整性能复现（评测客户端需按已有独立环境脚本准备，不改变推理环境）：

```bash
/opt/venv/bin/python scripts/run_mellum_full_campaign.py --gpu 0 --profiles sprint_graph_qk_router_prefix_metadata --budget-minutes 90
```

本地使用 EvalScope1.15 的兼容适配，A512/B50/C200；保存实际参数与统计。数据/模型在 `/data`，原始大张量、虚拟环境和模型不进 Git。精选结果随仓库提供，其余记录原路径及 SHA256，原始日志可在当前容器核验。历史实验脚本可能引用 `/workspace/results` 和独立评测环境，不是启动服务的依赖。
