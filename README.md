# Mellum 2.1：最终累计消融候选

基于冻结环境的BF16单卡适配。分支 `release/final-matrix`，完整源码在 `ATOM/`，启动不依赖其他工作树。**当前是待最终验收的交付候选，不宣称C通过**：最近开发筛查最佳为43/60；此前①②组合完整正确性通过，不能替代最终组合的验证。

题目要求材料置于根目录：[技术报告](TECHNICAL_REPORT.md)、[AI说明](AI_USAGE.md)、[工作日志](WORKLOG.md)。内部实验与取舍保留在 `report/`、`docs/`，最终消融执行规则见[计划65](docs/plan/65_final_delivery_matrix.md)。

## 一条命令启动

在仓库根目录：

```bash
./start_server.sh --gpu 0
```

默认等于消融最后一行 `step0123`，端口8000，单卡、32槽、eager、BF16、无MTP。此默认仅表示所测最终候选，不表示已合格。可显式选择任意累计行：

```bash
./start_server.sh --row step0 --gpu 0
./start_server.sh --row step01 --gpu 0
./start_server.sh --row step012 --gpu 0
./start_server.sh --row step0123 --gpu 0
./start_server.sh --row step0123 --prepare-only
```

| 行 | 槽数 | 启用内容 |
|---|---:|---|
| step0 | 1 | 参考数学路径；最终源码关闭三项优化重跑 |
| step01 | 32 | ①参考兼容批处理、共享routing与专家group4 |
| step012 | 32 | ①＋②参考RoPE与KV单次写入 |
| step0123 | 32 | ①②＋③SWA group4、decode元数据复用、prefix、chunk1024/d4、断连回收 |

③包含数个协同机制，分别披露局部证据，累计矩阵只衡量整体增量；不冒充每个机制的独立收益。三个点的最终计认由题目评审决定。`scripts/mellum_delivery_matrix.py` 是入口、性能与正确性测试共用的配置定义。

历史保守/激进入口保留为 `start_legacy_server.sh`，不属于本轮累计矩阵，不混用其性能与当前正确性。

## 完整实验

`run_delivery_matrix.py` 支持分卡按行运行、`--prepare-only` 和 `--resume`；`--kind all`依次运行完整性能及每行本地正确性。预算是每个性能campaign或每行正确性阶段的上限，不是整条命令总时长。失败任务及超时保留，不伪造尾延迟；已完成性能任务可校验后跳过，失败正确性需新建运行以保护证据。

结果在 `/workspace/results/mellum_delivery_matrix/`，包含原始日志、源码/数据哈希、完整分母和 `matrix.md`。双卡用于独立实验，每个服务仍N=1；共享主机干扰与本地客户端口径保留，结果不是裁判空闲机器三次中位数。HumanEval执行官方固定套件的生成代码评测，复用已核验HF参考，不升级推理依赖。
