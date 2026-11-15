# 17 直接索引decode：仅一次D200完整生成补证

## 要回答的问题

[性能14](../../report/performance/14_moe_routing_and_indexed_decode.md)中indexed_decode局部decode32约4.892×加速，额外分配64.20→3.71MiB；但正常greedy16的S32/L32位置匹配94.34%/93.95%，相对CPD下降超过计划16自设的2个百分点，未进入D200。

小集不等于完整200条，也不能将该自设筛选当题目90%门槛。用户批准下一步：**不改候选，只补一次固定200条、自然greedy64的HF比较**。本轮不重跑微基准/完整12-worker矩阵，不自动继续ABBA、HTTP、16k或HumanEval；先拿到缺失证据，再决定性能路线。

## 冻结的实验配置

入口：[run_mellum_indexed_d200.py](../../scripts/run_mellum_indexed_d200.py)，复用现有[engine worker](../../scripts/run_mellum_batch_campaign.py)。

- 数学实现保持commit `874115a`的indexed_decode：CPD的L请求shape、prefill按请求专家GEMM与批内共享路由保持，decode使用直接索引Triton GEMV。既有SWA优化关闭。
- prepare时核对上批归档指纹中的全部ATOM推理源码、runtime配置、模型配置与权重索引；候选控制配置与原screen逐项相同。只增加独立调度和分析脚本，不修改候选、attention/cache/sampler。
- D200原200条输入token、次序、来源、64token预算与缓存HF输出不变，HF SHA仍为`407cab507b32e45191f6cd5431709824a5b9f845442936dfc018c3668a4d7df1`。完整cases列表hash与上批D200归档一致。
- 单卡TP/DP1、BF16/eager、maxseq32、maxlen4192、max batched tokens16384、util0.35、prefix/chunk关闭；正常temperature0/ignore_eos=False，不强制历史，不缓存期望输出作推理。
- 一次真实有界32窗口完成200请求，继承seq-ID映射和丢失/重复完成检查、5631权重加载审计与真实batch审计。另核验layer0实际decode为`indexed_triton`且观测requests32，拒绝将其它后端结果误认作此候选。

## 本轮如何判断

**唯一分支门槛：本地developer200总体token位置匹配≥90%。** 不使用计划16的相对CPD下降2pp/0.5pp限制，也不把每个分层各≥90%作为新的隐藏硬门槛。

同时完整报告：总体及短/长、HumanEval来源/scenario A来源的位置/完整序列匹配、首差、EOS/输出长度、完成数与实际batch。输出不足64仍保留HF分母，不删除失败条目。完整序列是否≥90%单独标记，公开集身份/计数歧义保留；“位置通过”不是官方全部正确性验收或HumanEval pass@1通过。

复用上批已验证CPD和共享路由D200原始输出文件，核验hash；生成后离线比较逐题相对HF的匹配增减、相对两基线的完整输出相同数。**不重新运行两基线**，也不将不同worker单次wall比值当作性能加速比。

状态区分：

- `developer_position_passed`：本轮执行有效，总体位置≥90%。
- `developer_position_below_90`：本轮执行有效，但总体位置<90%。这是开发数据结论，不冒称裁判隐藏集结论。
- `execution_invalid`、`failed`、`blocked_environment`、`partial_budget_limited`或`analysis_failed`：执行、环境、预算或分析问题，不按准确性不通过计。

一次D200不当作重复稳定性实验。无论结果如何，本入口不会据此启用生产候选或自行修改kernel。

## 执行方式、预算及恢复

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_indexed_d200.py
```

预计3–6分钟（估算非承诺），默认单job上限900秒、累计预算20分钟。只有一个GPU job，不自动重试、不启动第二轮实验；继承15秒心跳、仅清理自有进程组、原子manifest和产物hash校验。

结束只需回传根`RESULT ... output ...`。根目录提供`summary.json`、`report.md`、`baseline_comparison.json`，原始D200逐条结果/审计/日志在`jobs/indexed_D200/`。

只检查输入/指纹、无GPU：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_indexed_d200.py --prepare-only
```

中断后恢复：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_indexed_d200.py --resume /workspace/results/mellum_indexed_d200/实际目录
```

源码、输入、候选/环境配置及成功产物hash不变才可恢复；完整成功job只重做分析，不启动第二次GPU。允许一次由用户显式resume触发的失败重试，总启动数最多2，累计预算不重置；硬中断落盘前尾段时间无法精确补计。已失败尝试保留，不覆盖原始数据。

## 交付验证

本轮只改脚本入口与指纹覆盖，没有数学实现变更。CPU测试覆盖：90%精确边界、序列/分层不替代总体门槛、单worker无下游分支、GPU环境失败不记准确性失败、完成后恢复不再启动GPU；同时回归既有MoE/批次调度测试。prepare-only验证原200输入与候选/基线hash。

本工具环境无`/dev/kfd`，真实GPU运行仍由用户终端执行。24项相关CPU测试通过（新入口4项、既有MoE/批次调度20项），`py_compile`与`git diff --check`通过。最终prepare-only目录：`/workspace/results/mellum_indexed_d200/20261115T132810810022Z`。

## 执行结果

运行`20261115T133034290474Z`一个worker、约3.79分钟完成，200条均64token，实际indexed decode/batch32通过审计。总体位置78.2266%、完整序列53.5%，未达到本轮90%位置门槛。按预定边界停止当前版本晋级，无追加性能/HTTP/HumanEval。详见[性能15](../../report/performance/15_indexed_decode_d200.md)。本计划已执行，不默认重跑。
