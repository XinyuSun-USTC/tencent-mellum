# 51 参考路径KV单次写入筛选

## 目的与实现边界

独立分支`step1/qualified-kv`，工作树`/workspace/tencent-mellum-qualified-kv`，基于已通过验收的主版本提交927d2ba开发；不修改正在测ABC的交付工作树。

候选profile `r_reference_kv_once`与`r`只有`MELLUM_DIRECT_KV_WRITE:0→1`的差别。复用已有`mellum_direct_kv.install_direct_kv_rope`，将参考旋转后的K/V直接写入既有分页缓存，替代“原有旋转/写入后再覆盖K”的路径。专家、共享路由、参考Norm、分离QKV、SDPA、eager、32槽均不变；关闭resident系数、QK融合、原生注意力、graph、prefix。此阶段复用既有内核而非开发新内核，新增的是主参考路径的最小接入与分层对照工具。

## 实验顺序

1. **真实旧/新缓存通路对照**：构造实际PagedAttentionImpl，旧路径安装reference-KV纠正，候选路径安装direct-KV；两者使用同一模型RoPE参数和输入。覆盖full/YaRN与SWA/default两类，1/6/32/1024 token，物理页交界、负slot、1023/1024及16384位置，strided V。检查返回Q/K、完整物理K/V缓存、逻辑读回、未写入位置及候选输入不变。该步要求精确相等是对“只改变复制路径”的局部语义检查，不是新增模型生成门槛。三次局部计时取中位数，双方计时均包含相同的输入恢复，避免旧内核原地旋转污染重复测量。
2. **完整developer200与长组**：只运行候选，复用已验证HF参考，32槽/64输出；同时比较保存的32槽主版本输出。长组及16k单列；不跑HumanEval，不提前宣布最终验收通过。核对5631加载、28层参考专家/attention、113个参考Norm，以及实际prefill/decode的KV单次写入事件。
3. **可选端到端短对照**：R→候选→候选→R，每次独立启动同一工作树的服务，A32/B4、每条输出32 token。输入过滤、stream、温度及ignore_eos检查沿用数据库校验，唯一缩短项为输出长度。保留实际吞吐和延迟，不用本批替代正式A512/B50/C200或其增益。

任一局部检查失败停止后续；候选200位置率低于90则不跑性能。长输入单列观察，不另增局部全模型bitwise门槛。通过后状态为`screen_passed_full_acceptance_pending`，表示筛选完成，仍需最后组合的HumanEval及正式性能。

## 运行

仅在目标卡空闲后运行：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-qualified-kv/scripts/screen_mellum_reference_kv.py --gpu 1 --with-perf --budget-minutes 40
```

不测HTTP短性能可省略`--with-perf`；GPU参数可显式指定0/1。输出在`/workspace/results/mellum_reference_kv_screen/<运行号>/`：`cache/results.json`、`candidate_screen/*_generation.json`、`paired_perf/R1,K1,K2,R2/`、根`summary.json`与源码/输入/参考指纹。执行预算40分钟，CPU准备单独记录；首次内核编译可能延长，不将预算当成预计时长。

## 结果与切换规则

- cache对照失败：只分析布局/写入或返回Q/K，暂不扩展全模型定位。
- cache通过而生成退化：利用保存主版本输出和相同前缀定位，首先检查真实模型cache布局及prefill/decode差异。
- 生成通过但端到端无稳定收益：该候选不计作有效优化点，转向参考注意力辅助数据复用，不微调几十组参数。
- 生成和性能均有价值：保留为优化②，随后在独立分支开发优化③，再验最终累计组合。

本文件不记载尚未运行的GPU结果；CPU单元测试及CPU输入准备不替代GPU验证。
