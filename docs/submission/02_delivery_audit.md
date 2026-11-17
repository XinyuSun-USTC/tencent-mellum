# 当前交付核对

提交工作树为`/workspace/tencent-mellum-delivery-dual`，分支`release/dual-mode`。根入口`start_server.sh`默认性能优先＋step0123，通过`--mode precision`切换精度优先。三份正式材料为根目录`TECHNICAL_REPORT.md`、`AI_USAGE.md`和`WORKLOG.md`，历史研究与内部记录保留于docs、report及artifacts。

## 配置与实测

两种模式共享单卡32槽BF16 Step0，MTP与双卡模型并行关闭。性能路线①为原生计算与packed投影，②为路由、RoPE/KV、融合与Graph，③为前缀兼容/缓存及prefill元数据复用。精度路线①为Torch专家decode group4，②为参考RoPE/KV单次写入，③为decode元数据复用与参考SWA四请求分组。

性能最终组合A1135.6885token/s/GPU，B TPOT p50 4.8985ms，C200/200，C TPOT/TTFT p99 11.3619/1026.0455ms；开发200位置率66.8438%，未达到90%，HumanEval147/164，HF145/164。

精度最终组合A74.7620token/s/GPU，B TPOT p50 51.2821ms；开发200位置率96.8125%、长组256/256、HumanEval148/164，均通过本地对应检查。C三轮均1/200成功，其余各199条180秒超时，不提供有效完整负载尾延迟中位数。正式报告分别列出两条路线的累计矩阵，旧第三项未混入最新数据。

## 已完成验证

- 112项相关CPU测试通过；13种合法开关组合从真实入口完成prepare-only核验。
- 两种模式各四行累计GPU烟测通过，检查5631张量加载、HTTP/SSE、usage、实际模式及端口回收。
- 正式文档链接及当前配置、证据索引、烟测源码哈希由`python scripts/verify_submission.py`核验。
- 模型、虚拟环境和完整results目录不纳入Git；必要摘要、trace、逐题判定及SHA索引保留。各分支及完整提交历史继续保存，不squash。

## 提交边界

当前实现可以作为实际交付的双模式研究版本，但尚无同时满足90%token一致率与C200全完成的配置。页数由最终排版核对。裁判隐藏集及空闲机器三次中位数为最终复测，不能由本地数据代替。最新交付分支仍需推送远程并明确告知评审所用分支或提交。

证据入口：[技术报告](../../TECHNICAL_REPORT.md)、[双模式集成](15_dual_mode_entrypoint.md)、[GPU烟测索引](../../artifacts/submission/dual_mode/gpu_smoke/index.json)。
