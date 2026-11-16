# 单卡集成：SWA group4、取消修复与C减负

## 选择依据

SWA group4+metadata在同轮隔离decode批32达到60.3174token/s，相对R1/R2均值42.8573提升40.74%。三候选D200/长组全部与已验收参考组合的实际序列相同，但未独立跑完整HumanEval。交错取消role/token两例均保住128token存活输出、回收787块，取消ID移除后无新step，恢复请求8token完成，最终队列0/0。

这些支持集成，但60.32仍低于单卡C的76.8token/s需求，不能跳过容量筛查或宣称完成。用户选择继续单卡，P/D暂缓。

## 独立工作树与变量

工作树 `/workspace/tencent-mellum-swa-abort-integration`，分支 `step1/swa-abort-integration`，基于a939efe并保留e15b691的独立引入历史。原计算、取消与已验收参考工作树不修改。

| profile | 固定计算与生命周期 | 额外机制 |
|---|---|---|
| r_swa4_abort | metadata一次解析、SWA group4、专家group4、KV单次写入、断连ABORT | 无prefix/chunk |
| r_swa4_abort_prefix | 同上 | 参考前缀缓存 |
| r_swa4_abort_prefix_chunk1024_d4 | 同上 | 前缀缓存、1024分段、每段后最多四轮decode |

仅配置开关组合，没有在集成中引入新的计算内核或改动数学。全局注意力仍按请求执行。分段桥接、KV布局与单卡eager/无MTP/无KV传输限制保持。

## GPU安排

每个profile先做现有四提示单独、混合与重复请求生成检查，包含16k；随后C60：原60条、rate0.3、输出256、单请求超时180。容量服务增加`--phase-audit`保存真实abort和step，记录其扰动，结果属于开发筛查，不是正式ABC。

GPU0只测第一行；GPU1顺序测后两行。每个服务单卡，N仍为1，两卡用于独立实验。预计一行约7–12分钟，两行约15–25分钟，预算30/60分钟不是单请求超时。

集成回归171项CPU检查通过。子方案已有GPU证据不自动赋予新组合；C60未全完成时不追加C200或整套HumanEval。完成时仍须更长容量验证与最终完整正确性。若仍不足，保留取消回收的正确性修复，继续定位剩余decode执行成本，不把前缀潜力直接当加速比。
