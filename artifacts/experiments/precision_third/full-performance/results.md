# 精度优先第三项实验

基线为已通过的参考①②。候选只增加decode元数据复用，或再增加SWA group4；无prefix、chunk、Graph和原生专家。

完整A512/B50

| 配置 | A token/s | A完成数 | B TPOT p50 ms | B完成数 | 隔离decode8/32 token/s |
|---|---:|---|---:|---|---|
| qualified_candidate | 74.7620 | 512/512 | 51.2821 | 50/50 | — |

隔离decode为最后prefill之后的实际callback交付，不是混合C容量；带阶段审计。短测只作筛选，完整A/B仅对通过正确性的候选执行。
