# 精度优先第三项实验

基线为已通过的参考①②。候选只增加decode元数据复用，或再增加SWA group4；无prefix、chunk、Graph和原生专家。

| 候选 | token位置一致率 | 整序列一致 | 长组/16k匹配 | HumanEval（HF145/164） | 绝对差 pp | 完整检查 |
|---|---:|---|---|---|---:|---|
| r_decode_meta | 96.8125% | 174/200 | 256/256；64/64 | 148/164 | 1.8293 | local_candidate_checks_passed |
| r_swa_decode4 | 96.8125% | 174/200 | 256/256；64/64 | 148/164 | 1.8293 | local_candidate_checks_passed |
