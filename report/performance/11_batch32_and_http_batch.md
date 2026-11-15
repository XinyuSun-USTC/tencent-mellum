# 11 B0容量至32与HTTP多请求：执行可用，准确性仍待代表性验证

## 结论

执行/容量里程碑到达：真实decode batch8/16/32完成，无本轮记录的OOM或请求丢失；HTTP形成真实batch2，正常完成前端记账修复符合本轮pending上界。**不能据此宣称批量正确性或正式A/B/C基准通过**：bs32小集token位置匹配89.45%，同输入副本出现不同生成。

下一步先以固定developer200、HF缓存参考做一轮真实批量正常生成对照。通过代表性生成检查后才建立可作为成绩的性能基线；开发profile可并行采短片段，但必须标记准确性待核验。避免因小集差异直接展开无限bitwise追踪，也不因容量可用就忽略任务门槛。

Offline `20261114T235336450680Z`，HTTP `20261115T032334158083Z`；源指纹与当前`3e48282`下文件复核一致。[原始轻量证据](../../artifacts/performance/11_batch32_and_http_batch/manifest.json)，[计划13](../../docs/plan/13_serving_batching_and_workload_baseline.md)。本轮仅分析产物/归档，无代码数学修改或新GPU任务。

## 容量与生成

reference B0、request batching=1、max_num_seqs32、TP/DP1、BF16/eager、prefix/chunk关闭，权重5631/5631。四个不同输入/预算重复组成容量阶段，不是独立32条新样本；单次模型加载依次容量阶梯。

| 阶段 | 实际prefill batch | 实际decode batch | 位置匹配 | 完整序列 | 阶段wall s |
| --- | --- | --- | --- | --- | --- |
| 单请求4例 | 1 | 1 | 64/64 | 4/4 | 逐例2.70/1.93/0.63/0.63 |
| pair | 2 | 1/2 | 46/48 | 1/2 | 2.04 |
| ragged4 | 4 | 1/2/4 | 62/64 | 3/4 | 2.30 |
| reverse_reuse | 4 | 1/2/4 | 64/64 | 4/4 | 2.31 |
| capacity8 | 1/7 | 2/4/8 | 128/128（100%） | 8/8 | 3.06 |
| capacity16 | 2/7 | 4/8/16 | 254/256（99.22%） | 15/16 | 4.56 |
| capacity32 | 1/5/7 | 8/16/32 | 458/512（89.45%） | 26/32 | 7.83 |
| late_arrival | 2 | 1/2/3 | 62/64 | 3/4 | 2.40 |

prefill预算4096导致分多次prefill，不表示decode未达32。迟到插入时原队列仍有1请求，随后decode3；反序/请求结束后复用完成。阶段wall含不同输出预算、验证审计落盘及请求长度，**不是正式吞吐、TPOT或A性能加速比**，不能按本表直接外推A512×256。

bs32详细：humaneval_0的8个副本中4个16/16、4个14/16；humaneval_1中6个32/32、2个9/32；window1023/1025各8副本均8/8。每个短HumanEval提示词有两种输出，首次差异分别在第15/10个token。说明batch/位置相关敏感性真实存在；本轮未获得首差前逐层对照，不确定归因于投影、MoE、注意力/缓存或其组合，不能直接断定是可忽略舍入。

这不是题目正式公开集，89.45%不直接宣布正式失败；但也不能将capacity8/16通过外推到32，或用当前代码已执行就认定B0准确性验收完成。继续按token位置与完整序列双口径记录，不新增全logits精确要求。

## HTTP batch4配置实测

max_num_seqs4、reference桥接、实际prefill/decode均观察到1/2。客户端只有两个同时请求，因此本轮证明HTTP并发到2，不宣称实际HTTP batch4已经观测。

- 两条提示词各非流/stream4次，文本与HF展示全部一致，usage16/32正确，SSE均有DONE。
- 并发两请求：humaneval_0文本与参考不同，humaneval_1一致，usage均正确；对应offline批量差异，未当作协议成功掩盖生成差异。summary.review_required仍True。
- chat原始reasoning开头换行差异保留，外围reasoning空白归一后相同，最终content不做宽松归一，DONE正常。
- `max_logged_pending_requests=2`、frontend_pending_bound_ok=True，支持正常完成登记清理修复有效；不是所有取消/异常路径或KV内存释放的完整证明。
- owned服务及EngineCore正常关闭，无本轮执行错误。异常断开/abort仍未验证。

## 下一步边界

先构造developer200相同输入/64token输出的B0多请求对照，复用现有HF，不重跑参考；记录每请求实际batch、到达/结束顺序及输出，区分总体/输入长度分层，不把重复4提示词当代表性集。至少覆盖真实decode32与正常EOS，batch变化也要记录。

若开发集位置口径达到90%，继续负载基准准备，保留数值差异说明，不追求额外bitwise精确。若明显不足，只在首差前同历史的少量请求做有界定位，先核对共享投影/MoE shape及请求索引/KV，不重新启动一套全层全样本诊断。无论结果如何，最终公开集身份/分母与HumanEval规则边界不变。

当前不启动完整A/C或将B0结果作为已合格成绩；可准备独立客户端/少量profile，但开发测量必须标记正确性状态。基线桥接引入的执行策略与成本继续单独披露，不能藏成新内核的免费收益。
