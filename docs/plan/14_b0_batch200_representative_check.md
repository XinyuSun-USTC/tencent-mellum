# 14 B0真实batch32下的固定200条代表性检查

## 目的

容量已到32，但重复四条的小集位置匹配89.45%，见[性能11](../../report/performance/11_batch32_and_http_batch.md)。本轮只做一次代表性开发集对照，避免把小集分叉直接判正式失败，也避免容量通过即宣称B0可作为已验收基线。模型数学、服务、调度及点1实现都不修改。

## 输入与参考

复用Step0 `20261113T104458011376Z`：200条、普通greedy64、HF5.10.1、当时Step0与HF token/logits完全一致。检查完成状态、模型config SHA、HF环境、输入mask/绝对positions、64输出预算、无强制前缀、名称唯一；HF reference JSON SHA还与已纳入Git的性能07归档校验和比对。

保持原顺序、原token IDs，不重新编码或套模板，不把参考输出作为推理输入。164 HumanEval+36场景A仍是developer200，不是已确认身份的官方公开集。输入63–4113，168条<1024、32条≥1024。只比较token，不读取/重存12800个全词表logits，不重复HF GPU任务。

## 真实请求窗口与B0设置

新增`scripts/validate_mellum_batch200.py`，单次引擎加载：TP/DP1、BF16、eager、prefix/chunk关闭、无MTP，MELLUM_REQUEST_BATCHING=1、SWA_OPT=0。max_num_seqs32，prefill总token预算16384，max_model_len按最长输入+64向16对齐，KV仍为真实分页缓存。至少80GiB free才加载。

初次提交32条，每次收到完成请求后从队列补足未完成窗口，直到200全部提交/完成；每次提交显式记录seq ID与输入索引，检查实际Sequence输入前缀，再按ID映射输出，不能假定完成顺序等于输入顺序。保存提交/完成次序、相对时间和当时outstanding，校验窗口不超过32、没有重复/未知/丢失请求。

审计来自layer0真实forward metadata，warmup后清空审计，报告prefill/decode实际batch直方图，并要求观测到decode32；客户端窗口32不直接等同GPU batch32。末尾额外forward可能增加观测次数，直方图是forward观测而非生成token统计。审计有同步/落盘开销，整个wall明确不是正式吞吐或A/B/C成绩。

## 比较与判断

普通temperature0、max_tokens64、ignore_eos=False；提前EOS保留并计入位置差异分母，不删除难样本。每请求输出期望/实际token、结束原因、首差位置、匹配位置/整序列；总体及输入长度、HumanEval/场景A来源分层均列出两种比例。

90%位置与90%整序列两种歧义口径同时报告，不自行加全logits精确门槛，不将4条重复样本的89.45%代入该集合。`batch200_comparison_complete`表示执行和证据完整，不表示准确性正式通过。GPU运行采用冻结Torch版本，并检查与缓存HF共享的Torch版本一致、权重加载5631/5631。

若代表性位置比例达到要求，继续性能客户端/基准准备并保留差异说明；若明显不足，只对少量首差前同历史样本定位共享投影/MoE shape或索引/KV问题，不展开全层全样本逐位追踪。正式公开集与HumanEval口径边界不变；本轮不启动完整A/C或全164 Thinking生成。

## 执行与产物

```bash
/opt/venv/bin/python /workspace/scripts/validate_mellum_batch200.py
```

预计数分钟，15秒心跳、1800秒总阶段上限。产物`/workspace/results/mellum_batch200/<id>/`：plan.json、results.json、comparison.json、request_progress.json、batches.jsonl、loading_audit.json及batch200.log。每次完成打印累计进度。未知输出/丢失请求/未达实际batch32/运行时异常会中止并保留结果；accuracy未达90%本身仍完整报告，不伪装运行错误。

`--max-seqs 1/4/8/16`仅供后续有界定位，默认本轮只跑32，不先扫所有大小。请勿与使用内部8026端口的其它Mellum验证/服务同时运行。

## 当前检查

请求窗口补充、乱序完成ID映射、未知完成拒绝、提前EOS分母、实际batch统计及已有桥接/服务相关15项CPU通过；编译/空白检查通过。prepare-only `20261115T033338654371Z`通过。GPU在工具中仍不可见，未启动GPU或声称实测结果。
