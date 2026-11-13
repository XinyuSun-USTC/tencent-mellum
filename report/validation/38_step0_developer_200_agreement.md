# 38 Step0：200条开发生成一致性对照

## 目的与口径

长输入64token回归已通过（报告37）。本阶段用200条固定prompt检查普通生成的一致性规模，不再添加数值定位开关。**该集合不是已确认的官方200条公开验收集，也不是HumanEval pass@1评测。** 其结果可作为开发正确性证据，不能冒称题目官方通过。

来源：`/data/datasets/HumanEval.jsonl`全部164题的prompt字段，包装为完成Python函数的用户请求；加`scenario_a_prompts.jsonl`原文件顺序前36条prompt。保留source_id、来源文件SHA256，不使用canonical_solution/test字段构造prompt、不执行生成代码，不筛选已知容易通过的样例。场景A材料是题目提供的性能数据，未确认其公开集身份，因此整体明确标注开发集合。

构造器`scripts/prepare_mellum_acceptance_inputs.py:build/main`。固定输入`results/mellum_validation_inputs/developer_200_v1/inputs.json`，200条不同原始prompt，实际token长度63–4113，均使用本地官方chat template、enable_thinking=True，normal greedy、最多64token、正常EOS。输入ID、全1 attention_mask和绝对position固定供双方共用。不同既有固定输入不得覆盖。

## 运行与输出

沿用correctness-candidate的同一数学路径，HF普通generate与ATOM正常生成，无forced计划。保存原始prefill及逐步完整logits，校验实际HF prefill grouped_mm / decode batched_mm。比较现有token_summary的完整序列一致数及token位置一致比例，两者都报告，避免擅自解释题目≥90%的分母。EOS提前停止时必须如实报告实际生成数量，不能补齐伪token。

此规模最多12800个输出token。保留完整逐步张量可能占约10GiB，已检查工作区约4TiB可用。运行显著长于10条例，不承诺固定耗时。新增`run_mellum_validation.py --stage-timeout`，默认仍300秒；本轮设3600秒是**每个阶段的保护上限，不是预计耗时**。15秒心跳、worker死亡监测及子进程清理不变；不要因原300秒过短误杀规模验证。

运行器CPU5项通过；实际输入生成和prepare-only检查通过。GPU本阶段未启动。命令：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite diagnostic --inputs /workspace/results/mellum_validation_inputs/developer_200_v1/inputs.json --correctness-candidate --stage-timeout 3600
```

失败优先读取已存首差和原始logits，避免无证据重跑。通过后仍需独立HumanEval HF/ATOM配对执行及服务交付验证；不能把使用HumanEval题目做64token一致性比较算作pass@1评测完成。

## 200条GPU结果：完整通过

运行`results/mellum_validation/20261113T104458011376Z`，ATOM`results/atom_mellum_gpu/20261113T105239431753Z`，status=tokens_and_logits_exact。实际200条，每条64个输出token，无EOS提前截短；200/200完整序列精确（100%）、12800/12800 token位置精确（100%）；12800个采样前完整logits向量全部逐值精确，200个独立prefill logits误差均0。加载5631/5631，无missing/unexpected，sampling_mode=normal、ignore_eos=False，未使用强制历史。

无论按完整序列比例或token位置比例，本开发集合均高于题目90%门槛；但集合身份限制仍存在，不能替代未获得的官方指定集结论。HumanEval题目在此只比较64token生成一致性，未做完整答案生成和执行，不等同pass@1。

阶段结论：**核心Step0适配及当前开发生成一致性验收已完成**；另有8192边界/16k各64token精确证据（报告37）。完整题目正确性门槛仍欠有效HF/ATOM HumanEval配对结果，最终服务交付也未验收。

## 工作重心调整

重新核对题目后，完整logits/中间层逐值精确只是诊断工具，不追加为题目验收条件。200条结果良好后结束同类数值定位和集合扩展，固定当前适配候选及其配置。下一主线推进性能测量所需的最小服务入口和profiling，选择最多3个有瓶颈证据的优化点；HumanEval作为正式门槛单独补齐，可与性能分析交错推进。无需先完成一套额外的完整并发正确性测试才允许profiling；实际测量负载所需的批量能力仍需实现并验证。尚未完成HumanEval前不宣称题目全部正确性门槛通过。

本次只核验产物并更新阶段记录，不启动新GPU任务、不修改模型数学、不自动中断其他运行。阶段状态用“核心适配完成、正式验收待HumanEval、开始转向性能分析”，避免将所有最终交付项无限扩展为Step1前置要求。
