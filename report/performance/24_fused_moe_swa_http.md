# 24 FusedMoE与SWA CK组合HTTP结果

GPU1批次20261115T223459375865Z diagnostic_complete，smoke与A64/C20全成功。加载审计、source hash及全部原始产物复核通过；launch profile=fused_moe_swa_ck，SWA开关1，原生MoE日志保留。相同输入manifest与上一轮性能23逐项核对相等。

| 指标 | FusedMoE | +SWA CK | 本轮变化 |
| --- | --- | --- | --- |
| A64输出tokens/s/GPU | 57.0039 | 59.6120 | +4.58% |
| A64 TTFT p99秒 | 10.0410 | 6.2729 | -37.53% |
| A64 TPOT p99秒 | 0.53744 | 0.51917 | -3.40% |
| C20输出tokens/s/GPU | 30.9853 | 34.4001 | +11.02% |
| C20 TTFT p99秒 | 22.6893 | 3.8190 | -83.17% |
| C20 TPOT p99秒 | 0.44394 | 0.36259 | -18.32% |

A实际decode batch32，C19；双方A64/C20全部完成。SWA候选针对长prefill，TTFT改善显著、总体吞吐改善较小，符合优化范围；decode仍使用原参考桥，尚未解除该路径成本。小B2 TPOT约61.4ms，上一轮59.0ms，不能声称单流延迟全面改善。

## 结论与限制

保留FusedMoE+SWA CK为已有收益组合。以上为同一小负载/输入、不同时间单次观察，不是最终无干扰ABBA或正式A512/C200结果；两轮另一张GPU任务不同，不能排除共享资源影响。C20成功不证明C200长期稳定；p99为小样本诊断。SWA内核改变输出，组合尚无完整功能评估，不能直接继承FusedMoE单项的HumanEval成绩。

[比较数据与证据](../../artifacts/performance/24_fused_moe_swa_http/comparison.json)。全部SQLite、请求和服务日志保留原路径。

## 下一步

GPU0继续prefix probe，不改其工作区。GPU1使用现成fused_moe_native_attention profile，测试完整原生prefill/decode路径与FusedMoE组合。仍smoke→A64→C20/35分钟，不重跑HF/完整ABC，不以前期D200差异阻止性能筛选。若收益显著，随后复用现有HF164功能参考，只补候选功能评分；若无明显收益，保留已测SWA组合，精力转prefix/chunked方案。最终选择有实测收益的配置，正式门槛独立核验。
