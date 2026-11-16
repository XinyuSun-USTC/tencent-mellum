# 31 原生组合后的下一优化：恢复整批投影执行

完整ABC性能30已全部成功，但A121.2755tokens/s仅为给定官方638.2827的19.0%。继续优化执行开销，暂不重复完整ABC或再做大规模数值追踪。

## 源码依据与自包含改动

现有fused_moe_native_attention profile仍配置campaign groups=L。ATOM/atom/model_config/mellum_campaign.py:Campaign.__init__/wrap/split_call把QKV、O和router按请求边界分开调用，batch32因此反复发射小GEMM。它此前用于隔离批量shape数值影响，不是模型语义必须要求。原生Attention已去掉参考ragged路径，下一步验证这一兼容控制的性能成本。

新增显式profile=fused_moe_native_batched，只把campaign groups由L改为空；其余环境配置与原生组合逐字段相同。保持独立Q/K/V投影、参考RoPE/K写回、BF16权重和KV、FusedMoE、原生varlen/paged、greedy规则、TP/DP1、eager、prefix/chunked关闭。关闭本候选即选回原profile。未修改既有已验工作区或底层算子。

这是让当前batch中的投影按整批执行，未改题目固定batch/并发/输出参数。预期减少CPU发射与小GEMM数量，但没有预先承诺收益；整批GEMM可能改变数值，需以功能指标评估，不能直接继承旧组合144/164。

## GPU1命令

```bash
/opt/venv/bin/python /workspace/tencent-mellum-native-batched/scripts/run_mellum_candidate_capacity.py --profile fused_moe_native_batched
```

独立worktree基于9a3b3ca，固定GPU1、HTTP8151/engine8726；smoke→A64并发32→C20率0.3，35分钟执行预算。与性能26/30对照，保留实际batch、原生路径、全部请求/失败及源码hash。不改任何评测参数、不重跑HF、不追加完整ABC。

16项相关CPU检查通过：新profile正确选FusedMoE/native Attention，唯一配置差为L控制取消，原profile不变，未知组合拒绝/launcher prepare与容量失败停步保持。真实输入prepare-only与最终hash校验通过。GPU尚未执行。

## 决策

有显著吞吐或C尾延迟改善，则复用HF164仅跑候选功能评分；不以token位置或张量精确前置阻止评分。无收益则停止此候选，不为放弃版本做长评测。当前prefix在原生路径上有小C集TTFT改善，后续只在最快且功能合格的配置上整合。最终还需完整规模对比、空闲环境复测和独立题目正确性项。
