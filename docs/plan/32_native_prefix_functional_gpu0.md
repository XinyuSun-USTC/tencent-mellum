# 32 GPU1性能测试期间：GPU0补齐原生prefix功能评估

GPU1正在独立native-batched工作区测试整批投影，保持运行源码不变。GPU0评估已具有C20性能收益的原生Attention+FusedMoE+prefix，复用固定HF参考，不重新生成HF。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-prefix-functional/scripts/run_mellum_humaneval_pair.py --attention native --prefix-caching --candidate-gpu 0 --reuse-hf-run /workspace/results/mellum_humaneval_recovery/20261115T213706098181Z --budget-minutes 35
```

## 实际配置与边界

独立prefix-functional工作区基于c34a96a，与性能29原生prefix数学相同；功能入口基于e55cd7a扩展候选GPU、prefix开关和命中审计。固定GPU0/engine9126，无HTTP，不与GPU1的8151/8726冲突。保留L请求级投影控制，本轮不提前混入仍在测性能的整批投影候选。

完整164题、关闭thinking、最多2048token、batch32，与HF引用的inputs hash一致。加载5631源张量审计、实际batch32、aiter_varlen/paged/FusedMoE路径核验保留；额外记录native cached prefill实际命中量。若短提示词未形成完整块命中，功能评分照常完成，但不能把该结果声称所有缓存行为均获验证。

GPU0候选强制复用HF引用，CLI拒绝在同卡另启HF。官方隔离评分/提取规则与此前一致，不用token位置或逐位张量差异前置阻止。35分钟执行预算；引用hash在prepare阶段检查，HF评分仅CPU执行，不占另一张卡。

## 核验与结果分支

10项CPU检查通过，覆盖既有完整功能计数、引用hash拒绝、native路径审计、prefix开关约束与GPU0/HF冲突拒绝。真实164题prepare-only、现有参考输入与完成hash验证通过。GPU尚未执行，不改冻结依赖。

若相对HF差距≤2pp，保留prefix为可用组合；GPU1若整批投影获得明显收益，随后在该更快配置上组合并补最终功能评估。若本轮超标，先核对命中、截断/提取失败和逐题变化，不立即回到无界浮点追踪。功能结果不替代公开200/16k独立要求；最终配置再完成累计消融与空闲正式规模复测。
