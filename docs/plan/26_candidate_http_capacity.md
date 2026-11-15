# 26 HumanEval恢复期间：GPU1候选HTTP与容量诊断

目的：让GPU0完成HF功能参考，同时用GPU1检验已有实测收益的FusedMoE在真实HTTP负载下的服务和容量。此批不依赖HumanEval通过，不等同正确性验收，不修改运行中的HF恢复工作区。

## 一次运行

```bash
/opt/venv/bin/python /workspace/tencent-mellum-next/scripts/run_mellum_candidate_capacity.py
```

独立next工作区；GPU1限定，HTTP8151/engine8726，避免HF恢复冲突；模型BF16、eager、TP/DP1、maxseq32、context16640，prefix/chunk关闭。使用已开发的fused_moe profile，不安装E的Torch包装器，记录批量/native调用和5631权重加载审计。

客户端仍为已审计EvalScope1.15独立环境。沿用题目性能协议thinking=True、stream、temperature0、ignore_eos、min/max256；用户关闭thinking/2048的选择用于当前HumanEval配对，不把功能评估协议混入ABC。原始命令与dataset adapter差异保留。

1. smoke：B2/A8/C2（C含最长输入），验证流式请求/usage/token数、真实batch和成功数。
2. A_small：64请求/并发32，沿用原A筛选/循环manifest前64项。检查actual decode batch32。
3. C_small：按token长度排序取20个等分位点，包含最短和最长；0.3到达率、180秒单请求超时。此选样不依据模型输出；不等同C200分布或尾延迟验收。

任何阶段协议/成功数失败就不再运行下一阶段，保留失败率与成功子集指标。每阶段重新启动并关闭自有服务；35分钟执行总预算（CPU输入准备另计），最长可显式调至45分钟。无需重跑完整86分钟ABC；没有以小样本输出官方比分。

## 开发与证据

复用run_mellum_abc_baseline.stage_worker，新增plan中profile/server_port/engine_port可选字段；旧R默认路径和端口不变。新入口独立source hash、输入hash、客户端审计、模型审计、原始请求DB、阶段日志、产物hash。记录HF并行可能产生CPU/内存带宽干扰，最终正式对比需空闲环境。

24项相关CPU测试通过，包括旧ABC原协议/恢复、profile互斥、端口和采样、容量失败停止后续阶段。未执行GPU服务；本轮不改变模型数学、默认配置或正式验收门槛。

## 分析决策

- 先看smoke成功、实际batch与模型加载，再看A64吞吐及C20成功率/排队。小样本p99只作诊断。
- 若A仍低或C超载，对照服务batch/native路径确定是否有效启用优化；不以离线1.47倍直接推算HTTP成绩。
- 结合恢复后的HumanEval选择最终候选；功能指标合格也不替代题目其它公开集指标。
- 后续优先一次FusedMoE+SWA CK组合消融，或有明确源码边界的prefix修复（研究06）；不同时铺开大量新内核。
