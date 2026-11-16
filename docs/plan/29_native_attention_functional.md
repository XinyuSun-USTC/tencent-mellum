# 29 GPU1：FusedMoE+完整原生Attention的功能评估

前置性能：20261115T231450301731Z HTTP smoke/A64/C20全部成功；A64为121.4966tokens/s，较FusedMoE+SWA CK的59.612约2.04倍；C20 TTFT p99为1.7263秒、TPOT p99为0.13478秒。确认layer0实际aiter_varlen prefill与paged_attention_triton decode，无参考Attention回退。

## 命令（仅GPU1）

```bash
/opt/venv/bin/python /workspace/tencent-mellum-native-functional/scripts/run_mellum_humaneval_pair.py --attention native --reuse-hf-run /workspace/results/mellum_humaneval_recovery/20261115T213706098181Z --budget-minutes 35
```

独立native-functional工作区基于9a3b3ca，模型数学与刚测HTTP候选保持一致；仅扩展功能评估入口。GPU1/engine8626，与GPU0 prefix的8251/8926分离。强制显式复用HF，不会在GPU0启动参考生成。冻结ATOM环境、FusedMoE、参考gate/路由、独立QKV与L控制保留；native Attention设置关闭旧ragged桥与SDPA decode。

164题全部生成，关闭thinking、最多2048token、batch32，与既有HF参考协议相同。参考完整性、输入与HF模型源码hash在prepare阶段核验；评分沿用官方隔离执行及相同提取规则。没有token位置/张量相等前置门槛；记录截断/提取失败、逐题增减、双方通过题数与2pp检查。

## 开发核验

候选审计区分kind=moe与kind=attention；MoE验证FusedMoE，Attention同时验证aiter_varlen prefill与paged decode，禁止参考回退。旧脚本不能直接把混合native日志全部要求为FusedMoE，否则产生假失败，本轮已修正。实际decode batch32仍强制核对。

8项相关CPU测试通过，包括旧功能评分/协议/序列化、原生开关与实际后端审计、HF参考hash变更拒绝。真实164题prepare-only及现存HF引用检查通过。未启动GPU生成。

## 结果分支

本地HumanEval差距≤2pp则优先保留完整原生Attention作为性能主线，随后考虑与prefix组合以及单项/累计消融；不自动继承单MoE的功能成绩，也不宣称公开200独立门槛通过。若差距超标，先看差多少及截断/提取，不立即启动全模型浮点追踪，可选择已有SWA组合做功能对照、保留达标且有收益路径。GPU0缓存路线继续，不修改其源码。
