# 25 HumanEval期间的并行工作与结果后的交付顺序

用户要求充分利用等待时间。正在运行的HumanEval及HF恢复工作区保持不变；并行CPU开发在`/workspace/tencent-mellum-next`。不额外占用GPU、不改冻结库、不自动切生产配置。

## 本轮完成

1. **修复HF导出并保留候选工作**：原HF因loading_info包含set而序列化失败；独立修复df33885复用既有json_default，不改模型/输入。命令仅补HF/GPU0，等待并复用原ATOM164题和已完成评分，原目录保持；见下方恢复命令。
2. **候选服务profiles**：统一r/fused/SWA CK/native Attention及组合的开关与manifest，避免旧Torch E wrapper误装到Fused专家，禁止不兼容混配。只完成CPU核验，尚未HTTP/GPU实测。
3. **C真实前缀分析**：共有3820token，完整块的结构复用量占输入37.15%；说明prefix caching值得评估，但已定位reference KV覆盖晚于gather的兼容性缺口，未盲开缓存。详见[研究06](../../report/research/06_parallel_preparation_and_prefix.md)。

### 当前需要的HF补跑

```bash
/opt/venv/bin/python /workspace/tencent-mellum-delivery/scripts/recover_mellum_humaneval_pair.py --run /workspace/results/mellum_humaneval_pair/20261115T210153552943Z
```

只用GPU0补HF，原GPU1候选不停止/不重跑。共享原inputs hash、关闭thinking/2048协议和HF模型源码hash；修复仅归一化加载诊断。独立结果`mellum_humaneval_recovery/<UTC>`，完成后给出完整配对评分。仍90分钟总体预算，HF完成即保存可复用hash；候选失败则保留HF不伪造完整配对。

## 结果后的执行分支

### A. 先解释完整功能结果

核对164完成数、HF/候选通过题数、绝对百分点差、逐题增减、截断/提取失败。HF零分或大量预算截断时先审协议有效性，不调整提取规则去迎合测试结果。功能通过不替代公开200/16k要求；token位置差异也不再阻止功能评分。

### B. 优先收敛已有高性能配置

若FusedMoE功能差距<=2pp，先固定fused_moe服务配置，完成HTTP/SSE/usage与真实batch32协议检查，再评估fused_moe_swa_ck一个组合。SWA prefill在本地HumanEval短prompt不激活，不将此功能分数外推成长输入通过；组合仍做16k与已有开发200报告。

若差距>2pp，复用已完成HF参考，只选择一个已有替代实现做候选侧功能评估，优先有实测性能依据的路线；不重跑HF，不同时展开多候选矩阵。isolated贡献方案若考虑，需要补性能证据，不能套旧Fused收益。

若功能差异很小但token位置仍低于题目门槛，如实保留两个指标，不宣称整体通过、不暗改口径；根据剩余时间决定是否只保留独立达标子项或向出题方确认具体public集/evaluator含义，代理不擅自联系他人。

### C. 先小型真实负载，再完整ABC

服务配置固定后，以原EvalScope协议做小型A/C诊断（例如A64/并发32、C20/同0.3到达率），含256输出和代表性长输入。小规模用于发现排队/超时，不能冒称正式A512/C200成绩。不要每次微改重跑86分钟。

若C仍超载，重点评估prefix复用：先修K覆盖/gather顺序和cached-prefill metadata，再启用缓存，验证绝对位置/窗口/页表及EOS/驱逐，不把关闭guard当修复。实际命中率和净收益需测量，37.15%结构值不是承诺。

最终只固定一个配置，空闲环境完整ABC一次本地复测；裁判“三次中位数”与本地单次测量明确区分。消融至少含参考/R、MoE、Attention/SWA、最终组合；prefix如实施，诚实计为策略优化点，总数最多三个。

### D. 并行收尾材料

形成<=4页技术报告、1页AI说明、复现命令与开关表。报告包括放弃的indexed/逐元素/原生Attention/归约实验，保留正确性和性能失败数据，不只展示最佳微基准。Git保留完整分支历史，接受代码合并后再安排推送，不改当前运行快照。

## 本轮边界

已有15项CPU服务/profile/前缀计算测试通过；HF修复另有6项CPU检查及恢复prepare-only。当前没有新增性能或功能分数，没有GPU服务/缓存实测；不会在两张卡繁忙时偷偷追加任务。
