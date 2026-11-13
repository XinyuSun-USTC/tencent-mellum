# 16 专家累加实验与HF实际分派核验

原始产物：`results/mellum_expert_math/20261111T160212992901Z`。

四例HF激活+FP32累加变体相对保存HF输出的相对L2为0.00004332、0.00004972、0.00003675、0.00004725；最大绝对差0.00024414–0.00048828。相同激活但逐专家BF16累加时，相对L2为0.00343–0.00361；其他激活/加权候选也未更好复现HF。前者仍非逐元素完全一致，不能声称已经复现全部专家算术或证明所有权重加载一致。

## 更正此前HF源码解释

此前验证15只阅读`MellumExperts.forward`正文，对HF执行顺序的描述不完整。独立HF环境`transformers/models/mellum/modeling_mellum.py`第294行的`@use_experts_implementation`可替换其forward：

- `transformers/integrations/moe.py:use_experts_implementation`保存原始forward，并通过`self.config._experts_implementation`在调用时分派。
- `transformers/modeling_utils.py:PreTrainedModel.get_correct_experts_implementation`在未显式指定时首先选择`grouped_mm`，检查不通过可回退`eager`。
- `transformers/integrations/moe.py:grouped_mm_experts_forward`最后恢复token/Top-k顺序，再执行`weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)`。源码注释明确其采用FP32累加，区别于低精度index_add。

现有HF导出脚本未显式指定专家后端，旧结果也未记录实际选择。因此：默认规则及数值结果强烈支持原参考采用grouped_mm，但这仍是推断，不能把旧运行后端写作已记录事实；不能继续用类正文eager实现解释全部参考结果。

## 下一步直接调用官方实现

`scripts/diagnose_mellum_expert_math.py --official-backends`使用独立HF 5.10.1解释器，构造官方`MellumExperts`，载入原始第0层64个专家的gate/up/down（不构造完整12B模型），固定已保存HF输入和HF路由。分别显式设置`grouped_mm`、`eager`，实际调用装饰器分派后的官方forward并保存输出，与旧HF及ATOM-HF路由重放结果比较。

这一步避免继续依赖我们自写的六组公式推断实际HF后端。如grouped_mm复现旧HF而eager不同，则确认参考路径；仍不自动说明ATOM专家内核错误，后续应根据实际精度语义选择Step0正确性路径。暂不改变正常ATOM实现或放宽验收标准。

`scripts/export_mellum_suite_hf.py`现已把实际`model.config._experts_implementation`写入运行报告和reference环境字段，旧产物保持原样，不追写假定后端。

本轮语法检查通过，套件/比较器/运行器/路由CPU24项通过。未执行GPU。普通终端命令：

```bash
/workspace/venvs/hf-reference/bin/python /workspace/scripts/diagnose_mellum_expert_math.py --official-backends
```

使用HF独立环境是为了调用官方专家实现；ATOM冻结环境未升级。返回新的`RESULT`目录后核验四条官方后端结果。
