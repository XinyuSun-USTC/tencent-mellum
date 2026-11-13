# 14 完整MoE同输入结果与路由受控比较

真实结果：`results/mellum_moe_replay/20261111T150103043063Z/results.json`；原始参考为`results/mellum_validation/20261111T140735437405Z`。

四例MoE输入和router logits均逐元素完全一致，MoE输出相对L2分别0.00446145、0.00446578、0.00448031、0.00450158。这证明首层MoE计算路径在这些相同输入下仍有差异，不能完全归因于Attention传入误差；但不能单凭此结果认定专家权重错误或某个内核错误。专家权重格式、路由权重精度、矩阵乘/激活/求和均仍需区分。

## 已准备的受控实验

`ATOM/atom/model_ops/moe.py:FusedMoE.select_experts`已有`custom_routing_function`入口，优先于原生路由。`ATOM/atom/model_config/mellum_diagnostics.py:make_replay_router`通过该入口只干预重放的首层，不修改共享MoE实现或正常Mellum路由：

1. `rounded_native`：原生`rocm_aiter_topk_softmax`选专家并计算权重，保留其ID及顺序，仅将权重转为logits dtype（BF16）再转FP32。后者满足AITER权重张量接口，同时保留BF16舍入值。此组用于隔离权重舍入。
2. `hf`：FP32 softmax、torch.topk、归一化、BF16舍入、转回FP32供原生专家路径使用，ID转Int32。此组同时对齐HF路由选择/顺序与权重；它和上一组的差别不只有专家集合，解释时不能过度归因。

两组固定HF保存的同一MoE输入，专家权重加载和FusedMoE计算路径不变。若第一组误差缩小，说明路由权重精度是贡献因素，仍不自动证明全部根因；若两组仍有差异，继续比较专家中间值及加权求和。不得预先将此诊断实现作为生产修复，也不调整验收阈值。

`check_atom_mellum.py`要求路由干预必须配合显式MoE输入重放，正常入口清除干预环境变量。`replay_mellum_moe.py --routing-study`顺序运行两组独立引擎，结果分别标记routing模式，生成token不参与正确性判断。四份输入和router是否仍精确一致须同时检查。

配置/模型CPU28项通过，覆盖原生ID保持、BF16舍入及HF公式。GPU实验尚未执行。本轮没有改变正常模型数学路径。

执行：

```bash
/opt/venv/bin/python /workspace/scripts/replay_mellum_moe.py --run /workspace/results/mellum_validation/20261111T140735437405Z --routing-study
```

会输出两组`RESULT diagnostic_complete output ...`，请保留两个结果目录。该状态只代表诊断数据采集完成，不代表正确性通过。
