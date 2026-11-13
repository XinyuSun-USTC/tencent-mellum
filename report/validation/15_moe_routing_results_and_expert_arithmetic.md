# 15 路由受控结果与专家算术隔离

结果来源：`results/mellum_moe_replay/20261111T155738667201Z`（rounded_native）和`20261111T155808236095Z`（hf）。四例输入、router logits仍精确一致。

| 样本 | 原生MoE相对L2 | 仅舍入原生权重 | HF完整路由 |
|---|---:|---:|---:|
| 0 | 0.00446145 | 0.00416847 | 0.00414789 |
| 1 | 0.00446578 | 0.00413924 | 0.00413238 |
| 2 | 0.00448031 | 0.00429793 | 0.00431079 |
| 3 | 0.00450158 | 0.00427681 | 0.00427484 |

路由对齐仅小幅改善，剩余约0.41%–0.43%；不能再把主要差异归因于路由权重精度，也未证明它与其他误差独立或可线性相加。暂不将实验路由写入正常适配实现。

## 源码依据与下一实验

HF独立环境`transformers/models/mellum/modeling_mellum.py:MellumExperts.forward`：按专家顺序取token，BF16 gate/up线性投影；`self.act_fn(gate) * up`；down线性投影；乘BF16路由权重；向`zeros_like(hidden_states)`执行`index_add_`。因此激活输出、乘法及逐专家累加的舍入位置值得单独观察。

ATOM `atom/model_ops/moe.py:UnquantizedFusedMoEMethod.apply`经`FusedMoE.select_experts`后调用`aiter.fused_moe.fused_moe`，使用加载后shuffle的w13/w2权重。AITER `/app/aiter-test/aiter/fused_moe.py:fused_moe_`根据metadata走单阶段或`fused_moe_2stages`。仅据Python调用不能确定选中GPU内核的每个舍入点，因此不声称已确认AITER累加精度或某内核错误。

新增`scripts/diagnose_mellum_expert_math.py`，固定四份HF输入及HF路由，逐专家从原始safetensors读取第0层gate/up/down权重（不加载完整模型或ATOM引擎），用PyTorch GPU执行六个算术组合：

- HF激活顺序：BF16 SiLU结果再乘BF16 up；或候选融合顺序：FP32 SiLU与乘up后一次回BF16。
- BF16加权结果逐专家BF16累加；BF16加权结果FP32累加后回BF16；或投影结果与权重在FP32相乘并累加后回BF16。

候选融合顺序是隔离变量的数值实验，不是已确认的AITER内核复刻。结果分别与保存的HF输出和完整HF路由下的ATOM输出比较；首先检查HF公式重放能否复现HF，不能复现时须先调查重放本身。保存每条输入选定专家的gate/up、激活、down及weighted中间值，供后续进一步对照。未读取ATOM shuffle权重的数值，因此本轮实验也不能单独证明加载后专家权重一致。

本轮仅完成语法、CLI检查，没有GPU设备访问，没有新GPU结果，也未改变正常ATOM模型代码。普通终端执行：

```bash
/opt/venv/bin/python /workspace/scripts/diagnose_mellum_expert_math.py
```

输出为`results/mellum_expert_math/`新目录；`diagnostic_complete`只表示采集完成，不是正确性验收。若某算术变体更接近ATOM，只作为后续核查线索；不能直接据此替换全部专家计算。
