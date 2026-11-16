# 48 利用空闲GPU0：专家/Attention管线最小对照

systems线仍在GPU1跑reference_control，不修改/终止它。已完成的快速子结果：基线65.36%、关闭prefix64.79%、Torch路由66.67%、分离QKV62.27%、batch1为64.10%；均未接近90%。Norm线最好72.91%。不继续单独调整这些已无足够改善的开关。

独立/workspace/tencent-mellum-core-diagnosis，分支step1/core-path-diagnosis，基于6485506。原脚本、性能候选、HF参考不改。新增core_nn/nt/tn/tt四项只用于诊断：首字母为Attention管线n(native)/t(reference SDPA)，次字母为专家n(AITER FusedMoE)/t(Torch HF grouped/batched专家)。

## 固定条件与四项

统一batch1、eager、参考Norm、独立QK Norm、分离QKV、Torch路由、prefix关闭、metadata关闭。只交换Attention管线及专家实现：

| profile | Attention | 专家 |
| --- | --- | --- |
| core_nn | 原生varlen/paged及direct KV | AITER FusedMoE |
| core_nt | 原生varlen/paged及direct KV | HF语义Torch专家 |
| core_tn | 参考SDPA/参考KV路径 | AITER FusedMoE |
| core_tt | 参考SDPA/参考KV路径 | HF语义Torch专家 |

Attention对照包含其RoPE/KV缓存外围，不能把改善直接归因于某个单独FA内核。四项保留同一权重映射；构造审计必须实际看到28个对应MoE类和28个prefill函数，5631加载通过。没有静默fallback或自适应按prompt挑后端。

## 少量固定诊断输入

从原开发200按来源及输入长度分层，固定12条HumanEval+8条场景A，覆盖各长度排序端点；不依据HF输出/差异筛选，不向模型提供参考token。每项greedy64，只这20条，不重跑长组、HF或HE。该20条的高位置率不能冒充原200/正式验收，后续仍需完整200确认。

CPU输入/配置/引用全部记录hash，候选每项独立加载清空KV，实际profile/数学后端回执保留。每项上限10分钟，总执行预算25分钟，CPU准备另计；预计5–12分钟，不能保证固定时间。

## 空闲GPU0命令

```bash
/opt/venv/bin/python /workspace/tencent-mellum-core-diagnosis/scripts/run_mellum_core_diagnosis.py --gpu 0 --budget-minutes 25
```

只用GPU0，engine9826；GPU1 reference_control仍用9926，两者不争卡/端口。工具无GPU，本轮只开发与CPU准备。

## 决策

- native Attention+Torch专家恢复：先聚焦专家数学/归约，尽量保留原生Attention与其它加速。
- reference Attention+native专家恢复：先聚焦Attention/RoPE/KV，保留专家加速。
- 只有reference/reference恢复：保留此正确控制，检查两类误差的组合影响，选择最小必要回退，再补完整200。
- reference/reference仍异常：先比较与正在跑的R控制、输入和投影/采样配置，不盲目继续改kernel。

新模型名只为诊断开关，未升级依赖/未改现有装饰模型。50项相关CPU检查与实际20条准备通过；没有GPU结论。此前HF快速Norm后备仍未满足条件，不启动。
