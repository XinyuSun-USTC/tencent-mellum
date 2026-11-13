# 12 扩展回归失败与路由证据

原始结果：`results/mellum_validation/20261111T132520303449Z`；ATOM：`results/atom_mellum_gpu/20261111T132612955255Z`。数据来自用户普通终端，本轮工具未启动GPU任务。

## 结论

11条各64新token中，仅binary_search、long_8192、multiply三条完全一致；其余八条分叉。全部初始prefill top1一致，但全词表logits不一致，最大绝对差0.5–5.75。Step0正确性仍未完成，停止扩大覆盖，回到短输入首差定位。

| 输入 | 首个不同新token（从1计数） |
|---|---:|
| h_0 | 15 |
| h_1 | 10 |
| h_10 | 23 |
| h_100 | 24 |
| frequencies | 25 |
| word_problem | 40 |
| window_1025 | 10 |
| long_16384 | 29 |

比较器统计704位置中394相同（55.97%），分叉后两端历史已不同，此比例不是同前缀精度，也不是官方验收指标。此前9条168token一致仍是有效历史结果，但不足以证明更长、多样输入正确。

## 同输入路由结果

`router/results.json`含HF/ATOM各两份保存的第0层logits，每份重复三次：所有专家集合与HF一致，原生ID重复稳定。Top-8边界并列仍为9/24及6/18行，但这四份输入并未表现出集合差异；不能据此归因于并列选择，也不能推广到全部层和全部输入。

`scripts/check_mellum_router.py:hf_route/main`比较结果：原生FP32权重与按原生专家ID计算并转BF16的HF权重最大绝对差分别为0.000975191593170166、0.0009764134883880615。源码路径见[验证11](11_mellum_expanded_and_router_replay.md)。该证据支持继续检查权重舍入，但尚未证明它导致本轮生成分叉；未修改模型路由或放宽验收阈值。

## 下一步：四条短探针

`scripts/diagnose_mellum_mismatch.py:prepare`增加`--cases`，从真实参考生成两条短HumanEval来源输入的原始prompt与首差前公共前缀：

- h_0：152和166输入token（追加14个HF新token）。
- h_1：142和151输入token（追加9个HF新token）。

每条只生成1 token，双端保存逐层输出及第0层子算子/选定权重。原始HF cached next token已独立记录，必须同时审查HF cached与整段prefill是否自身一致；不能把HF内部差异全部归咎于ATOM。选择器及既有诊断CPU测试3项通过，prepare-only实际输入长度已核验。

执行：

```bash
/opt/venv/bin/python /workspace/scripts/diagnose_mellum_mismatch.py --run /workspace/results/mellum_validation/20261111T132520303449Z --cases h_0 h_1 --stages
```

首先比较相同输入的embedding、首层各算子和最终logits；若MoE仍是主要差异点，再固定同一hidden states隔离完整MoE并对路由权重精度做受控比较。当前尚无完整MoE同输入重放结果，不把路由权重差当作已确认根因。
