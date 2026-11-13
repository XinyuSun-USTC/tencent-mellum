# 13 短探针结果与完整MoE同输入重放

原始套件：`results/mellum_validation/20261111T140735437405Z`；ATOM：`results/atom_mellum_gpu/20261111T140748118711Z`。用户普通终端执行；本轮仅分析与准备下一诊断。

## 四条探针

| 探针 | 原始HF cached下一token | 本轮HF整段prefill生成 | ATOM生成 |
|---|---:|---:|---:|
| h_0_prefix_0 | 23 | 23 | 23 |
| h_0_prefix_14 | 48 | 410 | 48 |
| h_1_prefix_0 | 23 | 23 | 23 |
| h_1_prefix_9 | 581 | 581 | 5472 |

h_0再次显示HF cached与整段prefill自身不同，不能把整段prefill直接当成cached逐token完全一致的基准。h_1的HF两者一致、ATOM不同，仍有待定位问题。

h_0_prefix_14的HF logits：410=28.0、48=27.875；ATOM：48=28.25、410=28.125。h_1_prefix_9的HF：581=29.875、5472=29.75；ATOM两者均29.625。ATOM采样器选择5472而torch.argmax返回581，故比较器`top1_exact=true`与`generated_ids`不同可以同时成立。并列时不同选择不能单独证明采样器错误；该处HF与ATOM的logits本身仍不一致。

四例embedding、首层input norm输入/输出、Q projection和Q norm精确一致；K/V projection有少量差异。选定九项权重全部逐元素一致。Attention context相对L2约0.00060–0.00076；MoE输入约0.00143–0.00172，MoE输出约0.00572–0.00796（前三例范围，完整值以JSON为准）。最终norm四例相对L2分别0.07358、0.07174、0.08663、0.09145。误差从首层累积，但MoE输入不同，不能由输出差异反推MoE内核有错。

## 下一诊断：固定完整MoE输入

新增`ATOM/atom/model_config/mellum_diagnostics.py:install_moe_input_replay`，仅在显式设置`MELLUM_MOE_REPLAY_INPUT_DIR`且首层stage capture有效时，通过forward pre-hook替换首层MLP输入。数据读取保存的HF `moe_input.pt`，必须与当前输入shape/dtype完全一致。dummy和decode不替换；正常验证入口未指定重放参数时清除该环境变量。既有模型数学路径未调整。

`scripts/check_atom_mellum.py --moe-replay-inputs`只接受单token探针并要求stage/layer capture，状态为`moe_replay_complete`，不返回正常greedy验证通过。`scripts/replay_mellum_moe.py`封装ATOM执行并比较保存的HF/新ATOM `moe_input`、`router_logits`、`moe_output`及选定权重。其结果是诊断采集状态，不是模型正确性验收；替换后的生成token无验收意义。

这样可以在真实加载、原生权重格式和FusedMoE路径下隔离首层完整MoE的同输入误差，无需先修改路由权重精度。若输入精确一致而router不同，先检查gate；若router也相同而MoE输出不同，再隔离权重舍入、专家矩阵乘及累加顺序。九项权重审计不含全部专家权重数值，不把它当成专家权重全量数值证明。

配置/模型CPU测试27项通过，新增测试覆盖无活动capture时不替换、同shape/dtype替换、原输入不被修改、dtype不符拒绝、移除hook后恢复。脚本语法与ATOM diff检查通过。GPU重放尚未执行。

普通终端执行：

```bash
/opt/venv/bin/python /workspace/scripts/replay_mellum_moe.py --run /workspace/results/mellum_validation/20261111T140735437405Z
```

结果位于新建`results/mellum_moe_replay/`目录，子进程完整日志为`atom.log`。下一步根据同输入重放证据确定修正，而不是继续扩大样本或把已有分叉归为一般BF16误差。
