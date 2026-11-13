# 33 索引8层router投影首差与局部候选

## 重放证据成立

原始产物`results/mellum_layer_tail/20261112T163058978676Z`，来源全层观察`20261112T152156609360Z`。HF/ATOM两路后半段重建均逐元素精确复现各自真实层输出，interpretation_gate=inspect_components，允许根据分量归因。

| 环节 | 结果 |
|---|---|
| 相同层输入及Attention Q/K/V/context | 精确一致 |
| O投影 | 精确一致 |
| 残差 | 精确一致 |
| post norm | 精确一致 |
| router logits | 64元素中1个不同 |
| 相同HF norm/router输入的MoE | 精确一致 |
| 自然输入的MoE输出 | 不同，max_abs0.03125、relativeL2约0.00663427 |
| 两路完整层输出重建 | 各自精确复现真实输出 |

不同logit位于expert44：HF=-2.546875、ATOM=-2.5625，差0.015625。同一组选中专家ID[44,17,32,1,14,40,59,30]保持不变，但expert44归一化BF16权重0.234375→0.2314453125，其他权重也随归一化改变，进而产生MoE及层输出差异。这不是本例专家ID切换，也不是相同输入下专家计算不一致。

该事实支持对齐router线性计算，不支持再次替换O投影或更改Top-k并列策略。它说明本数据上相同输入/权重的AITER tgemm和HF F.linear数值不同，不把跨实现BF16差异自动定性为AITER bug。此前四份第0层投影一致不能推广到全部层和单tokendecode；本次提供了首个具体反例。

## 最小显式候选

`mellum_projection.py:install_reference_router/reference_router_forward`仅为Mellum gate实例绑定BF16 F.linear，保留原Parameter、bias、参数名、weight_loader和forward hooks。拒绝量化输入/非BF16。`MellumSparseMoeBlock`通过MELLUM_ROUTER_BACKEND=torch启用，默认aiter不变，未修改共享线性层、O/lm_head或其他模型。

`check_atom_mellum.py --torch-router`要求torch MoE；decode运行器透传该flag并核对实际worker设置，继续同一separate QKV、CPU RoPE、参考K写回、分页缓存SDPA路径。无其他数学改动。

模型相关CPU45项、运行器相关5项通过，新增router参数/加载回调/观察器保持及输出对照测试；语法/diff检查通过。GPU候选尚未运行，不能直接据轻量重放宣称32位置已通过或Step0完成。

## 下一验证

复用原四输入、各8位置的相同HF cached参考，只改变router线性路径，检查完整32位置（含跨窗口）：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_decode_validation.py --reuse-hf-run /workspace/results/mellum_decode_validation/20261112T123830320864Z --reference-kv-rope --sdpa-decode --torch-router
```

保留worker死亡检测、心跳和超时。若完全精确，再进入候选整理及无强制正常生成回归；若仍不同，继续首差定位。受限单请求/短prefill边界仍存在，长输入、多请求、服务与正式验收不能跳过。
