# 19 原生MoE/Attention：明显提速，但两候选D200均未达90%

[计划21](../../docs/plan/21_native_runtime_pilot_delivery.md)双卡批次完成；独立实现commit cb09aed。GPU0 MoE累计1609.39秒（26.82分钟），GPU1 Attention累计1587.21秒（26.45分钟），均3个worker、未截断预算。原配置不切换，本轮仅分析/归档，不追加GPU任务。

## 性能与生成

固定相同输入、每请求256输出/ignore_eos，每配置两次、预热后取wall中位；R与候选为独立加载进程，不是ABBA或正式ABC。两卡并行且共享主机资源，只作候选探索性能。

| 候选 | S1加速 | S32加速 | L32加速 | 批量几何均值 | D200位置一致率 | HF完整序列 |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 FusedMoE | 1.0817× | 1.5492× | 1.3878× | 1.4663× | 9262/12800=72.3594% | 82/200=41% |
| 原生Attention | 1.0241× | 1.2902× | 1.6312× | 1.4507× | 9136/12800=71.3750% | 82/200=41% |

两个候选均跨过性能晋级线并完成200正常greedy64。长输入位置分别57.76%/61.28%，短输入75.14%/73.30%。首token均200/200与HF及历史R相同，但后续自回归分叉明显。与R完整输出相同分别83/200、80/200；不能把HF和R两种比较混用。

**当前版本均不满足本地开发200的位置90%筛选，停止晋级组合/ABC。** 这是接受浮点差异后仍需面对的生成门槛，不是额外要求中间张量精确。当前结果不能直接宣称官方公开集失败，也不能因性能好而宣称正式正确性通过。

## 重复生成的不同表现

| 256输出负载 | MoE两次输出不同 | Attention两次输出不同 |
| --- | --- | --- |
| S1 | 1/1，首差0起14 | 0/1 |
| S32 | 31/32 | 0/32 |
| L32 | 29/32 | 0/32 |

各track的R两次输出均相同。MoE S1两次都是prefill1/decode1，S32两次都是prefill32/decode32；L32两次prefill批大小分布也相同。全模型重复存在差异，值得优先核验Fused专家计算重复性；但批大小分布相同不等于所有内部执行条件完全相同，**尚未做固定同输入单层重复，不能仅据此断言某个AITER kernel有bug或已证明原子归并是根因**。

Attention在所测两次生成中稳定，不能外推全部运行确定性；它同时替换prefill/decode，当前尚未区分两者对D200分叉的贡献。

## 执行与完整性

两边加载均5631/5631、missing/unexpected为空。native审计各性能1566调用、D200460调用，真实decode32。MoE为FusedMoE/UnquantizedFusedMoEMethod，w13/w2 shuffle标记通过；Attention的layer0为aiter_varlen/paged_attention_triton。这是layer0类/分发证据，不是所有层具体内核trace；full层不能由layer0窗口层日志推断。

独立重算每批49产物和238源码指纹，两批共98/476，全部一致。[MoE证据](../../artifacts/performance/19_native_runtime_pilots/moe.json)、[Attention证据](../../artifacts/performance/19_native_runtime_pilots/attention.json)。原始结果路径分别：

- `/workspace/results/mellum_native_moe/20261115T184205841135Z`
- `/workspace/results/mellum_native_attention/20261115T184214571268Z`

## 下一步范围建议

高性能方向有实测收益，应继续有界改进，不退回外围小操作；但两项约1.45倍收益也不能直接相乘，不能推断会解决官方约15倍吞吐差距或C超时。

1. **MoE**：优先用固定真实输入/路由做短重复回放，记录实际one-stage/two-stage路径及输出差异。若能明确选择当前依赖已有的更稳定执行/归并方式，只改这一项后正常生成200复验；不要求完全等同HF，不无限追逐bitwise。
2. **Attention**：将当前合并改动拆为“参考prefill＋native paged decode”与“native prefill＋参考decode”两个有界消融，直接正常生成判断。复用已有参考，不再跑完整ABC；有单项达标才做256性能确认。
3. 两条目前都低于90%，不直接组合并长跑HumanEval/ABC。完整HumanEval参考可独立推进，但不能用其潜在分数抵消当前生成一致率缺口。
4. 每条只做一轮明确消融/修正，有依据才追加；如仍不达门槛，保留R和已具备条件的SWA CK，收敛可交付组合，不擅改题目指标。
