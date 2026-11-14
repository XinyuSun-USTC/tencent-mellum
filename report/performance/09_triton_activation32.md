# 09 Triton激活子集生成：结束本轮Attention替代实现筛选

## 决策

完成计划12的一条替代路线筛选。Triton局部更接近SDPA，但生成改善有限、局部慢于CK，不足以继续为该路线重复完整200和ABBA。保留CK/Triton为同一Attention优化点的可选实现，整体开关默认仍关闭；不新增bitwise验收门槛，也不将32条子集低于90%直接解释为正式公开集失败。

下一主线推进服务与多请求能力，建立场景A并发32/C到达过程的真实基准，再依据批量profile选择第二算子优化点。最终选择/默认设置与正式准确性验收在候选固定后处理；当前不强行认定CK或Triton为正式合格版本。

运行`/workspace/results/mellum_swa_32/20261114T204611749113Z`，代码`26e67dc`；[轻量证据](../../artifacts/performance/09_triton_activation32/manifest.json)，局部速度/数值见[08](08_triton_swa_screen.md)，CK200见[07](07_representative_correctness.md)。本轮无新GPU任务或模型修改。

## 配置与结果

实际SWA_OPT=1、SWA_IMPL=triton；normal greedy、ignore_eos=False，每条64token，加载5631/5631无missing/unexpected。复用同一HF5.10.1参考，源reference SHA与CK200完全相同；只运行developer200中32条输入≥1024的scenario A，未重跑168条不激活样本。

| 指标（同32条） | CK | Triton | 变化 |
| --- | --- | --- | --- |
| token位置匹配 | 1166/2048，56.9336% | 1262/2048，61.6211% | +96个，+4.6875个百分点 |
| 完整序列匹配 | 7/32，21.875% | 10/32，31.25% | 净增3条 |

逐题位置匹配13条改善、8条下降、11条相同。完整序列新增匹配scenario_a_3/9/18/28/35，同时scenario_a_7/29从CK匹配变为不匹配；不是逐题单调改善。

32条是刻意选择的激活子集，其本身不是题目公开200集合，不以其“90%检查false”宣称正式失败。Triton没有实测完整200，本报告不将此前CK168条结果拼接成“新Triton200实测成绩”。同样，不能把CK既有完整200的93.11%转称为Triton成绩。

## 性能与数值的取舍

前轮局部1K/4K relative L2约下降9.9/353倍，但本轮只有4.69pp位置匹配改善。这印证单层张量距离不是全模型greedy一致率替代指标；21层窗口、多层路由和自回归可放大差异，也不能仅凭此数据确定各层的独立贡献。

已知局部Triton较CK耗时约1.63/1.35倍；它仍快于SDPA，但尚无整模型Triton TTFT/TPOT成绩。当前准确性收益不足以让这一有界筛选继续扩成大规模配置搜索。保留实现/失败与改进证据，不删除可供后续服务消融的代码。

## 交接

1. 冻结这一轮单请求Attention比较，不自动追加GPU评测或全HumanEval。
2. 检查现有OpenAI server及引擎多请求路径与reference单请求限制，设计最小批量支持；不得用串行循环冒充并发32。
3. 以bs1回归、双请求不同长度/结束时刻及实际分页KV验证接入正确性，再建立服务A/B/C负载基准。批量引起的shape/调度/执行策略变化需要独立记录，不把收益无条件藏进基线。
4. 用批量profile决定下一算子优化，不按bs1 MoE/Attention占比直接预定。最终准确性以题目token匹配和HumanEval门槛为准，保持集合/协议边界清晰。
