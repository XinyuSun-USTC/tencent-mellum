# 53 候选筛选后续验收与C容量验证

## 当前并行工作的边界

KV与专家分组两项GPU实验保持原源码及参数不变。本阶段仅在交付工作树增加CPU分析、后续运行工具和说明，不修改候选源工作树。原始失败和新运行分别保存。

工具`scripts/followup_screened_mellum.py`通过`--source-root`导入对应候选工作树，检查已结束screen状态、被选profile、输入/源码/产物哈希及实际配对性能证据。运行中或未通过的screen、未通过的专家组、被修改的源码与结果均会拒绝。结果不会自动晋级为合格主版本。

## 决策顺序

1. 等当前两项筛选结束，先核验数学路径、200条和长输入，再比较实际性能与波动。只选择有价值的候选，不为每个参数组合跑全套HumanEval/ABC。
2. 对有价值的单项或已验证组合，先做C20短容量筛查；它可暴露明显超时，但不能证明持续0.3req/s能力。
3. 若多个单项可组合，先在新工作树验证组合200条及长输入。只对最终有价值的配置做完整HumanEval，避免重复长评估。
4. 完整验收通过后才测正式C200；保持rate0.3、256输出与单请求180秒。C200全完成是晋级条件，失败计数与success-only分位数分开报告。
5. 最后固定同一配置测完整ABC和对应累计消融。原始R正确性通过但C失败，仍作为基准，不标为满足全部要求的主版本。

## 工具参数

以下为模板；screen未结束时不能运行，PROFILE必须是实际通过的profile，GPU必须空闲。

```bash
# C20：筛查，不代表正式C通过
/opt/venv/bin/python scripts/followup_screened_mellum.py --source-root SOURCE_TREE --profile PROFILE --screen-run SCREEN_RUN --phase capacity --count 20 --gpu GPU --budget-minutes 20

# 最终选定配置的完整HumanEval+200条+长组
/opt/venv/bin/python scripts/followup_screened_mellum.py --source-root SOURCE_TREE --profile PROFILE --screen-run SCREEN_RUN --phase acceptance --gpu GPU --budget-minutes 120

# C200：必须给出同profile已通过的完整验收
/opt/venv/bin/python scripts/followup_screened_mellum.py --source-root SOURCE_TREE --profile PROFILE --screen-run SCREEN_RUN --phase capacity --count 200 --acceptance-run ACCEPTANCE_RUN --gpu GPU --budget-minutes 30
```

当前工具支持`r_reference_kv_once`与`r_expert_decode2/4`。若最终组合新profile，必须先完成其实际screen再接入同样的校验流程，不套用单项通过结果。`--prepare-only`只核验和准备文件，不启动GPU。容量阶段直接读取对应源码的客户端结果数据库，不修改总超时、输出长度、到达率或完成分母。

## 并行CPU发现

流式清理CPU复现见`report/validation/45_stream_cleanup_cpu.md`。提前终止未调用前端清理，且引擎取消链路未接通，可能放大过载。但其线上影响未经量化，本轮不将它混入两项在跑候选，也不据此宣称能解决计算容量不足。
