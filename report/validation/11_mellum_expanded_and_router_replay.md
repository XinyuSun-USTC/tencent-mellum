# 11 扩展回归与同输入路由诊断

当前依据：[验证覆盖总表](10_step0_validation_status.md)。已有9条开发输入、168个新token与HF匹配；prefill logits最大绝对差仍达6.0，Step0数值验证尚未闭合。本轮只扩充验证脚本，未修改ATOM模型数学路径，未执行GPU任务。

## 扩展集

`scripts/prepare_mellum_suite.py:build` 新增 `expanded`，共11条，每条greedy输出上限64 token（EOS可能提前结束）：

| 输入 | 实际输入token数 | 来源 |
|---|---:|---|
| h_0 / h_1 / h_10 / h_100 | 152 / 142 / 175 / 165 | 本地HumanEval第0、1、10、100条函数题 |
| multiply / word_problem | 21 / 33 | 自建数学题 |
| binary_search / frequencies | 28 / 25 | 自建代码题 |
| window_1025 / long_8192 / long_16384 | 1025 / 8192 / 16384 | 已有固定token边界输入 |

已用真实tokenizer完成prepare-only，产物：`results/mellum_validation/20261111T130517739835Z/inputs.json`。HumanEval来源仅用于输入多样性；未执行题目测试或计算pass@1，这不是正式HumanEval评测，也不是官方200条正确性验收集。

`scripts/compare_mellum_results.py:token_stats` 新增逐位置匹配数、公共前缀长度和比例；汇总写入comparison及summary。分叉后逐位置偶然相同也计为匹配，因此比例只能辅助观察；不能替代同前缀logits比较或自行成为验收阈值。

## 同输入路由诊断

源码依据：`ATOM/atom/model_ops/moe.py:FusedMoE.select_experts` 的softmax分支调用 `ATOM/atom/model_ops/topK.py:rocm_aiter_topk_softmax_impl`，分配FP32权重并调用 `aiter.topk_softmax`。独立HF环境 `transformers/models/mellum/modeling_mellum.py:MellumTopKRouter.forward` 执行FP32 softmax、Top-8、归一化，再将权重转回router logits dtype。本模型BF16，两条路径的输出权重精度存在差别；尚未证明这就是剩余数值差异的原因。

修正后短输入结果 `results/mellum_validation/20261111T120512707283Z` 中，保存的ATOM第0层router logits在Top-8/第9名边界存在并列：python_add有9/24行，arithmetic有6/18行。并列时不同算法可能选取不同专家，单凭这个统计不能判定实现错误。

新增 `scripts/check_mellum_router.py:hf_route/main`：分别读取HF与ATOM各两份已保存router logits，每份相同输入送入HF公式和原生AITER路由调用，记录专家集合、按原生专家ID重新计算的HF舍入权重误差、边界并列和三次原生调用的ID稳定性。缺少预期输入直接失败，避免空诊断成功。

该脚本只分配小型路由张量，不加载模型或完整专家权重。它隔离的是路由选择与权重阶段，尚未比较相同hidden states下完整MoE矩阵计算、专家求和或全部层。因此 `diagnostic_complete` 只表示证据采集完成，差异可以存在，不能读作正确性通过。

## 执行与判读

普通终端运行：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite expanded --router-audit
```

顺序为固定输入准备→独立路由诊断→HF参考→ATOM→比较。所有日志、路由JSON、输入和比较结果保存在新建run目录。默认不保存16K逐层dump，避免不必要的存储开销。最终`tokens_exact_logits_review_required`仍意味着数值问题待审查；任何生成分叉需查看首差位置并继续同前缀定位。

本轮CPU检查：套件、比较器、运行器、路由帮助函数共24项通过。GPU路由和11条扩展生成尚未执行。工具进程不可见GPU，下一步由普通终端执行上述命令，再根据原始产物决定是否需要调整路由或隔离完整MoE；不预先修改Top-k并列策略或数值阈值。
