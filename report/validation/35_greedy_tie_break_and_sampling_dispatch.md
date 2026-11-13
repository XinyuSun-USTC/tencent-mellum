# 35 普通多样生成：并列最大值与greedy调度修复

## 已保存的GPU证据

运行`results/mellum_validation/20261112T205013366006Z`，ATOM`results/atom_mellum_gpu/20261112T205052781277Z`。10条每条64token，8/10完整序列精确，629/640位置token一致。全部10条prefill完整logits精确；同历史共630个完整采样前logits向量全部逐值精确，nonexact_positions=0。停止比较分叉后不同历史，不把其数值作为模型同输入比较。

两个首分叉位置均为完全相同向量上的最大值并列：

| 样例 | 输出位置（从0计） | 最大logit | 并列token ID | HF选择 | ATOM选择 |
| --- | --- | --- | --- | --- | --- |
| stable_unique | 59 | 37.25 | 8599、8943 | 8599 | 8943 |
| identity_explanation | 57 | 34.25 | 392、2516 | 392 | 2516 |

window_1023和window_1024均完整64token精确，包含越过1024窗口的decode。本轮失败已有直接采样证据，不需要再次修改MoE/attention数学来解释这两个首分叉。

## 源码链与修复

`ATOM/atom/model_engine/model_runner.py:ModelRunner.prepare_sample`在CPU由原温度判断all_greedy，之后才把温度clamp到SAMPLER_EPS，并把all_greedy传至Sampler。

`ATOM/atom/model_ops/sampler.py:Sampler.forward`此前在top_ks/top_ps均None时直接调用`_temperature_sample`，后者调用AITER `mixed_sample_outer_exponential`；只有过滤路径`_topk_topp_sample`才判断all_greedy并执行torch.argmax。因此同一greedy语义因是否带过滤参数而进入不同路径。保存数据证明无过滤路径在这两次精确并列中未选择HF argmax所选的首个索引；不额外推断AITER内部归约实现。

修复在`Sampler.forward`分流前判断all_greedy并执行`logits.argmax(dim=-1).to(torch.int)`，复用已有过滤路径的greedy规则，不改变logits、返回dtype或随机采样路径。它适用于该共享采样器的纯greedy批次，影响范围不局限Mellum；混合greedy/随机批次和随机抽样仍需独立验证，不能据此声称全部采样模式已验收。这是正确性调度修复，不是性能优化。

## 验证与下一轮

新增`ATOM/tests/test_sampler_greedy.py`读取并执行真实Sampler类的AST以避免CPU环境导入GPU AITER扩展；共3项通过。覆盖98304词表及两个实际失败并列ID、有/无过滤时都不进入随机kernel、非greedy无过滤仍走温度采样。该测试验证CPU分支语义，不替代GPU集成。git diff --check通过。

下一轮沿用完全相同的10条输入重跑普通候选，不调整阈值或参考方式，核验640个token及所有同历史原始logits。GPU尚未执行修复后回归。

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite diagnostic --inputs /workspace/results/mellum_validation_inputs/normal_generation_10_v1/inputs.json --correctness-candidate
```

完整Step0仍需长prompt、并发/服务和题目正式正确性验证；当前prefill候选仍受限单请求、输入≤1024。

## 修复后GPU回归：10例完整通过

运行`results/mellum_validation/20261112T213129468143Z`，ATOM`results/atom_mellum_gpu/20261112T213209402440Z`：tokens_and_logits_exact。10/10完整序列精确，每条实际64输出token，总计640/640 token精确；640个采样前完整logits向量全部逐值精确，nonexact_positions=0，全部10个独立prefill向量max_abs=0。此前两个并列分叉用例也完整通过，无需降低阈值。

实际模式normal、ignore_eos=False，无强制历史；加载5631/5631且missing/unexpected为空。每条HF实际prefill为grouped_mm、后续63次decode为batched_mm。window1023/window1024两条均覆盖跨窗口的64token普通生成。此结果支持本次greedy调度修复在当前单请求候选上的集成正确性。

阶段结论：已完成当前受限候选的多样短输入及1024边界普通生成闭环，可以停止重复这组定位实验。仍不是完整Step0：prefill仍受限≤1024，缺少题目16k输入、并发/服务和正式正确性指标验证。下一优先项是将已验证的数学路径扩展到长prompt，先覆盖1025及8192附近，再到16k；在能力补齐前不能直接对现候选运行长输入。之后再核验多请求/服务入口并整理可复现适配版本，不进入Step1性能优化。
