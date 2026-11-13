# 64 token与真实窗口/上下文边界验证准备

日期：2026-11-11。状态：输入、脚本、CPU回归完成；本轮GPU套件尚未执行。不是边界验证通过报告。

## 固定开发集

[prepare_mellum_suite.py](../../scripts/prepare_mellum_suite.py)用实际Tokenizer固定input IDs，生成长度由token计数而非字符数决定。long/window使用确定性的非均匀注释记录加最后问题，保留ChatML起止和thinking生成头；原始token列表是权威输入，不依赖decode后再次encode恢复边界。均为自建开发集，不是题目200公开prompt。

| 套件 | 输入长度 | 输出上限 | 本轮CPU准备结果 |
|---|---|---|---|
| short | 已有24/18token两条 | 各64token | [输入](../../results/mellum_validation/20261111T104924699102Z/inputs.json) |
| window | 1023/1024/1025 | 各8token | [输入](../../results/mellum_validation/20261111T104929656824Z/inputs.json) |
| long | 8191/8192/8193/16384 | 各4token | [输入](../../results/mellum_validation/20261111T104934767670Z/inputs.json) |

窗口集覆盖prefill边界与后续decode跨窗；长集覆盖实际YaRN位置。它们不会单独证明所有attention位置正确，若输出无分歧但logits有明显差异，仍需算子/相同前缀诊断。

## 执行与artifact

[run_mellum_validation.py](../../scripts/run_mellum_validation.py)用固定两套解释器依次执行prepare→HF→ATOM→比较，GPU任务顺序执行，日志写独立运行目录。每组单独启动，避免未排错时运行全部长输入。

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite short
```

short结果复核后，再分别执行相同命令的`--suite window`、`--suite long`。可用`--gpu 1`切换物理卡；只有`--prepare-only`不启动GPU。本轮工具未暴露/dev/kfd或/dev/dri，实际GPU命令需普通容器终端。

- [HF导出](../../scripts/export_mellum_suite_hf.py)：固定真实input IDs/mask/position，SDPA BF16，完整加载审计，greedy；保存末prompt位置全词表logits和生成token。generate在全1mask下使用同一0起始位置语义。
- [ATOM入口](../../scripts/check_atom_mellum.py)：新增--reference并由输入长度推导运行context/batched token上限，max_num_seqs1、prefix/chunked关闭，其他首版配置保持；读取原token，不重新模板化。
- 新Mellum.compute_logits中以MELLUM_PREFILL_LOGITS_DIR控制诊断，只保存真实prefill末位置，跳过dummy/warmup和decode，批大小非1拒绝；默认无路径时不落盘。每次独立引擎的prefill序号对应顺序请求，当前不适用于批处理/分块。
- 结果candidate.json与HF reference.json遵守已有schema v1，可以复用比较器。token不一致时仍比较已保存artifact，报告第一处分叉和prefill数值；不会自动重跑或修改模型窗口。

## 判据

输入、配置hash、mask、位置和生成设置必须一致；greedy token要求精确，长度不一致也判差异。logits检查形状、finite、最大绝对误差、RMS与top1。

尚未校准BF16数值容差；默认零容差仅用作诊断。`tokens_and_logits_exact`表示均精确；`tokens_exact_logits_review_required`表示token完全一致但浮点值有差异，不能宣称数值验证通过。非数值协议错误、token/shape/finite错误均失败；日志保存后再决定是否需要teacher-forcing/分层诊断。后者本轮尚未新增完整导出工具，不能将prefill末位置比较当作已完成所有decode步骤的同前缀比较。

## CPU验证

- 配置/模型测试22通过（新增prefill诊断测试覆盖dummy/decode跳过、输出last-row、序号和batch限制）。
- 套件/比较器/顺序运行器20通过（包括3项新固定长度、确定性和源输入不变测试）。
- 三组实际HF Tokenizer生成inputs，长度核对无误，旧参考不覆盖。
- py_compile及git diff --check通过，无GPU执行、无依赖更改。

## 风险与范围

当前源码prefill窗口传参存在此前静态差1疑点，尚无运行证据，本轮不预先改共享后端。长输入可能出现内存或kernel限制；HF使用SDPA和只保留末位置logits，ATOM仍需正常prefill内存，错误时保留日志不盲重跑。

尚未执行HTTP、多请求、prefix/chunked开关与正式HumanEval/200prompt验收。Git提交仍受缺少作者身份影响，不伪造身份。

## short执行结果更新

2026-11-11：RMSNorm修正后两条各64token与HF精确一致；prefill logits仍不相同，状态tokens_exact_logits_review_required。详见[RMSNorm回归](09_mellum_rmsnorm_rounding_fix.md)。下一组为window，long尚未执行，剩余数值差异保留为未解决问题。

## window真实GPU结果（2026-11-11）

已直接复核[summary](../../results/mellum_validation/20261111T121139220014Z/summary.json)、[比较结果](../../results/mellum_validation/20261111T121139220014Z/comparison.json)及[ATOM结果](../../results/atom_mellum_gpu/20261111T121158149181Z/results.json)。

| 输入长度 | 新token数 | greedy相等 | prefill max abs | RMS |
|---:|---:|---|---:|---:|
| 1023 | 8 | 全部一致 | 2.5 | 0.6463040 |
| 1024 | 8 | 全部一致 | 2.25 | 0.8244857 |
| 1025 | 8 | 全部一致 | 2.5 | 0.5130764 |

三例合计24token一致，prefill top1一致，logits仍不精确，状态tokens_exact_logits_review_required。覆盖这些prefill/decode跨窗样例，不等于逐位置mask严格验证通过；少量greedy样例不一定对差1敏感，此前静态窗口参数疑点继续保留。long尚未执行，下一步运行8191/8192/8193/16384开发样例。全部结果仍不是正式200prompt/HumanEval验收。

## long真实GPU结果（2026-11-11）

已直接复核[summary](../../results/mellum_validation/20261111T124426925046Z/summary.json)、[comparison](../../results/mellum_validation/20261111T124426925046Z/comparison.json)及[ATOM结果](../../results/atom_mellum_gpu/20261111T124459502159Z/results.json)。

| 输入token | 输出token | greedy精确相等 | prefill max abs | RMS |
|---:|---:|---|---:|---:|
| 8191 | 4 | 是 | 6.0 | 2.5177882 |
| 8192 | 4 | 是 | 2.75 | 0.7191570 |
| 8193 | 4 | 是 | 4.0 | 1.9497243 |
| 16384 | 4 | 是 | 4.5 | 2.3181140 |

四条均能完成加载、prefill和decode，无本次执行错误，16个新token相等，prefill top1一致；仍为tokens_exact_logits_review_required。4token覆盖很短，尤其thinking开头易一致，不能据此确认长输入完整生成能力、YaRN/SWA所有语义或正式16k验收。数值差异继续作为未解决项，不认定为已校准的BF16正常范围。
