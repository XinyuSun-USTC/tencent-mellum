# 37 Step0增强：长prefill滑窗语义扩展

## 改动与边界

针对报告36的长度能力缺口，扩展`ATOM/atom/model_config/mellum_attention.py:sdpa_single_prefill`：移除n>sliding_window直接拒绝的guard，在n>=sliding_window时构造`tril().triu(1-window)`布尔band mask，保留原先explicit mask及KV head展开的HF调用布局。对query位置i，只允许`j<=i且j>i-window`，包括当前token在内最多window个key；full attention仍使用无显式mask的causal SDPA。

源码依据：独立HF5.10.1 `transformers/masking_utils.py:sliding_window_overlay`返回`kv_idx > q_idx - sliding_window`，`sliding_window_causal_mask_function`将其与causal条件合取。n==window时新band和原三角mask完全相同，不主动改变此前验证路径。

原参考RoPE、QK normalization、写入实际分页KV及decode路径不变；本轮只扩展单请求、完整未缓存prefill。缓存prefill、chunked prefill和多请求decode仍不支持，不能称完整服务能力补齐。候选运行器提前长度限制从1024改为16384，属于开发验证上限，不是GPU精度验证已通过的声明。

当前band使用稠密n×n布尔mask，16k单mask约256MiB，此外还有展开KV及SDPA临时内存。未进行性能优化，也不以CPU小张量测试保证16k GPU内存/运行时间；先用窗口边界递进验证。

## 检查

- ATOM Attention CPU8项通过，包括超过窗口时只平均允许区间的等分数样例、显式mask、KV布局/写回与原有RoPE测试。
- 独立HF CPU5项通过，新增band与官方sliding_window_causal_mask_function在阈值、超过阈值、多窗口长度上的逐元素比较；其余官方专家对照/观察测试保留通过。
- 运行器CPU5项通过，输入>16384仍在GPU前拒绝。
- git diff --check通过。
- window suite prepare-only：`results/mellum_validation/20261112T222827265167Z`，1023/1024/1025输入，每条最多8输出token，固定HF/ATOM原始ID协议及逐步logits观察。

本轮未启动GPU，长输入GPU正确性仍待验证。报告36原审计产物保持不变；audit脚本已将长输入门槛记为not_verified，CPU只探测小尺寸超过窗口的prefill支持，避免审计脚本为能力探测分配16k稠密mask或误称旧限制仍存在。

## 下一轮

先执行：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite window --correctness-candidate
```

验证1023/1024旧边界回归及1025首次越窗prefill，普通生成、HF实际专家切换及完整原始logits均继续核对。通过后再验证8191/8192/8193与16384，不将窗口小测试直接推广为16k通过。之后补可复现服务入口/多请求能力与200条/HumanEval验收，保持Step0阶段，不开始性能优化。

## 首次越窗GPU验证通过

运行`results/mellum_validation/20261113T085534271528Z`，ATOM`results/atom_mellum_gpu/20261113T085556427546Z`：1023、1024、1025输入各实际生成8token，3/3完整序列、24/24 token及24个采样前完整logits向量全部精确；3个独立prefill logits max_abs=0。加载审计5631/5631，无缺失/意外项。1025首次越窗prefill已具GPU外部参考证据，不能再将所有>1024输入写成不支持；尚不能推广为16k通过。

下一组long suite包含8191/8192/8193/16384，覆盖YaRN原始位置边界与16k，每条最多4个普通生成token。该组是长输入能力smoke，最多16个输出位置，不代替题目200条前64token及16k子集完整验收。prepare-only `results/mellum_validation/20261113T085802164110Z`，输入固定、候选设置不变。执行：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite long --correctness-candidate
```

长输入比上轮更重，继续观察15秒心跳和阶段日志；每阶段现有300秒超时及子进程清理保留。若失败先检查产物与首差，不直接重跑或改变阈值。

## 8192边界与16k能力smoke通过

运行`results/mellum_validation/20261113T085844629113Z`，ATOM`results/atom_mellum_gpu/20261113T085919838094Z`：实际输入8191、8192、8193、16384，每条实际4输出token。4/4序列、16/16 token、16个采样前完整logits向量及4个独立prefill logits全部逐值精确，5631/5631加载审计通过，无缺失/意外项。

当前已有真实checkpoint GPU证据证明16k输入可执行，不能再将16k“不支持”当作现状。报告36是增强前的历史审计快照，不覆盖新实现。尚未完成的是更长decode回归及正式200条/HumanEval/服务验收。

为覆盖题目64token长度，准备`results/mellum_validation_inputs/long_generation_64_v1/inputs.json`：保留本次4个固定输入的全部ID/mask/position，仅将max_new_tokens从4调整为64。此集合仍是开发长输入回归，不冒称官方16k公开子集。下一命令：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite diagnostic --inputs /workspace/results/mellum_validation_inputs/long_generation_64_v1/inputs.json --correctness-candidate
```

通过后停止增加同构长度定位样例，转入200条规模对照、HumanEval配对与服务能力补齐。无新的数学改动，不进入Step1。

## 长输入64token回归通过

运行`results/mellum_validation/20261113T103154043872Z`，ATOM`results/atom_mellum_gpu/20261113T103233957636Z`：8191/8192/8193/16384各实际64输出token，4/4完整序列、256/256 token、256个逐步完整logits向量及4个prefill logits精确。该组长输入与64步普通生成闭环完成，后续进入200条开发对照（报告38），不继续重复同构长度实验。仍不是官方集合或HumanEval pass@1验收。
