# 06 功能评估期间的并行准备：候选服务接口与场景C前缀复用

本轮不启动GPU、不修改正在运行的HumanEval工作区。发现HF端loading_info的set序列化遗漏后，在独立`/workspace/tencent-mellum-delivery`修复并交付reference-only recovery（df33885）；原ATOM生成可继续，原目录不改。另建`/workspace/tencent-mellum-next`用于后续源码/CPU开发，不改修复工作区的运行源码。

## 1. 实际C数据存在有分量的共享前缀

CPU用原始C数据、冻结tokenizer及已核验chat_ids处理，保持performance协议thinking=True：200条均属dev-shared-prefix-0，共有前3820token，按block16为238个完整块（3808token）。

- 全部输入2039880token，范围4068–16027。
- 每组首请求之外的公共块复用量为199×3808=757792token，约总输入37.1489%。
- 这是组内全体共同前缀的结构复用量，未计缓存容量/驱逐/到达时序，也未统计更深的两两/子组前缀；**不是实际命中率、统一上限或37%端到端加速承诺**。

[CPU证据](../../artifacts/research/06_parallel_preparation/scenario_c_prefix_potential.json)，原数据SHA38fc340a76fd48c025bf5806376bdabbc7762f6b5c48d4509bbb109af2cfe265。脚本`scripts/analyze_mellum_prefix_potential.py`可复现，normalize BatchEncoding为input_ids，不能把len(BatchEncoding)=2当成token长度。

C仅输出就要求平均0.3×256=76.8tokens/s/GPU，此外还需处理长输入。R在A测得41.58token/s不能直接当C容量，但已提示压力；新候选的短离线加速不能直接乘这个数当C实测。前缀复用是有数据依据的后续候选，比没有依据地继续做外围小算子优化更值得关注。

## 2. 为什么当前不能直接打开prefix开关

冻结`ATOM/atom/model_ops/attention_mha.py:PagedAttentionImpl.rope_cache`约288行，在has_cached时调用`_gather_prefix_and_concat_kv`；后者约298行按分页表收集cached+new K/V。框架已具备基础能力。

但Mellum当前`mellum_attention.py:install_reference_kv_rope`的顺序是：先算参考qr/kr，调用original rope_cache，**再**overwrite_reference_keys，最后返回新qr/kr。如果original已经gather cached+new，参考K修正晚于gather，且返回new-only K与full V可能不匹配。旧ragged桥和native_prefill的uncached检查正是当前支持边界，不能删掉guard就宣布缓存正确。

建议的最小改造边界：在Mellum局部适配器中让reference K覆盖先于prefix gather；保留框架的页表/槽位管理、V写入和gather实现，再让varlen后端使用真实cu_q!=cu_k及bottom-right causal。必须验证新token绝对位置、cached+new拼接、窗口边界、不同请求/EOS和缓存复用/驱逐。此处仅规划，**没有实现或启用prefix**。

## 3. 已完成CPU开发：显式服务profiles

新增`scripts/mellum_execution_profiles.py:profile_settings`，扩展`serve_mellum.py --profile`：

| profile | MoE | Attention | 控制 |
| --- | --- | --- | --- |
| r | Torch共享路由 | reference ragged SDPA | LE |
| fused_moe | 原生FusedMoE | reference | L |
| r_swa_ck | Torch共享路由 | reference＋SWA CK prefill | LE |
| fused_moe_swa_ck | FusedMoE | reference＋SWA CK | L |
| native_attention | Torch共享路由 | varlen/paged | LE |
| fused_moe_native_attention | FusedMoE | varlen/paged | L |
| isolated_moe | 贡献独立归约 | reference | L |

这些是既有路径的可审查配置，**不是七个已验收优化点**。isolated和native Attention保留作消融，不因接口存在就升级为合格版本。profiles禁止与手工campaign/implementation混用；fused配置不安装E的Torch专用wrapper；native Attention禁用旧SDPA/ragged桥。默认旧服务行为保留。

支持显式`--disable-thinking`，不改变性能测试的默认thinking=True。`--batch-audit`同时记录native路径。仍为BF16/eager、TP/DP1、prefix/chunk关闭；新profile没有绕过模型现有检查。

CPU prepare示例（**不占GPU**）：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-next/scripts/serve_mellum.py --profile fused_moe --batching --max-num-seqs 32 --prepare-only
```

15项服务/profile/前缀计算CPU回归通过；尚未启动这些profile的HTTP/GPU，不将CPU准备当服务实测。后续无需再临时手改环境变量，即可开展相同配置的HTTP/性能消融。

## 4. 尚不优先做的工作

Graph/compile受现有参考Attention的tolist/item/gather、L请求拆分、动态campaign控制等约束，仅打开图开关没有源码依据；量化/MTP/双卡模型会增加验收范围。当前两张卡用于HumanEval，优先完成现有候选功能取舍及真实服务接入。是否继续MoE舍入修复，应由完整功能结果和剩余时间决定，不默认再进入数值搜索。
