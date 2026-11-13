# 05 Step0基线配置与复现说明

版本说明：本文工作树/HEAD叙述属于Git迁移前Step0收尾快照。当前已纳入总仓库，实际提交、标签及路径见[07仓库管理](07_project_repository_management.md)；原始运行与指纹保持不变。

## 使用范围和版本

本文件是已通过本地正确性检查的Step0候选配置入口。阶段结果见[报告39](../../report/validation/39_step0_summary.md)，原始规则见[题目摘要](../requirements/01_assessment_summary.md)。单卡TP1/DP1、BF16、eager，顺序单请求；输入上限在当前验证运行器中为16384。不将默认ATOM启动等同于本候选。

- 工作副本`/workspace/ATOM`，分支`feature/mellum-support`，HEAD `26f23a0bdbf50b9010c07ee3a71171189723e767`。
- 适配文件仍有暂存/未暂存/未跟踪改动，HEAD只是原始基线。复制仓库必须包含这些文件；仅checkout该commit不能复现候选。
- 本次源码、脚本和原始产物指纹见[清单](../../report/validation/39_step0_manifest.json)，是整理时当前工作树快照，不是回填的历史测试commit。
- ATOM解释器`/opt/venv/bin/python`，冻结Transformers 5.2.0；参考解释器`/workspace/venvs/hf-reference/bin/python`，Transformers 5.10.1。两者必须分开使用。
- 只读权重`/data/models/Mellum2.1-12B-A2.5B-Thinking`；数据与配置哈希随输入artifact保存。原始权重、ROCm/PyTorch/AITER不升级。

## 实际候选设置

`run_mellum_validation.py --correctness-candidate`选择以下配置，并在结果中核对worker实际设置：

| 配置 | 值/行为 |
| --- | --- |
| `MELLUM_MOE_BACKEND` | `torch` |
| `MELLUM_QKV_BACKEND` | `separate` |
| `MELLUM_ROUTER_BACKEND` | `torch` |
| `MELLUM_PREFILL_BACKEND` | `sdpa_cpu_rope_diagnostic` |
| `MELLUM_REFERENCE_KV_ROPE` | `1` |
| `MELLUM_SDPA_DECODE` | `1` |
| `MELLUM_HF_GENERATION_EXPERTS` | `1`：prefill grouped、decode batched |
| QK/RoPE/cache与allreduce/norm融合 | `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0`、`ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0` |
| Unified attention | `ATOM_USE_UNIFIED_ATTN=0` |
| 引擎 | TP1/DP1、enforce_eager=True、level=0、kv_cache_dtype=bf16、max_num_seqs=1、gpu_memory_utilization=0.35 |
| 缓存功能 | prefix caching=False、chunked prefill=False；实际分页KV保留 |
| 生成 | temperature=0、normal sampling、ignore_eos=False、无forced计划 |

CPU逆频率在模型初始化建立，旋转/SDPA和专家计算仍在GPU。带diagnostic名称的后端参与正常数学计算；这是命名沿革，不是“只观察不干预”的模式。后端选择具有HF语义依据，但不能自动当作性能优化成果。

max_model_len由输入长度加输出预算向block边界取整、至少256；max_num_batched_tokens不小于该值。加载器线程池关闭以固定审计执行，PYTHONPATH优先`/workspace/ATOM`。具体附加环境、动态目录及引擎值以各ATOM results.json和`check_atom_mellum.py:main`为准。

## 已有运行与复现命令

以下命令启动真实GPU任务，需在有ROCm设备的普通容器终端执行。本文整理时未重新运行。独立HF解释器只用于参考子进程；主运行器用冻结解释器。

短输入最小回归：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite short --correctness-candidate
```

滑窗阈值（1023/1024/1025，每条8输出token）：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite window --correctness-candidate
```

固定长输入完整64token回归：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite diagnostic --inputs /workspace/results/mellum_validation_inputs/long_generation_64_v1/inputs.json --correctness-candidate
```

固定200条开发回归：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite diagnostic --inputs /workspace/results/mellum_validation_inputs/developer_200_v1/inputs.json --correctness-candidate --stage-timeout 3600
```

原始验证结果分别见报告39结果表。输入JSON存在时直接复用，不重新分词或改prompt。若迁移后丢失输入，可用`prepare_mellum_acceptance_inputs.py --output <新路径>`重建200条集合，但必须核对源文件、tokenizer和结果哈希；不能宣称未核对的新集合与旧实验完全相同。长输入64token集合来自既有long输入，只改输出预算，保留其来源字段。

运行器支持`--prepare-only`，只检查/准备输入，不启动推理。`--stage-timeout`是每个子阶段的秒数上限，默认300，200条使用3600；15秒心跳、worker死亡检查、进程组清理保留。超时不是数值正确性失败，先检查阶段日志和完成产物。

## 单独运行ATOM

若已保存HF reference，可用下列入口复现对应候选，不重复加载HF：

```bash
/opt/venv/bin/python /workspace/scripts/check_atom_mellum.py --reference /workspace/results/mellum_validation/20261113T104458011376Z/hf/reference.json --moe-backend torch --qkv-backend separate --prefill-backend sdpa_cpu_rope_diagnostic --reference-kv-rope --sdpa-decode --torch-router --hf-generation-experts --dump-generation-logits
```

这仍是离线Engine生成验证，不是已验收的HTTP服务命令；普通`/workspace/scripts/atom`只保证工作副本导入路径，尚无本候选服务正确性/性能产物。服务入口将作为Step1测量准备工作处理，不编造已验证启动命令。

## 结果读取和回归政策

- 新运行创建唯一`results/mellum_validation/<run>/`，保存inputs、HF reference、summary、comparison、generation_comparison及阶段日志；实际ATOM结果由summary.atom_run定位。
- 加载审计要求5631源槽全部成功、无missing/unexpected。不能将回调成功解释为全权重数值审计。
- `tokens_and_logits_exact`是当前开发严格比较标签；`diagnostic_complete`、`forced_history_logits_exact`、HTTP health均不能替代普通生成对照。
- 比较器只在历史相同的位置比较采样前logits，含首次分叉位置；历史分叉后不再将logits差异解释为同输入数值误差。
- Step1每次改动先跑与影响有关的CPU检查和短/窗口回归；改变长上下文、KV、SWA/YaRN时必须补长输入；候选最终配置再跑200条及题目要求指标。需要失败证据时才启用层dump与重放，避免每轮全量日志和GPU重跑。
- 题目门槛是token/任务指标，完整logits逐值精确是定位工具。优化后非零logits误差需评估，但不能仅凭它判定题目验收失败；不通过更改数据、停止规则或分母掩盖token/任务退化。

## 未覆盖项与交接

当前候选只支持单请求完整未缓存prefill、单请求单token decode；prefix/chunked prefill、并发批量、compile/CUDAGraph、TP>1、FP8、MTP未由上述证据覆盖。长prefill的稠密band mask和batched专家复制存在资源开销，需profiling后判断优化优先级。

200条是164 HumanEval prompt加36提供的场景A prompt，不等于200道HumanEval或确认的官方公开集。完整HumanEval答案生成、代码提取/执行及有效参考pass@1未完成；不能采用被Rejected的3/164分数。最终性能评分相对出题方Baseline；本候选用于内部Step0与各优化点的归因。当前可以推进Step1，不将未确认的外部材料变成无限等待条件。
