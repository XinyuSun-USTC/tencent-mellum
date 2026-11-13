# 21 QKV打包证据与成对prefill候选验证

## 实际投影结果

原始结果：`results/mellum_projection_math/20261112T085922910671Z/results.json`。参考重放精确且所有重复精确；torch打包形状改变部分输出，而相同打包shape的AITER与torch输出完全一致。

| 探针 | Q不同元素 | K不同元素/最大绝对差 | V不同元素/最大绝对差 |
|---|---:|---|---|
| h_0_prefix_0 | 0 | 6 / 7.62939e-6 | 3 / 2.38419e-7 |
| h_0_prefix_14 | 0 | 6 / 1.90735e-6 | 0 / 0 |
| h_1_prefix_0 | 0 | 5 / 0.000488281 | 9 / 0.000244141 |
| h_1_prefix_9 | 0 | 5 / 0.000488281 | 11 / 0.000244141 |

表中为torch打包vs分开投影。所有分开torch Q/K/V均精确复现保存HF；四份固定输入上的O/router两种内核都精确复现HF。这些事实不支持“全面替换tgemm”作为当前修复，也不证明tgemm在全部层/shape与torch一致。少量K/V差异尚未被证明能解释完整模型分叉，必须通过受控模型前向验证。

## 显式候选：只拆执行，不拆加载

新增`ATOM/atom/model_config/mellum_projection.py:install_separate_qkv/separate_qkv_forward`。对既有QKV模块绑定实例forward，以原打包weight的Q/K/V切片分别执行F.linear，再拼回原输出结构；保留原Parameter、weight_loader、参数名、state_dict和forward观察器。支持可选bias，检查canonical BF16形状，拒绝x_scale与非BF16调用。

`ATOM/atom/models/mellum.py:MellumAttention.__init__`通过`MELLUM_QKV_BACKEND=separate`启用，要求TP1。默认packed不变；不修改共享QKVParallelLinear。check_atom_mellum和run_mellum_validation提供`--qkv-backend`并记录实际值。不改O/router/lm_head，不在未验证时把候选切成默认。

## 一次运行收集两个完整prefill对照

新增`scripts/run_mellum_prefill_study.py`：

1. 复用经过哈希/形状核验的四条固定探针，保留token/mask/position及强制公共前缀；移除旧generated_ids/logits路径，均只生成1 token。
2. 在独立HF环境导出一次新参考及层/子算子结果。HF导出脚本现显式指定`experts_implementation=grouped_mm`，避免自动回退；仍使用SDPA并记录实际专家后端。
3. 在相同参考和默认Attention下依次运行packed、separate两个ATOM控制，明确固定torch MoE、aiter prefill和第0层观察，不混入SDPA或分步旋转选项。
4. 保存每组token/logits、完整层输出及选定权重比较。各只有四个prefill下一token，不能把内部历史名称short_greedy_passed理解为short64回归通过。

结果采集允许token不匹配（它是测量值），但子进程失败/缺少完整产物不能冒充diagnostic_complete。无自由decode，不据下一token四例相同宣称Step0或KV正确。

## 本轮检查与后续判据

模型相关CPU35项、脚本相关8项通过。新增测试验证权重/loader/state_dict/钩子保留、bias切片、非法类型/shape拒绝，以及固定历史保留、陈旧输出移除。语法与ATOM diff检查通过。

CPU prepare-only产物：`results/mellum_prefill_study/20261112T090247658002Z/results.json`。没有GPU运行，没有修改默认后端或数值阈值。

普通终端执行：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_prefill_study.py
```

优先检查separate组QKV是否精确复现HF、加载/选定权重是否保持一致；再比较prefill logits及逐层误差。如果QKV对齐但后续仍不同，保留为局部证据并定位最早后续差异，不立即跑expanded；如果误差没有整体改善，也不以局部精确直接提升默认。固定历史decode与独立输入仍属后续工作。

## 成对prefill真实结果

产物`results/mellum_prefill_study/20261112T095954484085Z`，packed子运行`20261112T100009010337Z`、separate子运行`20261112T100037552818Z`。四条均为固定历史的一步prefill，无自由decode。

分开QKV四例首层Q/K/V全部精确，选定权重两组均精确。下一token由packed的3/4变为separate的4/4，但logits没有全面改善：

| 输入 | packed logits RMS | separate logits RMS |
|---|---:|---:|
| h_0_prefix_0 | 0.637468 | 0.260327 |
| h_0_prefix_14 | 0.488468 | 0.642796 |
| h_1_prefix_0 | 0.150792 | 0.204311 |
| h_1_prefix_9 | 0.418597 | 0.490429 |

separate最终norm相对L2仍约0.0676–0.0912。局部投影对齐成立，但不能声称它解释全部prefill差异。保持默认packed，不直接恢复expanded，也不以4个下一token匹配当作完整回归通过。

## 复用参考，补齐组合对照

已有证据分别证明MoE同输入可精确、QKV分开可精确、HF旋转+SDPA同输入可精确，但此前没有在同一完整prefill中同时启用这些候选。下一次只补两个组合：separate+sdpa_diagnostic、separate+sdpa_rope_diagnostic，默认torch MoE不变；与本次separate+aiter构成相同参考下的Attention组合对照。

`run_mellum_prefill_study.py --attention-study --reference-run`现支持只读复用本轮HF产物。复用前核验已完成状态、输入文件哈希、配置、记录数量、token/position/mask/生成设置及明确的HF5.10.1/grouped_mm后端，记录当前参考文件和tensor哈希。当前哈希快照是新实验来源记录，不冒称旧运行已保存这些哈希。每组检查实际后端settings及5631加载审计。没有复制或改写旧HF结果，也不重新加载HF模型。

脚本相关CPU9项通过，覆盖复用输入/position不符拒绝；语法检查通过。准备产物`results/mellum_prefill_study/20261112T100506583597Z`，组合GPU实验未执行。本轮未改变模型默认路径。

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_prefill_study.py --attention-study --reference-run /workspace/results/mellum_prefill_study/20261112T095954484085Z
```

优先判断完整prefill最早差异和logits，而不是四个下一token是否碰巧一致。sdpa_rope仍复用AITER缓存，KV写入/实际decode仍沿原路径；即便prefill改善，也不表示cache/decode闭环完成。若组合仍不对齐，利用保存层输出固定相同输入重放最早差异层，不继续扩大自由生成。
