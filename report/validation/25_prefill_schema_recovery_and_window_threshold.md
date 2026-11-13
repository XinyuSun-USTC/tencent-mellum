# 25 Prefill协议缺项恢复与1024边界调用修正

## 错误与无GPU恢复

原始运行`results/mellum_validation/20261112T120608015005Z`，ATOM`results/atom_mellum_gpu/20261112T120626754201Z`已正常完成。比较器拒绝新prefill输入中遗漏的enable_thinking，属于生成器协议缺项，不是GPU推理异常。

`scripts/prepare_mellum_suite.py`已为新提示和已有窗口输入填True、raw代码输入填False。该字段记录模板处理，raw项没有模板开关调用；实际模型输入由保存的token IDs决定。新增validate_inputs，在生成输出及HF导出加载模型前校验必需字段，避免计算后才发现缺项。不放宽比较器协议要求。

新增`scripts/recompare_mellum_prefill.py`仅按已知prompt_source补齐派生JSON的元数据，把logits相对路径解析到原始文件；原输入、HF/ATOM JSON及所有张量保持不变，source/derived哈希和字段修正记录保存。未知来源拒绝自动补齐。原summary仍保留artifact_comparison_error，不改写成成功。

离线产物：`results/mellum_recomparison/20261112T120859851152Z`。没有重新加载模型或运行GPU，原logits文件的SHA256校验通过。

## 离线结果

- 下一token10/10一致。
- 六条新内容、raw16/17以及window1023共9条的完整logits精确一致，所有层观察精确。
- 只有window1024的logits不同：最大绝对差1.75、RMS0.579476。
- 全部300份状态中29份不同，均属于1024样例的28层及final norm；该例embedding仍精确，首差在layer_00。

因此已有9条新增/边界输入的完整prefill证据；不能把short_greedy_passed或10个下一token相同理解为全套数值通过。

## 边界执行语义核查与修正

直接在本地HF5.10.1 CPU调用create_sliding_window_causal_mask：1023输入返回None；1024返回形状(1,1,1024,1024)的bool mask，逐元素等于下三角。官方sdpa_attention_forward在存在mask时repeat_kv，而不启用enable_gqa；显式mask时is_causal=False。

当前诊断SDPA此前对所有n<=1024都使用mask=None、is_causal=True、enable_gqa=True。边界mask在语义上仍是相同因果可见范围，但调用方式可能选择不同低精度计算路径；尚未证明存在窗口多/少看一个token的问题。

`ATOM/atom/model_config/mellum_attention.py:sdpa_single_prefill`现在仅在n达到sliding_window时构造显式下三角bool mask、展开KV头，并设is_causal=False/enable_gqa=False；窗口以下仍走原调用，full层不引入滑窗mask。n>window仍拒绝，KV/decode仍不变。这是匹配参考边界调用的显式候选修正，默认AITER未改，GPU效果尚未验证。

## 检查及下一步

模型相关CPU37项通过；脚本相关11项通过，覆盖缺元数据提前拒绝、恢复不修改原件/数值字段、未知来源拒绝以及边界mask/展开KV/SDPA选项；语法/diff检查通过。HF mask阈值接口实际CPU验证完成。不能以CPU选项检查替代GPU数值结果。

下一次执行用于验证边界修正，而不是重跑以修复比较器异常：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite prefill --qkv-backend separate --prefill-backend sdpa_cpu_rope_diagnostic --dump-layers
```

重新检查10例完整logits及layer_observation_summary；精确后再进入强制同历史decode/KV。Step0仍未完成。
