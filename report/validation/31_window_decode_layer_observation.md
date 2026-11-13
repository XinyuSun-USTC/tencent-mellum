# 31 滑窗首decode的逐层成对观察

## 已确认的缩小范围

轻量重建`results/mellum_decode_boundary/20261112T145752408259Z`四例首层Q、实际K/V、SDPA context全部精确；其中window1023在首decode逻辑长度1024时也精确。前轮完整模型最终logits在该点仍不同，因此应检查后续算子/层，不能据旧全模型误差继续修改首层缓存。

## 一次有界的观察任务

`run_mellum_decode_validation.py --layer-study --steps 2`仅选window1023，执行一次prefill与首decode；保留正确Q/K/K写回、separate QKV、torch MoE和cached SDPA候选，默认模型路径不变。不再跑其他三例或8步完整基线。

HF新增`scripts/mellum_decode_observers.py:observe_decode_layers`：

- 对embedding、28层和final norm注册只读输出hook，保存两个cached前向的状态，uncached内部对照时关闭capture。
- 临时包装当前进程的SDPA接口，配置仍为sdpa，mask构造路径不变；在首decode保存真实传入的Q/K/V、mask及原函数输出context。
- finally恢复原SDPA注册和移除hooks，不改依赖源码文件。该接口替换仅观察，不改变Attention结果。

ATOM新增`ForcedDecodeProbe.install_layer_hooks`，在实际forward/logits配对活动期间保存同名状态；Decoder的hidden+residual按BF16实际materialize，final norm只保存normalized输出。`--decode-layer-capture`只允许配合forced SDPA decode。已有同缓存native/SDPA快照扩展到各层，计划外terminal不重复采集。

运行器自动比较60份层状态，以及28层×Q/K/V/context共112项首decode快照，输出layer_state_comparison.json、attention_state_comparison.json及first_nonexact_state/first_nonexact_attention。先看首差，不用最终token相同代替层级证据。

## 检查与成本边界

- ATOM相关44项通过，新增只读layer hooks的残差materialize及非活动期不覆盖检查。
- 运行器相关5项通过，包含单窗口输入选择。
- 独立HF小模型2项通过，新增观测前后cached/uncached logits精确不变、registry恢复、层与Attention快照shape检查。
- 语法/diff检查通过，prepare-only产物`results/mellum_decode_validation/20261112T151428620767Z`。

本次必须重新采集HF，因为前轮没有保存后续层快照；不是重复导出已有参考字段。需要一次HF与ATOM模型加载，但各只采一个样例的两个位置，仍保留15秒心跳、worker死亡检测和阶段超时。GPU尚未执行。

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_decode_validation.py --layer-study --steps 2 --reference-kv-rope --sdpa-decode
```

下一步根据最早非精确项：输入已不同先回到前层；相同输入/权重但Q/K/V不同查投影/缓存；Q/K/V精确但context不同查mask/SDPA；context精确但层输出不同查O投影、残差、norm及MoE。暂不增加自由生成或性能工作，Step0未完成。
