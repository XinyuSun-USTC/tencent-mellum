# 08 一轮Triton窗口筛选：进入32条生成对照

## 结论

冻结AITER Triton default窗口路径编译/执行成功，有资格进入正常生成验证：相同真实QKV下，比CK更接近SDPA，仍保留明显局部加速。**局部误差下降不等于生成通过，不要求bitwise才允许继续。** 保留CK，尚不切换默认实现，也不重跑ABBA/全200/全HumanEval。

运行`/workspace/results/mellum_swa_triton_screen/20261114T200217286077Z`，源码对应`9596dc1`。源码指纹已与该提交及冻结依赖核对，两份trace SHA复核。[证据归档](../../artifacts/performance/08_triton_swa_screen/manifest.json)，[计划12](../../docs/plan/12_bounded_triton_swa_screening.md)。本轮离线分析后接入可选实现，但GPU结果只来自筛选，未执行新整模型实验。

## 探针与执行路径

MI308X/gfx942、Torch2.10/ROCm、Triton3.6.0。n1025/window1024首key脉冲探针通过：第0位置GQA映射精确、query1023含左边界、query1024完全排除首key；两份真实QKV reference精确回放。

trace实际出现`_attn_fwd_IS_CAUSAL_1_NUM_Q_HEADS_32_NUM_K_HEADS_4_...SLIDING_WINDOW_1023`，另有BF16输出/FP32 LSE相关fill kernel。不能把fill都当attention内核；计时包含实际包装调用及分配/填充，未扣除这些成本。参数使用(1023,-1)+causal，依据冻结实现对right的限制及实际mask语义，不复用CK的(1023,0,0)。

## 同输入局部结果

relative L2以SDPA为目标。计时3次预热、各10次、SDPA/CK/Triton轮转、CUDA event，无profiler；下表中位数。

| 输入 | CK relative L2 | Triton relative L2 | SDPA ms | CK ms | Triton ms | Triton相对SDPA加速 |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 0.00568982 | 0.00057586 | 0.5386 | 0.2046 | 0.3328 | 1.62× |
| 4096 | 0.00568943 | 0.00001611 | 13.0860 | 0.8011 | 1.0848 | 12.06× |

1K差异从约0.569%降至0.0576%，约降9.9倍；4K从约0.569%降至0.00161%，约降353倍。4K最后位置逐位精确，但不外推所有层/所有生成位置。两长度最大绝对差均0.0009765625。

Triton比CK慢：1K约1.63倍耗时，4K约1.35倍耗时。若生成更稳定，这可能是合理取舍；不能将局部12×说成整模型12×，没有16K或整模型性能新证据。本次只筛一条现有替代实现，不继续扫tile/config或追求数值逐位精确。

## 有条件进入32条对照

接入`MELLUM_SWA_PREFILL_IMPL=ck|triton`（默认ck），仍须`MELLUM_SWA_PREFILL_OPT=1`才激活；整体优化默认仍关闭。只改变符合窗口阈值的SWA prefill，Full/短输入、reference RoPE、KV、decode、MoE保持。非法实现拒绝，无静默fallback。44项相关CPU通过，未改冻结依赖。

复用开发200中32条≥1024输入及同一HF输出，normal greedy64，不重跑HF：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_swa_200.py --eligible-only --swa-prefill-impl triton
```

prepare-only：`/workspace/results/mellum_swa_32/20261114T204321161713Z`。实际运行预计约2–3分钟，15秒心跳，单阶段3600秒上限。产物`results/mellum_swa_32/<id>/`明确标记激活子集，空below_window组为null比例，不冒称完整200通过。核对实际MELLUM_SWA_PREFILL_IMPL=triton与5631加载审计。

对照CK既有32条位置1166/2048=56.9336%、序列7/32。若生成改善明确，再补完整200及端到端收益；若改善不足，记录放弃并结束这轮Attention筛选，转向服务/批量能力。不因局部更精确而自动采用，也不将32条当正式公开集门槛。
