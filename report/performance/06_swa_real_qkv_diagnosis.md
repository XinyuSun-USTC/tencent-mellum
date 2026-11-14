# 06 SWA真实首层QKV：差异已缩小至窗口内核数值实现

## 决策

当前CK local候选继续关闭。两份真实首层输入可逐位回放Step0，差异在相同Q/K/V下已出现；无需继续全模型逐层定位才能尝试替代窗口实现。窗口边界改变不能单独解释观察结果。下一步优先用已保存QKV筛选另一兼容实现，必要时研究精度策略，不重跑模型采集/ABBA/200条。

运行`/workspace/results/mellum_swa_layer/20261114T145253662684Z`完成，保存源码SHA与两份QKV SHA均已复核。参考capture为Step0正常首层，非重新加载HF；输入源已在[性能05](05_swa_hf_triangulation.md)确认Step0/HF192token及logits精确。采集observer重建同公式RoPE后Q/K，后续SDPA精确复现原输出作为有效性检查。

[轻量归档及原始QKV清单](../../artifacts/performance/06_swa_real_qkv/manifest.json)，包含两个短原始trace，不将QKV大张量纳入Git。本轮只分析用户GPU产物，无新GPU任务或模型数学修改。

## 单层结果

以下relative L2以比较目标为分母。FP32 oracle是显式分块matmul/softmax、关闭TF32，另将其输出舍入BF16观察；不是正式准确性oracle，不以更接近FP32替代HF验收。

| 输入 | reference精确回放 | CK vs reference | reference vs FP32 | CK vs FP32 | CK最后位置 vs reference |
| --- | --- | --- | --- | --- | --- |
| ctx_1024 | 是 | 0.005690 | 0.001676 | 0.005574 | 0.005600 |
| ctx_4096 | 是 | 0.005689 | 0.001674 | 0.005594 | 0.005406 |

两长度CK vs reference最大绝对差均0.001953125。将FP32 oracle最终舍入BF16后，reference相对差约0.00100–0.00105，CK仍约0.00567；仅最终输出cast不能解释全部差异。不能据此断言具体中间tensor使用了哪种舍入或某一步代码有错。

## 窗口控制与内核证据

1K输入、window1024时，可见集合就是全causal；正确的local与full不应因窗口范围产生数学差异，但CK vs reference仍约0.005690。将window设1025时可见集合保持相同，AITER会根据left>=seqlen归一化窗口，回放误差降至0.000062。这表明相同数学可见集合、不同执行选路可能有明显数值差异；该1025控制没有单独trace，因此不确定声称其具体执行内核。

4K输入下window1023/1025会改变可见集合，相对reference误差分别约0.005716/0.005712，与正确window1024的0.005689相近，且并未消除差异。不能为追求相近输出把窗口擅改为1025。

两份正确window1024的短trace各含一个GPU kernel，均为`ck_tile::kentry`/`FmhaFwdKernel`/`BlockFmhaPipelineQRKSVSAsync`，实际CK Tile选路已确认。导入或LoadKernel日志中出现fmha_v3符号不能推翻实际trace；不再将候选泛称为已执行v3内核。

## 结论与下一步

事实：差异在同一真实QKV的attention执行内已出现；不只是RoPE采集、上游投影或采样器分流。推断：当前证据更支持归约/中间精度/舍入等数值执行差异，而非单纯窗口差一。未证明：具体是哪步舍入、GQA实现细节、所有层的差异分布；首层证据也不表示首层就是全模型全部分叉的唯一来源。

下一步直接复用保存的1K/4K QKV，尝试现有兼容窗口实现（优先评估冻结依赖提供的其它路径），同时核验数学窗口和GQA语义，记录reference误差与局部耗时。只有局部候选有希望再接整模型64token三方对照。若通用fused实现仍无法满足HF生成门槛，考虑保留Torch数值路径的窗口执行改造；不升级依赖、不放宽门槛、不修改其它模型模块。
