# 优化探索、失败定位与工程取舍

本页补充技术报告第3节。讨论对象为已实现并实际测试的路线，另将仅调研或暂缓的方案独立列出。下述收益各自绑定原负载与对照，不相加或连乘为最终收益。开发200指既有本地集合，子集检查不改称完整200结果。

## 1. 窗口注意力：从局部提速到激活样本验证

以参考profile中16K prefill的SDPA占比72.21%为依据，先接入AITER CK窗口内核，检查左窗口边界、GQA映射、真实Q/K/V回放，再做端到端ABBA。CK的16K×64离线TTFT约3996→1426ms，但长生成出现分叉。继而筛选冻结AITER的Triton实现，1K/4K局部误差较CK下降约9.9/353倍，4K调用相对SDPA约12.06倍。

真实激活32条的64token位置一致率仅从56.93%提高至61.62%，且有样本改善与退化并存。CK完整开发200曾为93.11%，但其中168条短输入不激活该优化，不能据总体分数忽略激活子集。Triton只完成该32条生成筛查，未拼接短样本制造一个“完整200”结果。停止继续扫描tile/config，保留可选实现，后续转向服务批量能力。

实现：`mellum_attention.py`、`mellum_swa_triton.py`。证据：[ABBA](../../report/performance/04_swa_prefill_abba.md)、[QKV诊断](../../report/performance/06_swa_real_qkv_diagnosis.md)、[局部筛选](../../report/performance/08_triton_swa_screen.md)、[激活子集](../../report/performance/09_triton_activation32.md)。

## 2. 专家执行：共享路由、直接索引与融合路径

**共享路由保留。** 将真实批次的softmax、top-k及归一化共享执行，保留逐请求GEMM形状，开发200输出与原控制逐条一致。该机制最终纳入r32工程基线，不再计入优化①。专家group4是此后单独加入的执行粒度优化。

**索引GEMV未保留。** Triton indexed GEMV直接读取专家权重，减少选中权重的物化。decode32局部时间8.030→1.641ms，调用额外allocated峰值64.20→3.71MiB；两者均为局部形状回放，不是整模型带宽或显存测量。短输出不能揭示完整风险：前16token总体匹配曾达95.34%，完整64token仅78.23%。因此停止晋级，不以局部低误差或4.892倍加速替代生成门槛。

实现：`mellum_moe_optimization.py`、`mellum_indexed_gemv.py`。证据：[共享路由与局部测量](../../report/performance/14_moe_routing_and_indexed_decode.md)、[完整200复核](../../report/performance/15_indexed_decode_d200.md)。

**FusedMoE及固定归并未保留。** 原生专家批量几何加速约1.4663倍，开发200为72.3594%。进一步固定权重、输入和路由，对prefill150、decode1、decode32三形状各重复12次，观察到输出变化；更换排序仍未消除。源码和trace确认所测CK输出归并使用AtomicAdd，但未把全部差异都归因于它。

随后将逻辑top8展开为独立贡献输出，再用明确的加权与top-k归约：局部重复最大差降至0，完整开发200提高至74.1797%，仍未达标。这项修复还改变了展开行数、临时存储及执行形状，不能描述成仅替换原子指令；没有新候选速度结果，也未借用旧FusedMoE加速比。该过程说明，重复确定性与参考一致性是不同问题。

实现：`mellum_native.py`、`mellum_phase_moe.py`。证据：[原生试验](../../report/performance/19_native_runtime_pilots.md)、[排序与重复性](../../report/performance/20_native_phase_diagnostics.md)、[独立贡献归约](../../report/performance/21_isolated_moe_reduction.md)。

## 3. 原生注意力：阶段拆分与模块交叉对照

整套原生Attention批量几何加速约1.4507倍，但开发200为71.3750%。保持参考专家，仅替换prefill或decode，两组一致率分别为72.5547%和77.2500%，未得到可直接晋级的阶段组合。

进一步在固定20条输入上交叉切换专家和Attention管线：原生/原生69.84%，原生Attention/参考专家71.17%，参考Attention/原生专家77.03%，参考/参考100%。该控制说明诊断链可恢复一致，也提示两条路径均需进一步定位；它包含KV等外围逻辑，不能把差异全部归为FlashAttention kernel精度。20条交叉诊断不当作完整200验收。

证据：[阶段结果](../../report/performance/20_native_phase_diagnostics.md)、[四格诊断](../../report/validation/43_core_attention_expert_matrix.md)。当前参考兼容路径继续采用Torch SDPA。

## 4. 投影、Graph与融合：性能有效，整体资格不足

恢复整批投影的短A64吞吐由121.4966增至439.6727token/s/GPU，约3.619倍，表明按请求小GEMM发射具有实际成本。围绕此路线继续实现和测试packed QKV、原生routing、RMSNorm、驻留RoPE、QK-Norm/RoPE融合、decode Graph、prefix与metadata复用。Graph使用ATOM已有捕获/重放机制，并检查真实replay记录，而非仅移除eager开关。

组合实验完成A512/B50/C200，A为1139.7781token/s/GPU，B TPOT p50为4.86523ms，C成功样本TPOT p99为11.50206ms且200/200完成。然而对应完整开发集仅62.85%位置一致率，不能作为符合全部门槛的主版本。

因此没有把这批组件整体迁入参考兼容配置。结论限定为该组合未通过，不能认定Graph、融合或整批投影各自必然错误；也不能将性能提升分别归给其中某一项。参考单次KV写入后来重新隔离验证并纳入②，说明被放弃的是未合格组合，而不是将所有相关技术一概否定。

实现与方案：`mellum_projection.py`、`mellum_fused_rope.py`、`mellum_fastpaths.py`、`mellum_prefill_metadata.py`，以及[Graph计划](../plan/37_optimization_sprint.md)。证据：[整批投影](../../report/performance/31_batched_projections.md)、[缓存组合](../../report/performance/34_batched_prefix.md)、[KV单次写入](../../report/performance/35_direct_kv.md)、[完整ABC](../../report/performance/36_combined_full_abc.md)。

## 5. 正确但无实质收益的逐元素合并

保持prefill每请求两次grouped_mm形状，合并SiLU、加权、逆排列和求和，decode仍走参考路径。开发200全部输出与控制相同，短/长输入各四次配对测量中，耗时分别变化−0.66%和+1.08%，几何收益0.99796倍，未达到预设1.05倍筛选目标。

停止该版本，而非继续增加验证次数。它没有减少GEMM次数，新增拼接和临时存储可能抵消收益，但本轮没有足够trace证据将无收益唯一归因于cat。该试验提供了“不改变主要执行成本时，局部操作合并未必有效”的实际依据。

证据：[完整记录](../../report/performance/18_prefill_elementwise.md)。

## 6. C调度与缓存：保留方向，淘汰无效配置

四轮decode配额下，chunk512/1024/2048的C60分别完成21/24/22，对照1/60；一轮配额及单独缓存未解决超时。公共前缀实际复用约35.47%输入token，不等于同比减少总时延。较小分段有利于缩短单次prefill，但也增加调度次数并延后准入。由此选择chunk1024/d4进入候选，而非继续无界扫参。

隔离decode、取消交错与最新profile又推动SWA group4、metadata once和断连回收接入。最终C200改善至48/200，仍152失败；完整组合开发200为75.9453%，长输入43/256。该方向没有被整体放弃，但仍在做子机制隔离，不能继承单项通过结论。保留失败与当前诊断，不将其包装成已经验证成功的第三项。

证据：[分段](../../report/performance/38_c_chunked_prefill_screen.md)、[缓存与配额](../../report/performance/40_c_prefix_and_quota_screen.md)、[SWA及取消](../../report/performance/43_swa_decode_groups_and_abort_interleaving.md)、[最终矩阵](../../artifacts/submission/final_r32_matrix/matrix.md)。

## 7. 仅调研或暂缓的方向

双卡P/D、FP8、MTP和SWA KV物理截断未进入完整实现与验收，不能列作已完成优化。P/D增加缓存传输与角色调度，还需面对按两卡放大的负载；FP8引入量化路径与误差预算；MTP需要额外资源且要求开关前后greedy一致；物理截断涉及分页缓存所有权和生命周期，不能等同于按窗口读取。

在候选人明确优先单卡、交付时间受限的条件下，集中完成已有方案的验证与收敛。暂缓依据是实现成本、验证成本和当时证据，不是未经测试就断言这些技术没有收益。
