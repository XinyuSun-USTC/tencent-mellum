# 23 MoE专家贡献独立输出与显式归约候选

依据[性能20](/workspace/tencent-mellum/report/performance/20_native_phase_diagnostics.md)：固定输入/路由/排序下CK two-stage仍存在重复差异，trace和冻结CK源码确认输出AtomicAdd归并。只推进一个有依据的修正，不重新展开排序/Attention矩阵。

## 实现与语义

独立worktree `/workspace/tencent-mellum-moe-isolated`，分支`step1/moe-isolated-reduction`；原R、原native和诊断工作区不改。默认关闭，新开关`MELLUM_MOE_BACKEND=fused_isolated`。

`mellum_native.py:isolated_fused_experts`：

1. 保留原gate、FP32 softmax/top8/renormalize、BF16路由权重舍入（FP32格式传输）。
2. 每个token的8条专家分配展开成8个pseudo-token，每个pseudo-token仅一个预先选定的专家。hidden按路由顺序repeat，专家ID reshape为[N*8,1]，native路由权重全为1。
3. 使用冻结AITER `fused_moe`，复用既有FusedMoE参数/loader/after-loading shuffle，不改库、不重复shuffle、不引入量化。输出为[N*8,H]，每行对应一个专家贡献。
4. reshape回[N,8,H]，在外部执行BF16路由加权和PyTorch top-k求和。对应R的加权/求和表达式，不把8专家归并留给native输出原子加。

`MellumIsolatedMoeBlock`复用原Fused模块的参数与gate，direct AITER compute绕过其原top8合并。模型的逻辑top-k仍是8，不是将模型改成top1。层0审计同时记录logical_topk=8、native_topk=1、expanded_rows=N*8、shuffle标记。

**不声称消灭所有atomic指令**：native内部仍可有AtomicAdd或split-K，但每个pseudo-token只对应一个逻辑专家，移除的是不同专家向同一token输出竞争累加。展开后M/topk和内核选路会变化，GPU回放必须记录实际stage metadata。激活和GEMM舍入仍可与HF不同，即使重复稳定也不保证生成达标。

## 代价与边界

需要重复hidden、保存专家独立输出与显式weighted张量，临时内存随N*8增长；不复制专家权重。当前max_num_batched_tokens16384、BF16/H2304意味着仅expanded hidden或独立contribution各约576MiB，另有激活和weighted临时张量，不能称作零额外显存。是否有净性能收益由256输出测试判断。

本轮Attention、L控制、RoPE、BF16 KV和scheduler维持R；不启用prefix/chunk/Graph、FP8/MTP或双卡推理。取消E逐请求专家控制，保留L。

## 运行

```bash
/opt/venv/bin/python /workspace/tencent-mellum-moe-isolated/scripts/run_mellum_isolated_moe.py
```

固定GPU0、独立引擎端口8526。失败早停时预计数秒至约5分钟；若进入全部性能任务，预计15–30分钟。累计上限45分钟、每worker15分钟、最多4次启动，每15秒心跳。

自动顺序：

1. **局部回放**：同前轮layer0真实权重与输入，decode1/32为显式标注的末行形状回放，prefill150；固定路由各12次、记录metadata及短真实trace。仅finite且三组重复精确才继续，未对HF张量距离设验收阈值。此门槛检查本候选试图消除的重复变化，不是增加“必须等同HF”的门槛。
2. **D200**：固定HF缓存、正常greedy64、真实batch32、加载5631审计、独立贡献路径审计；token总体位置>=90%才继续，完整序列/长短/来源全部保留。不加相对R下降限制，不缓存输出/强制HF历史。
3. **性能R/候选**：同一套S1/S32/L32原token输入、固定256输出/ignore_eos、预热后各一次。batch几何均值>=1.10×且每组>=0.95×为探索性收益筛选。独立进程单次测量，不是ABBA、正式ABC或统计显著性结论。

输出根`/workspace/results/mellum_isolated_moe/<UTC>`；回传最后RESULT和目录。`--resume <目录>`需同源码/输入/工作区，已完成阶段产物hash校验后跳过，累计预算/最多4次启动不重置。

## 停止条件与交付检查

- 固定回放仍不稳定或出现非有限值：停止，不自动试其他实现。
- D200低于90%：停止，不跑性能/完整HumanEval/ABC，不因为数值差异减小而放行。
- 通过生成但无实质速度收益：如实保留正确性和开销证据，不切默认。
- 通过者仍须后续组合/16k/完整HumanEval及最终空闲环境ABC；本批次不自动完成这些长任务。

CPU检查126项通过（含路由展开/单专家unit权重/外部加权求和、dtype、实际路径证据、回放→生成→性能门槛及恢复），py_compile和diff检查通过。prepare-only目录`/workspace/results/mellum_isolated_moe/20261115T202806058832Z`。工具无GPU设备，GPU实测待用户终端执行，尚无本候选稳定性/生成/速度结论。
