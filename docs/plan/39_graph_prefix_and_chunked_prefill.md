# 39 利用GPU等待时间开发Graph前缀缓存与调度实验

独立/workspace/tencent-mellum-scheduling，分支step1/graph-prefix-scheduling，基于2dc21b2；正在运行的fusion工作区完全不改。只做源码、CPU测试及CPU输入准备，不使用任何GPU。

## 优先级与兼容分析

当前Graph已明显减少decode成本，下一步针对场景C长prefill干扰：先测原生前缀缓存，再分别试2048/4096 token的chunked prefill预算。不是更改评测并发、输入/输出长度或到达率，而是内部调度参数；client保持A64/并发32、C20/0.3req/s/256输出。

直接打开开关不可行：mellum_direct_kv拒绝has_cached，native_prefill拒绝Q/K不同长度。复用c34a96a/e4cfb91已开发的native_prefix_geometry/audit_native_prefix，但保留新的单次KV写入，不能退回旧路径的二次旋转/覆盖写入。

实现：先按absolute positions旋转并写入新K/V，再收集本批完整逻辑K/V；SHUFFLE复用ATOM _gather_prefix_and_concat_kv，NHD使用已有gather_reference_kv逐请求索引，避免旧框架路径将整个KV池permute.contiguous。得到矩形Q/K后，继续原生varlen bottom-right causal与SWA左界1023。decode保持现有Graph，缓存gather仅在真实prefill+has_cached分支执行。三种新增profile均默认关闭，单次写入、Norm、Packed、Torch router与参考RoPE不变；不提前把未返回的路由/融合候选当赢家。

## 计划批次（现在不运行GPU）

- prefix track/GPU0：sprint_graph→sprint_graph_prefix；观察真实复用token及C20延迟，同时检查A64无明显退化。
- chunk track/GPU1：sprint_graph→sprint_graph_chunk2048→sprint_graph_chunk4096；prefix关闭，只改变内部chunk预算。记录真实cached suffix事件，不把后续chunk的历史token误称跨请求prefix命中。
- 每候选A64/C20，独立服务/相同输入；失败保留继续。预算建议45/60分钟，CPU准备另计。结果/workspace/results/mellum_scheduling_sprint。
- 启动脚本自动设置enable_prefix_caching或enable_chunked_prefill与max_num_batched_tokens；Graph保持1/2/4/8/16/32，level0，TP/DP1。没有静默fallback。
- C20必须有实际cached suffix事件；这同时证明计算走兼容路径，但chunk后缀不能充作跨请求复用率。最终比较需从引擎日志独立核对prefix命中。

## CPU验证与待解决风险

40项相关CPU测试通过：非连续物理块NHD逻辑顺序、写入新K/V后再gather、矩形absolute位置/边界拒绝、native kernel的Q/K长度与窗口参数、原路径回归、启动配置隔离。首次测试发现局部cached变量与驻留tensor同名，已在本分支改为cached_suffix并回归通过；未触及运行工作区。

尚无GPU结论。SHUFFLE现有gather含repeat_interleave，NHD逐请求gather有发射/同步成本，chunk拆得太小可能增加总prefill开销；混合prefill/decode与最后小chunk仍需真实调度检验。不能仅依据开关存在宣称兼容或收益。先复用已测底层算子，不新增自定义分页gather内核扩大变量。

## 结果返回后的安排

先分析当前fusion两条线，若有显著赢家再把相应开关接到本分支；如无需改代码即可使用Graph+Torch基线，则保持现有候选。空闲卡才启动本批，绝不与同卡现有作业争抢。仍不逐组件跑HF；集中筛选后固定最快组合，再统一题目正确性、完整ABC与正式消融。若chunk路线上调度/元数据不兼容，保留失败并优先完成prefix，不开展无界重构。
