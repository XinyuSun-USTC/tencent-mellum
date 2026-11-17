# 最佳单卡组合的受控瓶颈定位

## 背景与目标

386d124的C60最佳组合完成43/60，仍17失败，不能晋级。下一步固定 `r_swa4_abort_prefix_chunk1024_d4`，收集短trace，调查全局attention、SWA、专家、投影及主机间隙。优先用测量决定实现方向，不继续大规模参数盲扫。

## 实现

独立工作树 `/workspace/tencent-mellum-c-bottleneck-profile`；默认关闭的profiling新增requestwise_decode与grouped_swa范围，原数学路径不变。复用已确认的worker profiler启停回执，必须拿到GPU kernel和Mellum模块标记才接受trace。每次保存源码/输入哈希并检查运行后未变化。

GPU0：前16条真实C输入同时入队，各生成16token。GPU1：先4条输入，在实际engine审计确认至少4轮decode后插入另外4条长输入，各16token。混合组四个pass各用独立8条提示，避免整条提示重复命中掩盖prefill；共享前缀仍保留。两组均先完整预热，再两次未启用profiler的测量，最后一次trace，单组上限20分钟。预期约3–10分钟，但首次trace导出可能更久。

这是受控burst，不是泊松C，不报告官方性能或正确性通过。decode组所有pass共用同批提示的暖前缀，侧重稳定decode成本。mixed组使用不重复提示，允许真实共享前缀命中，实际处理的新token数以批次审计为准；受控插入仍不能外推正式泊松干扰。无profiler测量仍有阶段审计与标签hook，不是无插桩正式benchmark。必要时根据trace发现追加针对性冷输入实验，不能直接外推。

## 判读与后续实现候选

先确认实际prefill/decode记录、输出数、模块归因和启停回执。CPU范围含异步等待，不等于GPU耗时；重叠模块范围不可相加。分析GPU kernel归因及时间线间隙：

- 全局attention/KV gather主导：优先批量读取、保留单请求SDPA的方案；改变SDPA形状的计算分组另行筛查。
- SWA gather/SDPA主导：调查重复KV扩展与索引，不把此前CK误差候选直接当合格实现。
- 专家/投影主导：回到已通过的group4与逐请求投影，找减少调用开销的单项实现。
- 主机间隙主导：结合engine审计调查同步和调度，不能只看kernel总和。

profile只作定位。新优化先短数值检查与同负载性能对照，再决定C60；C60通过后C200，最后最终配置完整正确性、ABC与消融。未完成数据留空。
