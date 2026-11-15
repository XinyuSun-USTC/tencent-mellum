# 18 GPU1 prefill逐元素合并：输出保持，未获实质收益

运行`/workspace/results/mellum_prefill_optimization/20261115T170157802373Z`，独立worktree`/workspace/tencent-mellum-moe-prefill`、commit c22198d，GPU1/引擎8126，TP/DP1。4个worker全部完成，累计994.13秒（16.57分钟），未触及预算。结论：**停止当前prefill_shared_elementwise候选，不切默认、不追加ABC/HumanEval或微调循环。**

候选保留每请求专家sort/offset和两次grouped_mm的形状，集中SiLU、乘法、加权、逆排列、求和；decode保持R。数学实现位于独立分支`step1/moe-prefill`的`ATOM/atom/model_config/mellum_moe_optimization.py:shared_prefill_elementwise`；计划19亦在该分支，尚未合并主实验分支。

## 正确性与实际路径

R控制S32/L32各32条输出全部复现历史R。候选D200为200/200与R逐请求输出完全相同；与HF位置12392/12800=96.8125%、完整序列174/200=87%，长输入位置93.457%、短输入97.452%。这不是候选与HF200/200相同，也不是完整HumanEval功能分数。

layer0审计确认候选prefill确实走requestwise_gemm_shared_elementwise：screen16次、D20012次；decode仍requestwise_gemm_shared_route：screen128次、D200448次。实际batch核验通过，非静默回退后计时。完整加载5631/5631。

## R与候选配对ABBAABBA

无活动profiler、无逐步MoE audit；同worker预热，输出16token，短/长各4次，中位数：

| 输入组 | R秒 | 候选秒 | R/候选 | 候选耗时变化 |
| --- | --- | --- | --- | --- |
| S32 | 12.3795 | 12.2976 | 1.00666× | -0.66% |
| L32 | 24.0749 | 24.3343 | 0.98934× | +1.08% |

几何均值0.99796×，未达到预设1.05×收益筛选。不是测得显著回退的统计声明；结果仅说明这一批配对测试没有值得保留的收益。实际各组重复输出稳定。

GPU0 ABC与本批时间有少量交叠，主机共享资源可能影响速度，未独立测其因果贡献。但本轮无实质收益，不为当前候选追加空闲复测；最终胜出方案才安排正式空闲环境性能复测。

## 下一方向

本轮未减少专家GEMM次数，新增cat/临时存储也有成本。结果与此前prefill专家mm大量发射的profile相符，但没有新trace可量化具体抵消比例，不断言cat独自造成全部损失。

结合性能17：A仅官方吞吐6.51%，C199超时；后续优先研究专家GEMM/BMM执行和decode选权重的实质代价，使用接近256输出的有界负载衡量prefill/decode比例，避免只靠16token总wall选择最终方向。已有indexed版本D200失败，不原样重跑；替代方案需明确数学差别和生成门槛，有界筛选后再进完整ABC。停止外围逐元素小优化，不增加正确性精确度门槛，不重复现有完整基线。

独立核验90个产物和237个源码指纹全部一致；[轻量证据](../../artifacts/performance/18_prefill_elementwise/summary.json)保留分组、计时、原文件hash和执行结论。原results未改，本轮未启动新GPU任务或修改推理代码。
