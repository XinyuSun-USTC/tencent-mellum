# 52 参考专家decode小批量分组

## 动机

参考r32正确性已通过，但完整C200仅1成功、199条180秒超时。其A吞吐41.3351 token/s，低于C到达负载每秒76.8输出token的工作量，且C输入更长。降低服务槽数可能改变延迟/吞吐取舍，不能直接解决容量不足。需要减少参考执行组织中的重复调用，而不是重新运行相同完整ABC。

## 实现

独立工作树`/workspace/tencent-mellum-qualified-expert-groups`，分支`step1/qualified-expert-groups`。仅深化已有参考MoE执行组织，不新增原生专家或改变数学公式。

新增`r_expert_decode2/4`：基于r，仅向campaign配置加入`decode_expert_group_size=2/4`。主路径保留：线性投影L逐请求，prefill专家逐请求，参考Norm/SDPA/RoPE、共享路由、eager、32槽、无prefix；direct KV关闭。仅在每个真实请求一个decode token时，将相邻2/4个请求交给同一次`MellumTorchExperts.forward_routed`。保留SiLU、乘法、BF16阶段和Top-k归约；底层BMM批量维变化仍可能产生数值差异，必须实际验收。

`decode_expert_bounds`保留原token顺序和尾部不足整组的请求，拒绝MTP多token和非法组大小。真实请求边界仍用于其他模块与调度，不改变线性投影分段。layer0审计同时记录真实请求数、专家调用组数、实际组大小；prefill必须保持组大小1。

## 筛选

1. 每个候选完整运行developer200和长组，保留16k单列，复用已校验HF参考。
2. 核验完整加载、实际32请求与分组执行、参考专家/注意力/Norm。
3. 未达到200位置90%的候选不继续性能。
4. 通过候选执行R1→候选各一次→逆序候选各一次→R2的缩减性能对照，A32/B4，输出32token；不是完整ABC，不声明C容量修复。
5. 只有有明确收益的候选才进入完整HumanEval及C负载验证；失败保留记录，不改变180秒、0.3req/s或256输出的正式设置。

运行（仅在目标GPU空闲时）：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-qualified-expert-groups/scripts/screen_mellum_expert_groups.py --gpu 0 --groups 2 4 --with-perf --budget-minutes 60
```

结果位于`/workspace/results/mellum_expert_group_screen/<运行号>/`。该批是C容量修复的候选筛选；`screen_passed_full_acceptance_pending`只代表200检查与短性能流程完成，不能代替最终正确性和C200。
