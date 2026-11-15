# 19 GPU1并行MoE prefill候选：共享逐元素操作

## 目标和工作区

GPU0继续计划18的R配置A/B/C，原目录`/workspace/tencent-mellum`不改。本轮独立worktree为`/workspace/tencent-mellum-moe-prefill`，分支`step1/moe-prefill`，从`2d996ba`建立。共享Git历史，未合并到原分支；不修改冻结/app、/opt依赖，不启用双卡推理。

仅使用物理GPU1，TP/DP1，独立引擎端口8126，无HTTP服务。原ABC用8051/8026。子进程PYTHONPATH显式指向本worktree的ATOM，HF参考和历史R输出只读复用，不重新生成。当前工具无/dev/kfd且禁止socket，不能执行GPU验证，交付用户终端一次启动。

## 源码和profile依据

- `ATOM/atom/model_config/mellum_experts.py:MellumTorchExperts.forward_routed`：prefill每请求分别sort、histc/cumsum、两次grouped_linear、SiLU/乘法、逆排列及top-k求和。
- `ATOM/atom/model_config/mellum_moe_optimization.py:optimized_experts`：R仅合并route，仍逐请求调用上述专家计算。
- 性能14真实trace：R专家约占累计kernel时间75.98%，prefill专家82.56%；prefill专家mm发射96582次。累计kernel时间不是端到端占比。
- 性能13显示请求shape联合控制有正确性收益，贸然合并请求的GEMM存在已知风险。本轮不重复indexed_decode失败方案，不预设AITER或换GEMM精度安全。

## 唯一候选及取舍

开关`moe_optimization=prefill_shared_elementwise`，沿用groups=LE、phases=prefill/decode。

`shared_prefill_elementwise`保留每请求专家排序、histc/offsets和两次grouped_mm的输入行数、权重、组内顺序；先执行所有请求gate/up GEMM，拼接后统一SiLU和乘法，再以原请求边界切片执行down GEMM，集中加权、逆排列和top-k求和。保留BF16激活/乘法的分开舍入，不融合成FP32计算。decode保持原R的按请求BMM。

直接减少激活、加权、逆排列和输出求和等重复小操作；sort/histc与专家GEMM数量没有减少，因此**不能宣称已解决prefill主要GEMM瓶颈**。新增cat有开销，峰值临时存储增加；是否盈利由配对计时判断，不预设5%收益。额外存储按当前scheduled tokens增长，实验max_num_batched_tokens仍16384，不按所有请求最大上下文同时展开。

合并逐元素张量形状也可能改变实现：CPU BF16小样例精确，FP32出现约1e-9差异；测试中FP32容差只用于CPU公式检查，不是题目/GPU验收阈值。不承诺GPU bitwise一致。

## 单次有界实验

```bash
/opt/venv/bin/python /workspace/tencent-mellum-moe-prefill/scripts/run_mellum_prefill_optimization.py
```

默认gpu=1且拒绝gpu=0；不依赖终端cd。预计15–40分钟，累计预算最多60分钟、每worker最多15分钟、最多4个worker，每15秒心跳。运行目录`/workspace/results/mellum_prefill_optimization/<UTC>/`。

1. R控制：K4/S32/L32，正常greedy16（K4原预算），各两次；真实batch32、重复稳定，S32/L32与已归档R逐请求完全相同。基线复现失败即停止。
2. 唯一候选同样筛选。S32/L32各位置>=90%、K4位置>=90%、重复稳定；审计layer0真实多请求prefill执行候选、decode仍为R。未通过即结束，不展开数值修复矩阵。
3. 通过后仅候选D200，正常greedy64、200完整记录。输入/参考/历史R hash固定；报告总体位置、完整序列、长短及来源分组，以及与R逐条相同数。总体位置>=90%才进入计时，不添加相对R下降2pp筛选。
4. 通过后同worker、相同输入，R与候选ABBAABBA，S32/L32各4次、输出16token、预热、关闭MoE逐步审计和profiler。几何均值>=1.05、每组>=0.95作为本轮收益筛选；不是题目附加正确性门槛。

不启动HTTP、完整HumanEval、ABC或大trace；不重测CPD/B0/indexed或CK。参考R复现和D200足以决定这一小改动是否值得继续，失败则保留证据并结束候选。

## 恢复和最终验收边界

`--resume <本批次目录>`：完成job按产物hash跳过，失败尝试归档；累计预算保留、max4启动次数不重置。源码/input/config变更拒绝恢复。工作树source指纹含新增入口和全部ATOM数学源码。共用底层Runner保留自有进程组超时清理、失败日志和阶段记录。

两卡并行仍共享主机资源，本轮速度仅用于候选筛选；最终计分数据需空闲环境重测。候选通过D200/ABBA也不等于最终HumanEval、16k或组合正确性通过。最终配置固定后再集中做这些验收，默认后端本轮不切换。

交付核验：110项相关CPU回归通过，py_compile及git diff --check通过；prepare-only目录`/workspace/results/mellum_prefill_optimization/20261115T165902806916Z`，实际HF/历史R产物与输入预算checksum通过。无本轮GPU正确性或速度结论。
