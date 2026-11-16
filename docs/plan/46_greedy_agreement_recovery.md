# 46 有界恢复greedy200一致率，保留性能成果

已知最终候选本地开发200位置62.85%，低于90%，不能以HumanEval更高抵消。数据仍未确认等同官方公开集，但当前不足以证明通过。停止继续扩张优化；本批只测原200及长边界组，复用HF，不跑HE、完整ABC或逐层tensor dump。

独立/workspace/tencent-mellum-agreement，基于25350e6；已测候选和原始结果不改。默认生产开关不变。新MELLUM_HF_QK_ROUNDING=1让Triton QK归一化先cast BF16、再乘权重并cast BF16，匹配HF MellumRMSNorm的阶段边界；不新增kernel，不关闭Graph/融合，未测前不认定根因。

## GPU0：Norm为主的累加修复

```bash
/opt/venv/bin/python /workspace/tencent-mellum-agreement/scripts/run_mellum_greedy_recovery.py --track norm --budget-minutes 35
```

1. 原最终候选32并发基线重测。
2. 仅融合QK采用HF舍入边界。
3. 在2上把剩余Norm恢复参考实现（仍Graph，仍融合QK），区分其它Norm/残差舍入影响。
4. 在3上切回Torch自定义路由，观察组合贡献。

这是有明确依赖的累加诊断，不冒称每一行仅一个独立优化点；若3有明显改善再考虑用同样HF边界实现高性能Norm，而非永久依赖拆分Torch计算。

## GPU1：系统因素与控制组

```bash
/opt/venv/bin/python /workspace/tencent-mellum-agreement/scripts/run_mellum_greedy_recovery.py --track systems --budget-minutes 45
```

顺序原候选batch32、仅关闭prefix、仅Torch路由、仅分离QKV、原候选batch1；若以上无任何200位置≥90%，再运行R参考数学控制组batch1。R不是历史Step0、不用于正式性能行，只检查验证管道与已知参考路径能否恢复高一致率。较慢控制组单独上限20分钟，独立失败保留，其余阶段15分钟上限，执行受总预算约束。每项独立加载引擎，防止跨候选KV污染；每项内部200后跑原长组。

单batch仅用于定位batch/GEMM形状贡献，不把单流速度代替原并发32性能。整个脚本只记录位置率、整序列数、长组与真实runtime，不从诊断耗时捏造ABC分数。worker明确仅tokens/long，CPU测试防止误跑HumanEval。

## 检查与决策门槛

46项相关CPU检查通过；Triton仅RoPE、原Norm舍入、HF Norm舍入三种constexpr专门化均在gfx942离线编译成功；CPU真实参考准备通过。GPU侧要求5631加载、实际Graph/路由/融合回执和113 Norm后端符合profile。所有源码/参考/输入/批大小与suite配置有hash，失败继续。

没有改变90%、HF参考、prompt、输出预算，也没有要求模型中间张量逐位相等。只做原题相关输出指标的A/B。不自动重跑HF/HE。预计Norm线10–20分钟，systems线15–35分钟（取决于是否跑慢控制），执行预算35/45分钟，CPU准备另计。

- 一旦有batch32的≥90%候选，优先只对最有价值者短测A64/C20，评估正确率/速度权衡，再做一次统一验收；原来的性能结果不直接继承给改动后的模型。
- 若只有batch1恢复，优先修复批形状相关差异，保留单卡推理，不用缩小评测并发冒充优化。
- 若所有快速分支都低于90%、参考控制恢复，下一步再聚焦专家或Attention具体数值路径。不到这个条件，不全面回滚或重开逐层精确排查。
- 若参考控制也异常，先核对输入、模板、采样与运行配置，不能继续盲改内核。

所有新结果都是诊断，未宣称已经解决62.85%差距；HE高分与绝对差口径单独保留。
