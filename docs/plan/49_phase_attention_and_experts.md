# 49 两卡并行：Attention与专家的prefill/decode阶段定位

最新四格诊断20261116T144338670827Z已完整结束，52项产物哈希一致，实际28个MoE类/prefill函数与5631加载通过。固定20条结果：原生/原生69.84375%，原生Attention/参考专家71.171875%，参考Attention/原生专家77.03125%，参考/参考100%（20/20序列）。结合参考R完整200/长组100%，两条原生管线都需要更细定位，不能把差距只归给Norm或单一整块模块。

## 新批次（原20条，不扩充验收）

独立/workspace/tencent-mellum-attn-phase，基于29063ee；不改此前GPU作业源码，不升级依赖/改已有装饰模型。

GPU1，参考专家固定，只交换原生Attention的阶段：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-attn-phase/scripts/run_mellum_core_diagnosis.py --gpu 1 --profiles phase_prefill phase_decode --budget-minutes 15
```

- phase_prefill：原生prefill，参考SDPA decode。
- phase_decode：参考SDPA prefill，原生paged decode。

GPU0，参考Attention固定，只交换原生专家的阶段：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-attn-phase/scripts/run_mellum_core_diagnosis.py --gpu 0 --profiles expert_prefill expert_decode --budget-minutes 15
```

- expert_prefill：AITER专家只用于prefill，decode用HF语义Torch专家。
- expert_decode：prefill用HF语义Torch专家，AITER专家只用于decode。

两卡分别engine9926/9826、batch1/eager、参考Norm、分离QKV、Torch路由、独立QK Norm，prefix关闭。只20条×64输出，约每卡3–6分钟，预算各15分钟，CPU准备另计。不是正式200通过，也不是性能成绩；不跑HF、HE或完整ABC。

## 专家阶段诊断实现与限制

MellumPhaseMoeBlock保留原FusedMoE注册权重及框架原有加载后shuffle，另持有不注册为checkpoint目标的canonical Torch参考副本。双loader从同一原始checkpoint shard分别加载两种布局，不能拿shuffle后的权重冒充canonical。每层必须收到192个来源分片后才允许forward；注册权重仍5631源张量，阶段backend及canonical加载数有实际回执。

这会额外占约一份专家权重的显存，**只为短诊断，不是交付优化或额外模型资源**；仍在同一卡，不允许Graph或非BF16。默认分支不启用此模块。CPU测试验证双布局加载、注册键不重复、阶段分派；GPU真实数值/加载与内存仍待实际执行，失败保留，不静默fallback。

Attention阶段复用已有install_native_attention(prefill_only/decode_only)；除构造函数名外还核验decode实际调用参考ragged或paged后端。专家两阶段必须分别看到FusedMoE/Torch实际事件，不能只根据profile名认定。

53项相关CPU检查通过，实际20条准备、py_compile/diff通过；无新GPU结论。原候选性能/正确性证据不更改。

## 决策顺序

先找能安全保留的阶段；对诊断结果最好的组合补原200，不直接把小集通过算验收。若都未到高一致率，再比较参考/参考的共同外围或kernel中间BF16边界；不再盲目调prefix/packed或全面回滚。快速HF Norm后备条件未满足，继续暂停。
