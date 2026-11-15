# 22 原生候选的有界拆解：固定MoE回放与Attention分阶段消融

依据[性能19](/workspace/tencent-mellum/report/performance/19_native_runtime_pilots.md)：两条主路径约1.45倍提速但D200约72%/71%，MoE重复生成不同。下一步只解决明确问题，不新增优化路线，不要求中间张量等同HF，不启动完整ABC/HumanEval或直接组合候选。

## GPU0：固定输入/路由MoE回放

```bash
/opt/venv/bin/python /workspace/tencent-mellum-native-diagnostics/scripts/run_mellum_native_diagnostics.py --track moe
```

预计数分钟（首次JIT可能更久），累计预算15分钟、单worker10分钟、最多2次启动。只加载layer0真实专家权重，canonical与shuffled副本分开；输入为已保存HF layer0真实向量，decode1/32用末行构造明确标注的decode形状回放，另有ragged150行prefill形状。不声称这些是32条完整自然decode轨迹。

路由一次计算并固定，hidden/weights/ids在12次调用前后严格检查未被修改；每次调用后同步，记录相对首轮的精确相等/最大差、有限值，以及相对Torch grouped_mm/bmm公式的数值距离。这里的精确比较只用于判定重复性，不是候选必须匹配HF内部数值的验收要求。

两个独立worker分别使用冻结AITER预置Opus和CK排序（AITER_USE_CK_MOE_SORTING=0/1）。此为单一排序变量对照，不修改库文件、不引入新内核、不改变生产配置。记录实际get_2stage_cfgs返回的run_1stage、block_m、ksplit、stage1/stage2 callable/参数；观察器只透传原函数结果，结束恢复。排序记录只取num_valid_ids[0]以内已初始化token-id前缀，不比较未初始化尾部。每worker另取一次decode32实际GPU trace，和重复检查分开，不用其耗时作性能分数。

保存样例输入、固定路由、首输出和Torch参考，供下一轮有依据地分析。`diagnostic_complete`只表示两项回放完成，不表示完整模型正确性通过。如果固定回放稳定，也不能排除其它层/真实decode输入下的差异；排序hash变化也不是内核bug的充分证据。

**本入口不自动修模型、不再跑全模型MoE200、不开启HumanEval/ABC。** 看结果后只在有明确依据时选一次修正/后端再正常生成检验。

## GPU1：Attention只改一个阶段

```bash
/opt/venv/bin/python /workspace/tencent-mellum-attention-diagnostics/scripts/run_mellum_native_diagnostics.py --track attention
```

两份工作区固定相同源码，实验期间不修改源码；GPU1引擎端口8426，GPU0层级回放不启引擎/端口。GPU1预计8–15分钟完成两项生成；有通过项才测256性能，可能总计20–35分钟。累计上限45分钟、每worker15分钟、最多5次启动。

1. `MELLUM_ATTENTION_EXECUTION=native_prefill`：原生varlen prefill，reference_ragged_decode。保留参考RoPE/KV修正、R专家，不用旧SDPA dispatch wrapper代替显式分阶段安装。
2. `native_decode`：reference_ragged_prefill，原生paged decode；保留参考prefill所需raw QK、CPU频率和mask/GQA行为，不在这阶段改变prefill。
3. 两项各跑固定D200、正常greedy64；核验200请求完成、真实decode32、native/reference两半均有实际路径证据，全部分组和整序列报告。总体token位置>=90%作为本地晋级条件，不添加相对R下降限制。
4. 无通过项即结束。有通过项才各做S32/L32、固定256输出/ignore_eos、预热后一次计时，同时新跑R相同负载一次作对照。最多2个D200+1个R性能+2个候选性能。不重跑失败项的256性能；这是探索性单次计时，不是ABBA或正式ABC。

两阶段在`mellum_native.py:install_native_attention(mode)`显式选择：prefill_only/decode_only/both。预期的reference半阶段不是“静默回退”；日志同时核对该阶段确实为reference，防止两个阶段仍一起改。默认reference和已有native=both行为保留。

## 产物、恢复和停止

输出根分别`/workspace/results/mellum_native_moe_diagnosis/<UTC>`、`mellum_native_attention_diagnosis/<UTC>`，回传各自RESULT与目录。失败日志、输入/源码hash、完成阶段产物hash和累计预算保留。`--resume <目录>`要求同工作区/track/源码/输入，已完成job跳过，最多启动次数和预算不重置，失败尝试归档。

本轮没有GPU实测权限：工具无/dev/kfd、/dev/dri，只有CPU接线与调度检查。CPU测试覆盖模式的两个相反阶段、路由/窗口现有回归、通过/失败晋级、恢复、只做两项MoE回放、metadata描述和native/reference证据。最终验收仍为正常生成、16k、HumanEval与题目性能，不由局部误差替代。

交付核验：120项相关CPU测试通过，py_compile/diff检查通过；MoE prepare-only `20261115T195403222539Z`、Attention prepare-only `20261115T195402782769Z`通过。实现基于53b6833，排序前缀观察补丁分别1cac5e2/c300a25，两份源码树相同；GPU尚未运行。
