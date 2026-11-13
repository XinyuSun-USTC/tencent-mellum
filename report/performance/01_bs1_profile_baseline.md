# 01 首轮bs1性能Profile：基线与优化方向

## 结论

首轮采集有效，模型为MI308X/gfx942、冻结Torch2.10 ROCm/Transformers5.2，bs1、BF16、eager，单次加载后完成1024/4096/16384三组。当前证据支持**Attention优先于MoE作为第一个实现候选**，尤其针对长输入prefill；Graph仍是候选但不根据profile空隙直接认定CPU瓶颈。场景A并发32未测，不能把bs1排序直接外推到最终吞吐得分。

本报告不是优化后成绩，尚未实现新的加速kernel。原始结果`/workspace/results/mellum_profile/20261113T180921043675Z/`，采集commit `c557aee0a75564e7a264483c6da3024c4a76c8bf`，当时工作树干净。配置/范围见[08实验协议](../../docs/plan/08_profiling_experiment.md)，可读轻量产物见[归档](../../artifacts/performance/01_bs1_profile/manifest.json)。

## 采集有效性及解析修正

三份trace均有实际GPU kernel、CPU prefill/decode与Mellum标签，文件SHA256已与worker记录核对，未重新运行GPU。各组约5.4万个kernel，模块归因覆盖98.89%、99.43%、99.86%；加载审计5631/5631通过，引擎正常关闭。实际GPU可见设备名AMD Instinct MI308X，arch=gfx942:sramecc+:xnack-。

首次读取时发现ROCm Kineto同时输出`user_annotation`与`gpu_user_annotation`，名字与External id相同。初版解析器把后者视为CPU范围，造成区间次数/时长翻倍，并可能覆盖CPU owner。现修正为GPU镜像不参与CPU区间及launch归因，新增对应合成回归；未改变原始trace、worker结果或原分析JSON。

修正派生结果位于`/workspace/results/mellum_profile_analysis/20261113T181952529842Z/`，manifest记录原trace与新parser SHA256。报告使用修正结果，不使用原summary中6664个模块标签或2次prefill/32次decode计数。实际每组3332个CPU模块标签、1次prefill和16次decode。trace请求实际返回16token，捕获17次forward，包含异步末尾额外一步；不能假装只有15次decode，或将其总时间按15除。此现象与此前引擎异步末尾前向观察一致，不以此判定生成错误。

## 无profiler计时

每组8token预热，两次无profiler64token普通greedy、ignore_eos=True；下表为两次算术平均，不是p99，也不是题目B（要求256输出/50请求）成绩。profile module hook在这些计时请求时未安装。

| 输入token | TTFT ms | TPOT ms | 整请求wall s |
| --- | --- | --- | --- |
| 1024 | 217.094 | 56.303 | 3.7644 |
| 4096 | 662.965 | 56.144 | 4.2002 |
| 16384 | 3994.279 | 60.865 | 7.8290 |

两次样例数很少，只能说明本轮重复接近，不能估计尾延迟或稳定统计分布。16k首token耗时明显上升，后续单token成本变化较小。

## 分阶段GPU kernel时间

以下百分比以各phase累计kernel duration为分母，不是端到端wall占比。Attention为最内层模块归因中的attention_swa+attention_full，已扣除单独标记的QKV/O投影；仍包括内部Q/K norm、旋转、cache整理等工作。MoE为专家执行，不含独立gate投影。

| 输入 | Prefill kernel累计ms | Prefill Attention / MoE | Decode kernel累计ms（16次） | Decode Attention / MoE |
| --- | --- | --- | --- | --- |
| 1024 | 127.54 | 25.9% / 58.9% | 361.22 | 51.4% / 27.0% |
| 4096 | 581.42 | 58.5% / 30.7% | 392.52 | 55.2% / 24.9% |
| 16384 | 3899.41 | 78.6% / 15.3% | 515.70 | 65.7% / 19.1% |

16k prefill细分：21层SWA路径71.1%，7层Full路径7.5%，专家15.3%，QKV4.1%。1024 prefill则专家占主导，说明不能以单一长度代表所有负载。

16k的`attn_fwd`累计约2.958秒，为整体trace kernel累计时间的大头；同一kernel名可能出现在多个层/phase，应结合模块/phase归因，而不是按名字猜算法。`aiter::unified_attention_with_output_base`是PagedAttention外层custom-op容器，内部仍能执行我们的参考SDPA；其出现在CPU trace不意味着已经启用了某个优化Attention后端。

当前SWA语义通过稠密mask实现，不保证底层kernel按window数量而非全长度节省计算。从源码结构与长度扩展耗时推断，专用窗口prefill有较强优化动机；其实际复杂度和硬件效率仍需候选内核实测，不能承诺16倍加速。

## CPU与同步：有线索，不夸大结论

三组kernel active union分别约0.490/0.975/4.416秒，kernel span约1.537/1.946/5.290秒，中间空隙约1.047/0.971/0.874秒。它们来自开启profiler的时间线，不能将空隙直接命名CPU耗时；也不能从无profile TPOT减去profile kernel平均值推算“可由Graph消除”的时间。

每组约4480次`aten::item`，其CPU累计约20ms；实际DtoH拷贝事件112次，不能把全部item都当GPU同步。CPU热点`_grouped_mm`、to/copy的长区间可能包含等待先前排队GPU工作的时间，不能将其直接解释为MoE kernel耗时。CPU调用时间与GPU时间也不能相加。

每次捕获decode forward的kernel累计约22.6/24.5/32.2ms，而profile CPU decode range约82–84ms/次；无profile实际TPOT仅56–61ms。Profiler/label开销和异步调度显著影响CPU时间线。因此Graph值得后续评估，但本轮不足以排在Attention前或承诺收益。

## 优化顺序更新

1. **Attention为第一候选**：先针对长prefill的滑窗执行和decode分页整理做自包含替换；保留已验证RMSNorm、Torch router、参考RoPE来源与严格greedy，不一次恢复所有旧融合开关。先以已有AITER/varlen/paged接口做受控可行性验证，再考虑Triton布局替代。
2. **MoE仍为第二主候选**：1024 prefill占59%，decode约19–27%，并发32时权重复制/专家复用可能更重要；需要可测batch基线确认。解耦gate与专家，先只换专家执行，不以一组开关混合归因。
3. **Graph/调度/prefix按后续证据选择**：不要根据一次profiler中的空隙就开始大规模Graph重构。尚需服务批量接入和A/C代表性负载，才能选择更偏吞吐或C尾延迟的第三点。

本轮不再要求GPU重复采集，只离线修正解析并保留报告。后续首个实际开发批次建议为Attention后端可行性/必要批量接口，先用少量短、窗口和16k回归守住语义，再做对应profiling对照；完整logits逐值一致不是新的优化门槛。正式evalscope A/B/C以及HumanEval待补事项保留，不将本次bs1表当最终成绩。

后续需求更新：用户要求加深并扩大Step0性能评测再开始优化，扩展矩阵、真实样本与细粒度采集见[计划09](../../docs/plan/09_expanded_profile_study.md)。本报告保留首轮结果，不将初步排序当最终决定。
