# 精度优先第三项：元数据复用与参考SWA分组

## 目标

候选人明确本路线继续在保持正确性的条件下优化，不以通过C200作为本轮目标。保持交付分支及已定稿文档不变，在独立工作树 `/workspace/tencent-mellum-precision-third`、分支 `experiment/precision-third-point` 进行验证。

基线为已通过完整本地检查的①②：Torch参考专家group4、参考RoPE/KV单次写入、参考SDPA、单卡32槽。候选均不启用prefix、chunk、Graph、原生Attention或FusedMoE，也不混入断连回收。

| profile | 相对①②的唯一新增机制 |
|---|---|
| r_decode_meta | 每次forward共享解析context length与position |
| r_swa_decode4 | 元数据复用＋满足条件的SWA按4请求执行SDPA |

此前两个候选已在200×64和长组筛查中与通过的①②实际序列完全一致；批32隔离decode曾从42.8573提高至60.3174token/s。本轮不据此预判完整HumanEval必过，完整验证与重复性能重新采集。

## GPU0：完整正确性

两候选分别自然生成既有200条×64token、8191/8192/8193/16384四条×64，以及HumanEval164（thinking关闭、max2048）。HF输入/输出及评分工具校验后复用，不重生成。分别报告位置率、整序列数、长组和16k、功能通过数与绝对百分点差。

接受标准为既有本地token90%与HumanEval绝对差2pp检查，并保留长输入检查。不是中间张量/logits逐位相同。完整A/B测量只允许该候选完整检查通过后执行。预计两组20–50分钟，预算60分钟为每候选上限。

## GPU1：有界性能筛选

顺序R1→metadata→SWA group4→R2；四次独立引擎/服务，基线前后夹持。每项短A32、B4，各输出256；另固定真实C输入做隔离decode batch8/32、输出64。隔离统计最后prefill之后的callback交付速率，带host阶段审计，不是C持续到达负载或容量结论。

不运行C200。短A/B衡量服务增量，隔离decode检验机制作用；正确性结果独立返回，短测速度不赋予候选资格。预计20–35分钟，总预算90分钟，端口独立于其他实验。两卡各自空闲后运行。

## 晋级完整测量

两线完成后选择“完整正确性通过且短测有收益”的候选，执行full-performance：仅A512/B50，不重跑HumanEval或C。需要--profile及已通过的--acceptance-run，脚本校验摘要、全部产物和原source/input身份后才允许运行。预计约30–50分钟，取决于A吞吐；预算90分钟。

若SWA候选未过而metadata通过，则保留metadata作为第三项；若均未过，保持原①②，不重复无界精确对齐。所有结果与源码/输入哈希保存，失败不覆盖。
