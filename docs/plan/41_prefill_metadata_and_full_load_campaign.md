# 41 GPU运行期间：prefill元数据复用与完整负载批次准备

独立/workspace/tencent-mellum-campaign，分支step1/final-performance-campaign，基于ee3e238。正在运行的scheduling目录不改、不占GPU；本批准备不是最终报告整理。

## 1. 新优化候选：一次forward内复用长度校验

native_prefill在28层重复读取GPU cu_seqlens、positions并进行主机校验，cached矩形几何也逐层重复。forward_context.py:set_forward_context每次创建新的ForwardContext；数据在这一次forward内跨层共享，因此新增mellum_prefill_metadata.geometry，将已验证的新token/完整key长度列表存到当次ctx。

MELLUM_PREFILL_METADATA_REUSE默认关闭；开启时key包含元数据tensor身份、可用的版本号、长度与最大长度。普通tensor发生原地修改会失效重验；inference tensor无版本号时按同一forward内元数据不可变的框架约定复用，绝不跨ctx保存结果。它不缓存模型输出、不识别prompt、不跳过KV写入或模型计算。decode不调用本模块，Graph不变。

NHD gather也复用已校验的逻辑key长度，减少层间context_lens.tolist；SHUFFLE底层gather仍有repeat_interleave等开销，本次没有宣称全部主机同步都消除。metadata.jsonl按进程首次实际校验/复用各记一次，不逐层写文件；性能候选必须看到reused=true才确认开关生效。

配对候选：Graph+QK融合baseline与metadata；prefix baseline与prefix+metadata；chunk2048 baseline与chunk2048+metadata。每条track GPU1、A64/C20，不逐项HF。等当前结果返回，仅选择有价值的一组对照，避免把三个track都无条件跑一遍。

## 2. 完整ABC批量入口

从cd2c3e3复用已有完整ABC profile/端口/resume能力，修复当前分支完整execute尚未传递profile到stage plan的缺口。新run_mellum_full_campaign.py接受显式profiles，在同GPU逐配置运行B50/A512/C200；输入模板只准备一次、各配置只读复用。

- stage失败保留并继续其他stage/配置；失败请求或部分成功尾延迟不当有效成绩。
- resume对完成stage核验产物哈希并跳过，只重跑失败/未完成；失败目录attempt_N原样保留。
- 完成必须加载5631项并有实际Graph/路由/QK融合回执，不能以配置名推断后端。
- 保存实际指标、超时、请求覆盖、源码/输入/客户端指纹与已披露协议差异；空闲最终复测与并行开发测量区分。
- 端口GPU0 8451/9426，GPU1 8551/9526；本次不运行GPU。
- C200每卡0.3req/s，到达过程本身约11分钟，完整ABC不可能像C20一样几分钟完成；不要为缩短时间修改到达率或请求数。

准备/未来执行示例（待当前结果决定profile及GPU空闲后使用）：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-campaign/scripts/run_mellum_full_campaign.py --gpu 0 --profiles sprint_graph_qk --prepare-only
/opt/venv/bin/python /workspace/tencent-mellum-campaign/scripts/run_mellum_full_campaign.py --gpu 0 --profiles sprint_graph_qk --budget-minutes 90
```

恢复时加--resume原运行目录，GPU与profiles必须一致；预算为本次恢复执行预算，不复用旧的已耗尽预算。当前命令中的profile仅是已测参考，不预先替代尚未返回的组合赢家。

## 3. 验证与后续决策

40项相关CPU检查通过，覆盖同ctx复用、跨ctx隔离、tensor变更重验、inference tensor、原Attention/直接KV/调度/路由、全量stage profile传递、失败后继续、resume跳过成功、失败attempt保留及产物篡改拒绝。py_compile和diff检查通过；未有新GPU结论。

先分析当前组合/调度结果：组合有收益即保留；prefix/chunk失败或退化则停对应候选。有明显prefill瓶颈才运行metadata配对筛选；不自动扩展到量化/MTP/双卡长周期改造。随后优先固定有实测依据的组合，跑完整ABC；集中进行题目正确性与最终消融，而非再次回到逐层浮点精确排查。
