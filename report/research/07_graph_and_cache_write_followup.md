# 07 整批投影实验期间的Graph与KV写入源码核查

本轮只读当前native-batched源码，不修改GPU1运行工作区、不启动GPU或安装依赖。用途是为性能31结果后的后续方向准备有依据的接入边界，未宣称Graph或新KV写入已实现/有收益。

## 1. 整批投影之后，空campaign仍会阻止正常Graph执行

ATOM/atom/model_engine/model_runner.py:ModelRunner.run_model（约1836行）只要model._mellum_campaign存在且enforce_eager=False就抛错。当前整批投影profile虽groups为空，ATOM/atom/model_config/mellum_campaign.py:install仍创建Campaign，因此不能直接去掉enforce-eager。可行边界是纯生产无控制/观察时不安装campaign，同时保留旧诊断配置；不能简单删除全局检查让未知控制进入Graph。

框架本身有capture_cudagraph（约2122行），选择batch bucket、静态输入/positions并在torch.cuda.graph内执行模型（约2264行）。ForwardContext.in_hipgraph约406行提供capture状态。首次可尝试level0 decode graph，prefill保持eager；是否需要piecewise compile须后续单独判断，不能把图捕获与全部编译融合一股脑开启。

## 2. 参考K写回使用动态长度布尔索引

ATOM/atom/model_config/mellum_attention.py:overwrite_reference_keys（约114–118行）先slots=slot_mapping.long，再valid=slots>=0、slots=slots[valid]，对keys[valid]进行cache高级索引写入。这产生动态长度临时张量，是capture兼容性风险；其GPU同步/分配成本需要实际trace测量，当前没有耗时证据，不据源码估算加速倍数。

同时install_reference_kv_rope先计算参考qr/kr，调用原rope_cache再覆盖K；原PagedAttentionImpl.rope_cache（attention_mha.py约257–286行）会再执行RoPE及reshape_and_cache。因而重复旋转/缓存写入是已存在的执行链，不是猜测。

可选的最小后续实现：直接把已算好的reference kr与原V交给冻结aiter.reshape_and_cache，保留相同BF16 NHD/SHUFFLE布局及slot/block归属。接口在/app/aiter-test/aiter/ops/cache.py:reshape_and_cache；对应csrc/kernels/cache_kernels.cu约185行在GPU内判断slot_idx<0并跳过padding，无需先在Python做动态有效token压缩。该方案有机会同时去掉重复旋转/写入和capture动态索引，但必须验证实际布局、跨block/负slot、所有SWA/Full层，以及prefix先写后gather顺序。不能直接继承旧功能结论。

## 3. 原生decode审计的tolist在audit开关判断之前求值

ATOM/atom/model_config/mellum_native.py:install_native_attention.decode（约152行）调用audit(self,...context_lens.tolist())，Python先求值.tolist，再进入mellum_batch.audit检查环境开关。因此即使不写audit日志，此处仍读取GPU context lengths；capture不应执行这段。下一版可先判断是否需要审计及是否in_hipgraph，再读取实际长度，并确保性能测量不会被每token磁盘/同步诊断污染。具体收益需同负载配对，不直接套入投影收益。

## 建议顺序

先等待整批投影的真实A64/C20收益，再评估一次reference KV直接写入（有明确重复链路），随后才做少量decode graph bucket的可行性试验；不立即开发全部编译/量化/多卡。GPU0同时补原生prefix功能证据，避免把等待时间耗在重复数值对齐。
