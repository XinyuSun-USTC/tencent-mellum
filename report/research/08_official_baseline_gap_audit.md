# 08 官方朴素Baseline与当前执行路径的差距审计

用户质疑低于朴素Baseline是否遗漏重要内容。本轮只读核对direct-kv/已测profile源码、官方baseline.json及既有实测；未改推理、未运行GPU。

## 已有因果证据

- campaign L将QKV/O/router按请求拆分（mellum_campaign.py:split_call/ Campaign.wrap）；取消它时ATOM模型源码与输入完全相同，A64由121.50到439.67，约3.62倍（性能31）。说明此前诊断兼容控制显著拖慢了服务，不能视为模型必需成本。
- reference KV路径先参考旋转，再original rope/cache，再覆盖K；改一次参考KV写入后A64由439.67到499.64，约+13.64%（性能35），另有真实缓存布局验证。重复工作是第二项已证实开销。

## 尚保留的基础慢路径

1. atom/models/mellum.py:MellumRMSNorm.forward（约52–75行）手写FP32 cast/square/mean/eps/rsqrt/mul/cast/weight，多次eager算子。每层input/post/Q/K四处加final，共113个Norm位置。不能把位置数当实际GPU kernel计数，需短trace确认。对照同仓qwen3_moe.py约214/319行使用atom.model_ops.layernorm.RMSNorm，后者forward约333行复用rmsnorm2d_fwd_/with_add_融合路径。
2. mellum_projection.py:separate_qkv_forward约8行仍3次F.linear加cat，MELLUM_QKV_BACKEND=separate；router仍reference F.linear。它们为Step0形状/舍入对齐引入，尚未在任务级容差下重新评估框架packed/tuned路径。
3. mellum_attention.py:reference_position_rope约30行每调用将CPU inverse_frequency转q.device；direct-kv闭包仍保存CPU inv，因此有每层每步重复常量传输的源码路径。字节数虽小，实际同步/发射成本需测，不能按字节直接判断影响。
4. mellum_native.py:install_native_attention.decode约152行在audit函数检查开关前先求值context_lens.tolist。诊断日志/读取应与纯性能路径分离，并处理Graph捕获。

## 如何理解朴素Baseline

题目只明确torch循环MoE、显式mask SDPA、eager/level0/MTP off等特征，不等于其所有基础Norm/GEMM也采用分解式参考实现。baseline.json没有提供作者实际Norm/投影源码，不能断言作者一定用了某具体融合核。自身额外控制/参考式路径已有源码与实测证据，应优先清理。

官方Baseline本身eager，因此缺Graph不能解释我们此前所有落后；Graph是后续额外收益候选。官方A638.28，最新开发A64为499.64，数值约78.3%；旧完整A121.28是过时较慢版本，不能混用说最新版本仍落后5倍。新配置仍需完整A512实测，亦不可把小集比值当正式成绩。

官方baseline.json的Rejected HumanEval说明针对其功能评估，不据此推断性能数字无效。客户端版本/请求模板/长输入过滤/到达过程差异沿性能16披露；没有证据将其作为主要性能差距解释。另一卡负载、CPU/I/O共享需最终空闲复测控制，不能代替源码归因。

## 优先级调整建议

在最新版本采少量decode步骤短trace，确认Norm/投影/常量传输与CPU间隙占比；独立开关恢复框架Norm、packed QKV/tuned GEMM及GPU常量驻留，先小负载性能，有收益再功能评估。与prefix整合/有界Graph可行性结合，但不同时泛化量化/MTP/多卡。基础执行路径审计应优先于再做复杂内核，既有Step0逐位定位工具保留在诊断侧。
