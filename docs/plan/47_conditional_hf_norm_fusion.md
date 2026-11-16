# 47 等待greedy诊断时准备HF语义融合Norm后备

独立/workspace/tencent-mellum-hf-norm，分支step1/hf-compatible-norm，基于6485506。当前norm/systems实验目录不改，未使用GPU。本次是条件后备开发，不是宣布Norm已定位为根因。

## 条件与目标

只有当前agree_norms或agree_norms_route的batch32实测200位置率≥90%，才考虑把其中拆分Torch Norm换成快速实现。若这些控制分支没有达标，不继续消耗GPU在该假设上；直接依据当前实验选择其它主因。所有HF参考、输入、64输出及90%要求不变，不扩展HF/HE/完整ABC。

## 已开发

mellum_hf_norm.HFCompatibleRMSNorm：参数仍仅weight，名字/shape不变，TP1/BF16、dim128及2304、保留ATOM可选residual接口。

Triton _hf_norm一个launch处理每行：

1. 有residual时FP32相加后先cast BF16，保留HF残差加法边界，写出单独residual_out。
2. 对BF16 combined做FP32均方、rsqrt、归一化，然后cast BF16。
3. 再乘权重并存BF16输出，避免把权重乘法提前到归一化的FP32段。

没有输出缓存或输入修改；非连续输入先contiguous，真实模型input/post/final通常连续。对dim2304按4096lane取mask、分母使用2304；禁用FMA。遵循相同阶段边界不等于GPU归约树逐位相同，后续依然看200输出门槛，不能由离线编译推断通过。

make_norm新增默认关闭triton_hf后端。agree_fast_norms仅把agree_norms的backend替换为triton_hf；agree_fast_norms_route同理保留Torch route。QK融合和HF QK舍入、Graph、Packed、prefix均保留。未修改既有装饰模型或冻结库。

## 条件执行入口

run_mellum_greedy_recovery新增hf_norm/hf_norm_route两条配对track：参考Norm控制对照→HF融合Norm。非prepare运行必须提供--evidence当前已完成诊断目录，且对应控制分支确实达到90%，否则拒绝启动。先GPU局部数学/残差/不改输入冒烟，再仅200+long，微观冒烟不代替题目门槛。

GPU微观覆盖128/2304维、1/32/1024行、非连续输入、有无residual，严格检查残差BF16输出及输入不变；Norm输出采用宽松局部错误保护，不把它冒称正式精度阈值。总预算包含微观阶段，源码/参考/配置指纹保持原机制。

当前只做--prepare-only，默认不提供立即运行的新GPU命令。等现有结果返回后，若条件满足，再选择其中一条配对，避免两条都无条件跑。

## 已验证与尚未验证

49项CPU检查通过，含权重接口、CPU/非法shape/TP拒绝、factory显式启用、profile差异及无积极证据禁止GPU启动。gfx942离线编译四种特化（128/2304×有/无residual）通过。数据准备与源码指纹已核验；没有GPU数值或速度结论。

下一决策：若现有诊断找到≥90%的batch32候选，先保留它，再短测速度；本后备仅用于有证据的性能恢复。若Norm控制无效，则不继续本方向，不重启逐层浮点精确排查。
