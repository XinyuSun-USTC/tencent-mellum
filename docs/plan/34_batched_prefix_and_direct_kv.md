# 34 两卡批次：整批投影+prefix与直接KV写入

前置：整批投影性能31 A64为439.67tokens/s；功能33为149/164，高于HF145/164，原绝对差2pp检查False保留，不改门槛。prefix已有性能29及功能32，但仍是L请求级投影。本轮整合已有收益，并评估一项有明确重复链路的局部优化；不启动新完整ABC或HF生成。

## GPU0：整批投影+prefix

独立/workspace/tencent-mellum-batched-prefix，基于c34a96a，引入0555b76的profile；不改模型数学文件，只释放L控制并保留原生prefix能力。profile=fused_moe_native_batched、prefix=True，独立QKV/参考RoPE/FusedMoE/原生varlen+paged保持。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-batched-prefix/scripts/run_mellum_native_prefix.py
```

自动：矩形/窗口GPU结构探针→真实三条C提示词cache off/on→HTTP smoke/A64并发32/C20率0.3。实际cached backend必须aiter_varlen_cached并且有命中；自由生成相同率只报告，不阻止性能测试。首轮不复用旧L控制的功能分数当最终组合成绩。

固定GPU0；probe8826、HTTP8251/engine8926，45分钟执行预算，CPU准备另计。结果/workspace/results/mellum_native_prefix/<UTC>，HTTP子目录capacity。对照439.67版本判断A是否回退、C尾延迟/命中是否改善。

## GPU1：参考K/V只写一次

独立/workspace/tencent-mellum-direct-kv，基于0555b76。新增mellum_direct_kv.install_direct_kv_rope：保留reference_position_rope公式，直接用已有aiter.reshape_and_cache写入kr和原V，不再先调用旧rotary/cache再覆盖K，也不在Python先做负slot动态布尔筛选。保持同一use_triton_attn/asm_layout选择及NHD/SHUFFLE布局；只支持BF16、独立QK norm、无flash layout、无cached/chunked prefill。非法shape显式拒绝，无静默回退。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-direct-kv/scripts/run_mellum_direct_kv.py
```

自动：微型GPU布局探针（Full/SWA、Q32/KV4/hd128、跨16token块、负slot跳过、不连续V stride、未写槽不变）→HTTP smoke/A64/C20。cache复制应精确，仅检查写入语义，不对自由生成增加bitwise门槛。真实服务prefill和decode必须记录aiter_reference_kv_once；审计每phase一次，不做每token额外日志。

固定GPU1、HTTP8151/engine8726；40分钟执行预算、CPU准备另计；结果/workspace/results/mellum_direct_kv/<UTC>，HTTP子目录capacity。profile=fused_moe_native_batched_direct_kv只增加MELLUM_DIRECT_KV_WRITE=1，其他开关同439.67版本，prefix关闭。默认旧路径保留。本轮没有开启Graph或compile；直接写入有机会移除一处capture障碍，收益由实测决定。

## 检查与取舍

GPU0相关74项CPU检查、GPU1相关59项通过；包括原生缓存绝对后缀/窗口、旧路径回归、直接写入选路、BF16布局/负slot/shape拒绝、实际profile和自动失败停步。真实数据prepare-only/源码与输入/独立客户端审计核验通过；工具无GPU，尚未执行新GPU实验，不宣称新性能或功能成绩。

两项可并行，端口和工作区隔离；对单次结果注明另一卡活动，不冒称最终无干扰成绩。先以小负载收益取舍：prefix有收益则保留；直接KV无收益停止，有收益后再与prefix整合，补最终功能/长输入回归。若还有明显CPU发射开销，再有界试decode Graph，而非同时开发全部编译/量化/多卡。最终固定配置跑完整ABC、累计消融与空闲复测，所有原绝对差/公开集独立要求继续如实记录。
