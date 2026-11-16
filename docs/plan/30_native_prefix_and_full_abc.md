# 30 两卡推进：原生prefix整合与高速组合完整ABC

原生Attention+FusedMoE已具备性能26和完整HumanEval28证据（144/164，对HF145/164，差0.6098pp）。本轮落实两项并行工作，不改既有已验分支及冻结依赖。

## GPU0：原生Attention+FusedMoE+prefix

工作区/workspace/tencent-mellum-native-prefix，基于prefix提交2ff73c6。新增原生cached-prefill：保留正确K写回、NHD按请求gather及SHUFFLE框架gather；cu_q与cu_k独立，验证真实绝对后缀positions/max lengths；仍使用AITER varlen GQA和window=(W-1,0,0)，decode保持原生paged路径。无SDPA静默回退。缓存开关显式启用，旧默认关闭。

源码依据：冻结/app/aiter-test/aiter/ops/mha.py:flash_attn_varlen_func约2644行明确规定causal bottom-right及window offset=seqlen_k-seqlen_q。不能把SDPA的普通非方形is_causal直接当其替代。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-native-prefix/scripts/run_mellum_native_prefix.py
```

一条命令执行：

1. 小型GPU kernel结构探针：Q1/Q3、不同KV长度、GQA/跨请求隔离、Full与W4/W1024窗口；零Q/K和可解析V验证右下因果/窗口边界。数值容差仅是结构探针，非题目正确性门槛。
2. 真实模型cache off/on：三条真实C提示词，16输出、重复/共享/逆序请求；必须完成且有真实命中、cached backend=aiter_varlen_cached。自由生成是否相同只报告，不作为性能前置门槛，避免原生MoE波动引发无界追踪。
3. 同协议HTTP smoke→A64/并发32→C20/0.3req/s，记录cache命中、实际原生后端、失败率和尾延迟。C阶段独立服务冷启动，不特殊预热。

GPU0固定；probe engine8826，HTTP8251/engine8926；45分钟执行总预算，最多显式调60，CPU准备另计。结构/执行/命中失败即停止，不自动追加HF/HumanEval/完整ABC。结果根/workspace/results/mellum_native_prefix/<UTC>，HTTP子结果capacity/summary.json。

## GPU1：不带prefix的已验高速组合完整ABC

独立工作区/workspace/tencent-mellum-native-abc，基于9a3b3ca，模型数学保持已验原生版本；新增benchmark profile与端口向每个stage传递，保留原R默认行为和恢复逻辑。显式fused_moe_native_attention，BF16/eager、TP/DP1、maxseq32，prefix/chunked/MTP关闭。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-native-abc/scripts/run_mellum_abc_baseline.py --gpu 1 --profile fused_moe_native_attention --server-port 8351 --engine-port 9026 --budget-minutes 90
```

smoke→B50→A512/并发32→C200/0.3，输出固定256/stream/ignore_eos/temperature0；独立EvalScope1.15及数据适配器沿用已审计版本，协议差异继续披露。单请求C超时180秒，失败保留，不把成功子集p99当完整成绩。完整规模可能运行数十分钟，90分钟执行预算，CPU准备另计；支持同参数--resume <目录>，源码/输入/已完成产物必须不变。

HTTP8351/engine9026与GPU0分离；结果根/workspace/results/mellum_abc_optimized/<UTC>。本轮是题目规模的本地开发测量，不是裁判最终成绩；并行GPU0可能影响共享CPU/内存带宽，最终固定配置需空闲环境重复测量，不能把本轮单次结果声称三次中位数。

## CPU交付检查

GPU0原生prefix相关72项CPU检查通过，包含后缀/窗口语义、cache写入顺序、默认关闭路径、服务开关、自动批次结构失败停步、token分叉不阻止容量测试。GPU1 ABC相关22项CPU检查通过，特别验证worker实际收到profile/隔离端口，避免误测回R路径。两边py_compile/diff及真实数据prepare-only、源码/输入/客户端指纹检查通过。工具无GPU设备，不宣称新GPU内核或组合已运行。

## 结果后的取舍

先比较GPU0组合相对性能26的C20收益与A回退，同时以GPU1完整规模判断真实容量。缓存若无增益，不为凑第三点保留；若有收益，补必要功能/长输入检查并进入累计消融。C仍有prefill导致decode停顿时，再有界评估chunked prefill；CPU发射成为主要瓶颈时才考虑Graph。已知公开200独立问题仍需报告；HumanEval通过不替代它。最终只固定一个配置，准备报告/AI说明/复现命令。
