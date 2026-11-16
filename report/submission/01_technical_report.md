# Mellum/ATOM 阶段技术报告（交付版）

作者：XinyuSun-USTC。环境：题给 ROCm7.2.4/Torch2.10、ATOM、AITER；Mellum2.1-12B-A2.5B-Thinking BF16，单卡。本文中文说明与4页英文PDF表达相同结论；完整研究记录另附，不计入精简报告。

## 1. 完成内容与交付边界

完成 Mellum 配置注册、模型实现、权重映射和5631源tensor完整加载审计，以及 ATOM OpenAI server 的一键启动。历史 Step0 通过本地参考对比并保存标签；后续性能工作形成原生后端、调度/graph、融合和cache兼容实现。默认 `conservative` 为后期reference control，保留正确性复现基础；`aggressive` 为快速实验组合。模式开关独立于 Step0/优化点，不据此增加优化点数量。

**未完成整体正确性门槛下的性能闭环。** 激进组合在developer200前64 token仅62.85%，因此其速度不能作为通过题目门槛的最终成绩。保守预设此精确配置没有单独HumanEval及完整ABC；不能拼接两种配置成绩。正式公开集口径、隐藏集3次取中位数由裁判确认；本地结果只作为开发证据。

代码入口：`scripts/serve_submission.py:build/main`、`start_server.sh`；模型：`ATOM/atom/models/mellum.py` 中 `MellumAttention`、`MellumDecoderLayer`、`MellumForCausalLM.validate_loaded_weights`。接口同时覆盖非流式/流式输出及usage；冒烟检查使用真实交付入口。

## 2. Profiling 与优化选择

Torch profiler先按模型组件标注，再用固定输入微基准、AB/ABBA和全量ABC逐步验证。bs1 trace有18627个kernel，组件归因98.3%；活跃kernel170.569ms/跨度641.084ms，间隙470.515ms（间隙不全等同CPU时间）。MoE58.372ms、SWA41.228ms、QKV20.007ms、两处Norm合计25.183ms。bs32活跃1573.961ms/跨度2027.532ms，其中MoE714.275ms、QKV331.458ms、SWA258.752ms。该trace以prefill工作为主，不能直接解释为纯decode占比。

工程探索机制较多；以下按三个瓶颈方向叙述，而不是声称已经完成三组独立合格优化及题面累计消融。

| 方向 | 实现、预期与实测 | 关闭方式 |
|---|---|---|
| 原生批处理/专家/attention及数据路径 | 复用AITER原生专家、attention，按连续token执行投影，直接写分页KV，减少逐请求与布局操作。A64原生批处理439.67，相比前一原生L约121.5为3.62x；directKV进一步499.64，+13.64%。不同数学路径有BF16差异，核心矩阵显示attention与专家均贡献分叉 | disable-native-moe / disable-native-attention / disable-direct-kv / separate-qkv / reference-norm |
| decode graph与融合 | 减少多层小kernel的host launch与中间访存。graph同卡screen620.995→943.26，+51.9%；QK Norm+RoPE融合941.63→1062.46，+12.83%；native routing ABBA均值939.49→1008.94，+7.39%。这些是不同受控实验，百分比不可直接连乘 | disable-graph / disable-qk-fusion / disable-native-routing / disable-resident-rope |
| prefix兼容与元数据复用 | 复用KV前缀和prefill元数据，改善长共享前缀TTFT；不是缓存输出。完整ABC从无prefix组合到最终组合：C TTFTp99 1.4883→1.0260s，改善31.06%；A1129.38→1139.78，+0.92%。两机制在这组全量数据中一起变化，不能分别归因 | disable-prefix / disable-metadata-reuse |

来源：冻结ATOM已有模型、AITER kernel与HF原生Mellum数值语义；vLLM/SGLang/新版ATOM用于阅读参考。第三方kernel不主张原创。具体来源与许可证见根目录SOURCES.md，源码拆分见 `ATOM/atom/model_config/mellum_{native,fastpaths,direct_kv,fused_rope,prefix,prefill_metadata}.py`；graph接入见 `ATOM/atom/model_engine/model_runner.py`（以仓库实际路径为准）。

放弃/未选：chunk2048/4096造成吞吐/TTFT退化；indexed专家、仅prefill替换等未形成满意收益；CK/Triton SWA有数值分叉；reference Norm与阶段混合能改善局部token但未达到最终90%。双份专家权重只用于诊断，不用于交付预设。未为了通过公开集改变答案或缓存生成输出。

## 3. 性能测量与累计消融缺口

最终激进profile `sprint_graph_qk_router_prefix_metadata`，单卡槽32，A512/B50/C200；源配置与后端执行审计留存。独立EvalScope1.15客户端对题面命令做版本兼容适配：数据字段、A筛选循环至512、stream参数、Poisson归一化等均记录；不声称与裁判隐藏集完全等价。另一张GPU可能同时做验证，非裁判空闲机三次中位数。

| 指标 | 题给baseline | 本地激进最终 | 比较 |
|---|---:|---:|---:|
| A token/s/GPU | 638.282724 | 1139.7781 | +78.57% |
| B TPOTp50 | 未提供 | 4.865ms | 不计算提升 |
| C TPOTp99 | 41.265ms | 11.502ms | 延迟下降72.13% |
| C TTFTp99 | 993.549ms | 1025.998ms | 延迟增加3.27% |

A512、B50、C200均成功，无请求失败；C吞吐76.8348受到达率约束，不作为独立饱和性能宣传。prefix实际199事件、757792缓存token；图回放与原生routing/QK融合均有实际事件，非只设置环境变量。

题面要求的累计消融表必须保留缺口，而不拿不同实验规模填满：

| 题面累计行 | A token/s/GPU | B TPOTp50 | C TPOTp99 / TTFTp99 | 状态 |
|---|---:|---:|---:|---|
| 历史Step0，1卡 | 未测 | 未测 | 未测 | 标签f382471保留，不以reference-control替代 |
| Step0+优化1 | 未完成同口径全ABC | 未测 | 未测 | 有局部screen，不能冒充累计消融 |
| Step0+优化1+2 | 未完成同口径全ABC | 未测 | 未测 | 有graph/融合受控实验 |
| Step0+优化1+2+3 | 1139.7781* | 4.865ms* | 11.502 / 1025.998ms* | *仅最终实验组合，尚非已认证三点累计行；token门槛失败 |

因此提交的是可运行代码、正确性回退配置和已量化的优化探索；不是宣称所有必填实验已经齐备。

## 4. 正确性、诊断与复现

HF5.10.1独立参考虚拟环境与冻结推理环境隔离。固定token输入，先比较greedy前64，再用同前缀/teacher-forcing、逐层hook、同QKV与同专家输入回放区分累计误差与首个算子差异。精确加载审计覆盖全部5631源tensor；未仅凭能生成文本判定正确。

| 门槛 | conservative r槽1 | aggressive最终 |
|---|---|---|
| developer200前64 | 12800/12800，200序列精确 | 8045/12800，65序列精确，失败 |
| 长输入8191/8192/8193/16384 | 256/256 | 138/256；各55/9/11/63 |
| 16k单列 | 64/64 | 63/64 |
| HumanEval164，thinking off，2048 | 该精确预设未测 | 150/164=91.46%；HF145/164=88.41% |
| MTP / 双卡推理 | 未启用/未使用，不主张支持 | 同左 |

HumanEval候选提升5题，但绝对差3.05pp大于题面BF16 2pp，不能自行改成“只限制退化”而宣称通过。200条逐位置对齐与HumanEval功能指标是两个不同门槛；功能好不能豁免前者。长组整体90%为内部额外检查，题面要求16k单列，本报告不把它另造为官方独立门槛。

专家同输入回放曾定位grouped_mm/激活/FP32归约，修复后exact；投影packing、RoPE系数构造和HF生成语义也曾造成差异。最终核心20条矩阵：native/native69.84%、native/Torch71.17%、reference/native77.03%、reference/reference100%。阶段混合最好86.875%，仍未提供合格高速组合证据；不因诊断集规模小宣称完整失败定理。

复现：`./start_server.sh --mode conservative`；激进显式`--mode aggressive`。每组件开关见README；启动时launch.json固定实际配置/源码哈希。`scripts/smoke_submission.py`验证真实入口加载/HTTP/SSE，不能代替完整验收。证据见artifacts/submission/index.json；原始tensor与外部模型不进Git。完整Git历史、Step0标签、每日WORKLOG、报告与AI说明一起交付。后续首要缺口是同配置满足token门槛，再做规定的三点累计ABC和裁判复测。
