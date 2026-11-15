# 20 基于成熟实现的Step1主路径改造与验收

依据[研究05](../../report/research/05_optimization_implementation_sources.md)。本页是下一轮实施方案；本次仅完成调研和文档，未实现下面的新候选，未启动GPU。

## 目标

在冻结ATOM/AITER中复用Qwen3 MoE、原生分页Attention，减少参考式逐请求计算/数据收集。最终交付最多三个独立优化点和消融，不把框架支持的所有开关一起开启。当前R作为回退；不再投入prefill逐元素合并，当前indexed_decode失败版本不原样重测。

## 两卡开发拆分

| 卡 | 主候选 | 初始保留 | 初始移除/替换 |
| --- | --- | --- | --- |
| GPU0 | BF16 FusedMoE | 已验Attention和必要L控制，优先保留Torch gate/路由语义 | E按请求拆分、Torch专家grouped_mm/bmm和选权重物化 |
| GPU1 | 原生Attention | R专家，分层RoPE语义、BF16 KV | ragged SDPA逐请求桥、decode gather/repeat KV；窗口使用已有验证过的CK/原生后端 |

两个工作区独立，显式GPU绑定、引擎/HTTP端口分离，固定源码指纹与输入；不升级依赖、不改冻结/app、不改已有compile装饰模型文件。完整HumanEval HF参考可安排在卡空闲时运行，不与同卡性能计时混跑。

## 实施检查

### MoE

1. 在Mellum局部模块接入FusedMoE，沿用64专家/top8/H2304/I896、SiLU、softmax/renormalize、无bias、无共享专家。现有MELLUM_MOE_BACKEND=aiter仅作接入参考，不能继续套用Torch专用campaign wrapper。
2. 保证权重映射、after-loading shuffle和metadata恰好一次；5631源张量审计继续保留。选少量真实输入确认expert ID、权重、有限输出；不要求参考张量bitwise。
3. 整批专家执行必须有实际路径/批量证据，不允许按请求调用FusedMoE后把结果称作整批优化。路由保留或替换明确记录。

### Attention

1. 将“参考ragged桥”和“原生batch能力”分开配置；为原生路径添加独立batch审计，避免无意回退。
2. 核对S/S/S/F、head_dim128、Q32/KV4、W1024、full YaRN scale；W还是W-1依据各内核参数约定，不能跨框架照抄。
3. 优先复用冻结版PagedAttentionImpl的varlen/paged后端及已有SWA CK；RoPE/cache写入与实际KV布局必须匹配。
4. 第一版不启用prefix、chunk、Graph。主路径稳定后仅在实测表明必要时追加；不将它们未经验证塞进同一候选。

## 有界实验与晋级

每条主线只实现一个主候选。先关闭profiler，以真实短/长输入、batch1/32、包含256输出的少量请求验证速度和功能，不重跑完整ABC。仅当实际吞吐或延迟显示值得继续的收益，再运行固定200正常greedy64。

200按总体token位置>=90%作本地筛选，同时完整报告整序列、长短/来源分组、相对R输出差异；不添加相对R最多下降2pp筛选，不把中间张量相等当硬门槛。题目正式公开集口径未确认的限制保留。

收益足够且200具备条件后，最多一个组合：+FusedMoE+原生Attention。组合不能自动继承两个单点的正确性；补200、16k、完整HumanEval164同协议HF对照。代码执行由既有隔离评测机制承接。完整HumanEval BF16差距2个百分点，不用前64token或4题pilot替代。

错误诊断限于有证据的权重布局、路由/激活/累加、窗口/RoPE/cache语义修复，不展开无限逐层对齐。候选不达门槛就保留原始数据、停止或退回独立通过项；不改变prompt/评分/失败分母。

## 最终交付

- 固定一个最终配置，一条命令启动ATOM OpenAI服务。
- 独立开关及消融：参考/R、MoE、Attention、二者组合；计点诚实披露，不把每个子算子拆成优化点。
- 最终完整A/B/C在空闲条件执行；使用官方Baseline，C失败必须报告，不以成功子集p99冒充通过。
- 技术报告<=4页，AI说明1页，worklog和Git记录对应；详细研发报告为附属证据。

当前没有新增GPU命令。本页完成选型基础，具体入口、预算和恢复说明须在下一轮代码/CPU检查完成后交付。
