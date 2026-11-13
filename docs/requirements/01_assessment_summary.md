# 原题要求摘要

来源：[原始题目](2026%20CSIG云一计算加速团队实习校招面试题.pdf)，8页。
SHA256：`16b292e791d6d782d9f5cafe79bf2073cbd4f24f32b255d711194170513a7536`。
本页为2026-11-10读取结果的整理，不是新规则；歧义以原题为准。PDF曾通过标准库提取文本，未进行图像渲染检查。以下原数字节号指[历史阅读记录](../../report/archive/initial_research/02_atom_research_original.md)。

## 阶段与限制

- Step 0：冻结环境下，在ATOM中单卡正确推理并通过正确性门槛。
- Step 1：最多三个针对已识别瓶颈的优化点，可独立开关、消融；任何双卡推理并行计一个优化点。
- 最终提交为ATOM OpenAI兼容服务；HF/vLLM是参考方向，不能用其他引擎代替交付。
- 性能对比使用给定官方Baseline，自己的Step0用于开发与增量归因。
- fused MoE选择/调优等在原题中属于优化示例；组件复用本身不构成自动免计的规则。当前工程取舍见[Step0方案](../plan/01_step0.md)。

### 正确性门槛（PDF第6页）

| 检查 | 原题要求 |
|---|---|
| 公开prompt greedy比对 | 200条公开prompt，前64个token与HF Transformers参考输出逐token一致的比例≥90% |
| HumanEval | pass@1与参考值差距≤T；BF16为2个百分点，FP8为5个百分点 |
| MTP | 开启时的greedy输出必须与关闭时逐token完全一致 |
| 长上下文 | 一组16k输入prompt单独统计 |
| 双卡 | 同样的检查在双卡配置也必须通过；并行不得改变greedy输出 |
| 最终复测 | 裁判空闲机器、隐藏集、3次取中位数；禁止缓存输出或公开集硬编码 |

这里的“2/5个百分点”是绝对pass@1百分点，不能写成相对2%/5%。原题未明确200条中的“一致比例”是所有token位置的聚合比例，还是完整64-token序列匹配的prompt比例；也没有写明不同EOS长度/特殊token的计数细则。具体实现应以发题reference/evaluator为准，不自行选更宽松的分母。

当前**参考输出文件、参考HumanEval数值和具体evaluator仍未定位**，不能把baseline.json中Rejected的3/164或模型卡HumanEval+91.5代入。

MTP严格一致性意味着不能用当前源码的relaxed MTP接受机制满足门槛；若未来选择MTP优化，需保持严格路径并单独验证。局部FP32公式/内部KV一致性是定位工具，不能替代HF token与HumanEval的两项正式检查。

### 三个场景与固定评测命令（PDF第3–6页）

| 场景 | 固定负载与规模 | 指标 / 计分 |
|---|---|---|
| A | 512×N条，输入512–4k、输出256、ignore_eos、并发32×N | 总输出吞吐/N，tokens/s/GPU，性能分30 |
| B | 并发1，输入1024、输出256；命令number=50 | TTFT/TPOT p50/p99，必须报告，不直接计分 |
| C | 200×N条，Poisson open-loop，rate=0.30×N req/s；输入4k–16k、共享prefix，输出256 | TPOT p99、TTFT p99，性能分10；延迟不再除卡数 |

N为声明使用的推理卡数。场景A/C负载按卡数扩大，不能只拿双卡总吞吐与单卡比较。B并发不随卡数变。

原题给出的工具是 **evalscope perf**：

- A（第4页）：parallel=32×N、number=512×N，dataset指定scenario_a_prompts.jsonl，min/max prompt length=512/4096，max/min tokens=256，stream=True，extra_args含ignore_eos=true、temperature=0.0。
- B（第5页）：parallel=1、number=50、dataset=random，min/max prompt length=1024，输出同上。
- C（第6页）：open-loop、rate=BASE_RATE_R×N、number=200×N、dataset=line_by_line、dataset-path为scenario_c_prompts.jsonl、输出同上，total-timeout=180。

因此正式benchmark设计应以原题这些脚本/实际裁判版本为准；ATOM benchmark_serving或自写工具只能用于开发诊断，不能未经核对替代固定评测。仍需检查本机evalscope版本/参数语义、SERVER_URL实际endpoint、A/C多卡如何读取/扩展固定JSONL、C的到达过程和timeout含义；这些都是复现细节，不是擅改评测参数的理由。本轮未执行脚本。

原题提到的intended_words不等于实际token长度，正式长度/过滤由给定脚本与tokenizer处理；不得手改prompt长度以提高得分。

### 环境、参考资源与交付（PDF第1、7–8页）

- 冻结Docker镜像、ATOM commit、AITER、ROCm/PyTorch等环境及Transformers版本；不换镜像、不升级依赖。只用分配的两张GPU，不得使用外部GPU。
- 允许第二张卡跑HF参考、做profiling，双卡推理本身不是门槛。不能把参考任务误写成“必须双卡并行”。
- 原题指出HF原生MellumForCausalLM、vLLM 0.23.0起CUDA实现可作为阅读参考；这是题目提供的参考方向，不意味着当前冻结Transformers5.2.0已经可加载。前面本机AutoConfig阻塞的源码事实仍成立，不能通过升级解决。
- D0发题材料按原题应包括镜像/权重/基线/评测脚本/**参考输出**/评分规则；现有目录尚未定位所有材料。后续应优先确认发题参考输出与evaluator来源，而不是继续自行猜测正确性oracle。
- 可引用开源/上游PR，须标注来源；候选人之间不得共享代码。允许AI辅助，但有单独说明要求。
- 代码交付：一条命令启动server、每个优化点的开关说明。
- 技术报告≤4页：profiling证据、为什么选这些点/放弃哪些、预估与实测收益、消融表。当前D1为内部研发记录，不能直接充当最终4页报告。
- AI使用说明1页：工具用途、错误建议、如何发现；按天记录工作、阻碍与决策（项目唯一日志为worklog.md，提交时映射D1–D7），保留完整Git历史、不squash，日志对应提交。
- 消融表：Step0、+点1、+点1+2、+点1+2+3，记录卡数、A tokens/s/GPU、B TPOT p50、C TPOT/TTFT p99；少于3点仍应如实列出实际配置。
- 答辩20分钟（10分钟讲+10分钟问）；周期一周，D7 24:00交付、D8清场复测、D8–D9答辩。具体起始/截止日期不能由本地当前日期擅自推断。
