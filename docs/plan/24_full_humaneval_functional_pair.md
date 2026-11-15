# 24 完整HumanEval功能配对：停止以token位置筛选阻挡任务评估

用户已确认：关闭thinking，最多2048输出token，优先控制耗时。本轮只推进完整功能评估，不新增内核修复。题目token一致性与HumanEval是独立指标；本轮不因已知token位置低于90%而拒绝评分，也不声称HumanEval通过可以代替公开集/16k要求。

## 一次启动，两卡并行

```bash
/opt/venv/bin/python /workspace/tencent-mellum-humaneval/scripts/run_mellum_humaneval_pair.py
```

独立工作区`/workspace/tencent-mellum-humaneval`、分支`step1/humaneval-functional-evaluation`。默认GPU0为官方HF5.10.1，GPU1为冻结ATOM5.2.0环境、FusedMoE候选，真实batch32，引擎端口8626。两卡各单卡推理，不是双卡并行模型，不改变计点。

候选为已有性能19约1.47倍批量收益的`fused_torch_route`：保留参考gate/路由、L控制和参考Attention，使用原生FusedMoE专家。当前不选择尚无性能数据的isolated贡献版本，不同时评估多个版本。候选已知D200位置72.36%仍完整评分；没有token对齐或logits/内部张量精确晋级门槛。

预计约30–60分钟，首次完整运行取决于实际输出长度，**全流程硬预算90分钟，含评分**。每15秒显示双方完成题数与日志字节数，HF逐题、ATOM按真实请求完成落盘。预算到达清理各自拥有的进程组，保留完成和部分结果，不杀其他任务。可以显式设置`--budget-minutes`，正数且最多240；不开启thinking时仍默认2048，双方共享设置。

## 输入与协议

- 完整164个原始HumanEval task_id，数据`/data/datasets/HumanEval.jsonl`，已核验同固定OpenAI版本；hash、唯一性和全部完成数量继续验证。
- 沿用已有chat包装：“Complete this Python function. Return a complete Python implementation in a single Python code block.”加原prompt；不将测试/canonical_solution给模型。
- 两边使用同一份已tokenize input_ids、greedy、EOS28、pad0、max_new_tokens2048、正常EOS终止，不强制参考历史，不缓存输出。
- `enable_thinking=False`，CPU实际检查模板suffix为assistant后`<think>\n\n</think>\n\n`，不是仅写一个未生效的布尔参数。
- **本地协议，不冒称与未提供的裁判HumanEval生成协议完全一致**。思考开关、chat包装、预算、提取规则均保留，未来官方协议明确后可复用代码重测，不将本地分数当成模型卡分数。

## 完整生成与评分

先运行已部署评分器self-test，再并行HF/候选生成。HF只用官方MellumForCausalLM generate、sdpa/grouped_mm起始配置，不转用冻结TF、不保存全词表logits/逐层张量；记录版本和加载审计。ATOM不读取HF输出，独立输入驱动正常生成、加载5631及native路径/batch32审计。

沿用既有`extract_completion`：去特殊结束标记，若存在thinking块则需结束后才提取；固定选择定义entrypoint的首个Python代码块，无测试结果参与选择。保留原始文本、token ID、提取状态、预算截断。提取失败仍计入164分母，记为失败，不悄悄剔除。

复用`/workspace/tools/human-eval`固定commit `b9453e8f1865df0a0057a1383767caa400acfbcf`、独立评分环境和既有隔离sandbox。前置/评分时验证上游源码SHA，自检需恰好1/2通过；环境缺失或隔离失败就报错，不绕过已有隔离执行生成代码。

每题一个completion，官方pass@1，同时输出HF/候选通过题数、绝对百分点差、candidate gains/losses和逐题变化。164题差3题=1.829pp、差4题=2.439pp；BF16本地检查按绝对差<=2pp。HF零分时不将0比0判为有效通过，标记需检查协议。

截断、提取失败、单次生成的候选非确定性均如实报告，不另加隐形容差，也不拿这些指标悄悄删题。完整序列相同数仅为背景信息，不作为评分门槛。HumanEval功能通过不自动使目前公开200位置门槛通过。

## 输出和HF复用

结果根`/workspace/results/mellum_humaneval_pair/<UTC>`，含inputs/problems/manifest、hf/candidate generation.json、原始日志、提取文件、隔离官方评分和summary/report。两边完整记录必须逐task按输入和生成设置核验，禁止漏题/重复题/错位比较。源码和环境变化需标记失败。

已完成的HF reference即使候选失败也保存checksum供复用；以后可用：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-humaneval/scripts/run_mellum_humaneval_pair.py --reuse-hf-run /workspace/results/mellum_humaneval_pair/已完成HF的目录
```

复用必须input hash、完整164记录、设置和HF版本匹配；不会重跑HF。整个失败/部分批次不冒充完整配对。当前不提供部分生成的逐题恢复，避免预算/协议混合；若HF未完成需重生成，原目录保留。

本轮只开发/CPU检查；工具无GPU设备，真实生成和sandbox自检由用户终端命令执行。不安装/升级冻结依赖，不改变原R/其它候选工作区。

交付检查：10项相关CPU测试通过，1项真实Landlock隔离检查因工具进程缺少支持跳过（用户终端运行时仍强制自检）；prepare-only已核对164题、thinking=False、2048预算及真实模板后缀，py_compile/diff检查通过。CPU准备不代表模型或评分已运行。
