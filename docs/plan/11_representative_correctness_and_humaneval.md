# 11 候选准确性：200条开发集及成对HumanEval

## 当前决策

按用户要求，先以接近题目的评测判断候选价值，暂停继续追求微观数值精确。0.57%张量relative L2不是原题验收阈值；三条合成边界输入的分叉也不能直接代替公开集或HumanEval结论。CK候选默认保持关闭，但不因三条样本直接宣称正式准确性不通过。

两项独立报告：200条开发prompt前64token与HF匹配（位置/完整序列两种口径）；HumanEval BF16 pass@1与同协议HF差距≤2个百分点。开发集不是确认身份的官方公开集，HumanEval本地prompt/提取/预算也不是已经拿到的出题方脚本，均不得冒称正式验收。

## 关键覆盖事实

既有developer200：164条HumanEval提示词+36条给定scenario A，长度63–4113，其中168条<1024、32条≥1024；全部32条长输入来自scenario A。当前优化只作用于达到窗口长度的SWA完整prefill，decode无改动，因此必须同时报告32条激活子集，不以168条短输入稀释风险而不披露。

新HumanEval完整代码协议：164题提示词长度73–456，均不触发此prefill优化。即使生成长度超过1024，decode仍是未修改路径；功能分数可补齐评测链条，但不能证明长输入候选安全。四题pilot也不触发优化，需如实记录。此事实使200集合中的32条长输入结果更有诊断价值。

## 1. 开发200对照：复用已验证HF

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_swa_200.py
```

复用20261113T104458011376Z已通过Step0/HF精确比对的HF5.10.1输出，检查模型config及200份prefill tensor SHA、每条64token；仅运行新候选，不重新跑HF。正常greedy/EOS，不保存12800份全词表生成logits（避免额外磁盘/传输），logits数值精确不作为门槛。仍保留prefill tensor、全部输出token与比较结果，结构/非有限错误直接中止。

候选每条仍64输出，受监控进程15秒心跳、3600秒上限，预算约10–15分钟。输出`/workspace/results/mellum_swa_200/<id>/summary.json`：all、swa_eligible32、below_window168，分别含token位置及完整序列匹配；全200按两种歧义口径分别列90%检查，不能自行选更宽松一项冒充裁判口径。`developer_comparison_complete`表示测量完整，不等于通过。

本地prepare-only：`20261114T163611784460Z`。未运行GPU。

## 2. 获取官方HumanEval：独立环境，固定来源

```bash
/opt/venv/bin/python /workspace/scripts/setup_humaneval.py
/opt/venv/bin/python /workspace/scripts/score_mellum_humaneval.py --self-test
```

官方源码：https://github.com/openai/human-eval 。首条克隆到`/workspace/tools/human-eval`，记录精确commit、Python源码SHA至`/workspace/results/humaneval_environment/manifest.json`；后续可用`--revision <记录commit>`复现，拒绝修改过的上游工作树。建立`/workspace/venvs/humaneval-eval`，不在冻结环境pip安装/升级；只读复用已有numpy/tqdm，通过pth引用官方源码。不重新下载已有本地HumanEval数据。独立环境用于评分，不替代ATOM推理引擎。

官方工具提供evaluate_functional_correctness及逐题结果；其README明确要求隔离生成代码，execution中的reliability_guard也声明不是安全sandbox。因此评分脚本在独立CPU子进程中应用文件/进程/网络限制再调用官方函数，不把任意生成代码作为root在项目目录直接exec。

首选Linux x86_64 Landlock限制文件读写、seccomp只允许AF_UNIX作Manager IPC、no_new_privs、降到uid65534，内存2GiB/文件64MiB/进程64等上限；Landlock不可用时构建只读runtime拷贝并chroot，然后应用相同降权和网络限制。fallback需要用户终端具备CAP_SYS_CHROOT等权限，约复制百余MiB runtime，无mount及依赖升级。不具备隔离能力则失败，不自动无隔离执行。此为本地评测隔离，不声称通用生产安全认证。

自检只使用受控两题：一条canonical_solution、一条故意报错，必须得到pass@1=0.5，并在评测前探测项目写入/网络被拒绝。运行日志、官方源码版本与评分隔离模式保留。正式模型代码尚未执行，当前工具进程无Landlock syscall、无chroot capabilities，该隔离自检在此无法完成；用户终端需先跑上述自检，不支持时发回日志再处理。

当前工具网络代理不可连接，取消代理后DNS仍无法解析GitHub，故没有虚称官方工具已下载；需要用户有网络的终端执行setup。现有lm_eval/evaluate可见，但其humaneval模块会动态下载code_eval，未直接把它当作已就绪的官方工具。

## 3. 先4题完整生成与评分pilot

通过自检后执行：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_humaneval.py
```

选HumanEval索引0/1/10/100，双方完全相同输入/正常greedy/EOS，enable_thinking=True，max_new_tokens=2048。一次HF5.10.1导出完整生成，再运行候选，不保存每步完整logits。脚本在GPU前自动再执行小型evaluator自检，避免评分环境不通却先耗费生成成本。

每条保存全部生成IDs、原始文本、提取代码、提取状态、是否达到预算。固定提取规则：移除完整think段；未结束thinking判提取失败；优先第一个定义entry_point的Python代码块，否则第一个代码块或纯文本；保留函数体缩进，以原始prompt+completion做语法检查，不根据测试结果挑选或修补答案。完整函数可在原prompt后作为新定义覆盖原签名；也接受合法函数体续写。提取失败作为该题失败（明确抛错completion），不从分母排除。

使用官方问题test和check(entry_point)，每题单样本k=1，记录pass@1与逐题HF/候选得失变化。pilot不检查2个百分点是否“通过”，只检查链条、可完成代码及预算是否合理；预算截断、HF零分均要求审查，不把双零差距冒充有效准确率结论。`--generate-only`可显式只生成/提取，不得把其状态当已评分。

最坏情况下4题各用满2048预算，单ATOM端按56ms/token约7.6分钟，另有HF及加载/评分成本；建议预留约5–15分钟，实际随提前EOS而变。不是把64token思考片段拿来评分。

## 4. 全量条件与时间控制

pilot确认可得完整代码、提取稳定、隔离评分正常后，才考虑：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_humaneval.py --full
```

164题同协议，每题1输出，pass@1就是通过题数/164；3题差约1.83个百分点，4题差约2.44个百分点。比较绝对百分点差，同时保留HF过→候选不过及反向变化。由于本候选对所有HumanEval短prefill均不激活，是否值得立即重复两端全164需结合200结果与pilot判断，暂不自动运行。

重要预算：若164题都用满2048输出，仅ATOM端按当前TPOT估计就约5小时，另加HF。不得把全量当几分钟任务启动。可显式选择`--disable-thinking --max-new-tokens 1024`建立双方一致的另一种本地协议，但它不是原Thinking模式的直接复测，必须独立报告，不能悄悄切换来提升分数或节省时间后声称同口径。

## 已完成与未完成

已实现200候选运行、官方源码获取/环境、完整HumanEval生成/固定提取/成对评分、失败即停止的隔离、自检入口；CPU提取5项通过、隔离探测1项因工具内核能力缺失跳过，不算通过。200与4题pilot的prepare-only通过，full164的CPU token长度统计完成（73–456、激活数0）。模型数学没有修改，GPU未启动。官方源码获取、用户终端隔离自检与GPU实际生成/评分均待执行。
