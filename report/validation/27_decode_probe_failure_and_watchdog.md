# 27 Decode诊断挂起排查、探针修正与失败检测

## 实际事件

用户运行`results/mellum_decode_validation/20261112T123830320864Z`停留在STAGE atom。HF已导出四例各8步cached/uncached共64份logits。ATOM子运行`results/atom_mellum_gpu/20261112T123848966618Z`，首请求完成计划内8个logits观测后，另一次前向触发ForcedDecodeProbe的Unexpected extra decode step；ModelRunner于12:39:14报告进程死亡，上层subprocess.call仍等待，用户终端未返回。

这是诊断探针与失败传播问题，不是长时间正常计算，也不是HF仍在重复prefill。此前仅解释计算开销、未先检查日志的判断不充分；发现后明确通知用户在原终端Ctrl+C，不终止其他任务或泛杀GPU进程。

## 保留下来的有效观测

首请求chinese_inventory的8个原始logits均已保存。修正首步日志元数据后，8次实际forward输入/position/逻辑长度与HF计划一致。第0步prefill精确；第1次decode最大logits绝对差0.25，后续不同。只是一条请求的局部数据，不能当成四条基线完成，也不把强制采样token一致当作数值通过。

## 修正与约束

1. `ForcedDecodeProbe`允许一条末尾额外前向：仍要求输入为最后计划token、position符合下一位置，保存为terminal_*并标记unscored_terminal。返回真实logits、不强制评分，正式比较仍只读取step_000至step_007。超过一条、输入/位置不符或前向/logits不配对仍拒绝。此修正针对实际观察到的额外末尾前向，不放宽计划内数据一致性。
2. 新增`scripts/mellum_process_runner.py:run_stage`，用独立进程组启动自己拥有的子进程；监测worker死亡日志、每15秒心跳、默认每阶段300秒超时（可通过stage-timeout显式调整）。失败/超时/中断只清理本次子进程组，TERM后最多等待3秒再KILL，避免无限等待。未修改ATOM共享进程管理实现。
3. `run_mellum_decode_validation.py --reuse-hf-run`校验已完成HF状态、输入文件哈希、配置/版本/专家后端、步数和64份张量哈希后复用HF，不重算参考。
4. 同时发现旧HF首步trace的input_ids引用可变history列表，被后续append延长。数值计算当时已由torch.tensor拷贝，保存logits未受影响。export_case现按list(ids)快照；旧参考只在精确满足prompt+forced_tokens的已知模式时，于新运行生成元数据修正副本并记录源/派生哈希。所有实际原始输入及GPU张量不变。其他trace不符拒绝。

## CPU验证

- 探针3项通过：原始logits不被强制值污染、dummy/坏位置/配对检查、末尾记录不参与评分且可开始下一请求。
- 运行器/数据5项通过：模拟worker死亡、超时和正常退出；输入选择、旧日志元数据恢复不改变原始历史/张量路径及哈希。
- 独立HF小模型1项通过，增强首步input_ids快照断言，覆盖跨滑窗cached/uncached导出。
- 64份HF张量SHA256及首请求8次实际历史复核通过；语法/diff检查通过。

恢复GPU尚未执行。先Ctrl+C结束旧挂起命令、回到提示符，再运行：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_decode_validation.py --reuse-hf-run /workspace/results/mellum_decode_validation/20261112T123830320864Z
```

本次跳过HF，只重跑ATOM基线并离线比较。若仍失败会明确返回失败及日志路径，不继续等待。仍保留原KV写入/decode数学路径，不宣称这些路径已修复或Step0完成。
