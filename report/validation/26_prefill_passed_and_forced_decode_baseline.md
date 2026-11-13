# 26 新prefill集通过与强制同历史decode基线

## 实测prefill通过

原始套件`results/mellum_validation/20261112T122035305179Z`，ATOM`results/atom_mellum_gpu/20261112T122053931689Z`。10个输入的下一token、完整98304维logits全精确（最大误差与RMS均0），300份embedding/28层/final norm观察全精确。5631/5631加载审计通过，无缺项。1024阈值调用修正得到GPU验证。

连同此前四个固定探针，已有两组完整prefill精确对照，但当前仍是显式、单请求、窗口内候选，不能宣布Step0完成或decode/KV已正确，也不切换默认后端。

## 首轮强制历史验证设计

新增`run_mellum_decode_validation.py`，从已完整对齐的prefill运行选择4条固定输入：chinese_inventory(32)、raw_16(16)、raw_17(17)、window_1023(1023)。每条8个logits位置（prefill一次+decode七次），共32份，不是每条8次decode。覆盖短输入、16/17 block边界及逻辑上下文1023–1030跨过滑窗边界。

`export_mellum_decode_reference.py:export_case`在HF5.10.1、grouped_mm、SDPA下手动逐步前向：取cached argmax作为固定后续历史，保存每步cached logits，同时用相同完整历史重新prefill保存uncached logits，记录position、逻辑长度及HF第0/3层物理key形状。两种HF路径自身的差异独立报告，不以uncached值冒充cached逐步参考。手动前向不会因EOS自动停止，必要时继续馈入EOS；这是有意的诊断，不是正常生成评测。

`ATOM/atom/model_config/mellum_decode_probe.py:ForcedDecodeProbe`只在显式teacher-forcing环境变量启用：

- observe：验证真实forward的token IDs、position和单请求阶段，跳过dummy；顺序错误、漏配对、超计划步骤拒绝。
- force：先保存原始完整logits，再生成仅计划token可被采样的替代logits；原始logits不被修改，且后续CPU比较只读保存的原始值。

`MellumForCausalLM.forward/compute_logits`接入这两个诊断点。`check_atom_mellum.py --forced-decode`显式启动，ignore_eos=True以保证计划长度，状态为forced_decode_complete，不再用short_greedy_passed标记。正常入口清除相关环境变量，不改变自由生成。

运行器核验实际后端、完整加载审计、强制计划哈希、HF日志哈希与每步真实输入/position，再比较ATOM强制前的logits与HF cached logits。强制输出token相同是机制本身的结果，不是正确性证据。最终状态为forced_history_logits_exact或forced_history_logits_differ，保存首个非精确位置和HF cached/uncached差异。

本轮仍保留旧KV写入和decode计算，测的是该原路径在相同历史下的数值基线；没有声称已修正缓存。尚未对ATOM每个物理KV槽逐元素dump/审计，position日志和logits只能作为后续缓存定位入口，不等同于全量KV证明。

## 已完成的CPU检查

- ATOM相关39项通过，新增强制探针测试覆盖dummy跳过、真实logits先保存、强制值不污染原值、token/position和forward/logits配对检查。
- 脚本相关12项通过，包括从已对齐prefill选择四条正确输入。
- 独立HF环境小模型CPU1项通过：实际cached/uncached导出跨sliding_window，检查四步position/长度、文件及shape，未初始化CUDA。
- 语法与ATOM diff检查通过。
- prepare-only产物`results/mellum_decode_validation/20261112T122722722453Z`。GPU decode基线尚未执行。

## 普通终端命令

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_decode_validation.py
```

先HF、后ATOM顺序运行并各保存完整stage日志。无需重复prefill扩展集或自由生成。若有差异，先看首个不同位置及HF内部对照，再区分位置、KV内容/布局与decode内核；不根据被强制的token序列宣布通过。
