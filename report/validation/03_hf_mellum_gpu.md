# 真实checkpoint单卡BF16参考验证

日期：2026-11-11。当前状态：普通容器终端已完成真实checkpoint单卡BF16短输入验证，加载审计通过，两条16token greedy的cache开/关完全一致。下文保留前期设备限制和失败诊断。

## 执行边界与阻碍

用户已授权进入真实模型参考验证。当前工具进程torch.cuda.device_count()==0，且/dev/kfd、/dev/dri不存在，不能访问ROCm设备；未加载权重、未执行GPU任务，没有通过其他连接或权限绕过。不是模型执行失败，不应据此判断普通容器终端也没有GPU。

[本次预检结果](../../results/hf_mellum_gpu/20261111T043645260478Z/results.json)：status=blocked_no_gpu，exit=2。

## 可执行脚本

[validate_hf_mellum_gpu.py](../../scripts/validate_hf_mellum_gpu.py)

在可见GPU的普通容器终端运行：

```bash
/workspace/venvs/hf-reference/bin/python /workspace/scripts/validate_hf_mellum_gpu.py
```

默认选择空闲显存最多的逻辑GPU，要求至少40GiB空闲；可用`--gpu 0`显式选择。`--preflight-only`只检查设备，不加载模型。每次运行保存于results/hf_mellum_gpu的独立UTC运行目录。

## 运行内容与审计

- 固定独立Transformers5.10.1，离线读取现有模型/Tokenizer，使用原ROCm PyTorch。SDPA、BF16、单卡、eval，不导入ATOM，不修改依赖。
- 不使用需要accelerate的device_map；先CPU加载，再搬到选定GPU，需主机内存容纳加载过程。GPU显存检查不代替主机内存检查。
- 保存环境、官方模型源码hash、config/index/template指纹和loading_info；missing/unexpected/mismatched/error任一非空则停止。核对参数量12,149,923,072（BF16 index total_size/2）与dtype。
- 两条短输入：Python add函数和简单算术。真实chat template，enable_thinking=True；保存渲染文本、精确input IDs和mask。
- 默认每条greedy最多16新token，分别开启/关闭cache，要求生成ID精确相等。检查prefill logits有限，保存最后位置logits和top5。短预算可能只输出thinking开头，不用文本通顺度或是否答完判定数值正确。
- 保存结果与失败traceback，不盲目重试GPU失败。不包含性能benchmark、ATOM对比、16k、HumanEval或正式200条验收。

## 后续放行

需要普通终端的运行结果才能判断真实checkpoint是否可执行。若加载审计或cache一致性失败，保留本次输出诊断。通过仅意味着短输入参考路径可运行，后续仍需更长生成、边界和ATOM逐级比对。

## 首次普通终端执行与日志修复（2026-11-11）

用户终端已完成from_pretrained的权重加载进度311/311；紧接着保存loading_info时遇到set无法JSON序列化。异常发生在参数审计与model.to(cuda)之前，尚未执行模型GPU推理。不能将进度条完成解释为加载完整性已通过。

原因已用真实HF小模型CPU加载报告复现：missing_keys、unexpected_keys、mismatched_keys是set，error_msgs是list。原保存代码和异常分支都未处理set，导致二次异常掩盖诊断。

修复：JSON默认转换支持set/frozenset并稳定排序；先序列化再临时文件原子替换；异常先输出原traceback，JSON保存失败时保留文本兜底。加载审计仍直接判断原始集合，不吞掉非空错误。

CPU回归通过：真实HF loading_info、非空及嵌套set/frozenset，以及不可支持对象序列化失败时原results.json保持不变。未加载真实checkpoint，CUDA未初始化。可在普通终端重新执行相同命令，生成新的独立运行目录；旧失败记录不覆盖。

## 真实checkpoint短输入通过（2026-11-11）

用户在普通容器终端重新执行脚本；已直接读取[原始结果](../../results/hf_mellum_gpu/20261111T044314965290Z/results.json)复核，status=short_input_smoke_passed、phase=complete。

- 独立Transformers5.10.1、原ROCm PyTorch、GPU0、BF16参数、SDPA。
- 参数总数12,149,923,072，与index声明一致；missing/unexpected/mismatched/error均为空。
- python_add、arithmetic两条均生成16个新token，cache开/关greedy ID精确一致；prefill logits有限，已保存最后位置logits与top5。
- 本次记录load_seconds=6.0819，peak_allocated_bytes=28,249,155,072。这些是脚本阶段计时和PyTorch峰值分配，不是正式性能指标或总设备占用。
- 输出为thinking开头，16token尚未完成答案，不能据此判断代码功能或算术准确率。

结论：真实模型在当前ROCm硬件上的独立HF参考路径已跑通，无需回移模型源码。尚未验证16k、SWA跨窗真实输入、完整答案、HumanEval/200prompt或ATOM输出。后续可据此推进ATOM必要适配，并保留本次精确输入/输出作为最初回归样例。
