# 20 分步旋转实验结果与线性投影核查

真实套件`results/mellum_validation/20261111T205654256595Z`，ATOM`results/atom_mellum_gpu/20261111T205716526285Z`。sdpa_rope_diagnostic仍仅算术64token匹配，代码公共前缀37；prefill logits最大绝对差算术1.5、代码2.984375，RMS分别0.422499、1.497331。相较仅SDPA实验，分步旋转没有带来整体改善，不能提升为默认。

第9层算术输入相对L2由约0.001056降至0.000846，输出由0.009376降至0.002199；但最终norm相对L2由0.051451变为0.055237。代码final norm从0.075758变为0.069911，但全词表logits RMS更高。不同指标变化方向不同，不能以中间一项下降证明全模型正确。decode未改，prefill缓存复用AITER，因此实验仍非完整HF Attention路径。

## 线性层源码证据

`ATOM/atom/model_ops/linear.py:Linear.forward`无量化分支调用`aiter.tuned_gemm.tgemm.mm`；`ATOM/atom/model_ops/embed_head.py:ParallelLMHead.forward`也调用tgemm。HF Mellum投影使用`nn.Linear`/`torch.nn.functional.linear`。此外ATOM QKV打包为一次投影，HF分别计算。此前首层选定权重精确一致但K/V有少量差异，尚未完整分离线性内核与打包shape的影响。

新增`scripts/diagnose_mellum_projection_math.py`，固定四份保存HF input_norm输入及选定权重，比较torch分开QKV、torch合并QKV、AITER合并QKV；再固定HF attention_context及moe_input，分别用torch/AITER执行O投影与router，对照保存HF输出。该实验不载入checkpoint或引擎，只使用少量保存张量。它暂不覆盖lm_head、所有层及decode形状，也不把非零BF16误差自动判为bug。

已核对实际tgemm调用与保存权重/输入字段，语法及CLI检查通过。GPU未执行，正常模型后端未改变；SDPA及分步旋转继续保留为显式诊断选项。

普通终端：

```bash
/opt/venv/bin/python /workspace/scripts/diagnose_mellum_projection_math.py
```

优先确认torch分开投影能否复现HF，再看合并shape及AITER差异。根据实际结果决定是否增加Mellum专用PyTorch线性正确性路径，不在证据不足时替换全部线性层。

## 按统一协议补齐投影实验

执行顺序已整理至[04 正确性协议](../../docs/plan/04_correctness_protocol.md)。本轮没有GPU运行，也没有修改模型数学路径。

投影脚本新增CPU prepare-only：核对模型配置SHA256、四份第0层固定探针、5项权重及32项输入/期望输出张量（共37个文件）的形状、BF16和有限值，按reference顺序建立sample/name映射；保存全部张量SHA256、参考文件哈希、源参考环境和脚本指纹。旧运行未明确记录的stage_layer以legacy default标识，不伪造旧元数据。

实验默认每种算术重复两次，首先快照输出，防止共享缓冲区使重复检查虚假通过。除各后端相对保存HF的误差外，新增直接的packing_effect（torch合并vs分开）及kernel_effect（相同shape的AITER合并vs torch合并），避免混淆两个变量。输出包括不同元素数、最大绝对差、相对L2和自动汇总。

判读前置条件为torch参考投影重放精确、所有重复稳定；不满足则标记resolve_reference_or_repeat_instability_first，而不是归因内核。diagnostic_complete只表示采集完成，不是模型正确性通过。输出头尚不在本轮覆盖内。

本轮CPU4项通过：真实保存输入预检、错误shape/dtype/nonfinite拒绝、参考/重复失败的判读门槛、复用可变缓冲区的稳定性检查。语法检查通过。最终prepare-only原始产物：`results/mellum_projection_math/20261112T075612053958Z/results.json`，cuda_initialized_before_replay=false，尚无GPU投影数值结果。下一GPU命令仍为：

```bash
/opt/venv/bin/python /workspace/scripts/diagnose_mellum_projection_math.py
```

结果返回后按协议分支处理，不再先跑expanded或零散修改一个算子后自由生成。
