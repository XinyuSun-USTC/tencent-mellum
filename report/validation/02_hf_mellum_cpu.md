# HF Mellum 小模型 CPU 验证

日期：2026-11-11。状态：24/24检查通过；仅验证官方HF小模型，不代表真实checkpoint、GPU或ATOM正确性验收通过。

## 环境与复现

```bash
/workspace/venvs/hf-reference/bin/python /workspace/scripts/validate_hf_mellum_cpu.py
```

[脚本](../../scripts/validate_hf_mellum_cpu.py) · [原始结果](../../results/hf_mellum_cpu/20261111T034252494142Z/results.json) · [源码指纹](../../results/hf_mellum_cpu/20261111T034252494142Z/source_hashes.json)

Transformers5.10.1来自独立venv，PyTorch来自冻结ROCm环境。CPU FP32，seed=20261111，线程数2；HF离线，无完整权重读取、GPU初始化或ATOM代码改动。ATOM基线仍`26f23a0bdbf50b9010c07ee3a71171189723e767`，本轮git status干净。

## 模型规模与验证范围

模型43,936参数：4层S/S/S/F、H32、Q4/KV2、D8、4experts/Top2、expert intermediate16、vocab97。从真实config拷贝并缩小；保留分层Default/YaRN，测试窗口W4、YaRN factor16/original8。随机权重，非12B checkpoint。实际tiny配置、脚本hash见原始JSON，每次运行使用独立结果目录。

| 检查 | 数量 | 结果 |
|---|---:|---|
| 配置保存/AutoConfig重读关键字段 | 1 | 通过 |
| 长度3/4/5/9/15，逐token及分块cache对比整段无cache logits | 10 | 通过 |
| 分块3+6+2+4跨滑窗，cache对比 | 1 | 通过 |
| W4长度3/4/5/9、W1024长度1023/1024/1025的实际mask对比独立位置条件 | 7 | 通过 |
| eager与SDPA相同权重logits | 1 | 通过 |
| 左填充两请求，显式正确position_ids，对比各自独立执行 | 1 | 通过 |
| greedy cache开/关生成8token | 1 | 精确一致 |
| 官方packed checkpoint保存/加载 | 1 | 参数精确一致，logits误差0 |
| 模拟磁盘逐expert gate/up/down键，官方加载转换 | 1 | 参数精确一致，logits误差0，无missing/unexpected/mismatched/error |

logits比较阈值rtol=2e-5、atol=2e-6，是本轮CPU FP32回归容差，不是BF16或题目正式标准。主要比较最大绝对误差1.4901161193847656e-7。greedy比较是整数token精确相等。

窗口独立判据为`k<=q and k>q-W`，W包含当前token。长度1025时W1024最后一行可见1024个位置。这里只检查实际mask生成，不是在1025长度上运行真实模型。

跨窗口cache最终各层实际存储长度为[3,3,3,15]：此官方HF DynamicCache将SWA历史存储保留W-1，Full层保留完整历史。这是HF缓存行为，不能当作ATOM必须采用同样物理布局的要求；ATOM需保持等价可见位置和输出。

## 有效结论与限制

- 参考环境已超越配置/导入阶段，官方Mellum可在CPU执行随机小模型forward、缓存decode与generate。
- 官方加载器能处理当前checkpoint风格的逐expert键；本轮只用合成小权重验证转换，不证明真实5631张量已全部正确加载。
- 缓存、分块、padding、后端一致性有开发回归依据，不是独立于HF的数学oracle。随机小模型通过不能证明官方实现或未来ATOM没有共同语义错误。
- 未测试真实12B、BF16 GPU数值、真实Tokenizer/template生成、16k、HumanEval及题目200prompt；未比较ATOM。

## 下一步

准备真实checkpoint的单卡BF16短输入HF参考脚本：明确GPU选择、离线加载、完整加载审计、模板/input IDs/生成设置、输出与环境指纹；先短序列验证，再扩展cache/边界/长输入。无需回移官方源码，本轮没有启动该GPU阶段。
