# Mellum 2.1 on ATOM：源码分析与模型适配前期调研指南

> **文档性质：** 阶段性研发任务书（Research Brief）  
> **项目：** 腾讯 CSIG 云一计算加速团队校招实践考核  
> **当前阶段：** ATOM 源码分析、现有模型支持调查与 Baseline 实现定位  
> **目标模型：** JetBrains Mellum 2.1 Thinking  
> **目标推理引擎：** ROCm ATOM  
> **硬件平台：** AMD Instinct MI308X × 2  
> **工作目录：** `/workspace/ATOM`  
> **文档状态：** 当前阶段执行指南

---

# 1. 项目背景与研究动机

## 1.1 考核任务背景

本项目是腾讯 CSIG 云一计算加速团队提供的校招实践考核，核心任务是在指定的 AMD ROCm 环境中，使用 ATOM 推理引擎适配 JetBrains Mellum 2.1 Thinking 模型，并在此基础上开展推理性能优化。

考核提供的硬件资源为：

- 两张 AMD Instinct MI308X GPU；
- 每张 GPU 约 192 GB 显存；
- 两张 GPU 位于同一节点，通过 xGMI 互联；
- 预置 Docker 容器、模型权重、评测数据和基础性能数据。

考核主要分为两个阶段。

**Step 0：模型适配与正确性验证**

在冻结的软件环境中，使 Mellum 2.1 能够通过 ATOM 单卡正确推理，并通过规定的正确性门槛。

**Step 1：性能优化**

在 Step 0 的基础上，最多实施三个独立优化点。每个优化点必须针对一个明确的性能瓶颈，具有独立开关，并能够通过消融实验验证其效果。

本次考核不仅关注最终性能，也关注候选人理解陌生模型、理解陌生推理引擎、定位瓶颈、选择优化方向和使用实验数据验证结论的能力。

因此，本项目的研发原则是：

**先理解模型与推理引擎，再完成正确适配，最后通过 Profiling 和实验驱动性能优化。**

## 1.2 为什么需要进行 ATOM 源码分析？

当前目标模型 Mellum 2.1 的架构为 Qwen3-MoE 风格的 Decoder-only Transformer，但 ATOM 在出题时尚未将 Mellum 列入模型支持列表。

考核说明书明确指出，ATOM 已经具备适配 Mellum 所需要的相关基础组件：

- Qwen3-MoE 模型骨干；
- GPT-OSS / MiMo 的 Sliding Window Attention；
- DeepSeek / MiMo 的 MTP 投机解码；
- AITER GPU Kernel；
- Prefix Cache；
- Torch Compile 和 Graph Capture；
- TP / DP / EP 等并行执行能力；
- Profiling 和性能分析工具。

考核还提供了 ATOM 自带的 Model Support Guide 和 CLAUDE.md。

因此，本项目不应从零实现完整推理引擎，而应该充分理解并复用 ATOM 已有的架构、模型抽象和算子实现，针对 Mellum 与已有模型之间的差异进行必要适配。

## 1.3 当前研究的核心疑问

在正式开发之前，我们需要首先回答：

1. ATOM 的整体架构是什么？模型从启动到推理经历哪些模块？
2. ATOM 如何识别模型架构、解析配置并注册模型？
3. ATOM 是否已经包含 Mellum 或相关模型的适配实现？
4. 腾讯提供的 Baseline 实现是否存在于当前容器？
5. 如果 Baseline 实现不存在，ATOM 中哪些已有模型可以作为参考？
6. Qwen3-MoE、GPT-OSS、MiMo、DeepSeek 的哪些实现能够直接复用？
7. Mellum 的特殊 Attention、MoE、RoPE 和权重格式需要修改哪些模块？
8. 在不改变冻结环境的前提下，如何完成 Mellum 的最小化适配？

本阶段的最终目标不是立即写出 Mellum 模型，而是通过源码调查形成有证据支持的模型适配方案。

---

# 2. 当前工程环境与目录约定

## 2.1 原始 ATOM 与工作副本

当前容器中存在腾讯预置的原始 ATOM：

```text
/app/ATOM
```

这是腾讯提供的参考源码，只允许阅读和分析，**绝对不允许修改**。

我们已经完整复制了一份 ATOM：

```text
/workspace/ATOM
```

这是本项目唯一允许修改的 ATOM 工程目录。

当前工作副本的 Git 信息：

```text
Commit:
26f23a0bdbf50b9010c07ee3a71171189723e767

Tag:
v0.1.4

Branch:
feature/mellum-support
```

原始 ATOM 和工作副本在建立时均处于干净状态，工作副本保留了原始 Git 历史。

必须遵守：

- `/app/ATOM`：只读参考，禁止修改。
- `/workspace/ATOM`：开发和修改的唯一 ATOM 工程。
- 不得在 `/app/ATOM` 上运行会产生源码修改的操作。
- 不得执行 Git Pull、Reset、Checkout 等改变原始版本的操作。
- 不得切换工作副本到其他 ATOM Commit。
- 保留原始 Git 历史和后续完整开发提交记录。

当前 ATOM 使用 Python Editable Install 安装：

```text
Editable project location: /app/ATOM
```

因此，直接运行系统中的 `atom` 命令，默认使用的是原始源码。

为了让 Python 使用我们的工作副本，已经建立专用启动脚本：

```text
/workspace/scripts/atom
```

脚本内容：

```bash
#!/usr/bin/env bash

set -e

export PYTHONPATH="/workspace/ATOM${PYTHONPATH:+:$PYTHONPATH}"

exec /opt/venv/bin/python \
    -m atom.entrypoints.openai.api_server "$@"
```

后续所有 ATOM 服务启动均应通过：

```bash
/workspace/scripts/atom
```

不得直接使用默认的 `atom` 命令启动我们的开发版本。

## 2.2 其他关键目录

| 路径 | 用途 | 权限要求 |
|---|---|---|
| `/app/ATOM` | 腾讯原始 ATOM | 只读 |
| `/workspace/ATOM` | ATOM 工作副本 | 可修改 |
| `/app/aiter-test/aiter` | 腾讯预装 AITER | 只读 |
| `/opt/venv` | Python 虚拟环境 | 不修改 |
| `/opt/rocm` | ROCm 软件栈 | 不修改 |
| `/data/models` | 模型权重 | 只读 |
| `/data/datasets` | 评测数据 | 只读 |
| `/workspace/report` | 技术分析文档 | 可修改 |
| `/workspace/experiments` | 实验配置 | 可修改 |
| `/workspace/results` | 实验结果 | 可修改 |
| `/workspace/logs` | 实验日志 | 可修改 |

当前 Python 环境：

```text
Python: 3.12.3

PyTorch:
2.10.0+rocm7.2.4.git3d3aa833

HIP:
7.2.53211

Transformers:
5.2.0

ATOM:
0.1.4

ATOM Commit:
26f23a0bdbf50b9010c07ee3a71171189723e767
```

AITER 实际导入位置：

```text
/app/aiter-test/aiter
```

GPU：

```text
GPU 0: AMD Instinct MI308X
GPU 1: AMD Instinct MI308X
```

## 2.3 开发辅助工具

当前已经安装并验证：

| 工具 | 版本 | 用途 |
|---|---:|---|
| tree | 2.1.1 | 查看目录结构 |
| ripgrep | 14.1.0 | 搜索源码 |
| fd-find | 9.0.0 | 文件查找 |
| jq | 1.7 | JSON 分析 |
| tmux | 3.4 | 终端会话管理 |
| htop | 3.3.0 | 进程监控 |
| git | 已安装 | 版本管理 |

Codex CLI 已安装，版本为 0.167.0。

当前远程网络通过 Windows 本地代理和 SSH RemoteForward 提供。远程代理入口：

```text
http://127.0.0.1:11080
```

Codex 已经能够正常使用。

---

# 3. Mellum 2.1 模型背景

本节提供后续源码分析必须掌握的模型架构信息。

实际模型目录：

```text
/data/models/Mellum2.1-12B-A2.5B-Thinking
```

配置文件：

```text
config.json
```

权重索引：

```text
model.safetensors.index.json
```

模型权重由五个 Safetensors 分片组成，目录大小约 23 GB。

## 3.1 模型基本架构

| 参数 | 配置 |
|---|---:|
| Model Type | `mellum` |
| Architecture | `MellumForCausalLM` |
| 总参数量 | 12B |
| 激活参数量 | 2.5B |
| Transformer Layers | 28 |
| Hidden Size | 2304 |
| Intermediate Size | 7168 |
| Vocabulary Size | 98304 |
| Precision | BF16 |
| Max Context Length | 131072 |
| 原生上下文长度 | 8192 |

模型为 Decoder-only MoE Transformer。

## 3.2 Attention 结构

| 配置 | 数值 |
|---|---:|
| Query Heads | 32 |
| KV Heads | 4 |
| Head Dimension | 128 |
| QK-Norm | 启用 |
| RoPE Theta | 500000 |
| Sliding Window | 1024 |

每四层中：

```text
S S S F
```

其中：

- S：Sliding Attention。
- F：Full Attention。

28 层中：

- Sliding Attention：21 层。
- Full Attention：7 层。

两种 Attention 的 RoPE 参数不同：

**Sliding Attention：**

```json
{
  "rope_type": "default",
  "rope_theta": 500000.0
}
```

**Full Attention：**

```json
{
  "rope_type": "yarn",
  "rope_theta": 500000.0,
  "factor": 16.0,
  "original_max_position_embeddings": 8192,
  "beta_fast": 32.0,
  "beta_slow": 1.0,
  "attention_factor": 1.2772588722239782
}
```

因此，不能直接将所有 Attention 层视为同一种实现。

## 3.3 MoE 结构

| 参数 | 数值 |
|---|---:|
| Experts | 64 |
| Activated Experts | 8 |
| Expert Intermediate Size | 896 |
| Hidden Size | 2304 |
| Norm Top-K Probability | true |
| Shared Experts | 无 |

28 层均使用 Sparse MLP。

需要注意，普通 `intermediate_size=7168` 与 `moe_intermediate_size=896` 是不同的参数。

## 3.4 Thinking 与 Tool Calling

模型提供：

```text
chat_template.jinja
```

模板包含：

```text
<|im_start|>
<|im_end|>
<think>
</think>
<tools>
<tool_call>
<tool_response>
```

支持 Thinking Mode 和 Tool Calling。

模型官方 README 表明 Mellum 2.1 相比 Mellum 2 的主要变化在后训练，尤其是强化学习，而模型架构保持不变。

考核说明书指出模型具有单层 MTP Head，但官方 README 对 vLLM 中 Mellum MTP 支持的状态描述为仍在准备中。

MTP 的实际权重和 ATOM 接入情况尚未确认，后续需要独立调查。

---

# 4. 当前已完成的前期验证

## 4.1 Transformers 支持情况

当前 Transformers 版本：

```text
5.2.0
```

实际模型配置：

```json
{
  "model_type": "mellum",
  "architectures": ["MellumForCausalLM"]
}
```

已完成以下测试。

**AutoTokenizer：成功。**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=False
)
```

结果：

```text
Tokenizer class: TokenizersBackend
Vocab size: 98304
Chat template available: True
```

**文本编码解码：成功。**

测试文本：

```python
text = "def add(a, b):\n    return a + b"
```

编码后能够正确解码：

```text
Round trip: True
```

**Chat Template：基础验证成功。**

Thinking 开启和关闭时均能正确渲染模板。

**AutoConfig：失败。**

```text
ValueError:

The checkpoint you are trying to load has model type
`mellum` but Transformers does not recognize this architecture.
```

模型配置记录：

```json
"transformers_version": "5.8.1"
```

但当前环境是 Transformers 5.2.0。

结论：

- 当前 Transformers 可以加载 Tokenizer。
- 当前 Transformers 可以处理 Chat Template。
- 当前 Transformers 不支持通过标准 AutoConfig 注册路径识别 Mellum。
- 尚未通过 Transformers 完整加载 Mellum 模型。

禁止通过升级 Transformers 解决这一问题，因为考核明确要求冻结环境。

**本项目的最终推理引擎必须是 ATOM，而不是 Transformers。**

## 4.2 官方 Baseline

当前文件：

```text
/data/datasets/baseline.json
```

出题方 Baseline 配置：

```text
1 GPU
BF16
Eager
Compile Level 0
BF16 KV Cache
MTP Off
SWA On
```

考核说明书进一步指出，Baseline 采用：

- PyTorch 循环实现 MoE；
- 带显式 Mask 的 SDPA；
- 单卡 Eager；
- Compile Level 0；
- 不开启 MTP。

场景 A：

| 指标 | 数值 |
|---|---:|
| Requests | 512 |
| Concurrency | 32 |
| Output Tokens | 131072 |
| Total Time | 205.351 s |
| Throughput | 638.283 tokens/s/GPU |
| TTFT p99 | 5.010 s |
| TPOT p50 | 38.951 ms |
| TPOT p99 | 50.144 ms |

场景 C：

| 指标 | 数值 |
|---|---:|
| Requests | 200 |
| Arrival Rate | 0.30 requests/s/GPU |
| Output Tokens | 51200 |
| Total Time | 671.772 s |
| Throughput | 76.216 tokens/s/GPU |
| TTFT p99 | 0.994 s |
| TPOT p50 | 36.290 ms |
| TPOT p99 | 41.265 ms |

当前 `baseline.json` 没有场景 B 的基线数据。

HumanEval 原始结果为 3/164，但出题方明确标记为 Rejected，原因是没有使用正确的 Chat Template 和 Thinking Mode，不能作为有效正确性参考。

## 4.3 其他已发现的问题

1. AITER 导入时出现 OPUS GEMM 仅支持 gfx950、当前 MI308X 为 gfx942 的警告。尚未确认实际 Mellum 推理路径是否调用该模块。
2. APT 软件源与容器预装 RCCL、OpenSSH 版本存在差异。已避免执行全局依赖修复。
3. MTP 权重是否包含在 Mellum 模型目录中尚未确认。
4. 尚未加载 Mellum 完整模型权重。
5. 尚未启动 Mellum 推理服务。

---

# 5. 当前阶段的核心研究任务

当前 Codex 的首要任务是对 ATOM 进行深入源码分析。

**只读调查是本阶段的主要工作方式。允许在 `/workspace` 中新增分析文档，但暂时不允许修改 ATOM 功能代码。**

## 5.1 任务一：理解 ATOM 工程结构

研究入口：

```text
/workspace/ATOM
```

重点查看：

```text
atom/
docs/
models/
recipes/
scripts/
tests/
tools/
```

需要回答：

1. ATOM 的主要模块如何组织？
2. API Server 如何启动？
3. Engine 和 Model Runner 如何协作？
4. 模型从启动到执行推理的整体流程是什么？
5. Attention、MoE、KV Cache、Scheduler 分别位于哪些模块？
6. ATOM 与 AITER 的调用关系是什么？

优先阅读：

```text
CLAUDE.md
README.md
docs/architecture_guide.md
docs/model_support_guide.md
docs/model_run_guide.md
docs/model_ops_guide.md
```

尤其是 `model_support_guide.md`，需要仔细理解其规定的模型适配规范。

不要只输出目录树，需要结合实际代码梳理模块职责和调用关系。

## 5.2 任务二：分析模型注册和配置解析机制

重点调查：

- Model Registry；
- Config 解析；
- 模型类注册；
- 模型架构识别；
- 模型实例化；
- 模型权重加载；
- 是否依赖 Transformers AutoConfig；
- 是否使用自定义 Config 和模型注册机制。

需要明确回答：

**当前 Transformers 5.2.0 不支持 Mellum 的情况下，ATOM 是否能够独立解析 Mellum 配置并构建模型？**

不要根据猜测回答，必须找到实际调用代码。

输出需要包含：

- 核心文件路径；
- 类名和函数名；
- 调用链；
- 关键代码位置；
- 需要修改或扩展的接口。

## 5.3 任务三：寻找 Mellum 和 Baseline 的已有实现

这是当前最高优先级的调查之一。

首先确认当前 ATOM 工作副本中是否已经包含 Mellum 适配。

搜索：

```bash
rg -n -i "mellum|MellumForCausalLM" \
    /workspace/ATOM \
    --glob '*.py' \
    --glob '*.md' \
    --glob '*.json'
```

但不能仅凭没有 Mellum 字符串就认定不存在相关实现。

还需要调查：

- 是否存在通用模型注册入口；
- 是否存在未使用 Mellum 名称的相似实现；
- 是否存在专门的模型 Config；
- 是否有针对 Mellum 的脚本或补丁；
- 是否有 Baseline 相关的测试或实验记录。

尤其需要确认：

**腾讯公布的 Baseline 性能数据所对应的朴素 Mellum 适配实现，是否已经包含在当前容器或 ATOM 仓库中？**

需要区分以下情况：

A. 已经存在完整 Mellum 实现。

B. 只存在部分 Mellum 适配代码。

C. 没有 Mellum 代码，但已有可以复用的基础模块。

D. 当前仓库完全没有相关实现。

如果无法找到 Baseline 原始实现，必须明确记录调查范围和证据，不能凭空推断。

## 5.4 任务四：分析 Qwen3-MoE 参考实现

主要参考：

```text
atom/models/qwen3_moe.py
atom/model_config/qwen3_5_moe.py
atom/model_ops/moe.py
atom/model_loader/loader.py
atom/model_loader/weight_utils.py
```

具体路径需要以实际源码为准，不要假设文件名一定完全一致。

重点研究：

**模型结构**

- Model 类；
- Decoder Layer；
- Embedding；
- Final Norm；
- LM Head。

**Attention**

- QKV Projection；
- GQA；
- QK-Norm；
- RoPE；
- Attention Kernel；
- KV Cache。

**MoE**

- Router；
- Top-K；
- Expert；
- Token Dispatch；
- Expert GEMM；
- Expert Aggregation。

**权重加载**

- Safetensors；
- 参数名称映射；
- 权重切分和合并；
- Expert 权重组织；
- 权重加载接口。

最终需要判断 Qwen3-MoE 哪些模块可以作为 Mellum 的直接参考，哪些需要调整。

## 5.5 任务五：分析 SWA 参考实现

考核建议参考 GPT-OSS / MiMo。

主要调查：

```text
atom/models/gpt_oss.py
atom/models/mimo_v2_flash.py
```

以及相关 Attention 和 KV Cache 模块。

重点研究：

- Sliding Window Attention 的实现位置；
- 层类型如何配置；
- Sliding Window 如何传入 Attention Kernel；
- KV Cache 如何管理；
- Full Attention 和 SWA 如何共存；
- RoPE 配置如何区分。

特别需要对照 Mellum 的：

```text
SWA / Full Attention = 3 : 1
Sliding Window = 1024
Full Attention RoPE = YaRN
Sliding Attention RoPE = Default
```

确认现有实现能否直接复用。

## 5.6 任务六：分析 MTP 参考实现

考核建议参考：

```text
atom/models/deepseek_mtp.py
atom/models/mimo_v2_flash_mtp.py
atom/models/qwen3_5_mtp.py
```

具体以仓库实际内容为准。

重点研究：

- MTP Head 如何构建；
- Draft 参数如何加载；
- Target 与 Draft 的关系；
- Speculative Decoding 执行流程；
- 接受率和验证逻辑；
- 当前 ATOM 的 MTP 框架是否要求模型具备特殊接口。

同时检查 Mellum 的 Safetensors Index，确认 MTP 相关权重是否存在。

MTP 不属于 Step 0 的必要功能，当前只做技术调查，不急于实现。

---

# 6. Mellum 与 ATOM 的模块映射分析

完成上述任务后，需要形成一张详细的适配映射表。

建议采用以下格式：

| Mellum 模块 | Mellum 需求 | ATOM 参考实现 | 复用程度 | 适配工作 |
|---|---|---|---|---|
| Config | model_type=mellum | 待分析 | 待判断 | 配置注册 |
| Embedding | Vocab 98304 | 待分析 | 待判断 | 权重映射 |
| Attention | GQA、QK-Norm | Qwen3-MoE | 待判断 | 结构适配 |
| RoPE | Default / YaRN | Qwen3 / GPT-OSS / MiMo | 待判断 | 双配置处理 |
| SWA | 1024 Window | GPT-OSS / MiMo | 待判断 | 层类型及 KV Cache |
| Full Attention | YaRN | 现有 Full Attention | 待判断 | 长上下文配置 |
| MoE | 64 Experts、Top-8 | Qwen3-MoE | 待判断 | Router / Expert |
| LM Head | 不共享 Embedding | 现有模型实现 | 待判断 | 权重加载 |
| MTP | 单层 MTP Head | DeepSeek / MiMo | 待判断 | 后续调查 |

注意：在完成源码阅读之前，不要提前填写“可以直接复用”或“必须重新实现”。

所有判断都应附有具体代码依据。

---

# 7. Baseline 实现调查与复现计划

这一项需要单独形成调查结论。

## 7.1 已知 Baseline 特征

```text
1 GPU
BF16
Eager
Compile Level 0
BF16 KV Cache
MTP Off
SWA On
PyTorch Loop MoE
Explicit-mask SDPA
```

官方性能数据：

```text
Scenario A:
638.283 tokens/s/GPU

Scenario C:
76.216 tokens/s/GPU
```

## 7.2 需要调查的内容

1. ATOM 中是否有与 Baseline 对应的模型实现？
2. 是否有对应启动脚本？
3. 是否存在 PyTorch 循环 MoE？
4. 是否存在显式 Mask SDPA？
5. 是否存在 Mellum 专用权重映射？
6. 是否存在出题方提供的参考输出或测试代码？
7. 是否有其他目录保存了 Baseline 相关实现？

需要注意：

当前已知 `/data/datasets/baseline.json` 保存了性能数据，但不能仅凭它的存在推断原始实现代码也在容器中。

## 7.3 复现原则

如果找到完整 Baseline 实现：

- 优先考虑复用；
- 先核对启动参数；
- 确认代码与官方描述一致；
- 再制定复现方案。

如果没有找到：

- 记录已经检查的目录和搜索依据；
- 以 Qwen3-MoE 为参考；
- 制定最小化 Mellum 适配方案；
- 不自行编造官方 Baseline 的实现细节。

当前阶段暂时不运行完整性能 Benchmark。

---

# 8. Codex 当前阶段的工作边界

本节非常重要，必须严格遵守。

## 8.1 允许执行的操作

- 阅读 `/app/ATOM` 原始源码。
- 阅读 `/workspace/ATOM` 工作副本。
- 搜索源码和配置。
- 阅读模型权重索引。
- 分析 Python 类和函数调用关系。
- 查看 Git 历史和版本。
- 运行不修改环境的轻量级 Python 检查。
- 阅读现有单元测试。
- 编写源码分析报告。
- 在 `/workspace/report` 中创建和更新 Markdown 文档。
- 在 `/workspace` 中保存研究脚本和分析结果。

## 8.2 当前禁止执行的操作

**禁止修改以下目录：**

```text
/app/ATOM
/app/aiter-test/aiter
/opt/venv
/opt/rocm
/data/models
/data/datasets
```

**暂时禁止以下操作：**

- 修改 ATOM 功能代码；
- 修改 AITER 源码；
- 修改 PyTorch / Transformers / ROCm 版本；
- 执行 pip upgrade；
- 执行 apt upgrade；
- 执行 apt --fix-broken install；
- 切换 ATOM Commit；
- Git Reset；
- Git Pull；
- 删除或覆盖原始实验数据；
- 启动大规模 GPU Benchmark；
- 修改评测参数；
- 未经确认加载完整 Mellum 权重。

特别注意：

`/workspace/ATOM` 虽然允许修改，但当前阶段的目标是只读分析。不要因为 Codex 发现了潜在问题就直接修复。

## 8.3 不要过度搜索

ATOM 是一个较完整的推理引擎，源码中包含大量与本次任务无关的功能。

不要：

- 一次性读取整个仓库的全部 Python 文件；
- 无差别搜索所有底层 Kernel；
- 对 `/app` 或整个文件系统进行大规模递归搜索；
- 在不了解调用关系的情况下阅读大量孤立函数。

应当采用：

**工程结构 → 启动入口 → 模型注册 → Config → Model → Weight Loader → Attention / MoE → 参考实现**

逐步深入。

优先研究与 Mellum 适配直接相关的路径。

---

# 9. 当前阶段的输出要求

本阶段需要形成一份正式的源码分析文档：

```text
/workspace/report/D1_atom_architecture_and_model_support.md
```

这份文档是内部工程研发记录，供开发者和后续 Codex 反复阅读使用，不是腾讯要求的最终交付报告。

## 9.1 文档必须包含的章节

### 第一章：ATOM 工程结构与整体架构

- 项目目录结构；
- 核心模块职责；
- Server 启动流程；
- 推理执行总体流程；
- ATOM 与 AITER 的关系。

### 第二章：模型注册与配置解析

- Config 加载；
- Model Registry；
- 模型类查找；
- 模型实例化；
- Transformers 依赖关系；
- Mellum 注册所需接口。

### 第三章：权重加载机制

- Safetensors 加载；
- 参数映射；
- 权重转换；
- Expert 权重组织；
- Embedding 和 LM Head。

### 第四章：Qwen3-MoE 参考实现

- 模型结构；
- Attention；
- MoE；
- Kernel 调用；
- 权重加载。

### 第五章：SWA / Full Attention 参考实现

- GPT-OSS / MiMo；
- Sliding Window；
- KV Cache；
- RoPE；
- 不同层的 Attention 配置。

### 第六章：MTP 参考实现

- DeepSeek / MiMo；
- Draft Head；
- Speculative Decoding；
- Mellum MTP 兼容性。

### 第七章：Mellum 现有支持情况

明确回答：

- 是否存在完整 Mellum 实现；
- 是否存在部分适配；
- 是否存在官方 Baseline 实现；
- 哪些代码可以复用；
- 哪些能力尚不存在。

### 第八章：Mellum Step 0 适配设计

形成：

- 模型模块映射表；
- 文件级修改清单；
- 适配顺序；
- 正确性验证方案；
- 潜在风险。

### 第九章：待确认问题与后续实施计划

列出尚未解决的问题、优先级和建议的下一步实验。

## 9.2 源码引用要求

这是本次研究最重要的质量要求之一。

每个关键结论必须提供具体源码证据。

建议采用：

```text
文件：atom/models/qwen3_moe.py

类：Qwen3MoeForCausalLM

方法：load_weights()

作用：
负责将 Hugging Face Safetensors 参数映射到 ATOM 模型参数。

相关调用：
...
```

实际文件名、类名和方法名必须通过源码核对，不允许根据文档或印象编造。

如果适合，可以提供：

- 文件路径；
- 类名；
- 函数名；
- 关键代码片段；
- 行号；
- 调用关系。

对于关键调用链，应当明确：

```text
入口函数
  → 配置解析
  → 模型注册
  → 模型构造
  → 权重加载
  → Forward
  → GPU Kernel
```

需要区分：

- 源码直接证明的事实；
- 根据多个模块推导出的设计理解；
- 尚未通过运行验证的假设。

## 9.3 分析深度要求

不要仅仅输出：

“这个文件负责 Attention。”

而应该回答：

1. 该模块的输入是什么？
2. 输出是什么？
3. 在整个推理流程中由谁调用？
4. 它依赖哪些其他模块？
5. 内部关键数据结构是什么？
6. 哪些部分与 Mellum 对应？
7. 是否存在可以直接复用的实现？
8. 如果不能直接复用，具体差异在哪里？

分析需要服务于后续实际代码实现。

---

# 10. Codex 的执行方式

本阶段建议分批执行，不要试图一次性完成整个仓库分析。

## Phase 1：工程概览

先完成：

- 阅读 CLAUDE.md；
- 阅读 Model Support Guide；
- 阅读 Architecture Guide；
- 分析目录结构；
- 定位启动入口。

输出第一批分析结论。

## Phase 2：模型加载链路

重点追踪：

- Config；
- Registry；
- Model；
- Loader；
- Model Runner。

形成从模型路径到模型执行的调用链。

## Phase 3：参考模型分析

分别研究：

- Qwen3-MoE；
- GPT-OSS / MiMo；
- DeepSeek / MiMo MTP。

不要把所有模型实现混在一起分析。

## Phase 4：Baseline 和 Mellum 支持调查

重点搜索：

- Mellum；
- 模型配置；
- 朴素 MoE；
- SDPA；
- SWA；
- 权重映射；
- 启动脚本。

明确是否存在可复用的官方实现。

## Phase 5：适配方案设计

综合前面结果，形成 Mellum Step 0 的完整适配方案。

完成后先向项目负责人提交分析结论，不要直接开始修改代码。

---

# 11. 最终完成标准

只有同时满足以下条件，才认为当前 ATOM 调研阶段完成：

1. 已经理解 ATOM 的整体架构和主要模块。
2. 已经明确模型启动、注册、配置解析和权重加载流程。
3. 已经找到 Qwen3-MoE 参考实现。
4. 已经找到 SWA 和 MTP 参考实现。
5. 已经调查 Mellum 是否存在预置支持。
6. 已经调查出题方 Baseline 实现是否存在。
7. 已经建立 Mellum 与 ATOM 的模块映射关系。
8. 已经形成具体的文件级修改清单。
9. 已经设计 Step 0 的正确性验证方案。
10. 所有关键结论均具有源码依据。
11. 完整分析报告已保存到指定目录。
12. 工作副本和原始 ATOM 环境均未受到不必要的修改。

---

# 12. 给 Codex 的第一项具体任务

请从现在开始执行本指南。

**第一轮任务：只完成 Phase 1，不要尝试一次性完成全部研究。**

具体要求：

1. 阅读 `/workspace/ATOM/CLAUDE.md`。
2. 阅读 `/workspace/ATOM/docs/model_support_guide.md`。
3. 阅读 `/workspace/ATOM/docs/architecture_guide.md`。
4. 检查 ATOM 的主要目录结构。
5. 找到 API Server 的启动入口。
6. 找到模型配置、注册、加载和模型实现的主要目录。
7. 结合代码建立 ATOM 的初步架构图。
8. 形成第一批源码分析结果。

第一轮允许创建或更新：

```text
/workspace/report/D1_atom_architecture_and_model_support.md
```

但不要修改任何 ATOM 功能代码。

完成后，请输出：

- 已阅读的文档；
- ATOM 的整体模块划分；
- 核心入口及调用关系；
- 模型适配涉及的主要目录；
- 对 Model Support Guide 的理解；
- 下一轮应该深入阅读的文件及原因；
- 当前尚未确认的问题。

**不要在第一轮就给出未经源码验证的 Mellum 适配结论。**

我们的目标是让每一轮源码分析都建立在上一轮已经确认的事实之上，最终形成一份能够真正指导工程实现的技术方案。

---

## 最后：项目研发原则

请始终遵守以下原则：

1. 以 `/app/ATOM` 为原始参考，以 `/workspace/ATOM` 为唯一开发副本。
2. 以腾讯提供的实际环境和考核说明书为准，不擅自更换软件版本。
3. 优先复用 ATOM 已有模型实现和 AITER Kernel。
4. 先分析调用关系，再决定代码修改位置。
5. 先完成 Step 0 正确性，再开展性能优化。
6. 不为了追求性能而牺牲模型输出正确性。
7. 每个优化点必须有独立开关和消融能力。
8. 不把未经验证的假设写成实验结论。
9. 保留完整 Git 历史和实验记录。
10. 所有代码修改、实验结果和项目文档均保存在 `/workspace`。

**本阶段最重要的事情不是写代码，而是建立对 ATOM 的准确理解，并找到 Mellum 适配的最短、最可靠实现路径。**