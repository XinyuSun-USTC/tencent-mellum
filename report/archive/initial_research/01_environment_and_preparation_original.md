# D0 阶段工作记录：实验环境准备与前期技术调研

**项目：** 腾讯 CSIG 云一计算加速团队校招实践考核  
**考核任务：** Mellum 2.1 on ATOM  
**日期：** 2026-11-10  
**阶段：** D0 — 环境准备、模型分析与性能基线核对

---

## 1. 项目背景与阶段目标

### 1.1 考核任务概述

本次实践考核要求在指定的 AMD ROCm 环境中，使用轻量级推理引擎 ATOM，使 JetBrains Mellum 2.1（12B MoE，2.5B 激活参数）能够正确运行，并在此基础上最多完成三个独立的性能优化点。

硬件资源为同一节点上的两张 AMD Instinct MI308X GPU，每张 GPU 配备 192 GB 显存，并通过 xGMI 直连。考核允许使用单卡，也允许采用 TP、DP、EP、Prefill/Decode 分离等双卡并行方式，但最终性能指标需要按照 GPU 数量进行归一化。

整个任务分为两个阶段：

- **Step 0：模型适配。** 在冻结的软件环境中完成 Mellum 2.1 在 ATOM 上的单卡正确推理。这是后续优化的基础，不计入三个优化点。
- **Step 1：性能优化。** 在 Step 0 的基础上，最多实施三个针对已识别瓶颈的独立优化点，每个优化点需要具有独立开关并能够进行消融实验。

本次考核不只是要求完成代码实现，还要求候选人理解陌生模型和推理引擎、定位真实性能瓶颈、合理分配时间，并通过实验数据证明优化有效。

### 1.2 考核评价与交付要求

根据题目说明，主要评价内容包括：

1. 三个规定负载场景下的推理性能。
2. 模型正确性，包括与 Hugging Face Transformers 参考输出的比较、HumanEval 和长上下文检查。
3. 优化点的实际收益及独立消融结果。
4. Profiling 证据和性能瓶颈分析。
5. 优化方向选择、时间分配及放弃某些方向的决策依据。
6. AI 工具的使用方式及对 AI 错误建议的识别能力。

最终交付包括：

- 代码仓库：一条命令启动 Server 的脚本，以及每个优化点的开关说明。
- 技术报告：不超过 4 页，包含 Profiling、正确性检查、优化收益和消融表。
- AI 使用说明：记录使用的工具、工具发挥的作用、错误建议及其发现方式。
- WORKLOG.md：按 D1–D7 记录每日工作、问题和关键决策，并保留完整 Git 提交历史。

此外，需要进行 20 分钟项目答辩，其中 10 分钟展示、10 分钟问答。

本文件属于项目内部研发记录，用于保存环境准备、前期分析和后续技术工作的上下文，不属于上述正式交付报告。

### 1.3 D0 阶段工作范围

本阶段主要完成以下工作：

1. 建立 Windows 到腾讯远程容器的开发连接。
2. 检查 GPU、ROCm、PyTorch 和 ATOM 环境。
3. 补充常用 Linux 开发工具。
4. 检查 Mellum 2.1 模型权重和配置文件。
5. 验证 Tokenizer、Chat Template 的基础功能。
6. 阅读模型 README，理解模型定位和特殊推理行为。
7. 核对出题方 Baseline 和评测数据。
8. 识别当前环境及模型适配中存在的潜在问题。
9. 制定 ATOM 源码分析与 Step 0 模型适配计划。

本阶段尚未开展 ATOM 源码修改、Mellum 模型适配和正式推理性能测试。

---

## 2. 远程实验环境搭建

### 2.1 开发环境与服务器连接

本地开发设备为 Windows 笔记本电脑，使用 VS Code 作为主要开发环境，远程连接方式为 SSH。

腾讯提供的服务器连接参数：

| 配置项 | 内容 |
|---|---|
| 服务器地址 | 43.180.254.191 |
| SSH 端口 | 2211 |
| 用户名 | root |
| 认证方式 | SSH 私钥 |
| 私钥文件 | sunxingyu1810_ed25519 |
| 本地公网出口 IP | 202.141.181.10 |
| 容器主机名 | sunxingyu1810 |

公网出口 IP 已由腾讯加入访问白名单。若更换网络导致公网出口 IP 发生变化，需要提前联系对接人更新白名单。

Windows 本地私钥保存位置：

```text
C:\Users\ASUS\.ssh\sunxingyu1810_ed25519
```

SSH 配置文件：

```text
C:\Users\ASUS\.ssh\config
```

当前配置：

```sshconfig
Host tencent-mellum
    HostName 43.180.254.191
    User root
    Port 2211
    IdentityFile C:/Users/ASUS/.ssh/sunxingyu1810_ed25519
    IdentitiesOnly yes
```

后续可以直接执行：

```powershell
ssh tencent-mellum
```

完成远程登录。

首次连接时，SSH 提示确认服务器 ED25519 Host Key，当前本地记录的指纹为：

```text
SHA256:OySrTjWNbb/ctSRNOP7PkLFKqvthdgWnAb6u/r26ru0
```

该指纹尚需通过腾讯对接人独立确认。

### 2.2 容器目录与存储约束

登录后进入独立容器，初始工作目录为 `/root`。

腾讯明确指定 `/workspace` 为唯一可写工作目录，且该目录内容在容器重启后保留。

主要目录：

| 路径 | 权限 | 用途 |
|---|---|---|
| `/workspace` | 可读写、持久化 | 项目代码、实验记录、结果 |
| `/data/models` | 只读 | 模型权重 |
| `/data/datasets` | 只读 | 评测数据 |
| 其他系统目录 | 需要根据实际挂载和权限确认 | 系统环境 |

实际检查：

```bash
ls /
ls -lah /workspace
```

`/workspace` 初始为空，后续所有项目文件均在该目录中组织。

腾讯对容器环境的说明：

- 登录后进入个人独立容器。
- `/workspace` 是唯一可写目录。
- 模型权重位于 `/data/models`。
- 评测数据位于 `/data/datasets`。
- 环境中的 ATOM、AITER、ROCm、PyTorch 等组件已经冻结，不允许升级或替换。

### 2.3 远程开发工具

Windows 环境中已经确认存在 OpenSSH 客户端：

```text
C:\Windows\System32\OpenSSH\ssh.exe
```

SSH 登录测试成功，能够进入：

```text
root@sunxingyu1810
```

后续计划使用 VS Code Remote - SSH 直接打开远程 `/workspace`，通过远程终端进行代码开发和实验。

需要注意：SSH 命令行登录已经验证成功；VS Code Remote - SSH 和 Codex 在远程环境中的具体使用状态，需要在后续单独确认。

---

## 3. 实验硬件与软件环境检查

### 3.1 GPU 硬件检查

首先执行腾讯要求的 GPU 数量检查：

```bash
python -c "import torch; print(torch.cuda.device_count())"
```

输出：

```text
2
```

进一步检查：

```python
import torch

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))
    print(
        f"Memory: "
        f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
    )
```

结果：

```text
PyTorch: 2.10.0+rocm7.2.4.git3d3aa833
HIP: 7.2.53211
CUDA available: True
GPU count: 2

GPU 0: AMD Instinct MI308X
Memory: 191.98 GB

GPU 1: AMD Instinct MI308X
Memory: 191.98 GB
```

两张 GPU 均正常识别，显存容量与题目要求的 192 GB 基本一致。

需要注意，ROCm 版本的 PyTorch 仍然使用 `torch.cuda` 接口，因此 `torch.cuda.is_available()` 返回 True 是正常现象，并不代表当前运行的是 NVIDIA CUDA。

### 3.2 ROCm 硬件状态检查

容器中可以使用：

```bash
amd-smi list
rocm-smi
```

`amd-smi list` 输出的 GPU 信息：

| GPU | BDF | UUID |
|---|---|---|
| GPU 0 | 0000:22:00.0 | 01ff74a2-0000-1000-8017-01e0705b15cf |
| GPU 1 | 0000:30:00.0 | 68ff74a2-0000-1000-80ed-c45f40163460 |

`rocm-smi` 检查结果：

| 指标 | GPU 0 | GPU 1 |
|---|---:|---:|
| 温度 | 42°C | 39°C |
| 功耗 | 162 W | 159 W |
| VRAM 使用率 | 0% | 0% |
| GPU 使用率 | 0% | 0% |
| Power Cap | 650 W | 650 W |

当前 GPU 基本空闲，没有发现明显硬件异常。

ROCm SMI 显示的节点编号分别为 5 和 3，而 PyTorch 使用的逻辑设备编号为 0 和 1，这是不同工具的编号体系。

### 3.3 核心软件版本

当前容器实际环境：

| 软件 | 实际版本或路径 |
|---|---|
| Ubuntu | 24.04.4 LTS |
| 系统架构 | x86_64 |
| Python | `/opt/venv/bin/python` |
| PyTorch | 2.10.0+rocm7.2.4.git3d3aa833 |
| HIP | 7.2.53211 |
| Transformers | 5.2.0 |
| ATOM | `/usr/local/bin/atom` |
| ROCm 工具 | `/opt/rocm/bin/amd-smi` |
| ROCm SMI | `/opt/rocm/bin/rocm-smi` |

ATOM 已经能够执行：

```bash
atom --help
```

并正常输出 OpenAI API Server 的命令行参数。

从帮助信息中确认当前 ATOM 命令支持以下配置：

- Tensor Parallel / Data Parallel。
- Expert Parallel。
- Prefix Caching。
- BF16 / FP8 KV Cache。
- Compile Level 0–3。
- MTP / EAGLE3。
- Torch Profiler。
- Chunked Prefill。
- Two-Batch Overlap。
- 在线量化。
- Server Host / Port。

这些能力与考核说明书中介绍的 ATOM 功能基本对应。

### 3.4 AITER 架构警告

执行：

```bash
atom --help
```

时出现以下警告：

```text
RuntimeWarning:

aiter.ops.opus (a16w16) is gfx950-only;
detected arch='gfx942'.

opus_gemm_* calls will raise RuntimeError at invocation.
```

当前 MI308X 被识别为 `gfx942`，而该 OPUS GEMM 模块只支持 `gfx950`。

目前仅确认该模块在导入时产生警告，尚未证明 ATOM 的实际 Mellum 推理路径会调用它。

因此，当前将其记录为待调查的兼容性问题，而不是认定整个 AITER 环境异常。

**不要根据警告提示设置 `GPU_ARCHS=gfx950`，也不要尝试替换或重新编译 AITER。**

### 3.5 Linux 辅助开发工具

容器初始提供的开发工具较少，检查结果：

| 工具 | 初始状态 |
|---|---|
| git | 已安装 |
| curl | 已安装 |
| wget | 已安装 |
| gcc | 已安装 |
| g++ | 已安装 |
| cmake | 已安装 |
| ninja | 已安装 |
| python | 已安装 |
| tree | 缺失 |
| jq | 缺失 |
| ripgrep | 缺失 |
| fd-find | 缺失 |
| bat | 缺失 |
| htop | 缺失 |
| tmux | 缺失 |

容器操作系统为 Ubuntu 24.04.4 LTS，根文件系统为可写 OverlayFS，`/workspace` 为独立可写 ext4 挂载。

#### APT 安装问题

执行：

```bash
apt-get update
```

成功获取约 41.6 MB 软件包索引。

但直接执行：

```bash
apt-get install -y tree jq ripgrep fd-find bat htop tmux
```

遇到多个未满足依赖问题，主要包括：

- bat 依赖 libgit2-1.7。
- htop 依赖 libnl-genl-3-200。
- jq 依赖对应版本的 libjq1。
- tmux 依赖 libutempter0。
- OpenSSH Client 与 Server 版本不一致。
- ROCm 的 RCCL 版本与当前 APT 候选版本不一致。

其中 ROCm 组件的版本情况：

| 软件包 | 已安装版本 | APT 候选版本 |
|---|---|---|
| rocm-hip | 7.2.4.70204-93~24.04 | 相同 |
| rccl | 2.27.7-29e1567b | 2.27.7.70204-93~24.04 |
| rccl-dev | 2.27.7-29e1567b | 2.27.7.70204-93~24.04 |

OpenSSH 组件也存在版本不一致：

| 软件包 | 已安装版本 |
|---|---|
| openssh-client | 1:9.6p1-3ubuntu13.16 |
| openssh-server | 1:9.6p1-3ubuntu13.20 |
| openssh-sftp-server | 1:9.6p1-3ubuntu13.20 |

执行：

```bash
dpkg --audit
```

没有输出，说明没有发现未完成配置的软件包。

由于全局依赖修复可能改变冻结的 ROCm 和 OpenSSH 环境，因此没有执行：

```bash
apt --fix-broken install
```

#### 辅助工具安装方式

最终采用：

```bash
apt-get download <package>
dpkg -i <package>.deb
```

逐个下载和安装辅助工具，避免触发 APT 的全局依赖解析。

目前已成功安装：

| 工具 | 版本 | 用途 |
|---|---:|---|
| tree | 2.1.1 | 目录结构查看 |
| ripgrep | 14.1.0 | 源码搜索 |
| fd-find | 9.0.0 | 文件查找 |
| jq | 1.7 | JSON 处理 |
| tmux | 3.4 | 终端会话管理 |
| htop | 3.3.0 | 进程监控 |

`bat` 暂未安装，不影响当前开发工作。

执行版本检查：

```bash
tree --version
rg --version
fdfind --version
jq --version
tmux -V
htop --version
```

全部已安装工具均能正常运行。

注意 Ubuntu 中 fd-find 的命令名称为 `fdfind`，后续可以通过 Shell alias 配置为 `fd`。

**后续原则：允许合理补充基础系统工具，但不升级、不替换、不修复性修改腾讯预置的 Python、ATOM、AITER、ROCm、HIP、PyTorch 和相关 GPU 计算依赖。**

---

## 4. Mellum 2.1 模型分析

### 4.1 模型基本信息

模型名称：

```text
Mellum2.1-12B-A2.5B-Thinking
```

模型目录：

```text
/data/models/Mellum2.1-12B-A2.5B-Thinking
```

官方模型仓库：

https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking

根据模型 README，Mellum 2.1 的主要架构与 Mellum 2 保持一致，主要变化来自后训练，特别是强化学习。

后训练涉及：

- 数学、竞赛编程和科学任务。
- 工具使用和软件工程任务。
- 在真实代码仓库环境中使用 Shell 和文件编辑工具。
- 根据测试结果对模型行为进行奖励。

因此，Mellum 2.1 是一个面向复杂代码和 Agentic 任务的 Thinking 模型。

### 4.2 Transformer 主干结构

根据模型 `config.json`：

| 配置 | 值 |
|---|---:|
| model_type | mellum |
| architectures | MellumForCausalLM |
| Hidden Size | 2304 |
| Layer 数量 | 28 |
| Intermediate Size | 7168 |
| Vocabulary Size | 98304 |
| Model dtype | bfloat16 |
| 最大上下文长度 | 131072 |
| Embedding 是否共享 | false |

模型采用 Decoder-only Transformer 架构。

### 4.3 Attention 结构

配置：

```json
{
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "sliding_window": 1024,
  "use_sliding_window": true
}
```

使用 GQA：

- 32 个 Query Heads。
- 4 个 KV Heads。
- 每个 KV Head 对应 8 个 Query Heads。

模型每四层采用 3:1 的 SWA / Full Attention 分布。

28 层的具体排列：

```text
Layer  0-3:   S S S F
Layer  4-7:   S S S F
Layer  8-11:  S S S F
Layer 12-15:  S S S F
Layer 16-19:  S S S F
Layer 20-23:  S S S F
Layer 24-27:  S S S F
```

其中：

- Sliding Attention：21 层。
- Full Attention：7 层。
- Sliding Window Size：1024。

后续适配必须正确处理不同层的 Attention 类型，不能将所有层统一视为普通全注意力。

### 4.4 RoPE 与长上下文

配置文件中的 RoPE 参数：

```json
"rope_parameters": {
  "full_attention": {
    "rope_type": "yarn",
    "rope_theta": 500000.0,
    "factor": 16.0,
    "original_max_position_embeddings": 8192,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "attention_factor": 1.2772588722239782
  },
  "sliding_attention": {
    "rope_type": "default",
    "rope_theta": 500000.0
  }
}
```

需要特别注意：

- Full Attention 使用 YaRN。
- Sliding Attention 使用 Default RoPE。
- 两种 Attention 的 RoPE 配置不同。
- 原生上下文长度为 8192。
- 最大上下文长度为 131072。

因此，后续模型适配必须确保 RoPE 行为与层类型对应。

尤其需要验证 16K 输入下的长上下文正确性。

### 4.5 MoE 结构

配置：

```json
{
  "num_experts": 64,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 896,
  "norm_topk_prob": true
}
```

每层采用 Sparse MLP：

- 64 个专家。
- 每个 Token 激活 8 个专家。
- Expert Intermediate Size 为 896。
- Top-K 路由概率需要归一化。

28 层均采用 Sparse MLP。

这是后续 MoE Kernel 适配和优化的重要依据。

需要区分普通 Transformer 的 `intermediate_size=7168` 与 MoE Expert 的 `moe_intermediate_size=896`。

### 4.6 MTP 与投机解码

考核说明书明确指出 Mellum 具有一个单层 MTP Head，可以用于投机解码。

ATOM 本身已经具备 MTP / EAGLE 投机解码框架。

但模型 README 又指出 Mellum 2.1 的 MTP Head 在 vLLM 中的支持仍在准备中。

目前尚不能确定：

- 当前预置权重是否包含可直接使用的 MTP 参数。
- 考核环境是否已经提供额外的 MTP 适配。
- ATOM 是否存在 Mellum 专用 MTP 实现。

因此，MTP 暂时作为候选优化方向，后续需要通过权重索引和 ATOM 源码进一步确认。

---

## 5. 模型权重与 Tokenizer 验证

### 5.1 权重目录结构

模型目录包含：

```text
Mellum2.1-12B-A2.5B-Thinking/
├── README.md
├── chat_template.jinja
├── config.json
├── generation_config.json
├── model-00001-of-00005.safetensors
├── model-00002-of-00005.safetensors
├── model-00003-of-00005.safetensors
├── model-00004-of-00005.safetensors
├── model-00005-of-00005.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

目录总大小约 23 GB。

权重由五个 Safetensors 分片构成，并提供索引文件。

从文件结构看，模型配置、权重分片、Tokenizer 和 Chat Template 文件均已准备。

注意：目前只完成文件存在性检查，尚未执行完整模型权重加载。

### 5.2 Tokenizer 加载验证

使用当前预装 Transformers 5.2.0：

```python
from transformers import AutoTokenizer

path = "/data/models/Mellum2.1-12B-A2.5B-Thinking"

tokenizer = AutoTokenizer.from_pretrained(
    path,
    local_files_only=True,
    trust_remote_code=False
)
```

加载成功。

结果：

```text
Tokenizer class:
transformers.tokenization_utils_tokenizers.TokenizersBackend

Vocab size: 98304

Chat template available: True
```

词表大小与模型配置一致。

加载过程中出现：

```text
You are using a model of type `mellum`
to instantiate a model of type ``.
```

该警告目前没有阻止 Tokenizer 加载，但说明需要进一步检查当前 Transformers 的配置分发机制。

### 5.3 文本编码解码测试

测试代码：

```python
text = "def add(a, b):\n    return a + b"

ids = tokenizer.encode(
    text,
    add_special_tokens=False
)

decoded = tokenizer.decode(
    ids,
    skip_special_tokens=False
)
```

结果：

```text
Original: 'def add(a, b):\n    return a + b'

Token IDs:
[910, 846, 42, 99, 46, 321, 800, 233,
 308, 587, 255, 99, 523, 321]

Token count: 14

Decoded:
'def add(a, b):\n    return a + b'

Round trip: True
```

说明普通文本的编码与解码能够正确工作。

### 5.4 Chat Template 检查

模板使用：

```text
<|im_start|>
<|im_end|>
```

组织消息，并包含：

```text
<think>
</think>
<tools>
<tool_call>
<tool_response>
```

等特殊结构。

支持 Thinking Mode 和 Tool Calling。

测试：

```python
messages = [
    {
        "role": "user",
        "content": "Write a Python function that adds two integers."
    }
]
```

Thinking 开启时，生成：

```text
<|im_start|>user
Write a Python function that adds two integers.<|im_end|>
<|im_start|>assistant
```

Thinking 关闭时，生成：

```text
<|im_start|>user
Write a Python function that adds two integers.<|im_end|>
<|im_start|>assistant
<think>

</think>
```

两种模式均能够正常渲染模板。

注意：使用当前 Transformers 5.2.0 的 TokenizersBackend 时，`apply_chat_template(..., tokenize=True)` 返回结构与传统的 Token ID 列表存在差异。直接对外层对象调用 `len()` 得到的 2 并不是实际 Token 数量。

实际输出的 Encoding 信息分别显示：

- Thinking=True：18 tokens。
- Thinking=False：24 tokens。

后续统计时需要读取实际的 `input_ids`，不能直接使用外层对象长度。

### 5.5 AutoConfig 兼容性检查

执行：

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    path,
    local_files_only=True,
    trust_remote_code=False
)
```

失败：

```text
ValueError:

The checkpoint you are trying to load has model type
`mellum` but Transformers does not recognize this architecture.
```

模型配置文件中记录：

```json
"transformers_version": "5.8.1"
```

而当前容器实际安装：

```text
Transformers: 5.2.0
```

目前确认：

- AutoTokenizer 可以加载。
- Chat Template 可以渲染。
- 普通文本编码解码正常。
- AutoConfig 无法识别 `model_type=mellum`。

需要强调：这并不意味着 ATOM 一定无法支持 Mellum。ATOM 可能使用独立的模型注册、配置解析和权重加载机制。

后续应优先研究 ATOM 自身的模型支持机制。

**禁止通过升级 Transformers 解决该问题，因为考核环境明确冻结了软件版本。**

---

## 6. 评测数据与官方 Baseline 分析

### 6.1 评测数据集

目录：

```text
/data/datasets/
├── HumanEval.jsonl
├── HumanEval.jsonl.gz
├── baseline.json
├── humaneval_bf16.json
├── scenario_a_prompts.jsonl
└── scenario_c_prompts.jsonl
```

实际统计：

| 数据集 | 样本数 |
|---|---:|
| Scenario A | 512 |
| Scenario C | 200 |
| HumanEval | 164 |

均已完成基本文件存在性检查。

### 6.2 官方 Baseline 配置

`baseline.json` 记录：

```json
{
  "config": "1 GPU, eager, compile level 0, bf16 KV, MTP off, SWA on",
  "num_gpus": 1,
  "precision": "bf16",
  "rate_per_gpu": 0.3
}
```

基线配置：

| 参数 | 配置 |
|---|---|
| GPU 数量 | 1 |
| 模型精度 | BF16 |
| KV Cache | BF16 |
| 执行模式 | Eager |
| Compile Level | 0 |
| MTP | Off |
| SWA | On |

所有最终加速比均相对于出题方单卡 Baseline，而不是相对于我们自己的 Step 0。

### 6.3 场景 A：高并发吞吐

官方数据：

| 指标 | Baseline |
|---|---:|
| 请求数量 | 512 |
| 并发数 | 32 |
| 输出 Token 数 | 131072 |
| 总耗时 | 205.351 s |
| 总输出吞吐 | 638.283 tokens/s |
| tokens/s/GPU | 638.283 |
| TTFT p99 | 5.010 s |
| TPOT p99 | 50.144 ms |
| TPOT p50 | 38.951 ms |

吞吐计算：

\[
\frac{512\times256}{205.351}
\approx638.283
\]

与 Baseline 文件记录一致。

### 6.4 场景 B：单流延迟

场景 B 的评测要求：

- 并发数为 1。
- 输入长度为 1024 Token。
- 输出长度为 256 Token。
- 请求数量为 50。

当前 `baseline.json` 没有提供场景 B 的实际基线数据，需要后续确认是否存在其他基线文件。

### 6.5 场景 C：Agent 风格混合负载

官方数据：

| 指标 | Baseline |
|---|---:|
| 请求数量 | 200 |
| 每卡请求速率 | 0.30 requests/s |
| 输出 Token 数 | 51200 |
| 总耗时 | 671.772 s |
| 总输出吞吐 | 76.216 tokens/s |
| tokens/s/GPU | 76.216 |
| TTFT p99 | 0.994 s |
| TPOT p99 | 41.265 ms |
| TPOT p50 | 36.290 ms |

吞吐计算：

\[
\frac{200\times256}{671.772}
\approx76.216
\]

与 Baseline 文件一致。

场景 C 的特点是：

- 泊松请求到达。
- 输入长度为 4K–16K。
- 请求共享前缀。
- 长 Prefill 会干扰正在执行的 Decode。

数据中存在：

```json
"shared_prefix_group": "dev-shared-prefix-0"
```

说明该场景设计了共享前缀。

后续需要结合 ATOM 的 Prefix Cache 实现和实际命中情况判断是否能够获得收益。

### 6.6 HumanEval 基线的特殊情况

`baseline.json` 中：

```json
"humaneval_passed_bf16": 3,
"humaneval_count_bf16": 164,
"humaneval_pass1_bf16_rejected": 0.018292682926829267
```

同时：

```json
"humaneval_pass1_bf16": null,
"humaneval_pass1_fp8": null
```

出题方明确注明：

```text
Rejected: raw /v1/completions without the chat template
or thinking mode.
Not comparable to the model card HumanEval+ 91.5.
```

即原始测试没有使用正确的 Chat Template 和 Thinking Mode，因此结果被拒绝。

不能将 3/164 作为有效模型能力基线，也不能直接与模型卡 HumanEval+ 91.5% 比较。

后续必须严格按照考核指定的正确性测试协议进行验证。

### 6.7 三个场景的初步分析

根据 Baseline：

- 场景 A 更关注高并发吞吐和每卡计算效率。
- 场景 B 更关注单请求延迟。
- 场景 C 更关注长 Prefill 与 Decode 干扰时的尾延迟。

初步值得研究的方向：

| 场景 | 候选优化方向 |
|---|---|
| A | MoE Kernel、Compile/Graph、FP8 |
| B | MTP、Kernel Launch、CPU 调度 |
| C | Prefix Cache、SWA KV Cache、Chunked Prefill、调度优化 |

以上仅为基于负载特点的初步假设，尚未通过 Profiling 验证。

---

## 7. 前期发现的问题与风险

### 7.1 Transformers 不支持 Mellum Config

当前 Transformers 5.2.0 无法通过 AutoConfig 识别 `model_type=mellum`。

下一步需要确认 ATOM 是否拥有独立的模型注册机制，以及它是否依赖 Transformers AutoConfig。

不能直接升级 Transformers。

### 7.2 AITER OPUS 架构警告

AITER 导入时检测到 MI308X 为 gfx942，而 OPUS GEMM 模块仅支持 gfx950。

目前尚未确认 Mellum 推理路径是否会调用该模块。

后续根据实际执行调用栈决定是否需要进一步处理。

### 7.3 MTP 适配状态不明确

考核说明书指出 Mellum 存在单层 MTP Head，但模型 README 说明其 vLLM MTP 支持仍在准备中。

需要通过 Safetensors 权重索引和 ATOM MTP 代码进一步确认。

### 7.4 HumanEval 基线无效

官方 Baseline 中的 3/164 结果因没有使用正确的对话模板和 Thinking Mode 而被拒绝。

后续必须独立验证正确性测试协议。

### 7.5 场景 B 基线缺失

当前 `baseline.json` 只有场景 A 和 C 数据。

需要进一步检查是否有其他评测文件提供场景 B 的基准数据。

### 7.6 APT 软件源与预置依赖存在版本差异

当前容器中的 RCCL 和 OpenSSH 版本与 APT 候选版本不完全一致。

因此，后续安装辅助工具时需要避免触发全局依赖修复。

目前已通过单独下载和安装辅助工具的方式完成必要的开发工具准备，没有执行全局升级或修复操作。

---

## 8. 后续 ATOM 源码分析与 Mellum 适配规划

D0 阶段结束后，下一阶段进入 ATOM 源码分析。

### 8.1 ATOM 工程结构分析

首先需要定位：

- ATOM 命令入口。
- 源码实际安装目录。
- Model Support Guide。
- CLAUDE.md。
- 模型注册与配置解析。
- 模型加载和权重映射。
- Attention、MoE、KV Cache 和 Scheduler 等关键模块。

### 8.2 Qwen3-MoE 参考实现

重点分析：

- 模型类定义。
- Config 解析。
- Attention 实现。
- MoE Router。
- Expert 权重加载。
- AITER Kernel 调用。

### 8.3 SWA 参考实现

阅读 GPT-OSS / MiMo 的相关实现，重点理解：

- Sliding Window Attention 的实现方式。
- 不同层的 Attention 配置。
- KV Cache 管理。
- Prefill 和 Decode 阶段的行为差异。

### 8.4 MTP 参考实现

阅读 DeepSeek / MiMo 的相关实现，确认：

- MTP Head 如何加载。
- Draft 和 Target 模型之间的关系。
- Speculative Decoding 的执行过程。
- Mellum 的 MTP 权重是否可以直接接入现有机制。

### 8.5 Mellum Step 0 适配

在源码调研基础上形成具体适配方案：

1. Mellum 模型注册。
2. 模型配置解析。
3. Safetensors 权重映射。
4. Embedding 和 LM Head。
5. GQA、QK-Norm 和 RoPE。
6. SWA / Full Attention。
7. MoE Router 和 Expert。
8. Chat Template 和服务接口。
9. 单卡 BF16 正确性验证。

优先实现最小化的单卡 BF16、Eager、Compile Level 0 基础推理路径，暂不引入 MTP、FP8 和双卡并行。

---

## 9. D0 阶段总结

本阶段完成了正式模型适配前的实验环境准备和基础资料分析。

首先，完成了 Windows 到腾讯远程容器的 SSH 连接，并确认两张 MI308X 均能被 ROCm 和 PyTorch 正常识别。容器中的 ATOM 命令行可以正常运行，模型权重和评测数据文件均已准备。

其次，完成了 Mellum 2.1 的模型结构分析，确认其采用 28 层 MoE Transformer，具有 64 个专家、Top-8 路由以及 SWA / Full Attention 混合结构。特别确认了两类 Attention 使用不同 RoPE 配置，这是后续模型适配需要重点处理的部分。

再次，完成了 Tokenizer 和 Chat Template 的基础验证。当前环境可以正常加载 Tokenizer，词表大小正确，普通文本编码解码一致，Thinking 模板也能够正常渲染。但 Transformers 5.2.0 尚不能通过 AutoConfig 识别 Mellum，这一兼容性问题需要在 ATOM 源码分析阶段进一步研究。

最后，核对了出题方提供的场景 A 和场景 C 性能基线，确认分别为 638.28 tokens/s/GPU 和 76.22 tokens/s/GPU，并识别了 HumanEval 原始结果无效、场景 B 基线缺失等问题。

**当前阶段最重要的成果是完成了实验资源和任务目标的摸底，明确了后续 Mellum 适配需要解决的核心问题，并建立了以实际验证为依据的后续研发路线。**

下一阶段将正式进入 ATOM 源码阅读，从模型注册、配置解析和 Qwen3-MoE 实现开始，逐步形成 Mellum 的最小化适配方案。

---

## 附录 A：当前环境关键路径

| 资源 | 路径 |
|---|---|
| 项目工作区 | `/workspace` |
| 模型目录 | `/data/models/Mellum2.1-12B-A2.5B-Thinking` |
| 模型配置 | `/data/models/Mellum2.1-12B-A2.5B-Thinking/config.json` |
| 权重索引 | `/data/models/Mellum2.1-12B-A2.5B-Thinking/model.safetensors.index.json` |
| Chat Template | `/data/models/Mellum2.1-12B-A2.5B-Thinking/chat_template.jinja` |
| 场景 A 数据 | `/data/datasets/scenario_a_prompts.jsonl` |
| 场景 C 数据 | `/data/datasets/scenario_c_prompts.jsonl` |
| HumanEval | `/data/datasets/HumanEval.jsonl` |
| 官方 Baseline | `/data/datasets/baseline.json` |
| ATOM 命令 | `/usr/local/bin/atom` |
| Python | `/opt/venv/bin/python` |

## 附录 B：当前环境保护原则

1. 不升级或替换 ATOM、AITER、ROCm、HIP、PyTorch。
2. 不修改腾讯指定的 Docker 环境和评测参数。
3. 不使用分配 GPU 之外的外部 GPU 资源。
4. 不修改 `/data/models` 和 `/data/datasets`。
5. 项目代码、实验日志和结果统一保存在 `/workspace`。
6. 所有优化点必须有独立开关和消融能力。
7. 保留完整 Git 提交历史，不进行最终 squash。
8. 重要实验必须记录执行配置、代码版本和结果。
9. AI 生成的代码和技术结论必须经过人工审查及实际验证。
10. 对尚未确认的环境和工具使用限制，先核实规则再执行。