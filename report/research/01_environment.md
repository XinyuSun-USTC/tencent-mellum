# 环境与开发入口

状态：2026-11-10 已有调查证据的主题整理；本次迁移没有重新运行技术检查。
源码基线：`26f23a0bdbf50b9010c07ee3a71171189723e767`。ATOM 相对路径基于 `/workspace/ATOM`。
区分源码事实、源码推断、待运行验证与未找到；静态风险不是已复现故障。

本页提取已有记录，原文中的“本轮”指原检查批次；残留数字节号指[归档原文](../archive/initial_research/01_environment_and_preparation_original.md)，不表示工作日或新的执行计划。当前方案统一见[Step 0](../../docs/plan/01_step0.md)。

## 远程实验环境搭建

### 开发环境与服务器连接

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

### 容器目录与存储约束

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

### 远程开发工具

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

## 实验硬件与软件环境检查

### GPU 硬件检查

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

### ROCm 硬件状态检查

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

### 核心软件版本

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

### AITER 架构警告

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

### Linux 辅助开发工具

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

## 当前状态补充

上述版本和操作为历史检查记录，不代表每次阅读都重新检测。当前禁止升级冻结依赖；后续 HF 参考运行方式尚待核验。默认 editable 指向 `/app/ATOM`，开发入口为 `/workspace/scripts/atom`。模型/数据及原始仓库只读。通用工具安装记录不构成后续更换推理依赖的授权。
