# HF参考环境建立记录

建立日期：2026-11-10；更新：2026-11-11。当前状态：用户在普通终端成功安装5.10.1，真实配置解析与模型类导入通过；尚未执行模型forward或GPU推理。

## 边界与布局

用户确认允许独立新版HF仅用于正确性参考，ATOM开发/优化保持冻结（见[决策06](../../docs/plan/02_decisions.md)）。

- `/workspace/venvs/hf-reference`：新建标准venv，默认不启用system-site-packages。
- 本地site-packages添加`frozen_rocm_dependencies.pth`，追加`/opt/venv/lib/python3.12/site-packages`为只读共享依赖路径；本地包优先，不处理原环境的editable .pth。
- 仅在独立环境安装`transformers==5.10.1`，使用`--no-deps --ignore-installed`，防止pip卸载共享旧包或安装其他PyTorch。依赖兼容仍需安装后核查；不宣称绝对环境完全独立。
- 共享ROCm PyTorch实际导入版本`2.10.0+rocm7.2.4.git3d3aa833`，路径位于`/opt/venv`；分发元数据版本带`.lw`，两种版本串并不完全相同，检查时保留原值。
- 未加载权重、未构造模型、未启动GPU。

## 可复现入口

[安装脚本](../../scripts/setup_hf_reference.py) · [环境检查](../../scripts/check_hf_reference.py)

```bash
/opt/venv/bin/python /workspace/scripts/setup_hf_reference.py --prepare-only
/opt/venv/bin/python /workspace/scripts/setup_hf_reference.py
/workspace/venvs/hf-reference/bin/python /workspace/scripts/check_hf_reference.py
```

安装脚本也接受`--wheel /path/to/official.whl`，用于之后提供经过核验的官方离线包；本轮未取得wheel。新版安装后检查要求实际导入位于参考venv且版本精确5.10.1，再离线解析真实配置、导入Mellum类；不构造12B模型。检查通过也只代表配置和导入，不等于推理正确。

## 实际结果

1. venv创建成功。
2. pip dry-run与实际安装均失败：`No matching distribution found`，候选列表为空。
3. 独立Python访问PyPI元数据失败：`URLError ... Operation not permitted`。网页工具可读取[PyPI官方5.10.1元数据](https://pypi.org/pypi/transformers/5.10.1/json)，故不能将本机pip失败解释为该版本未发布。
4. 当前未安装参考环境自己的Transformers，导入会落到共享5.2.0。环境检查按预期返回退出码2、status=incomplete，不把fallback误判为可用参考。
5. torch来自原ROCm环境；`torch.cuda.is_initialized()`为false。

原始输出：[install.log](../../results/hf_reference_environment/install.log)、[environment_check.json](../../results/hf_reference_environment/environment_check.json)。没有通过升级权限或更换连接绕过网络限制。

## 下一步

取得官方固定版本安装包（及确有缺失的依赖），校验来源/hash；只写参考环境。安装后检查依赖要求、Mellum配置和类导入，再进行小尺寸CPU模型forward/cache检查。之后才加载真实checkpoint、短输入GPU生成与16k等验证。尚未执行这些步骤，当前不能产生可信参考输出。

## 安装完成与复核（2026-11-11）

用户在普通容器终端执行安装脚本成功。已复核保存的environment_check.json：新版5.10.1来自参考venv，真实MellumConfig解析和MellumForCausalLM导入通过，CUDA未初始化。2026-11-10的网络失败保留为历史记录，不再是当前阻碍。

进一步读取Transformers分发元数据，使用packaging.Requirement核对所有当前环境适用的必需依赖，未发现缺失或版本不符（不含可选extras）。原`/opt/venv`实际导入仍为Transformers5.2.0。检查没有构造模型或加载权重。

下一步为小尺寸CPU模型forward/cache/窗口验证；通过后再进行真实checkpoint GPU参考推理。当前不能把配置和导入通过等同于实际checkpoint可运行或适配正确。

## CPU执行进展（2026-11-11）

[小模型验证](02_hf_mellum_cpu.md)已24/24通过，含forward/cache/generate、窗口和加载转换。真实checkpoint/GPU仍未执行。
