# 38 Graph路由组合与QK/RoPE融合批次

独立工作区/workspace/tencent-mellum-fusion，基于3f81b6d。延续先优化、批量筛选、最终集中正确性/正式消融的决策；不启动HF/HumanEval/公开200或完整ABC。

## 上批证据与本批假设

优化批次20261116T120257485855Z_graph、20261116T120304274036Z_kernels，266项产物哈希全部核验，A64/C20全部请求成功。GPU0清理版620.9951→Graph943.2600tokens/s（+51.9%），C20 TPOT p99 58.0375→22.8411ms，TTFT p99 1.02605→0.94012秒；有实际1/2/4/8/32图回放记录。GPU1clean610.9447、unpacked593.3894、native-router628.0816tokens/s。保留Packed；原生路由+2.8%需在Graph上重测，不能叠乘推算。

## GPU0：Graph路由ABBA

```bash
/opt/venv/bin/python /workspace/tencent-mellum-fusion/scripts/run_mellum_fusion_sprint.py --track routing --budget-minutes 50
```

依次Graph+Torch路由A、Graph+AITER路由B、B重复、A重复。每次A64/并发32与C20/0.3req/s，256输出；重复独立启动服务，均为同卡、同输入。通过目录_repeat保留重复，实际profile为原profile。端口8251/8926。执行预算50分钟，CPU准备另计；预计约15–25分钟，实际取决于首次编译/捕获。

## GPU1：两级融合筛选

```bash
/opt/venv/bin/python /workspace/tencent-mellum-fusion/scripts/run_mellum_fusion_sprint.py --track fusion --budget-minutes 45
```

先在GPU1运行一次小型内核冒烟，然后Graph基线→Graph+仅RoPE融合→Graph+QK-Norm/RoPE融合。端口8151/8726。预计约10–20分钟，执行预算45分钟，CPU准备另计。

- mellum_fused_rope._norm_rope：一个Triton launch同时处理Q/K各token/head；THD BF16、D128，支持packed投影的跨token stride，输出独立连续张量，不原地改输入。按每个head做FP32 RMS归一化和权重，再回BF16；RoPE沿用各层CPU公式导出的FP32 inverse与attention_factor，保留BF16 cos/sin、乘法及加法边界，禁用FMA。不宣称逐位相等。
- 使用冻结Triton的portable libdevice.sin/cos，避免把长位置的范围处理交给快速近似三角函数；传入运行时positions，不按prompt或请求内容选择路径。
- 仅RoPE版保留现有AITER Q/K norm。QK融合版通过MellumAttention.forward默认关闭开关跳过两处独立norm；canonical q_norm/k_norm参数及加载键保留，以普通tuple引用交给直接KV写入路径，无新注册参数。
- mellum_direct_kv.install_direct_kv_rope选择融合函数，然后继续原AITER reshape_and_cache、varlen/paged Attention；不改缓存布局/槽位、窗口或专家计算。
- 独立开关MELLUM_FUSED_ROPE与MELLUM_FUSED_QK_ROPE，profile分别sprint_graph_rope/sprint_graph_qk；都保留Torch router，避免混合归因。

## 必要检查与边界

37项CPU检查通过；RoPE与QK融合两种constexpr专门化已在无GPU条件下成功编译为gfx942，缓存仅写/tmp。离线编译不能代替GPU运行。

check_mellum_fused_rope.py对1/33/1024token、非连续QKV、关闭/开启norm、跨16k位置做有限性、shape、输入不变及宽松数学冒烟。相对L2>5%会停止融合track，防止索引/旋转符号/漏norm等明显实现错误；这是新内核局部冒烟，不是模型验收门槛，更不改题目正确性要求。其结果不证明HumanEval通过。

每profile加载5631项权重、请求完成/256输出/真实batch、113norm/28packed投影、GPU驻留、原生路由、实际graph replay及真实融合调用均需核验；失败保留继续其它独立候选，绝不静默fallback。内核冒烟失败则不再花时间启动融合网络任务，GPU0独立继续。

源码/输入指纹覆盖新内核、模型、runner、实验脚本及kernel-smoke脚本。结果/workspace/results/mellum_fusion_sprint。运行期间不改工作区；工具无GPU设备，需终端执行。

## 后续

原生路由在Graph中无收益就停止晋级；融合有收益再与最快路由组合。随后在独立分支合入已测prefix机制（当前direct-KV不支持cached/chunked，不能直接开开关），必要时有界调度参数筛选。最后固定组合统一跑题目正确性/完整ABC/消融，不为每个微小变更重复HF。
