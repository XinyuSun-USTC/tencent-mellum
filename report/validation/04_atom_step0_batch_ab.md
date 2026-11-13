# ATOM Step0 第一批：配置接入与参考比较协议

日期：2026-11-11。状态：A已完成CPU配置接入与验证；B已完成初始参考artifact导出/比较器。尚未实现ATOM Mellum模型类、注册执行模型或GPU推理。

## 改动与依据

- `ATOM/atom/model_config/mellum.py:MellumConfig`：使用冻结HF5.2的初始化契约，保留model_type/architectures、explicit head_dim、layer_types/mlp_layer_types及nested RoPE；深拷贝输入，防止基类from_dict注入runtime字段污染调用者字典；校验维度、分组、expert top-k、窗口与RoPE，当前拒绝dense/MTP结构。
- `ATOM/atom/config.py:get_hf_config`：增加model_type=mellum窄分支，直接MellumConfig.from_dict；其他分支不改。不注册/升级第三方AutoConfig，不改checkpoint身份。
- `ATOM/tests/test_mellum_config.py`：真实配置/roundtrip、嵌套输入不变、非法输入及实际get_hf_config函数分支回归。入口函数使用真实源码AST和真实HF类，隔离其他引擎依赖；这不等同于完整Config/Runner初始化已通过。
- `scripts/export_mellum_config.py`：不同解释器导出同一模型的语义字段。
- `scripts/export_hf_mellum_reference.py`：将已有、加载审计及cache通过的HF GPU结果转换为schema v1；不重跑GPU。输出inputs.json、reference.json及完整末prompt位置logits副本，保留来源、hash与环境。
- `scripts/compare_mellum_results.py`：离线核对配置hash、样例集合、精确输入/位置/模板/生成设置、greedy token和完整末位置logits；检查shape、finite、文件hash并报告首个token分歧、max abs/RMS和top1。默认数值容差0，之后BF16容差需明确选择和记录，不以默认宽容掩盖差异。
- `scripts/tests/test_compare_mellum_results.py`：利用真实参考artifact测试正常比较及输入/token/数值/NaN/hash/样例集合/配置差异拒绝。

新配置字段默认值参考官方Mellum，文件只包含配置桥接，不含HF模型数学实现。AITER和原ATOM/冻结依赖未改。

## 执行结果

[汇总](../../results/atom_step0_batch_ab/summary.json)

| 验证 | 结果 |
|---|---|
| 新增配置测试 | 12通过 |
| 32个真实配置语义字段：冻结ATOM类 vs 官方HF5.10.1 | 全部一致，忽略版本记录、dtype统一为字符串 |
| 比较器测试 | 14通过 |
| 既有量化/配置测试 | 46通过 |
| 既有Kimi配置测试 | 收集阶段mock缺is_vllm；原始/app/ATOM同样失败，非本次新增分支回归 |
| HF参考自身严格比较 | 两条记录通过，token一致、logits误差0；仅证明导出比较链路，不是ATOM结果 |
| GPU与环境 | 本批未启动GPU；CUDA未初始化，冻结Transformers仍5.2.0 |

配置测试最初发现HF基类from_dict会改写输入字典并注入attn/experts implementation字段，已通过局部深拷贝修正，回归通过。未修复无关Kimi测试mock。

## 复现命令

```bash
/opt/venv/bin/python -m pytest /workspace/ATOM/tests/test_mellum_config.py -q
/opt/venv/bin/python -m pytest /workspace/ATOM/tests/test_quant_config.py -q
/opt/venv/bin/python -m pytest /workspace/scripts/tests/test_compare_mellum_results.py -q
/opt/venv/bin/python /workspace/scripts/export_mellum_config.py --backend atom --output /tmp/atom_mellum_fields.json
/workspace/venvs/hf-reference/bin/python /workspace/scripts/export_mellum_config.py --backend hf --output /tmp/hf_mellum_fields.json
```

当前已生成[inputs.json](../../results/atom_step0_batch_ab/hf_reference/inputs.json)、[reference.json](../../results/atom_step0_batch_ab/hf_reference/reference.json)、[自身比较结果](../../results/atom_step0_batch_ab/reference_self_compare.json)。参考来源为20261111T044314965290Z真实HF GPU run。导出器要求新输出目录，不覆盖已有结果。

## 协议与限制

schema_version=1，每条包含name、rendered_prompt、input_ids、attention_mask、position_ids、generation_settings、enable_thinking、generated_ids，以及prefill_logits路径/hash与logits_semantics。inputs.json不含生成答案，用于后续ATOM固定输入。参考是每条最后prompt位置的全词表logits，不是全部位置/全部生成步logits。

本次位置是全1 mask短样例的绝对位置；以后padding/分块/教师强制应显式提供位置并扩展协议，不能依赖此初版推导处理所有场景。B已完成最初固定输入、prefill和生成token比较；逐层输出与teacher-forcing导出留到模型对齐批次D实施，不宣称已经支持。当前没有ATOM candidate artifact，不能称HF/ATOM已一致。

## 下一批

进入C：独立MellumAttention/Decoder/Model/ForCausalLM、执行类注册、packed/expert加载映射与覆盖审计。RoPE参数桥接和逐层窗口必须在实现时验证；先以本批inputs和HF参考建立短prefill闭环，随后cache、窗口与长输入。GPU任务在有设备的普通终端执行，工具负责先完成源码/CPU检查。

## 提交状态

本批ATOM三个文件已暂存，提交因Git未配置作者姓名/邮箱失败；未伪造身份、未改全局配置。待用户提供身份后完成独立提交。workspace脚本/文档位于ATOM仓库外，当前仅在workspace保留，最终交付打包需一并纳入。
