# ATOM Mellum 模型接入：CPU检查与首次GPU入口

日期：2026-11-11。当前状态：真实ATOM单卡加载、warmup与两条短greedy/HF对比通过。下文保留实现与预检历史；尚未完成Step0完整验收。

## 实现

- `ATOM/atom/models/mellum.py`新增MellumAttention、MellumDecoderLayer、MellumModel和MellumForCausalLM。参考当前Qwen3-MoE接口独立实现骨干，不修改已有compile装饰模型；仅直接复用Qwen3MoeSparseMoeBlock、通用linear/norm/FusedMoE/Attention。
- 保持explicit D128、Q32/KV4、per-head QK-Norm、每层SWA/Full与nested Default/YaRN选择、residual契约、独立LM head。按mlp_layer_types已验证全sparse构建，不依赖decoder_sparse_step。首版明确仅TP1。
- `ATOM/atom/model_config/mellum_utils.py`为纯辅助模块：HF attention_factor转换为AITER attn_factor比例，避免重复默认mscale；源权重名→目标参数/QKV shard/expert shard映射。
- `ATOM/atom/model_engine/model_runner.py`增加执行类注册；load_model返回后、warmup前，对提供validate_loaded_weights的模型调用验证。其他模型没有该方法时不改变路径；load_dummy跳过审计。
- Mellum对参数weight_loader包装，只有原加载回调成功返回后记录(target, shard, expert)；从实际index要求所有源tensor对应的槽位覆盖，无静默缺Q/K/V或expert。`MELLUM_LOAD_AUDIT_PATH`指定保存JSON，缺项则启动失败。

加载回调覆盖不是数值全量核验。原回调若内部静默忽略、重排后数值错误或核函数出错，仍需GPU输出/逐层对照；目前没有宣称完成真实参数样本值检查或所有packed元素验证。

## CPU验证

[汇总与源码hash](../../results/atom_step0_batch_c/summary.json) · [RoPE结果](../../results/atom_step0_batch_c/rope_cpu.json)

- 配置/模型专项合计21通过（配置12，模型9），既有量化配置46通过。
- 实际28层构造函数在CPU替身算子下检查S/S/S/F、window与head_dim传递；Attention明确q_size4096/kv_size512。替身只证明接线，不证明真实GPU类可构造。
- 5631个实际index键唯一映射到5631个加载槽，其中expert5376、QKV84；expert目标与源FusedMoE.make_expert_params_mapping方法输出一致。此为映射/覆盖检查，不是合成数值packing已完成。
- 缺失expert审计会失败；成功回调记录，抛错回调不记录；普通参数通过真实CPUcopy测试。
- 源码检查注册入口、无compile装饰；py_compile、git diff --check通过。
- AITER RoPE采用源文件AST提取实际类与纯数学函数，避免导入GPU依赖；官方HF5.10.1独立参考。实际D128、theta500000、factor16/original8192，位置0/1/1023/1024/8191/8192/8193/16383，Default和YaRN的FP32 cos/sin比较最大误差均0。

RoPE仅证明FP32 cache构造与参数映射，不证明BF16 GPU旋转及Attention输出。当前kernel窗口边界风险仍待真实长输入验证，没有修改通用后端来假装修复。

复现：

```bash
/opt/venv/bin/python -m pytest /workspace/ATOM/tests/test_mellum_config.py /workspace/ATOM/tests/test_mellum_model.py -q
/opt/venv/bin/python -m pytest /workspace/ATOM/tests/test_quant_config.py -q
/workspace/venvs/hf-reference/bin/python /workspace/scripts/check_mellum_rope_cpu.py
```

## 首次GPU执行入口

[check_atom_mellum.py](../../scripts/check_atom_mellum.py)通过EngineArgs/LLMEngine真实入口构建，使用原始token列表，避免重复模板/tokenize。Python spawn；环境强制关闭加载线程池、QK/rope/cache融合、allreduce RMSNorm融合和unified attention；单卡BF16、level0/eager、prefix cache/chunked prefill关闭、max_model_len256、max_num_seqs1、gpu_memory_utilization0.35。

先验证5631项回调覆盖，再分别生成已有两个样例并与HF保存的16token精确比较。保存loading_audit.json和results.json。当前输出为token smoke记录，不是schema v1的完整logits candidate；prefill logits导出/teacher-forcing仍留给后续对齐诊断。

当前工具没有ROCm设备；[本次预检](../../results/atom_mellum_gpu/20261111T085248888768Z/results.json)以blocked_no_gpu退出，未加载权重。普通容器终端执行：

```bash
/opt/venv/bin/python /workspace/scripts/check_atom_mellum.py
```

默认HIP_VISIBLE_DEVICES=0，可用--gpu 1选择另一物理卡。脚本要求该卡至少80GiB空闲，用于首次独立调试；不修改其他任务或杀进程。实际会触发模型加载、GPU warmup和生成，错误需保存后诊断，不盲重跑。日志最后给出独立结果目录。

当前尚无一键服务验收；本入口为首次离线引擎验证，服务路径在后续E完成。Git作者身份仍待用户提供，已有暂存的A批文件保留，本批未伪造身份提交。

## 放行与后续

下一步需要读取普通终端运行结果，先确认真实加载审计、构造/warmup和短token输出。失败先定位模型接口/加载/核函数；若token不一致则进入相同前缀、分层/算子诊断。短token通过也不代表SWA1024、16k、批处理或正式门槛通过。原始冻结依赖与/app/ATOM未改。

## 首个真实ATOM短输入闭环通过（2026-11-11）

用户在普通容器终端执行后，已直接复核[结果JSON](../../results/atom_mellum_gpu/20261111T085551407745Z/results.json)和[加载审计](../../results/atom_mellum_gpu/20261111T085551407745Z/loading_audit.json)。

- status=short_greedy_passed；冻结/opt/venv解释器、Transformers5.2.0、原ROCm PyTorch，单卡。
- 5631个源tensor对应5631个成功加载回调槽，missing/unexpected均为空。它证明加载覆盖闭合，不等于所有真实元素已逐一比较。
- 用户日志显示构造、warmup、KV绑定、prefill/decode和引擎正常关闭完成。
- python_add输入24token、arithmetic输入18token，各生成16token；全部与官方HF保存的greedy token完全一致。
- 测试输入均未触及SWA1024边界。此结果不能证明跨窗/长输入、所有权重值、多请求或HTTP服务正确，更不代表题目200prompt/HumanEval/16k验收通过。

日志出现OPUS gfx950警告、模型type为空的配置警告和AITER类型提示等；本次没有因此发生执行失败，也未据此修改依赖/硬件目标。模型type警告的具体来源尚未定位，不将它当成Mellum身份被改写的结论；后续应按调用来源审计。

下一阶段：扩展64token回归、相同前缀teacher-forcing与必要层输出；优先验证真实1023/1024/1025窗口以及8192/16k位置，再做批处理与服务。是否需要具体修正取决于结果，不因两条通过而跳过验证。
