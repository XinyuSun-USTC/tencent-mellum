# 34 强制同历史32位置精确与正常生成候选回归

## 已完成的cached闭环

原始运行`results/mellum_decode_validation/20261112T164434545261Z`，ATOM`results/atom_mellum_gpu/20261112T164434590610Z`。四请求共32个强制历史位置（4prefill+28decode）的完整原始logits全部精确一致，first_nonexact=None，历史/position及5631/5631加载审计通过。

实际组合：separate QKV、torch MoE、torch router、CPU逆频率prefill、参考Q/K和K写回、读取实际分页缓存的SDPA decode。参考仍是HF5.10.1/grouped_mm/SDPA cached。覆盖三个短输入和跨block/1024滑窗边界的8位置；不将强制采样输出相同当作证据，精确结论来自强制前的完整logits。

这结束了本组固定历史数值定位，但不等于正常生成、长prompt、多请求、服务或正式Step0通过，也不是每个物理KV槽的全量审计。默认模型路径暂不切换。

## 正常生成预设

`run_mellum_validation.py --correctness-candidate`显式展开同一组后端设置，使用check_atom_mellum现有普通engine.generate，不传forced-decode；ignore_eos=False。运行器检查worker实际配置、sampling_mode=normal且无强制计划。冲突的显式QKV/prefill参数拒绝，不隐式混合候选。输入prompt大于1024在GPU前拒绝，因为当前prefill候选仍是受限路径，不能直接跑expanded中的长prompt。

预设只是组合入口，没有新增数学改动，没有改变不带预设时的默认行为，也没有把诊断受限路径冒称完整生产后端。

## 采样前logits被动观察

为避免正常生成若分叉又单独GPU取证，候选自动启用dump-generation-logits：

- HF generate使用return_dict_in_generate/output_logits，保存每步原始输出logits；HF可能以FP32保存BF16值，比较时提升到FP32，不以存储dtype差异伪造数值失败。
- ATOM `FreeGenerationProbe`只保存实际输入/position和原始logits，不接收计划、不覆盖logits、不改变token或EOS。
- 异步调度可能有末尾多余前向，原始文件可保留，但candidate列表只包含实际返回token对应的位置，不计末尾额外计算。
- `compare_mellum_generation.py`只比较相同历史下的位置，包括首个分叉位置，不比较分叉后不同历史；保存双方选择是否为各自最大值，便于区分精确并列与数值漂移。

generation_comparison.json保存逐步结果，summary的generation_logit_summary给出数量和首差。如果token/prefill相同但cached逐步logits不同，不能保持tokens_and_logits_exact，降为review_required。强制数据和正常生成数据不混记。

正常运行器也改用已测试的worker死亡检测、15秒心跳、单阶段300秒默认超时及自有进程组清理；ATOM普通token_mismatch退出码1仍保留完整比较，失败不会被当作进程成功或无限等待。

## 检查与下一步

- 模型相关CPU46项通过，新增被动观察不修改原始logits验证。
- 运行器/比较器/进程控制CPU9项通过，覆盖候选显式选择与长prompt提前拒绝、同历史截止/并列判读、日志哈希拒绝和失败监测。
- 独立HF CPU3项通过，包括generate启用原始logits捕获前后greedy token不变，使用实际API。
- prepare-only产物`results/mellum_validation/20261112T172319358182Z`，两条短样例各max_new_tokens64。
- 语法与diff检查通过；新候选正常生成GPU尚未运行。

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite short --correctness-candidate
```

两条各最多64新token，可能EOS提前结束；核验实际token数量、prefill以及同历史cached逐步logits，不额外保存大层状态。通过后才增加窗口内多样生成及补长prompt/多请求/服务能力，正式验收缺项继续披露，不进入Step1。

## 正常生成首轮失败与下一步定位

实际运行 `results/mellum_validation/20261112T173200224689Z`，ATOM `results/atom_mellum_gpu/20261112T173225862933Z`：两条prefill完整logits精确，arithmetic的64个输出token精确；python_add共同前缀37个token，第37号位置（从0计）分叉。128个位置103个相同，只有1/2序列完整相同。分叉后历史不同，不能把其logits差异作为同输入证据。

被动采样前观察共比较102个同历史位置，其中100个logits非精确；两个用例各自的第0步精确，最早差异在python_add第1步，max_abs=0.75。此前固定四历史32位置精确结论保留，但其覆盖不能推广到其他prompt。当前正常生成未通过，不能进入Step1，也不能据此认定单纯采样并列规则错误。

源码检查：HF正常生成入口为`scripts/export_mellum_suite_hf.py:main`的`model.generate`；此前手动cached入口为`export_mellum_decode_reference.py:export_case`。两者都传attention_mask且使用cache。HF本地`generation/utils.py:prepare_inputs_for_generation`负责切分输入/position，`_prepare_cache_for_generation`准备DynamicCache；`models/mellum/modeling_mellum.py:MellumModel.forward`在无cache时也建立带config的DynamicCache。仅阅读入口不足以确定根因；不能断言generate有问题或需要更换参考基线。

下一实验只取python_add（24token）的prefill与第一decode，各保存embedding、28层输出、finalnorm及decode的Q/K/V/attention输出，寻找最早差异。新增`run_mellum_decode_validation.py --case`允许从普通失败运行选一个已确认prefill logits精确的样例，仅限fresh两步layer-study；原四用例验证入口不变。该实验使用手动HF cached路径，需将其原始logits与本次保存的HF generate/logits及ATOM正常logits交叉比对：若手动路径复现差异，则按第一不同层定位；若手动路径精确而正常路径不同，继续核对实际调用/缓存/执行差别，不改验收条件。

CPU选择/历史恢复3项通过；prepare-only产物`results/mellum_decode_validation/20261112T173629407575Z`。未运行GPU，未修改模型功能代码。

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_decode_validation.py --prefill-run /workspace/results/mellum_validation/20261112T173200224689Z --case python_add --layer-study --steps 2 --reference-kv-rope --sdpa-decode --torch-router
```

## python_add两步逐层结果及跨运行核验

运行`results/mellum_decode_validation/20261112T181028200695Z`，ATOM`results/atom_mellum_gpu/20261112T181040490903Z`：两步原始logits、60个层状态、112个attention分项均无首差（first_nonexact_*为None）。

将其与173200普通生成的被动logits交叉比较：step0四方均精确；step1 **ATOM正常=ATOM强制=HF手动cached**，HF generate与HF手动cached最大绝对差0.75。四方该步argmax仍是233，step0均23。因此当前证据将首差定位到HF generate与手动forward的调用路径之间，尚未确定原因，不能称HF generate错误，亦不能替换参考路径以宣布通过。

新增HF-only `scripts/diagnose_mellum_hf_generation.py`：同一个官方模型实例内比较manual cached与正常generate两步，保存实际forward输入/position、stride、cache层类型、scalar kwargs、层状态和attention Q/K/V/context。只观察，不修改模型计算或缓存。语法检查通过，GPU待用户执行。命令：

```bash
/workspace/venvs/hf-reference/bin/python /workspace/scripts/diagnose_mellum_hf_generation.py --run /workspace/results/mellum_validation/20261112T173200224689Z
```

## HF generate的隐式专家后端切换：源码定位与候选修复

同实例运行`results/mellum_hf_generation/20261112T184743771424Z`复现step0精确、step1差0.75，输入token/position/attention_mask及DynamicCache类型符合预期。step1第0层Q/K/V/context精确，第0层输出已差0.000244140625，之后传播。原结果first_nonexact_state按文件名排序先列final_norm，不能据此说final_norm是最早差异；已将脚本排序修为执行顺序。

找到确定的调用语义差别：HF5.10.1本地`transformers/generation/utils.py:GenerationMixin._sample`在每次decode forward外包裹`self._optimize_model_for_decode()`；同文件`GenerationMixin._optimize_model_for_decode`在原experts_implementation为grouped_mm且device非CPU时调用`set_experts_implementation('batched_mm')`，finally恢复原值。prefill在该context外。因而运行结束才读取config会错过临时切换。此前手动cached全程grouped_mm并非正常generate的专家执行序列。此发现解释了为何手动验证精确并不确保generate精确，但新增路径GPU是否修复全部差异仍待验证。

HF实际专家数学见`transformers/integrations/moe.py:batched_mm_experts_forward`及`_batched_linear`：按token/topk顺序repeat_interleave，索引选中专家权重，torch.bmm做gate/up与down投影，BF16激活及路由乘积，reshape+sum。新增`ATOM/atom/model_config/mellum_experts.py:MellumTorchExperts.forward(...,batched=True)`同序实现；`ATOM/atom/models/mellum.py:MellumSparseMoeBlock.forward`仅在显式MELLUM_HF_GENERATION_EXPERTS=1时按实际context.is_prefill选择prefill grouped/decode batched。权重加载不变，不按token数猜阶段，不依赖新版HF包运行ATOM。

`check_atom_mellum.py --hf-generation-experts`显式启用；普通correctness-candidate预设增加该参数，并校验worker环境。手动强制历史运行器不启用，仍对齐其grouped_mm参考。HF普通导出增加每次实际forward的experts配置记录，候选运行器要求prefill grouped_mm、后续batched_mm，与返回token数量对应。没有修改HF generate或放宽正确性条件，默认ATOM配置未切换；batched路径会复制选中专家权重，当前仅受限单请求候选使用，不当作性能优化。

验证：独立HF CPU4项通过，含1/3token BF16与官方batched_mm专家逐值精确；ATOM专家/模型CPU19项通过；普通比较CPU1项、运行器CPU5项通过；语法检查通过。prepare-only `results/mellum_validation/20261112T191136196304Z`成功。未启动GPU。

下一步重跑同一short普通生成候选，检查实际后端记录、完整token和同历史逐步logits：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite short --correctness-candidate
```

## 普通short生成128位置精确通过

运行`results/mellum_validation/20261112T193650834240Z`，ATOM`results/atom_mellum_gpu/20261112T193716221730Z`，状态tokens_and_logits_exact。python_add与arithmetic各实际64输出token：128/128 token一致、2/2完整序列一致；同历史采样前完整logits共128向量全部逐值精确，nonexact_positions=0。两个独立prefill向量max_abs=0。实际sampling_mode=normal、ignore_eos=False，无强制计划，加载审计5631/5631，missing/unexpected均空。

HF每个样例64次实际forward记录：第0次grouped_mm，其余63次batched_mm。此结果验证了隐式专家切换定位与新增候选路径在这两条正常生成上的修复效果。原固定历史grouped_mm验证仍作为独立诊断记录保留。

这不是完整Step0验收：仅两个短prompt，单请求TP1/eager/BF16，prefill候选受限输入≤1024；尚需多样生成、长prompt、并发/服务及题目正式正确性指标验证，不进入性能优化。

下一组固定输入`results/mellum_validation_inputs/normal_generation_10_v1/inputs.json`，来自既有精确prefill10样例，保留原始IDs、mask、position和thinking元数据，只把每条max_new_tokens改为64：六条代码/数学/中文等短prompt、raw16/raw17，以及window1023/window1024。最多640输出token，EOS正常停止；边界样例decode会继续越过1024窗口。沿用同一数学候选及完整逐步logits精确比较，不扩大当前prefill支持边界。prepare-only `results/mellum_validation/20261112T204933481396Z`通过，未启动GPU。

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite diagnostic --inputs /workspace/results/mellum_validation_inputs/normal_generation_10_v1/inputs.json --correctness-candidate
```

后续10条多样普通生成结果及两处并列最大值采样调度修复，统一记录于[报告35](35_greedy_tie_break_and_sampling_dispatch.md)。
