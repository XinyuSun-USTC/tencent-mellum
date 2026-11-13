# 39 Step0阶段总结：Mellum在ATOM中的单卡适配

版本说明：本文工作树/HEAD叙述属于Git迁移前Step0收尾快照。当前已纳入总仓库，实际提交、标签及路径见[07仓库管理](../../docs/plan/07_project_repository_management.md)；原始运行与指纹保持不变。

## 阶段结论

**Step0适配开发完成，本地生成一致性验证通过，进入Step1。** 这是本项目的开发阶段结论，不代表已取得出题方正式验收结果。200条开发prompt的前64token完整序列一致率与token位置一致率均为100%；16k输入后的64token生成也与官方HF参考精确一致。尚未完成HumanEval完整答案的pass@1评估，开发集合是否符合出题方公开集口径仍待确认。缺少材料不能据此认定这些项目必然是隐藏测试。

本报告归纳最终实现、关键诊断与有效证据；实验01–38保留原编号和原始结果。复现配置以[基线说明](../../docs/plan/05_step0_baseline_and_reproduction.md)为主，不再从历史失败记录推导当前状态。题目依据见[要求摘要](../../docs/requirements/01_assessment_summary.md)。

## 目标、环境与范围

目标是在冻结AMD ROCm环境中，以最简单可行方式让Mellum2.1-12B-A2.5B-Thinking在ATOM正确单卡推理，为后续优化提供可比较版本，不刻意复刻出题方朴素Baseline。开发副本为`/workspace/ATOM`；`/app/ATOM`、AITER、ROCm/PyTorch原环境、模型与数据保持冻结边界。

- 执行环境：`/opt/venv/bin/python`，Transformers 5.2.0，ROCm PyTorch 2.10.0；GPU运行日志识别gfx942/MI308系列，单卡约192GiB。型号与版本以实际运行记录为证据，见[环境研究](../research/01_environment.md)。
- HF参考：独立`/workspace/venvs/hf-reference/bin/python`，Transformers 5.10.1，共用既有ROCm PyTorch。该环境仅用于参考，不替换ATOM冻结依赖。
- 模型：28层、S/S/S/F混合Attention、hidden_size=2304、Q32/KV4、head_dim=128、64专家/top8、专家中间维896、SWA1024；BF16、12,149,923,072参数。完整配置、分层RoPE与权重调查见[模型研究](../research/03_mellum_model.md)。
- 已验证组合：TP1/DP1、eager level0、BF16权重与KV、单请求逐条执行、完整未缓存prefill；关闭prefix caching、chunked prefill和MTP。

200条运行是200次顺序单请求，不是200请求并发；16k证据不能推广为所有长度、所有cache模式与所有服务负载均已验证。

## 工程实现

以下路径相对于`/workspace/ATOM`。符号及行为依据当前工作树，[文件指纹清单](39_step0_manifest.json)记录本次整理时的版本；行号可能随Step1变化，故主要引用类和函数。

| 环节 | 文件与符号 | 具体实现与依据 |
| --- | --- | --- |
| 配置识别 | `atom/model_config/mellum.py:MellumConfig`；`atom/config.py:get_hf_config`中的mellum分支 | 原Transformers 5.2.0缺少Mellum注册；新增局部配置类，读取真实model_type，不冒用Qwen身份 |
| 模型接入 | `atom/models/mellum.py:MellumAttention/MellumDecoderLayer/MellumModel/MellumForCausalLM`；`model_runner.py:support_model_arch_dict` | 新增独立模型文件，复用ATOM加载、调度、KV与输出头接口；TP非1显式拒绝 |
| 权重映射 | `atom/model_config/mellum_utils.py:mellum_load_target`；`MellumTorchExperts.weight_loader` | 将Q/K/V映射为qkv分片，将gate/up/down映射为w1/w3/w2与expert_id；目标槽包含参数、分片、专家标识 |
| 加载审计 | `MellumForCausalLM._install_load_audit/validate_loaded_weights`；ModelRunner加载后调用 | 原始回调成功返回后记录槽位；比对index的5631个源张量映射，检查missing/unexpected；这是加载回调完整性，不是全权重逐元素审计 |
| RMSNorm与残差 | `MellumRMSNorm.forward` | FP32 variance/rsqrt后先cast回输入dtype，再乘权重；残差相加遵守HF的BF16舍入边界 |
| QKV与router | `atom/model_config/mellum_projection.py:separate_qkv_forward/reference_router_forward` | 保留原参数和加载接口，分别以F.linear计算Q/K/V以及gate；避免打包形状与投影执行差异；O/输出头未做全局替换 |
| RoPE | `mellum_utils.py:aiter_rope_parameters`；`mellum_attention.py:cpu_inverse_frequency/reference_position_rope` | 翻译YaRN幅值比例；逆频率在CPU初始化，位置频率与三角函数在GPU FP32计算，旋转保留BF16运算边界 |
| Prefill Attention | `mellum_attention.py:sdpa_single_prefill` | HF布局Torch SDPA；滑窗阈值后显式展开KV heads，mask允许`j<=i`且`j>i-window`，支持超过1024的完整单请求prefill |
| KV与decode | `install_reference_kv_rope/overwrite_reference_keys/gather_reference_kv/sdpa_cached_decode` | 保留ATOM真实分页缓存和slot mapping，写回参考旋转K；decode按block table读取实际历史KV，滑窗截取合法区间并用SDPA计算 |
| 专家计算 | `mellum_experts.py:MellumTorchExperts.forward`；`MellumSparseMoeBlock.forward` | grouped prefill、batched decode，保持官方BF16激活、路由权重乘法与sum顺序；按实际context.is_prefill选择，不能按token数猜阶段 |
| Greedy采样 | `atom/model_ops/sampler.py:Sampler.forward` | all_greedy在是否需要过滤的分流之前执行torch.argmax；并列最大值选择最小词表索引，输出仍为torch.int |

普通默认启动不等于该已验证组合。当前仍使用带`diagnostic`名称的显式后端选项，名称来源于开发过程；不能因此认为它们只采集日志，也不能假设它们已自动成为默认生产配置。

## 多轮问题如何闭环

### 参考环境与接口

本地5.2.0不原生支持目标模型，先核实后使用独立5.10.1建立外部参考。处理加载报告中set不能JSON序列化、诊断脚本AITER路径与RoPE构造接口等工具问题。工具异常、blocked_no_gpu和diagnostic_complete分别记录，未当作模型正确性成功。见验证01–05。

### BF16舍入、专家与投影

同前缀比较将首差定位到第0层；保存输入和真实权重后确认RMSNorm先乘权重再cast的顺序不符合HF，专用实现修正。短token一致后扩展集仍分叉，因此继续用固定HF输入重放MoE，排除上游输入混杂。路由控制仅小幅改善误差；官方grouped_mm可精确重放，eager正文不能，证明必须检查专家装饰器实际分派。由此采用PyTorch专家正确性路径，而非反复调整路由阈值。见验证09、14–17。

同shape投影对照表明保存案例中的AITER与Torch可一致，而QKV打包形状改变K/V；增加separate QKV。后续窗口首decode逐层追到索引8层，Attention仍精确但router一项发生BF16变化，改变归一化权重；只修正gate投影，不把全部线性层无证据重写。见验证20–21、31–33。

### RoPE、Attention与真实缓存

同QKV交叉对照区分独立HIP旋转、实际融合旋转与prefill内核；分步旋转单独不足以闭环。CPU/GPU逆频率初始化有微小差别，影响sin/cos与BF16舍入；采用CPU逆频率后完整prefill精确。1024处HF由隐式causal/GQA切至显式mask/展开KV，候选按同布局处理。decode继续观察实际分页KV而非另建独立历史，先修正Q/K与K写回，再以读取相同分页缓存的SDPA定位后续层。见验证18–30。

### HF普通generate隐式切换

强制历史32位置精确后，普通生成仍不同。跨运行发现ATOM正常、ATOM强制与HF手动cached一致，而HF generate不同；同实例对照再次复现。HF5.10.1 `generation/utils.py:GenerationMixin._sample`在decode外进入`_optimize_model_for_decode`，非CPU上临时grouped_mm→batched_mm，结束后恢复。只读结束后的config会漏掉实际执行路径。候选对齐grouped prefill/batched decode，并逐次记录HF实际后端，short128位置闭环。见[验证34](34_cached_decode_exact_and_normal_generation.md)。

### 精确logits上的采样分叉

多样十例8/10完整序列一致，但全部630个同历史向量精确。两个首次分叉都是最大值并列：8599/8943与392/2516；HF选择较小ID。Sampler无过滤路径绕过已有all_greedy分支进入AITER温度kernel，修正入口后10/10、640位置精确。这个问题发生于共享采样器，不能把修复影响说成仅限Mellum；随机/混合批次不在本轮验证范围。见[验证35](35_greedy_tie_break_and_sampling_dispatch.md)。

### 长输入增强

早期正确性候选只支持窗口内输入，报告36当时的审计如实指出这一限制。随后去掉越窗拒绝并加入HF语义band mask，先过1023/1024/1025，再过8191/8192/8193/16384 smoke和64token回归。报告36是历史快照，不能再引用为“当前16k不支持”。见[验证37](37_step0_long_prefill_extension.md)。

## 最终有效GPU证据

以下都是普通生成，无强制token；精确向量指标来自采样前完整logits，不是强制历史产生的相同token。每行独立运行，集合有重复，不相加声称独立样本总数。

| 集合与实际输入 | 完整序列 | 输出token一致 | 逐步logits精确 | 验证运行ID |
| --- | --- | --- | --- | --- |
| 多样短输入与1023/1024 | 10/10 | 640/640 | 640向量 | `20261112T213129468143Z` |
| 1023/1024/1025，每条8token | 3/3 | 24/24 | 24向量 | `20261113T085534271528Z` |
| 8191/8192/8193/16384，每条64token | 4/4 | 256/256 | 256向量 | `20261113T103154043872Z` |
| 200条开发prompt，每条64token | 200/200 | 12800/12800 | 12800向量 | `20261113T104458011376Z` |

运行目录均在`/workspace/results/mellum_validation/<ID>/`；对应ATOM目录由各summary的atom_run给出，不以“latest”猜测。200条的ATOM结果为`/workspace/results/atom_mellum_gpu/20261113T105239431753Z/`。四组独立prefill向量均精确，5631/5631加载审计均无缺项。具体artifact及SHA256见[清单](39_step0_manifest.json)，含24个结果/输入/审计文件指纹。

200条由全部164题HumanEval prompt加原文件顺序前36条scenario_a prompt构成，实际输入63–4113token，均64个输出token，没有EOS提前截短。两种一致率分母都达100%，但这不自动确认集合符合出题方公开评测口径。详见[验证38](38_step0_developer_200_agreement.md)。

## 验证工具与正常推理的界限

`run_mellum_validation.py`依次启动HF、ATOM，固定相同ID/mask/position，保存独立结果；比较器核对schema、张量哈希、token与logits。`FreeGenerationProbe`被动保存正常输入和采样前向量，不替换token。`ForcedDecodeProbe`只在显式强制历史诊断时改采样logits，评分使用修改前值；不能作为普通生成正确性的替代。`mellum_diagnostics.py`与各diagnose/replay脚本用于固定输入、逐层首差及受控重放，非基线必需干预。

worker生命周期增加15秒心跳、死亡检测、可配置阶段超时和自有进程组清理，解决探针异常退出后父进程等待。运行器支持开发200条时更长超时。模型CPU回归、官方专家/掩码对照和运行器测试均有各批次记录；不同批次计数含重复，不相加宣传总覆盖。本轮只整理文档与指纹，没有重新启动GPU或重新执行全部测试。

## 题目门槛与阶段结束条件

| 原题项目 | 已有证据 | 当前解释 |
| --- | --- | --- |
| 冻结环境ATOM单卡推理 | 真实BF16权重和普通生成通过 | 核心适配开发完成 |
| 200条前64token ≥90% | 开发集合序列/token均100% | 开发对照通过，公开集口径尚待确认 |
| 16k单独统计 | 16384输入64输出token精确 | 已有长输入开发验证，不泛化为所有隐藏样例 |
| BF16 HumanEval差距≤2个百分点 | 未执行完整答案与测试 | 待参考/工具/口径确认与正式补证 |
| MTP、双卡 | 未启用 | 对应条件检查当前不适用 |
| 服务与性能交付 | 尚无本候选完整服务负载证据 | 后续主线工作，不是额外logits精确门槛 |

前64token相同不能证明完整函数相同；若完整代码相同且提取、测试、执行条件相同，pass@1理应相同。当前缺的是直接证据，并非已知HumanEval失败。`baseline.json`中的3/164已被标注Rejected，不作为有效参考值。

完整logits与中间层逐值一致是本次定位证据，题目没有将其设为所有优化必须满足的门槛。后续以题目token/任务指标为准；不再为了重复追求100%内部精确而无限延后Step1。

## 版本、复盘与Step1交接

工作分支`feature/mellum-support`，HEAD仍为`26f23a0bdbf50b9010c07ee3a71171189723e767`。包含暂存、未暂存及未跟踪的适配代码，不能仅凭HEAD复现已通过结果。历史日志记录Git作者身份缺失导致提交未完成；本轮不伪造身份、不自动提交。当前文件哈希与git status已保存，但不是推理发生时捕获的历史commit指纹，也不能替代完整可恢复Git归档。后续固化版本时须处理这一交接项。

方法上的收获是：先核对真实后端、以相同输入隔离首差、分开工具故障和模型错误、区分数值差异与采样差异。成本上的教训是早期对HF装饰器/生成context检查不足，以及将开发诊断目标扩大为验收标准，产生了过多实验与过时导航。

下一主线为：固定候选与启动配置，打通最小性能测量服务入口，处理实际负载所需的批量能力，采集profiling后再选择最多三个可独立消融的优化点。单请求稠密SDPA mask、选中专家权重复制、分开投影和Python控制流只是待测实现特征，尚无证据宣布哪一项是首要瓶颈或预估收益。最终加速比相对出题方Baseline，自己的Step0用于增量归因。

题目最终≤4页技术报告和1页AI说明在Step1结果出来后形成；本内部报告提供Step0素材。AI参与的接口误判、分派遗漏、串行试验与验证方法均如实记录，供最终说明引用。唯一按自然日维护的记录见[worklog](../worklog.md)。
