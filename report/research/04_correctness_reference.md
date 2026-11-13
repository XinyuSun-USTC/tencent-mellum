# 正确性资产与参考实现调查

状态：2026-11-10 已有调查证据的主题整理；本次迁移没有重新运行技术检查。
源码基线：`26f23a0bdbf50b9010c07ee3a71171189723e767`。ATOM 相对路径基于 `/workspace/ATOM`。
区分源码事实、源码推断、待运行验证与未找到；静态风险不是已复现故障。

本页提取[资产审计原文](../archive/initial_research/03_correctness_assets_audit_original.md)和[累计研究原文](../archive/initial_research/02_atom_research_original.md)；“本轮”指原检查批次，数字节号指对应原文，不表示工作日或新的执行计划。当前方案统一见[Step 0](../../docs/plan/01_step0.md)。

## 当前结论

HF 是原题认可的正确性参考。用户已确认只有题目和容器，没有额外附件；不继续将资产缺口作为用户需补交材料的事项。尚无可运行的可信参考，建立它与编写测试脚本同样必要。本页末尾已补充本地复核和官方v5.10.1来源调查；下一项任务是固定官方源码、实施最小兼容参考并做CPU验证。vLLM可作为源码与补充对照，不能直接替代HF验收。


日期：2026-11-10。只读检查容器文件与 Python 安装元数据；未启动 GPU、服务、评测或加载权重，未安装/升级依赖。

## 结论

容器中确有题目相关数据与历史结果，但本轮未找到可直接执行的完整正式正确性验证套件。不能把“未找到”扩大为全机绝对不存在，也不能把通用工具或历史 rejected 结果当作正式验收依据。

## 已确认资产

| 路径 | 证据 | 可用性 |
|---|---|---|
| `/data/datasets/HumanEval.jsonl`、`.gz` | 164题，字段 task_id/prompt/entry_point/canonical_solution/test | 有题目、标准解和测试，可作为后续评测输入；没有候选输出及完整评测协议 |
| `/data/datasets/humaneval_bf16.json` | count=164，passed=3，pass_at_1=0.018292682926829267；每题仅 task_id/passed/error | 历史执行结果，无生成文本、token、logits；不是可用参考输出 |
| `/data/datasets/baseline.json` | humaneval_pass1_bf16、humaneval_pass1_fp8 均 null；humaneval_note 明确 Rejected: raw /v1/completions without the chat template or thinking mode | 明确否定旧 3/164 作为有效基准；提供 A/C 性能值，不提供 HF token 参考 |
| `/data/datasets/scenario_a_prompts.jsonl` | 512条，id/prompt/intended_words | 性能负载 |
| `/data/datasets/scenario_c_prompts.jsonl` | 200条，另有 shared_prefix_group | C 性能负载；无参考 token、不能仅因200条认定为正式正确性集合 |
| `/data/models/Mellum2.1-12B-A2.5B-Thinking/chat_template.jinja` | ChatML、reasoning_content、think 等处理 | 后续参考与服务端必须统一模板；本身不是验证器 |
| 同目录 README.md:279–285 | JetBrains 自报 thinking mode 的 HumanEval+ 91.5 | HumanEval+ 与当前 HumanEval 不是同一个验收协议，不能代替题目参考值 |

历史错误统计：humaneval_bf16.json 中147题错误含缺失 pytest，153题含 ModuleNotFoundError。这只能说明历史运行存在问题；当前安装元数据显示 pytest=9.0.3，不能把旧错误当成当前环境缺包诊断。

## 容器工具及仓库脚本

通过 importlib.metadata 读取版本，未导入模型/推理框架：Transformers 5.2.0、lm_eval 0.4.12、evaluate 0.4.6、pytest 9.0.3；evalscope、human-eval、evalplus 未找到对应安装分发元数据。`/opt/venv/bin` 也没有 evalscope 命令。此结论针对当前 Python 环境，不断言所有可能环境不存在这些工具。

- `/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/humaneval/humaneval.yaml`：通用 HumanEval 任务，dataset_path=openai/openai_humaneval，原始 prompt，max_gen_toks=1024，带 stop 序列和 create_test filter。不是 Mellum 专用 harness；直接照搬可能重现模板/生成/代码提取口径问题，且默认数据路径可能触发远端取数。
- `/app/ATOM/scripts/run_gsm8k_eval.sh`：lm_eval local-completions、GSM8K；不是本题 HumanEval/200 prompt 验证器。
- `/app/ATOM/tests/test_prefix_cache_accuracy.py`：共享数学前缀、已知算术答案与重复请求一致性；不是 HF 逐 token 对齐。
- `/app/ATOM/tests/entrypoints/`：服务/API/解析等通用测试；未发现本题 Mellum 的 HF reference 比对入口。
- `/app/ATOM/atom/benchmarks/benchmark_serving.py`：通用服务性能 benchmark，不能替代模型数值验收。
- 当前 Transformers models 文件目录中未发现 Mellum 路径；Mellum checkpoint 目录没有 modeling/configuration Python 文件。冻结环境不能直接假设 AutoModel 能运行题目所述 HF Mellum。

## 搜索范围与边界

检查 `/app`、`/data`、`/workspace` 的项目/脚本/数据文件，`/opt/venv` 的入口与相关包源码，`/root/.cache`、`/usr/local/bin`、`/usr/local/share`、`/usr/local/lib`、`/srv`、`/mnt`、`/media`、`/var/tmp` 等的相关文件名。关键词包括 Mellum、HumanEval、baseline、correctness、reference、evalscope、prompt，以及题目数据文件名、前64token匹配相关文本。

大型第三方源码、ROCm 文档、编辑器/代理内部文件不作完整逐文件内容扫描；未读取凭据，未查询代理会话记录来充当外部证据。`/home/ubuntu` 无读取权限。模型仅核查文件清单/文本，未读权重 payload。本轮没有穷举 Git 历史对象或缓存内所有无扩展二进制，因此结论限定为上述已检查范围。

题目 PDF（临时文本 `/tmp/csig_assessment_text.txt`）中，正确性条款描述200条prompt、HF前64token参考、HumanEval和16k子集；D0条款列有“评测脚本、参考输出”。原题确实预期这些资产存在，但没有提供本机路径。本轮未找到对应完整资产。

## 验证路线

调查的后续执行顺序统一维护于[当前Step0方案](../../docs/plan/01_step0.md)，不在研究证据页重复维护。

## 文件指纹 SHA256

- baseline.json: `14674cbbc641709358e197bd5ad71c3cac8f41ed3932b3091266f5636b0dd6c8`
- humaneval_bf16.json: `399354335c6da2f1e57ca0f14133a509b60b8e99d7588b4bd97011dad2b9287b`
- HumanEval.jsonl: `1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2`
- scenario_a_prompts.jsonl: `f5cb156bb468757841a321142c8813d281c0732b494d70d3c59d5b415bf7489d`
- scenario_c_prompts.jsonl: `38fc340a76fd48c025bf5806376bdabbc7762f6b5c48d4509bbb109af2cfe265`


## 早期仓库与 Baseline 搜索证据

### 搜索清单、命令与结果

| 检查对象 | 实际范围 / 方法 | 结果与边界 |
|---|---|---|
| `/workspace/ATOM` 当前树 | 含隐藏与 ignored 文本，排除 `.git`、字节码，检索 Mellum/architecture、SDPA/mask、Baseline/关键性能值 | Mellum 名称 0 命中；SDPA/attn_mask 共 3 行，见 13.4；Baseline 泛词命中属于其他功能/CI/配方 |
| `/app/ATOM` 原始树 | 同样定向检索；工作副本 384 个 tracked 文件逐个做 SHA256 内容比较 | Mellum 0 命中；384 文件均存在且内容完全相同；两处 Git status 均干净 |
| 工作副本 Git 文件 | ls-files、非 ignored untracked、ignored 清单、patch/archive 后缀 | tracked=384；非 ignored untracked=0；ignored=42，源码/脚本/patch 类只见自动生成 `atom/_version.py`，其余主要为 egg-info/bytecode；tracked 无 .patch/.diff/.zip/.tar/.gz/.tgz |
| Git 引用与历史 | 本地 heads、354 个 refs 的名称；所有本地可达 refs 的提交 subject 含 Mellum、路径名含 Mellum；本地 main 的定向 grep | 只有 feature/mellum-support 分支名命中；提交 subject 与文件路径名历史无命中；不等于逐个扫描所有历史 blob 内容 |
| `/workspace/scripts` | 顶层仅 atom，完整读脚本 | 只是设置 PYTHONPATH 后启动通用 api_server，无 Mellum patch/特殊参数 |
| `/workspace/docs`、`report` | 顶层与定向文本核对 | 研究指南和 D0/D1；它们是研究记录，不是预置实现；未找到独立原始考核说明书/实现包 |
| `/app` 顶层 | 只列目录后选择已知 ATOM / AITER 文件 | 顶层有 ATOM、aiter-test、mori；没有大范围递归扫描 `/app` 或 MORI |
| `/data/datasets` | 仅 6 个现有文件，读取 JSON/JSONL schema 和统计 | 性能 JSON、HumanEval 题目/结果、A/C prompts；无 .py/.sh harness 或实现文件 |
| Mellum 模型目录 | 列出全部 15 个条目；读 README/config/generation/template 与索引 | 5 个 Safetensors 分片、配置/分词器/模板/索引/README 等；无 modeling/configuration Python、自定义 inference 子目录、独立 MTP 文件 |
| `/data/models` 顶层 | 列目录，不递归其他大模型 | 4 个模型目录，仅一个 Mellum 目录；不以其他模型文件推断 Mellum 参考实现 |
| 当前 venv 可能的模型参考目录 | 只查具体标准路径存在性与 mellum 名条目 | transformers/models 存在但无 Mellum 条目；vllm/model_executor/models、vllm/transformers_utils/configs、sglang/srt/models 这些路径不存在；不是对所有安装/源码位置的全盘排查 |

关键检索形式（在相应仓库根目录运行，仅为可复查记录）：

```bash
rg -n -i --hidden --no-ignore -g '!.git/**' -g '!*.pyc' \
  'mellum|MellumForCausalLM' .
rg -n -i --hidden --no-ignore -g '!.git/**' -g '!*.pyc' \
  'scaled_dot_product_attention|attn_mask' .
rg -n -i --hidden --no-ignore -g '!.git/**' -g '!*.pyc' \
  'explicit.*mask|naive.*moe|torch_moe|pytorch.*loop' .
git log --all --regexp-ignore-case --grep=mellum --format='%h %s'
git log --all --format= --name-only -- '*[Mm]ellum*'
```

对两处 Git status 的收尾检查设置 `GIT_OPTIONAL_LOCKS=0`，避免可选 index refresh。源码对比只读字节，不复制或覆盖任何原始文件。没有搜索凭证目录、Shell 历史或整盘内容；没有扫描 Safetensors payload/header、解包二进制缓存或所有 Git 历史代码。

### 当前/历史 Git 能证明什么

- 工作副本 HEAD=`26f23a0bdbf50b9010c07ee3a71171189723e767`，feature/mellum-support 也指向此提交；当前没有适配修改。分支名表达工作意图，不是实现已经存在的证据。
- 本地另有 main=`890dc0f12e3040a5550bf5b476faee3eb21635b3`，不是当前研究基线。未切换或合并；对 main 的 atom/scripts/recipes/tests 做 Mellum grep 无命中。
- HEAD 上 tag 包含 v0.1.4 与 v0.1.4-rc0；本轮没有根据 tag 切版本。
- 本地存储的 refs 共 354 个（heads/remotes/tags），Mellum 名称只出现在本地工作分支；`git log --all --grep=mellum` 无提交 subject 命中，Mellum 路径名历史也无命中。

以上只能排除本轮这些证据入口；不能推出“所有历史代码内容都没有无名称实现”。没有下载远端新 refs，也没有为了找一个可能分支而改变冻结 commit。

### 不依赖 Mellum 名称的候选排除

### SDPA / 显式 mask 候选

当前两份 ATOM 树对 `scaled_dot_product_attention|attn_mask` 均只命中：

1. `tests/test_profiler_regression.py:67`，类 `_TransformerBlock`，forward()：测试用 LayerNorm + QKV + GELU dense FFN，4 层/16 heads 的小模型（`:37–43,49–79`），没有 Mellum checkpoint/config 接入，也没有此调用的显式 attn_mask 参数。
2. `atom/models/qwen3_5_vl.py:8` 的文件说明及 `:101` 的调用，类 `Qwen3VisionAttention`，forward()（`:66–104`）：视觉 block 使用 QKV/RoPE/SDPA，不是 Mellum 文本 decoder，调用也没传显式 mask。

故这些命中不能作为官方“显式 Mask SDPA + Mellum”的实现证据。原始 `attention_mask` 泛词还出现在类型/TP计划等配置说明；不能仅靠名称把配置声明当成执行 mask 的代码。

### 循环 MoE 确实存在，但不是已定位的 Baseline

只读的 `/app/aiter-test/aiter/fused_moe.py:1864–1925` 有 **`torch_moe()`**：接收 hidden_states、w1/w2、top-k weights/IDs，转 FP32，逐 expert 做 mask → gate/up GEMM/activation → down GEMM，最后加权求和再转回原 dtype。逐专家循环明确在 `:1912–1925`。

但针对 `torch_moe` 的 ATOM 当前源码检索没有调用/导入命中；AITER 同文件的同名命中为这些参考函数定义和辅助 activation。前面已闭合的 ATOM Qwen 无量化运行链是 `UnquantizedFusedMoEMethod.apply()`（`model_ops/moe.py:559–616`）调用 fused_moe 或 modular kernel，而非 torch_moe。也没有找到把这个函数与 Mellum 配置、权重和 SDPA 拼接的启动入口。

因此它可以是后续算子参考的候选，**不能认定为产生官方性能数据的实现**。BF16 输出格式与内部 FP32 计算的参考行为也不能被简化为“与官方 BF16 Baseline 必定相同”。

ATOM 自身还有一些容易混淆的循环/开关：

- `moe.py:1366,1724,1754` 等 expert 循环用于量化参数处理；`:2227` 位于 `_online_quant()` 的权重量化处理，并非每步推理的朴素 MoE forward。
- `moe.py:891–892` 的 `ATOM_V4_TORCH_MOE` 命中在 Mxfp4MoEMethod.process_weights_after_loading() 中，只提前 return，不能据名字认定提供了 Mellum 的 PyTorch 循环执行开关。
- `fused_moe_triton.py:232` 的 triton_kernel_moe_forward 是 Triton MoE 包装，torch 排序/route 操作不等于逐专家 PyTorch GEMM Baseline。
- `v4_kernels/csa_translate_pack.py:155` 的 explicit mask 注释是 V4 index packing 的边界 mask，不是 Mellum SDPA。

### Generic Registry / plugin / patch 入口

除了名称检索，再核对机制：

- Phase 2 已证明独立 Server 的 get_hf_config 与首 architecture 查表链；当前 Mellum config 不能原样到达 model_class 构造。Qwen 可参考，但不存在“自动按近似架构执行未知 Mellum”的 fallback。
- `atom/plugin/register.py:13–33` 的 `_ATOM_SUPPORTED_MODELS` 及 SGLang 条件扩展中没有 Mellum；`plugin/vllm/model_wrapper.py:87–121` 选择已有 architecture，`ATOMModelBase.__init__():130–155` 使用 vLLM 配置，不是自动生成缺少的模型结构。相关注册文件全树 Mellum 检索也无命中。
- `atom/utils/debug_helper/ref_patch.py:3–29` 描述如何包装已有只读参考模型的 forward；`patch_method():88–118` 和 patch_block_forward():122 起是 dump 工具，不包含 Mellum 实现。文档中的 `/data/<model>/inference/model.py` 是模式示例，当前 Mellum 目录没有该目录，不能当作已找到参考模型。
- graph_capture_patch / spec_decode_patch 等文件属于框架集成补丁；没有 Mellum 专用 patch 文件、注册行为或脚本串联证据。

### Baseline 泛词命中

`baseline|638\.28|76\.216` 在单份当前树共 318 行、33 个文件；文件分布为 CI JSON/workflow、dashboard、benchmark scripts、其他模型 recipes、开发记录和 V4/prefill 注释。没有 Mellum 名称或目标适配链。

代表例子 `.github/scripts/plugin_benchmark_validate_baseline.py:20–36` 的 validate_result_dir() 检查 dashboard artifact 的 output_throughput / dashboard_publish_allowed；它不是 Mellum 模型或官方考核 evaluator。`.github/benchmark/models_accuracy.json` 的 accuracy_baseline 是各已支持模型的 CI 基线，不可替代 `/data/datasets/baseline.json`。

### 官方性能数据与可复现性缺口

`/data/datasets/baseline.json` 已逐字段核对：

| 内容 | 文件依据 | 可确认的事实 |
|---|---|---|
| 开始时间 | `:2` | 记录为 2026-10-26T20:07:50+0800 |
| 运行描述 | `:3–6` | 1 GPU、eager、level 0、BF16 KV、MTP off、SWA on；precision=bf16，rate_per_gpu=0.3 |
| Scenario A | `:7–15` | count=512，elapsed≈205.351s，output_tokens_per_s≈638.282724，TTFT p99≈5.009713s，TPOT p50≈38.951ms / p99≈50.144ms |
| Scenario C | `:16–23` | count=200，elapsed≈671.771582s，output_tokens_per_s≈76.216383，TTFT p99≈0.993549s，TPOT p50≈36.290ms / p99≈41.265ms |
| Scenario B | 整个文件 | 无 scenario_b 项 |
| 有效 HumanEval 指标 | `:26–31` | pass1_bf16/fp8 为 null；3/164 的旧结果被明确标记 Rejected |

**来源更新：** baseline.json本身没有“PyTorch循环MoE + 显式mask SDPA”这两个实现字段。早期调查只能依据指南转述；后来收到并阅读原题PDF，已确认原题确有该描述，见[题目摘要](../../docs/requirements/01_assessment_summary.md)。仍未定位对应原始实现代码。

JSON 也没有保存源代码 commit/patch、完整 server/benchmark 命令、软件环境快照、sampling/stop 参数、output length、prefill/cache 开关全量配置等。指南提供的额外描述可以指导后续设计，但这些数据文件**不足以单独恢复产生性能数据的运行过程**。

A/C prompts 文件轻量 JSONL 检查：

- scenario_a_prompts.jsonl：512 条，字段 id/intended_words/prompt。
- scenario_c_prompts.jsonl：200 条，另有 shared_prefix_group。
- 二者没有 max_tokens/output_len/input_tokens 等实测长度字段；intended_words 也不能当成 tokenizer token 数。

因此 count 与性能数据对上，只能确认材料相容，不能推断已复现 tokenizer 长度、并发、输出 token 数或吞吐。没有执行 benchmark。

### 模型目录与本地官方 README 的证据

当前 Mellum 目录的完整顶层清单已核对，没有 .py、inference/、单独 MTP checkpoint 或额外实现包。config 无 auto_map 的结论沿用 Phase 2；仅开启 trust_remote_code 不会从这个目录凭空取得自定义 Config/Model 类。

本地 `/data/models/Mellum2.1-12B-A2.5B-Thinking/README.md`：

- `:211` 表示相比 Mellum2 架构不变，主要变化在后训练。
- `:223–234` 列出 H/heads/experts/context/SWA 等模型概览，与本轮 config/index 的结构描述相容。
- `:239–250` 提供 vLLM serving 示例，但未随目录提供相应模型源码，不证明当前 venv 可运行这些命令。
- `:253` 将 vLLM MTP head 列为 coming soon；与 Phase 3 “当前 index 没有可识别 MTP 参数项”的证据相容，但不能据此断言所有发布渠道都无 MTP。
- chat_template.jinja:98–103 控制 generation prompt 与 enable_thinking=false 时的闭合 think 前缀；模板是 prompt 构造依据，不是 Attention/MoE 的权威数值实现。

这些只代表本地提供的 README 快照，不判断今天线上最新版框架的支持状态。本轮不查询远端、下载新代码或升级依赖；本地仍缺 Mellum Attention mask、RoPE、routing 的完整参考 forward。

### Baseline 调查判定表

| 调查项 | 判定 |
|---|---|
| 当前 ATOM 完整 Mellum 实现 | 本轮范围内未找到；独立运行入口还有已证实的 Config/Registry 缺口 |
| 当前 ATOM 部分 Mellum 功能适配 | 未找到；研究分支名、已有 Qwen/SWA 组件和启动包装脚本不算模型适配 |
| 官方 Baseline 原始实现 | 未定位，不能认定可以原样复用或复现 |
| 官方 Baseline 性能数据 | 存在，A/C 已核对，B 缺失 |
| Mellum 专用启动/patch/权重映射 | 已检查位置未找到 |
| PyTorch expert-loop 参考 | AITER torch_moe 存在，但未连接 Mellum/ATOM Baseline |
| 显式 Mask SDPA 的 Mellum 实现 | 未找到；现有 SDPA 命中是测试和视觉模型 |
| 正确性参考 | 有 HumanEval 题目、参考解、test；无有效 Mellum 模型输出 oracle / 完整考核 harness |
| MTP 资源 | 当前模型目录未发现额外 MTP 文件，index 结论见 Phase 3；独立来源仍未知 |

### 证据固定与收尾

本轮输入文件 SHA256（仅文本读取）：

| 文件 | SHA256 |
|---|---|
| `/data/datasets/baseline.json` | `14674cbbc641709358e197bd5ad71c3cac8f41ed3932b3091266f5636b0dd6c8` |
| `/data/datasets/humaneval_bf16.json` | `399354335c6da2f1e57ca0f14133a509b60b8e99d7588b4bd97011dad2b9287b` |
| Mellum `config.json` | `f43d018246094dee6dca455b3072766b6acbefd1ac8e7fe664df34710606a541` |
| Mellum `model.safetensors.index.json` | `a6ad930e5bb35ea92d5b734c4df4649bfcfdcd899f64b0a83efe40fe0ab9b60e` |

工作副本与原始 ATOM 的 384 个 tracked 文件 SHA256 均一致；无功能修改。检查工具为 rg、只读 Git 命令、标准库 Python；未执行原始目录中的任何项目脚本，未修改数据集/模型/依赖、未获取远端代码、未跑测试或 GPU。写入范围仅本报告。

## 本地 Transformers 与官方 HF Mellum 支持复核（2026-11-10）

### 结论与执行范围

当前 `/opt/venv` 的 Transformers 5.2.0 **不具备原生 Mellum 支持**；HF 官方后续实现确实存在。两者并不矛盾。官方v5.10.1发布将Mellum列入新模型，并关联PR #46112；v5.10.0已被撤回。不能将checkpoint写入的5.8.1当作最低支持版本。

本轮执行CPU导入、元数据/接口检查、离线AutoConfig，以及官方网页源码阅读；没有构造模型、读取权重payload、安装依赖或运行GPU。CPU探针结束时`torch.cuda.is_initialized() == False`。

复现命令：

```bash
/opt/venv/bin/python /workspace/scripts/probe_transformers_mellum.py
```

[探针脚本](../../scripts/probe_transformers_mellum.py) · [本次原始JSON](../../results/transformers_mellum_support/local_probe.json)。脚本设置HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE，未调用AutoModel或模型from_pretrained。错误消息中自带的pip升级建议仅是库输出，不是本项目实施建议。

### 本地运行时证据

| 项目 | 实际结果 |
|---|---|
| 默认python、python3 | 均为`/opt/venv/bin/`下入口 |
| 工作副本启动 | `/workspace/scripts/atom`明确exec `/opt/venv/bin/python`并设置工作副本PYTHONPATH |
| 元数据及运行时版本 | 均5.2.0 |
| 实际导入文件 | `/opt/venv/lib/python3.12/site-packages/transformers/__init__.py` |
| CONFIG_MAPPING_NAMES / MODEL_FOR_CAUSAL_LM_MAPPING_NAMES | 均没有mellum |
| 顶层模型类访问 | AttributeError：没有MellumForCausalLM |
| 模型源码目录 | 未发现`transformers/models/mellum` |
| 离线AutoConfig | ValueError：不能识别model_type=mellum |
| checkpoint扩展入口 | config没有auto_map，目录没有Python模型文件 |

checkpoint仍为`architectures=[MellumForCausalLM]`、`model_type=mellum`、`transformers_version=5.8.1`。trust_remote_code开关本身不能生成缺少的实现。

对`/opt`、`/usr/local/lib`、`/root/.local`和`/workspace`检查相关Python环境/transformers目录，跳过ROCm、Git与node_modules，找到的运行环境为`/opt/venv`；没有发现可直接使用的另一份Transformers。该范围不代表全机所有挂载均已穷尽。

### 官方来源、版本与固定基线

1. [HF官方模型文档](https://huggingface.co/docs/transformers/model_doc/mellum)记录贡献日期为2026-05-28，并公开Mellum配置、模型及CausalLM接口。
2. [官方v5.10.1发布说明](https://github.com/huggingface/transformers/releases/tag/v5.10.1)将Mellum列为新模型，关联[PR #46112](https://github.com/huggingface/transformers/pull/46112)，同时声明v5.10.0已撤回。由此将v5.10.1作为本轮已证实的正常发布支持基线；不建议使用撤回版本。
3. 发布页对应[完整提交](https://github.com/huggingface/transformers/commit/90c3ae54d448d4906b6167317ea5a7f5d48a232d)：`90c3ae54d448d4906b6167317ea5a7f5d48a232d`。本轮正文成功读取的是v5.10.1 tag URL；直接SHA源码URL未能获取，因此没有把“已解析提交”冒称为“本地已固定源码快照”。
4. [v5.10.1模型实现](https://raw.githubusercontent.com/huggingface/transformers/v5.10.1/src/transformers/models/mellum/modeling_mellum.py)、[配置实现](https://raw.githubusercontent.com/huggingface/transformers/v5.10.1/src/transformers/models/mellum/configuration_mellum.py)均成功阅读。以下分析使用该tag，避免混用不断变化的main。
5. PR正文/commits/API因工具抓取失败未能完整读取，**首次合并的精确SHA仍未确认**。本轮没有完成所有历史patch/预发布版的穷举，故不将v5.10.1表述为经逐版本测试得到的唯一最低版本。v5.9.0的configuration_auto已改从auto_mappings导入映射，仅搜索前者无Mellum不能证明该版本不支持。

网络限制：shell urllib无法联网，改用网页工具读取官方资料；未进行权限升级。未克隆完整仓库、下载权重或安装新软件。以后真正接入前需取得固定提交原文、保留Apache-2.0版权许可及兼容patch。

### 官方模型与当前checkpoint的语义对应

源码行号以v5.10.1原文件1-based为准，网页显示从0计数，人工核验时注意差1。

| 官方文件/符号 | 证据与适配含义 |
|---|---|
| `configuration_mellum.py:MellumConfig`，约29–123 | H2304/L28/Q32/KV4/D128/E64/K8/I896、untied；layer_types和mlp_layer_types可配置，与本地配置一致。默认layer_types为全Full，不能依赖默认值覆盖checkpoint的S/S/S/F；默认RoPE也不能代替本地参数 |
| `modeling_mellum.py:MellumAttention`，约206–266 | Q/K投影后在head_dim执行RMSNorm，再RoPE，再更新cache；attention缩放D^-0.5，按层类型传SWA |
| `MellumRotaryEmbedding`，约48–118 | 按层类型单独初始化频率；FP32计算cos/sin并乘attention_scaling，最后转回输入dtype |
| `MellumTopKRouter.forward`，约314–322 | 先router linear，FP32 softmax，再top-k，按配置归一化，再转回router logits dtype |
| `MellumExperts.forward`，约281–302 | packed gate/up投影按gate、up拆分，SiLU(gate)*up，down后乘路由权重并聚合；不是直接接受磁盘逐expert键的布局 |
| `MellumDecoderLayer.forward`，约371–389 | input norm→attention→residual，post-attention norm→MLP→residual |
| `MellumModel.forward`，约445–499 | DynamicCache、绝对位置、Full/SWA分别建mask和RoPE；各层依layer_types选择，最后norm |
| `MellumForCausalLM`，约576–665 | 独立lm_head，支持logits_to_keep及GenerationMixin；训练aux loss不是首版推理必要路径 |

这些是官方计算路径依据，不是当前checkpoint已通过数值验证的结论。配置与权重形状对应支持该实现作为候选参考，payload和实际forward仍未验证。官方forward示例含Qwen checkpoint遗留文字，不将示例输出视为本地Mellum参考答案。

### YaRN与SWA的关键依据

[官方v5.10.1 RoPE工具](https://raw.githubusercontent.com/huggingface/transformers/v5.10.1/src/transformers/modeling_rope_utils.py)：`_compute_yarn_parameters`约294–410，明确读取嵌套层类型参数；已提供attention_factor时直接采用，只有缺失时才计算默认缩放。与MellumRotaryEmbedding组合后，它乘在cos和sin上，并非再额外乘一次attention logits。本地5.2.0的同名函数也具有该字段处理。仍需比较完整频率计算和dtype，不能因字段同名就宣称数值完全一致。

[官方v5.10.1 mask工具](https://raw.githubusercontent.com/huggingface/transformers/v5.10.1/src/transformers/masking_utils.py)：`causal_mask_function`和`sliding_window_overlay`约65–86，组合条件为`k <= q`且`k > q-W`。故W1024包含当前token和最多1023个历史token。这个规则现在有HF直接依据；ATOM/AITER是否按同一规则执行仍待运行验证。

### 冻结5.2.0的兼容性：已确定的问题

| 接口 | 本地证据 | 官方v5.10.1调用 | 后续必要处理 |
|---|---|---|---|
| 配置生命周期 | `configuration_utils.py:57,166`，PreTrainedConfig有显式__init__，没有__post_init__（已CPU introspection） | MellumConfig使用strict和super().__post_init__ | 改为兼容的初始化桥接并验证字段/序列化，不能原样复制两个文件就结束 |
| mask函数 | `masking_utils.py:834,1038`，两个create_*mask必需cache_position | MellumModel只传config/inputs/attention_mask/past/position_ids | 显式桥接绝对cache_position，尤其测试prefill/decode、padding和滑窗偏移 |
| expert转换 | 本地`conversion_mapping.py:127–141`有qwen2_moe转换，`register_checkpoint_conversion_mapping():360`可进程内注册；本地没有mellum别名 | [官方转换表](https://raw.githubusercontent.com/huggingface/transformers/v5.10.1/src/transformers/conversion_mapping.py)含mellum→qwen2_moe | 显式提供该转换，审计所有expert键、顺序、shape及missing/unexpected；仅注册AutoConfig不足以正确加载 |
| 常用运行组件 | 本次CPU导入确认strict、expert/kernel decorators、maybe_autocast、merge_with_config_defaults、OutputRecorder/capture_outputs、AttentionInterface和DynamicCache存在 | 模型源码使用这些组件 | 降低移植范围，但“存在”不证明签名和行为完全兼容，后续小模型验证 |

当前Checkpoint逐expert gate/up/down与官方packed gate_up/down不同。必须保留独立参考的加载审计，不直接复用ATOM weight loader，否则可能共享错误。

另有generation、权重初始化、装饰器行为、cache更新和HF加载器的版本差异尚未逐一执行验证；上述清单是已确定项，不是“只改三处必成功”的承诺。

### 推荐路线及替代方案

**推荐下一步：固定官方v5.10.1源码，工作区建立最小兼容参考，使用当前冻结依赖，先做CPU配置/类导入及小模型检查。** 推理数学尽量保持官方原样，兼容patch集中于初始化、接口桥接和加载注册，独立于ATOM实现；不导入ATOM关键计算算子。优先选择HF内置eager/SDPA路径，不启用额外hub kernel。该路线是经调查后的建议，尚未实现、尚不能声称可运行。

- 当前环境直接AutoModel：已被Config/类缺失阻止。
- 独立conda/venv安装新版本：能隔离工程环境，但不自动满足题目冻结限制；本轮未创建，不视为确定方案。
- vLLM：暂作第二份源码依据；CUDA路径未验证ROCm可运行，也不替代题目HF基准，本轮不安装运行。
- 自写纯PyTorch参考：若官方兼容改动必须大幅改数学/框架才考虑；当前已有官方实现且本地多数依赖存在，尚无理由优先重写全部模型。

可信度边界：兼容版依然需要审计及一致性检查；尚无未经修改新版HF实跑输出可作为二次校验。若无法获得该输出，必须在报告披露，不能自动把兼容版称为出题人原始reference。

### 下一轮可执行的验证顺序（未执行）

1. 获取提交固定的官方源码及许可、记录hash，形成最小兼容patch。
2. 仅CPU加载真实配置，验证字段和roundtrip；导入模型类，不构造12B模型。
3. 构造小尺寸CPU模型，保留混合层与nested RoPE，测试forward、带/不带cache和窗口边界；检查位置桥接、packed权重映射。成功仅说明接口基本通。
4. 再进入真实checkpoint加载与短输入GPU参考推理，保存精确input IDs、生成参数、逐步token与必要logits；禁止以短句通顺代替数值证据。
5. 与ATOM逐算子/层/整模型对比，之后覆盖多请求、16k和一致HumanEval协议。16k优先SDPA或经过验证的数学等价分块，避免未经评估的全量eager分数矩阵。

本轮完成支持调查与兼容方案，不执行第1步之后的模型接入或GPU阶段。

## 路线更新：独立参考环境（2026-11-10）

用户已明确允许独立新版HF环境仅用于正确性验证，ATOM依赖继续冻结。此前最小源码桥接是冻结解释下的候选，现不作为优先路线；改用固定官方5.10.1与只读ROCm依赖。实际执行及网络失败见[环境记录](../validation/01_hf_reference_environment.md)，不能将已创建venv视为新版模型可运行。
