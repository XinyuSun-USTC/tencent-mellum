# Mellum 模型结构、权重与候选组件

状态：2026-11-10 已有调查证据的主题整理；本次迁移没有重新运行技术检查。
源码基线：`26f23a0bdbf50b9010c07ee3a71171189723e767`。ATOM 相对路径基于 `/workspace/ATOM`。
区分源码事实、源码推断、待运行验证与未找到；静态风险不是已复现故障。

本页提取已有记录，原文中的“本轮”指原检查批次；残留数字节号指[归档原文](../archive/initial_research/02_atom_research_original.md)，不表示工作日或新的执行计划。当前方案统一见[Step 0](../../docs/plan/01_step0.md)。

## 解释边界

配置与张量形状不等同于完整数学语义；仍需核对官方 HF 实现。以下组件分析用于必要适配，不预定 Step 1 优化点。当前五分片未见额外 MTP 权重，不推出所有版本模型都没有 MTP。

## Mellum 2.1 模型分析

### 模型基本信息

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

### Transformer 主干结构

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

### Attention 结构

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

### RoPE 与长上下文

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

### MoE 结构

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

### MTP 与投机解码

考核说明书明确指出 Mellum 具有一个单层 MTP Head，可以用于投机解码。

ATOM 本身已经具备 MTP / EAGLE 投机解码框架。

但模型 README 又指出 Mellum 2.1 的 MTP Head 在 vLLM 中的支持仍在准备中。

目前尚不能确定：

- 当前预置权重是否包含可直接使用的 MTP 参数。
- 考核环境是否已经提供额外的 MTP 适配。
- ATOM 是否存在 Mellum 专用 MTP 实现。

因此，MTP 暂时作为候选优化方向，后续需要通过权重索引和 ATOM 源码进一步确认。

---

## 模型权重与 Tokenizer 验证

### 权重目录结构

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

### Tokenizer 加载验证

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

### 文本编码解码测试

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

### Chat Template 检查

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

### AutoConfig 兼容性检查

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

### Qwen3-MoE 的模型数据流

完整阅读 `atom/models/qwen3_moe.py`，并沿实际调用核对相关算子。

### Backbone、Decoder 与 residual

`Qwen3MoeModel`（`:355–431`）为 Embedding → 本 PP stage 的 decoder layers → Final RMSNorm，CausalLM 包装器另持有 LM Head，forward/compute_logits 分离，接口见 Phase 2。

`Qwen3MoeDecoderLayer.__init__()` / forward()（`:265–352`）：

```text
hidden_states + residual
 → input_layernorm
 → Qwen3MoeAttention(positions, hidden_states)
 → post_attention_layernorm(hidden_states, residual)
 → SparseMoeBlock 或 Dense MLP
 → (hidden_states, residual)
```

这里把 residual 作为跨层数据传递，不能理解成漏了残差连接。`model_ops/layernorm.py:RMSNorm.forward():250–279,334–345` 分别处理纯 Norm、add+Norm 和 TP fused allreduce；最后 backbone norm 合并末层 residual（`qwen3_moe.py:429`）。

MoE 层选择条件是 layer 不在 mlp_only_layers、num_experts>0 且 `(layer_idx+1) % decoder_sparse_step == 0`（`:300–315` 附近）；代码不读取 Mellum 的 mlp_layer_types。Mellum config 有全部 sparse 的 mlp_layer_types，但没有 decoder_sparse_step。此差异需要配置/模型入口明确处理，不能默认不存在的字段会自动正确补齐。

另一个工程限制：包装器和 backbone 虽然都有 layer_type 参数（`:453–458`、`:357–362`），包装器构造 backbone 没传该参数，backbone 的 make_layers lambda 直接写死 Qwen3MoeDecoderLayer（`:375–381`）。不能假设只传入一个替代 decoder 类就能让现有链路使用它。

### Attention：输入、GQA、Norm、RoPE 与 Kernel

`Qwen3MoeAttention.__init__()` / forward()（`:142–262`）：

- 输入为 `[T,H]` hidden states 和 `[T]` positions；QKVParallelLinear 合并投影，再按 q_size/kv_size 拆分。
- head_dim 显式存在时优先使用，否则回退 hidden_size/num_heads；head/TP 划分与 KV replication 在 `:162–180`。QK scale 为 head_dim^(-0.5)。
- Q/K 分别按 `[T, heads, head_dim]` 做 RMSNorm（`:250–255`），V 不做 QK-Norm。
- `get_rope()` 构造一个该层使用的 RoPE（`:202–208`）；Attention 收到 rotary_emb，模型普通 forward 不重复显式旋转 Q/K。
- 输出 Attention tensor 为 `[T,local_q_heads×D]`，由 RowParallelLinear o_proj 返回 `[T,H]`（`:190–200,258–262`）。

普通执行链：

```text
Qwen3MoeAttention.forward
 → qkv_proj → split → q_norm/k_norm
 → base_attention.Attention → PagedAttention.forward
 → torch.ops.aiter.unified_attention_with_output_base
 → PagedAttentionImpl.forward_impl_server_mode
 → rope_cache（旋转与写 KV）→ dispatch_backend → Attention kernel
 → o_proj
```

`paged_attention.py:174–189` 将模型的 per_layer_sliding_window、RoPE、Q/K norm 传给 impl；`:222–224` 是 standalone custom-op 入口。`attention_mha.py:89–122` 读取 ForwardContext，reshape QKV，先缓存后执行 Attention。

可选 `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION` 默认 0（`utils/envs.py:50–52`）。开启后 Qwen 把 q_norm/k_norm 和 joint cos_sin_cache 交给 Attention（`qwen3_moe.py:209–229,243–248`），`attention_mha.py:142–226` 的标准 RMSNorm 路径调用 AITER fused_qk_norm_rope_cache_quant_shuffle。默认关闭时 Norm 在模型层、RoPE 在 cache 路径。两种执行方式需要各自验证，不能因接口相同推断数值等价已测。

`Qwen3MoeDecoderLayer` 当前直接取 `config.rope_parameters["rope_theta"]`（`:272–276`），没有 layer_types 分支；构造 Qwen3MoeAttention 时也未传 per_layer_sliding_window。Mellum 的嵌套 rope_parameters 没有顶层 rope_theta，因此只保留原 JSON 而更换 architecture，并不能原样走通这里。

**Mellum 尺寸依据（配置算术，非权重 tensor 实测）：** H=2304，Q heads=32，KV heads=4，D=128。Q 投影宽度应为 4096，K/V 各为 512，O 输入宽度为 4096；不能回退到 `2304//32=72` 的 head_dim。现有 QKVParallelLinear 能表达 H 与 Q heads×D 不相等的投影（`linear.py:1275–1320`），但 Mellum tensor 的实际 shape 本轮未读。

### Qwen3-MoE 的 Router、专家执行与权重

`Qwen3MoeSparseMoeBlock`（`qwen3_moe.py:85–139`）的输入/输出都是 `[T,H]`：

1. ReplicatedLinear gate：H → num_experts，无 bias、不传 quant_config，得到 `[T,E]` router logits。
2. FusedMoE 配置：num_experts、num_experts_per_tok、moe_intermediate_size、norm_topk_prob；use_grouped_topk=False、has_bias=False、reduce_results=False。
3. FusedMoE 返回加权专家输出；TP>1 且未启用 fused allreduce+Norm 时在该 block 显式 all_reduce；否则延迟到后续 Norm。

实际 BF16 无量化路径的调用链：

```text
FusedMoE.forward()                               model_ops/moe.py:2961
 → custom op → forward_impl()                    :3053
 → UnquantizedFusedMoEMethod.apply()              :559–616
 → FusedMoE.select_experts()                      :2862–2959
 → rocm_aiter_topk_softmax_impl()                 model_ops/topK.py:98 起
 → aiter.topk_softmax(..., renormalize, ...)
 → AITER fused_moe（或已初始化的 modular kernel）
```

`select_experts()` 的非 grouped、softmax 分支（`moe.py:2909–2917`）把 renormalize 传至 AITER；topK.py 分配 `[T,K]` FP32 weights 和 int32 IDs 并调用 topk_softmax。没有在这个 Qwen block 中逐专家 Python 循环执行 GEMM。

为闭合 token dispatch → GEMM → aggregation，仅定向阅读 AITER/fused_moe.py：`fused_moe_():258` 接收 hidden states、w13/w2、topk weights/IDs；`:369–393` 调 moe_sorting，生成 sorted token IDs、sorted weights、sorted expert IDs 与 output buffer；之后依据 metadata 分成一阶段或两阶段执行（`:396–450`）。`fused_moe_2stages():1512` 执行 stage1，再于 `:1769–1787` 把 sorted_weights 等交给 stage2 并返回 moe_out。具体 ASM/CK/Triton/FlyDSL 选择依赖 metadata 和运行形状，本轮不声称 Mellum 已选择或验证某个特定 GPU kernel，也未解决指南中 OPUS 警告的问题。

专家参数组织和 loader 已在 Phase 2 给出：gate/up 融为 w13，down 为 w2，按 expert_id 与 shard_id 加载。Mellum index 的命名与 `Qwen3MoeModel.get_expert_mapping():433–441` 的 gate/down/up 名称吻合，见 12.7；这只证明名称协议，不证明所有权重 shape、routing 数学和数值结果已验证。

Mellum config 的 E=64、K=8、I_expert=896、norm_topk_prob=true 对应 Qwen block 的四个参数接口；普通 intermediate_size=7168 不能替代 I_expert。没有索引证据支持添加 shared expert，见 12.7 的完整 key 分类。

### GPT-OSS 与 MiMo：分别可参考什么

| 特性 | GPT-OSS 源码 | MiMo-V2-Flash 源码 | 对 Mellum 的证据边界 |
|---|---|---|---|
| 按层选窗口 | `gpt_oss.py:124–138`，偶数层 window、奇数层 None | `mimo_v2_flash.py:294–332,390–391`，hybrid_layer_pattern==1 为 SWA | 可参考选择逻辑；都没有直接读取 Mellum 的 S/S/S/F layer_types |
| 不同 Attention 参数 | OAIAttention 使用一套 heads/rope_parameters | SWA 与 Full 可有不同 heads、head_dim、rope_theta、v_head_dim | MiMo 体现按层构造接口；Mellum 实际 heads 相同，不能搬入 MiMo 尺寸字段 |
| RoPE | `OAIAttention.__init__():80–91`，平面 get_rope(... rope_scaling=...) | `MiMoV2Attention.__init__():211–217`，只传 theta/partial factor，无 YaRN scaling；forward `:250` 显式应用 | GPT-OSS 可参考 YaRN 传递；MiMo 并不自动处理 Mellum Full YaRN |
| QK-Norm | OAIAttention 未建 q_norm/k_norm | MiMoV2Attention 也未建 QK-Norm | Mellum 的 QK-Norm 候选来自 Qwen，而非直接继承二者 |
| 特殊参数 | QKV/O bias + per-head sinks（`gpt_oss.py:95,102–118,135`） | 可选 SWA sink、V scale、V padding/truncation（`mimo_v2_flash.py:219–269`） | Mellum index 无这些 bias/sinks 项，不应无依据新增 |
| MoE | biased router、has_bias=True、ActivationType.Swiglu（`gpt_oss.py:205–225`） | FP32 gate + correction bias、grouped sigmoid（`mimo_v2_flash.py:98–140`） | 均不同于 Qwen 的非 grouped softmax 候选，不能整体替代 Mellum MLP |

MiMo 在 forward 中先旋转 Q/K，再把 rotary_emb=None 传入 Attention（`:236–250`），是避免重复 RoPE 的具体例子；注释解释其 head_dim=192 与 fused kernel 的限制。Mellum D=128 不自动继承这项限制，也不因此自动证明所有融合分支可用。

### SWA / Full 的真实后端、窗口与缓存

### 窗口从模型到 kernel

```text
模型构造 Attention(per_layer_sliding_window=W 或 None)
 → PagedAttention(...):paged_attention.py:34,184
 → PagedAttentionImpl.sliding_window=W 或 -1:attention_mha.py:76
 → rope_cache 判断 use_triton_attn
 → dispatch_backend 区分 prefill/decode 与 KV layout
```

`attention_mha.py:134–140` 的判断：只要 window!=-1，或 D!=128，或 Q heads==KV heads，就选择 decode 的 Triton/Gluon 分支；普通 D=128 GQA Full 可走 ASM。它不等同于用户开启了 `ATOM_USE_UNIFIED_ATTN`：**AiterBackend 内部本来就可能用 Gluon 做 SWA decode**。

实际分支（`dispatch_backend():644–660`）：

| 阶段 / layout | 函数 | window 的传递 |
|---|---|---|
| 非 flash-layout prefill | `prefill_attention():554–579` | `aiter.flash_attn_varlen_func(window_size=(W,0,0))` |
| flash-layout prefill | `prefill_attention_triton():581–642` | `unified_attention(window_size=(W-1,0))` |
| 非 flash-layout SWA decode | `paged_attention_triton():434–497` | `torch.ops.aiter.pa_decode_gluon(sliding_window=W)` |
| flash-layout decode | `paged_attention_triton():404–433` | `unified_attention(window_size=(W-1,0))` |
| 非 flash-layout Full decode（满足其他选择条件） | `paged_attention_asm()` / persistent 版本 | Full window=-1；block_size=1024 时选 persistent，不能与 SWA window=1024 混为一谈 |

### 静态发现：窗口计数可能差一

从本机 AITER 继续追踪，而不是仅比较参数外观：

- AITER/ops/mha.py 的 FlashAttnVarlenFunc.forward()（`:2453–2455`）把 window_size[0] 直接传 window_size_left；同文件 flash_attn_varlen_func() 的接口说明（`:2654–2656`）定义区间为 `[q_abs-left, q_abs+right]`，两端 inclusive。
- AITER/ops/triton/attention/unified_attention.py:144 定义 `SLIDING_WINDOW=1+window_size[0]`；对应 `_triton_kernels/attention/unified_attention.py:307–311` 的 mask 要求 `q_abs-key_pos < SLIDING_WINDOW`。
- AITER/ops/triton/gluon/pa_decode_gluon.py:5571 原样传 sliding_window 给 kernel；例如 SWA kernel `:2537–2543` 以 `context_length-W` 作为起点，终点为 context_length（exclusive）。

因此，在接口文档所述语义下，传 W=1024 给 Flash 的 left 可能保留“1024 个历史 token + 当前 token”，而 unified 的 W-1 和 Gluon 的 W 表示总计 1024 个 token。标准库区间算术示例：q_abs=2048 时前者 [1024,2048] 共 1025 个，后者 [1025,2048] 共 1024 个。

**性质：源码/API 语义不一致的静态风险，尚未 GPU 复现。** 本轮未深入 CK/ASM 的全部底层 mask 实现，也没有权威 Mellum forward 实现作为最终口径；不能将这一推导写成已测出的错误或提前决定修复。后续正确性阶段需验证 W 附近边界、prefill/decode 一致性，含有 cached prefix 与 chunked prefill 的情况。

AITER 的旧导入路径 `aiter.ops.triton.unified_attention` 实际通过 AITER/ops/triton/__init__.py:107 的兼容映射指向 attention/unified_attention；上述物理文件路径已经核对，不能因旧路径没有同名文件误认能力缺失。

### KV cache：SWA 不等于环形缓存

`AiterAttentionMetadataBuilder.allocate_kv_cache_tensors()`（`attentions/aiter_attention.py:363–408`）标准路径仍分配 `[2,L,physical_blocks,physical_block_size,KV_heads,D]`，没有按该层窗口 W 缩小容量。`build_kv_cache_tensor():488–516` 绑定各层 K/V slice。

MiMo 专用路径（`:379–386,446–487`）逐模块分配，原因是 heads/结构可能异构；依然按 runner 的全部 physical_blocks 分配，不是自动的窗口 eviction。不能把 MiMo 的“逐层不同分配”解释为 Mellum 的环形 SWA cache。

`CommonAttentionBuilder.prepare_prefill()`（`attentions/backends.py:334–435`）按 num_cached_tokens 到 context_len 构造绝对 positions 与 slot_mapping；有 prefix/partial cache 时准备 block tables 和完整 KV 长度。`PagedAttentionImpl.rope_cache():288–294` 在 has_cached 时调用 `_gather_prefix_and_concat_kv():298–392`，写新 KV 后 gather cached+new 成连续 K/V，再供 prefill Attention 使用。

这个链路提供 prefix/chunked prefill 的工程入口，但缓存量与 gather 临时内存仍随历史增长；本轮不把它等同于验证了 Mellum 长上下文性能或语义。

### Backend / 融合组合限制

`TritonMHAMetadataBuilder.build_kv_cache_tensor()`（`attentions/triton_mha.py:66–92`）显式拒绝 MiMo 的 per-layer allocation 路径；也拒绝 impl 同时带 rotary_emb、q_norm、k_norm 的融合 shuffle cache 路径。开启 unified backend 和开启 QK-Norm/RoPE/cache fusion 不是可任意组合的两个独立优化。

普通 cache 路径的 QK-Norm 顺序也应保持模型语义：Qwen 非融合时在模型层先 Norm，Attention 后做 RoPE；不能简单把 norm 从模型层搬到 `rope_cache()` 的其他分支而不检查先后关系。

### Default / YaRN RoPE 的实际复用边界

本轮定向阅读 AITER/rotary_embedding.py 的 get_rope、RotaryEmbedding 相关入口和 YaRNScalingRotaryEmbedding：

- `get_rope():1705–1749` 接收一个平面 scaling dict，以维度、max_position、base、scaling 参数与 dtype 为缓存 key；不负责从 Mellum 的 nested config 选择层类型。
- `:1776–1784` 从 rope_type/type 与 factor 选实现；default 分支 `:1801–1839` 构造普通 RotaryEmbedding；yarn 分支 `:1860–1877` 使用 original_max_position_embeddings、factor、beta_fast/beta_slow 等构造 YaRN。
- `YaRNScalingRotaryEmbedding._compute_inv_freq():647–670` 根据原始 context 做频率混合；`_compute_cos_sin_cache():672–681` 覆盖 original_max×factor 并在 cos/sin 上施加 mscale。Mellum 的 8192×16=131072 与此长度接口对应，但未做长位置 tensor 数值对比。

**幅度参数必须按语义判断：** AITER 读取的可选键叫 attn_factor（`:1866`），其 mscale=`(1+0.1×ln(factor))×attn_factor`（`:609–612,642`）；Mellum config 的键为 attention_factor，AITER 该分支不直接读取这个名字。

然而 Mellum 当前配置的 attention_factor=1.2772588722239782，恰好等于 `1+0.1×ln(16)`，AITER 默认 attn_factor=1 的总幅度也为该值（已用标准库 math 核对）。所以：

- 不能据键名不同直接宣布当前输出有幅度错误。
- 也不能直接把 attention_factor 原数值改名为 attn_factor，否则在该公式下会再次乘默认 mscale，造成双重放大。
- 这里只证明当前常数与 AITER 默认幅度相等；YaRN 的完整频率、rotation style、融合/非融合路径和精度仍需权威参考与运行验证。

已有模块可以表达每层持有不同 RoPE 对象，但 Mellum 按 layer_types 选择子配置尚不是 Qwen/GPT-OSS/MiMo 任一现有模型类的现成行为。

### Mellum checkpoint index：完整名称分类与 MTP 证据

只读文件：

- `/data/models/Mellum2.1-12B-A2.5B-Thinking/config.json`
- `/data/models/Mellum2.1-12B-A2.5B-Thinking/model.safetensors.index.json`

使用 json 解析 weight_map，没有打开 Safetensors。索引 metadata（`:3–4`）为 total_parameters=12,149,923,072、total_size=24,299,846,144 bytes；这是索引声明，不是本轮重算 tensor 大小。5 个索引指向的分片文件均通过 is_file() 确认存在，未检查其头部或张量完整性。

分类结果：

| 项目 | 数量 / 命名 |
|---|---|
| weight_map key 总数 | 5,631 |
| 主模型 layer IDs | 0–27，未出现 >=28 |
| 每层普通权重 | 9：input/post-attention Norm、mlp.gate、Q/K/V/O projection、Q/K norm |
| 每层专家权重 | 64×3=192：experts.0–63 的 gate_proj/up_proj/down_proj |
| 非 layer 权重 | 3：model.embed_tokens.weight、model.norm.weight、lm_head.weight |
| 总计校验 | 28×(9+192)+3=5,631；生成上述预期名称集合，与全部 weight_map key 精确相等，缺失=0、未分类=0 |

具体 key 例子：索引 `:5220` 的 model.layers.0.self_attn.q_norm.weight，`:5550` 的 model.layers.0.mlp.experts.0.gate_proj.weight，`:5378` 的 lm_head.weight。

这里的集合完全覆盖意味着，**当前 index 没有额外的显式 MTP 专用参数项**，不只是简单“搜不到 mtp 字符串”。辅助检索 mtp/nextn/predict/enorm/hnorm/eh_proj/shared_head/pre_fc/draft 均无命中，config 中也没有对应 MTP 配置字段。

结论范围：未发现 DeepSeek/MiMo/Qwen MTP 所需的独立融合投影、额外 decoder、norm 或 draft head 权重。不能凭考核说明写“当前模型目录已包含单层 MTP”；也不能断言分片未被索引的 tensor、其他目录的额外 checkpoint 或其他命名方案绝对不存在。本轮未扫描分片头部，也未做 Phase 4 的容器实现搜索。

名称兼容性方面，Embedding/LM Head、q/k/v_proj、q_norm/k_norm、mlp.experts.N.gate/up/down_proj 与 Qwen 的标准 loader 映射接口吻合。未读取 shape/dtype tensor 元数据，故不宣称实际加载成功或无转换差异。

### 三类 MTP 参考结构与 draft 权重

### DeepSeek MTP

`models/deepseek_mtp.py`：

- `DeepSeekMultiTokenPredictorLayer:43–103`：对 next-token embedding 和 previous_hidden_states 分别 enorm/hnorm，拼接 `[T,2H]`，ReplicatedLinear eh_proj → `[T,H]`，再进入 DeepseekV2DecoderLayer；合并 residual。
- `DeepSeekMultiTokenPredictor:106–163` 从 num_hidden_layers 开始创建 num_nextn_predict_layers 个模块，按 step modulo 选择；compute_logits 使用该层 SharedHead 的 norm + head。
- `DeepSeekMTP.remap_mtp_weight_name():227–231` 委托 get_spec_layer_idx_from_weight_name/rewrite_spec_layer_name（`:270–314`），从额外 model.layers.N 识别 draft 层，把 decoder 权重加 mtp_block 段并处理共享 embedding。

这是 DeepSeek 的 MLA/模型配置和权重协议，不是一个可脱离结构直接套在 Mellum 上的通用 Head。

### MiMo MTP

`models/mimo_v2_flash_mtp.py`：

- `MiMoV2FlashMTPLayer:20–103` 明确构造 SWA Attention + **dense** MiMoV2MLP；不是主模型 MoE 的简单复制。
- `MiMoV2FlashMTPPredictorLayer:106–162` 为 enorm/hnorm → eh_proj → block → residual 合并 → final_layernorm。
- `MiMoV2FlashMultiTokenPredictor:165–205` 维护从主层数开始编号的 ModuleDict；`MiMoV2FlashMTP.__init__():219–227` 当前断言 num_speculative_tokens==1。
- `remap_mtp_weight_name():293–331` 识别 `model.mtp.layers.N.*`，把 predictor keys、pre_mlp_layernorm、embed_tokens 和其余 decoder 权重映射到相应位置。

其 1 个 draft token 限制与模型含几个 predictor layers 是两个概念；不能从某个默认 num_nextn_predict_layers=3 推断当前能做 MTP3。

### Qwen3.5 MTP

`models/qwen3_5_mtp.py:25–105` 的 Qwen3_5MultiTokenPredictor：分别 pre_fc_norm_embedding/hidden，concat 后 ColumnParallelLinear fc，再 TP all_gather，进入强制 full_attention 的 Qwen3_5DecoderLayer 并做末 Norm；LM Head 在外层 Qwen3_5MTP。

权重过滤由 remap_mtp_weight_name（`:121–134`）和 `weights_mapping={"mtp.":"model."}`（`:119`）配合；与 DeepSeek/MiMo 的 checkpoint 格式不同。这里只研究接口，不推断 Mellum 必须采用其中一种结构。

### MTP 的注册、共享、提出、验证与限制

### Config 与加载协议

`SpeculativeConfig`（`config.py:807–826,860–930`）先将 model_type 映为 MTP type，再改 architectures / predictor 层属性；当前映射无 mellum。`spec_decode/eagle.py:24–31` 的 draft registry 也无 Mellum draft 类。

`EagleProposer.__init__():187–212` 从 draft config 查类；普通 MTP 将原 atom_config 交给模型构造，而 Eagle3 有独立 config/cache 分支，不能混用。`load_model():257–318` 对 MTP 调通用 loader(spec_decode=True)，随后按条件共享 Embedding/LM Head；Embedding 在 PP=1、shape 相同时直接替换为 target 模块，LM Head 在未记录加载时共享，另可提供 share_with_target hook。所依赖的 loaded record 局限见 Phase 2。

MTP 至少要求：额外 hidden_states 输入、与 target 配合的 positions/KV、compute_logits 接口、checkpoint remap 与共享策略、配置/draft 注册，以及拒绝后缓存/位置更新。不能只写一个名为 MellumMTP 的类就认定接入完成。

### 提出与 target 验证

`EagleProposer.propose()`（`eagle.py:330–489`）：

1. 使用 target token IDs/positions/hidden_states，置 is_draft=True，把 next_token_ids 写入对应位置，positions 加一。
2. 循环 mtp_k 次调用 draft forward；第一次选每请求最后 hidden state，之后使用返回状态；compute_logits 后用 argmax 提出 draft token（`:391–408`）。
3. 多步时调整 query length、context lengths、positions、KV slot / indices，并调用 builder.prepare_mtp_decode（`:410–482`）。
4. 返回 `[batch,mtp_k]` IDs。配置 predictor layer 数与 mtp_k 不等同；当前 proposer 调用没有显式传 spec_step_idx，不应假定循环自动轮换多个不同 predictor layer。

下一次 target 批次评分时，`calc_spec_decode_metadata():516–562` 计算 draft/bonus logits 索引和 draft IDs；ModelRunner.postprocess（`model_runner.py:1930–1959`）调用 RejectionSampler，并用 mtp_k-num_bonus_tokens 得拒绝数，后续调度根据结果更新 Sequence。

`model_ops/rejection_sampler.py:10–16,89–130,150–178` 显示默认是严格 greedy：逐项要求 draft ID 等于 target argmax，首次不符时输出该 target token 并停止后续接受，全部匹配则追加 bonus token。开关 ATOM_ENABLE_RELAXED_MTP 默认 0（`utils/envs.py:119–121`）；开启时有 top-10/delta=0.6 的放宽接受路径。

这段实现没有使用完整 draft probability 做通常的随机接受比率校正，不能笼统声称任意温度采样下保持 target 分布；本轮也没有运行温度或接受率实验。

接受率统计来自 `Scheduler.SpecStats.update()` / acceptance_rate（`scheduler.py:58–84`），按输出 token 数减一统计额外接受 token，除以累计 draft 数；`:1038–1048` 排除没有实际验证 draft 的 prefill 步。本轮没有任何实测接受率。

### 普通 MHA draft 的额外接口缺口

- `EagleProposer.propose():466` 的多步循环要求 metadata builder.prepare_mtp_decode。当前该方法只在 `aiter_mla.py:524`、`gdn_attn.py:524`、`deepseek_v4_attn.py:956` 中有实现；普通 AiterAttentionMetadataBuilder、CommonAttentionBuilder、TritonMHAMetadataBuilder 都没有定义。已核对 `plugin/attention.py:692–745` 的装饰器：独立模式继承 CommonAttentionBuilder，不会自动注入该方法。因此不能认定普通 Mellum MHA 的多步 draft 已具备完整框架；mtp_k=1 不进入这段步间更新，但也不因此自动具备模型/权重。
- 标准 `AiterAttentionMetadataBuilder.allocate_kv_cache_tensors()` 虽接收 num_draft_layers，实际标准 tensor 的 layer 维度用 hf_config.num_hidden_layers（`aiter_attention.py:363–405`）；MiMo 的逐模块分配和 GDN 等专用路径不同。新普通 MHA MTP 还需核对 draft KV 容量与 layer 编号，不能默认已经预留。
- 这些为静态接口缺口，不是已运行的 Mellum 错误；本轮不新增方法、不扩 cache、不尝试 draft 推理。

### 本轮补充证据：五个分片头部的全量布局核对

为减少设计假设，使用 Python 标准库对已知五个 Safetensors 文件逐个只读 **8 字节长度前缀 + JSON header**；每个 header 设置 16 MiB 上限。没有调用 torch/safetensors/AITER，也没有读取张量 payload。来源目录为 `/data/models/Mellum2.1-12B-A2.5B-Thinking/`。

| 分片 | JSON header bytes | header 中 tensor 数 |
|---|---:|---:|
| model-00001-of-00005.safetensors | 133720 | 1092 |
| model-00002-of-00005.safetensors | 147296 | 1202 |
| model-00003-of-00005.safetensors | 148208 | 1202 |
| model-00004-of-00005.safetensors | 148208 | 1202 |
| model-00005-of-00005.safetensors | 115192 | 933 |

总共读取头部 692,664 bytes。检查结果：

- unique keys=5631，重复 key=0；所有 dtype=BF16。
- header key 集合与 index.weight_map 完全相等，索引的分片归属无错配；不存在额外未索引 tensor。
- header shape 与按 config 推导的主模型全部 5631 个名称/shape 逐一一致，missing/unexpected/shape mismatch 均为 0。
- offsets 均在文件范围内且连续、不重叠，末尾与文件长度一致；每个 tensor 的 storage bytes 等于 2×shape 元素数。
- shape 总元素数=12,149,923,072，payload 声明总 bytes=24,299,846,144，与 index metadata 一致。

**更新 Phase 3/4 的边界：** 对当前这五个实际分片，已经可排除“额外 tensor 藏在 header、未列入 index”的情况；全部参数均被主模型名称集合解释，未见 MTP 专用参数。仍不能排除其他未提供 checkpoint 的 MTP，也没有检查 payload 值、NaN、损坏或加载后的数值。

实际磁盘 shape：

| 权重 | 每层 / 全局 shape |
|---|---|
| Embedding、LM Head | [98304,2304]，两者各有独立 tensor |
| input/post-attention/final Norm | [2304] |
| q_proj / k_proj / v_proj | [4096,2304] / [512,2304] / [512,2304] |
| o_proj | [2304,4096] |
| q_norm / k_norm | [128] |
| mlp.gate | [64,2304] |
| 每 expert gate_proj / up_proj | [896,2304] |
| 每 expert down_proj | [2304,896] |

## 当前结论更新

早期段落所述“MTP 尚待调查”已被本页后续 index/header 核对更新：当前提供的主模型5631个参数均已解释，未见额外MTP张量。AutoConfig失败是此前本地检查的结果，后续将再次核验安装路径、注册与官方实现版本；不能解释为HF官方不支持Mellum。

## 官方HF语义复核更新（2026-11-10）

已阅读官方Transformers v5.10.1 Mellum实现，确认QK-Norm→RoPE→cache、FP32路由softmax→top-k→归一化等数据流，以及HF窗口包含当前token的边界。官方来源、符号/位置、YaRN与冻结5.2.0接口差异统一见[参考调查](04_correctness_reference.md)。这是源码证据，尚未建立可运行参考或完成checkpoint数值验证。
