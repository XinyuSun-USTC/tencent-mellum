# 18 Torch后端扩展回归与Attention同输入定位

原始结果`results/mellum_validation/20261111T173427695865Z`，ATOM`results/atom_mellum_gpu/20261111T173519833603Z`。新torch后端仍有7/11条分叉，4条全匹配；704位置513相同，仅作分叉后对齐统计，非正式验收指标。

| 输入 | 原AITER公共前缀长度 | 新torch公共前缀长度 |
|---|---:|---:|
| h_0 | 14 | 14 |
| h_1 | 9 | 64 |
| h_10 | 22 | 64 |
| h_100 | 23 | 44 |
| frequencies | 24 | 24 |
| word_problem | 39 | 64 |
| binary_search | 64 | 60 |
| multiply | 64 | 64 |
| window_1025 | 9 | 34 |
| long_8192 | 64 | 8 |
| long_16384 | 28 | 28 |

旧结果`20261111T132520303449Z`。部分改善同时出现新分叉，不能据4/11相较3/11认定整体正确。long_8192第9新token就分叉；其原始prefill全词表RMS虽从1.014下降到0.576，仍未对齐。h_0还需保留已观察到的HF cached/prefill自身差异解释边界。暂停增加样本或生成长度。

## 下一步固定Q/K/V

首层同输入MoE已精确对齐，而正常short首层Attention context仍有小量差异。源码`ATOM/atom/model_ops/attention_mha.py`调用`self.rotary_emb(position,q,k)`，AITER `rotary_embedding.py:RotaryEmbedding.forward`再按后端选项分派；HF模型用`MellumRotaryEmbedding`及`apply_rotary_pos_emb`后进入SDPA。仅比较RoPE缓存不能证明旋转算术一致。

新增`scripts/diagnose_mellum_attention_math.py`，只读取四条保存HF探针的q_norm/k_norm/v_proj，固定同一Q/K/V。分别执行官方HF与实际AITER RoPE，再使用相同官方滑窗mask构造和`sdpa_attention_forward`计算context。先核查HF RoPE+SDPA能否复现保存HF context，再比较AITER RoPE引入的变化，同时对照旧ATOM context。保存旋转后Q/K中间值。

该脚本只处理142–166 token的第0层滑窗prefill，不加载模型权重或完整引擎。它隔离旋转与prefill Attention，不能证明full YaRN、跨窗mask、decode或KV缓存正确。AITER缓存在GPU构造，以匹配引擎默认设备；没有预先改模型RoPE或Attention实现。

语法与CLI检查通过，GPU未执行。普通终端：

```bash
/workspace/venvs/hf-reference/bin/python /workspace/scripts/diagnose_mellum_attention_math.py
```

若HF自身不能复现，先核对mask及SDPA路径；若旋转Q/K不同再定位舍入；若旋转一致但旧ATOM context不同，继续对齐Attention内核输入/输出。`diagnostic_complete`仅采集状态。

## 诊断入口导入修正

首次普通终端执行因独立HF环境未发现editable安装的aiter而失败，尚未产生Attention比较结果。新增`import_native_rope`，显式加入冻结源码`/app/aiter-test`及现有扩展目录`/app/aiter-test/aiter/jit`，不安装包或升级环境。通过路径解析确认aiter及module_aiter_core.so均来自原容器，语法检查通过。工具进程的完整AITER导入因GPU架构检测依赖rocminfo而受阻，不能声称已在工具内完成GPU导入验证；普通终端继续执行原命令。正常ATOM模型代码未改。

第二次普通终端已成功导入原AITER，但诊断脚本错误地向`MellumRotaryEmbedding.__init__`传入device。实际签名只接受config，现修为`MellumRotaryEmbedding(cfg).to('cuda')`，也匹配原HF先CPU构造再移GPU的流程。使用独立HF环境对四份真实保存Q/K/V在CPU完整执行构造、rotary forward、apply_rotary_pos_emb、mask构造及SDPA调用，输出shape/有限值四例通过，cuda_initialized=False。此检查仅验证接口及数据路径，不是GPU数值验证。正常模型代码未改。

## 首轮同QKV结果与实际融合分支核对

真实产物`results/mellum_attention_math/20261111T194826009449Z`：四例HF旋转+官方SDPA全部逐元素精确复现保存HF context，mask_is_none均为true。独立AITER HIP旋转相对HF的Q误差约0.00220–0.00224、K约0.00235–0.00239，最大绝对差分别0.125、0.0625；经同一SDPA后context误差约0.00186–0.00194。但是该结果不复现旧ATOM context，不能直接将独立HIP旋转差异认定为实际首层根因。

进一步源码：`ATOM/atom/model_ops/attention_mha.py:PagedAttentionImpl.rope_cache`因sliding_window!=-1设置use_triton_attn=True，在QK融合norm关闭时调用`aiter.ops.triton.fused_kv_cache.fused_qk_rope_reshape_and_cache`，而不是rotary_emb.forward的HIP路径。其物理源码位于`/app/aiter-test/aiter/ops/triton/fusions/fused_kv_cache.py`。use_flash_layout默认为False。此次实验只比较了独立HIP分支，对实际融合旋转尚未验证。

已在相同脚本补充实际Triton融合函数：相同Q/K/V、position/slot_mapping、BF16缓存、block_size16、NHD对应的5维K/4维V缓存、apply_scale=False、q_out/k_out原位。分别保存fused旋转后Q/K并通过相同SDPA比较context，字段为fused_q_rope/fused_k_rope及contexts.atom_fused_rope。缓存仅为小型诊断分配，不进入引擎。已核对实际函数签名及布局并通过语法检查；未GPU执行，未改变模型代码。仍用原命令重试。

若融合路径旋转与HF一致，则保留独立HIP差异作为旁支事实，主线转向实际Attention kernel；若不同，继续隔离缓存值与旋转舍入后才确定修复。不能用当前独立HIP数值替代实际首层证据。

## 实际融合旋转结果与prefill内核对照

真实产物`results/mellum_attention_math/20261111T195637190126Z`。四例HF旋转+SDPA仍精确复现参考；实际Triton融合旋转Q相对L2约1.11e-5–2.14e-5，K约2.78e-6–8.40e-6。融合旋转+相同SDPA相对HF context约3.59e-5–5.20e-5，而旧ATOM context相对HF约5.995e-4–7.595e-4。旋转差异存在，但该隔离实验不能解释旧context的主要差异；不同误差不可直接线性相减，也尚未证明完整模型生成问题来源。

源码`PagedAttentionImpl.dispatch_backend`：prefill且use_flash_layout=False时调用`prefill_attention`，其实际调用`aiter.flash_attn_varlen_func`，使用causal=True、scale=128**-0.5、window_size=(1024,0,0)、单请求cu_seqlens及对应最大/最小长度。decode另走paged_attention_triton，不由本次实验覆盖。

已向同脚本加入两个交叉组合：固定HF旋转Q/K+实际AITER prefill内核，以及实际融合旋转Q/K+实际AITER prefill内核。所有组合复用相同保存HF的Q/K/V，比较HF context和旧ATOM context，字段为contexts.atom_prefill_hf_rope及contexts.atom_prefill_fused_rope。与原ATOM输入仍有少量projection差异，不能要求该组合必然精确复现旧ATOM；主要用途是在相同输入下分离Attention算术与旋转。核对原调用参数，语法检查通过，GPU待执行，正常模型未改。仍执行原命令。
