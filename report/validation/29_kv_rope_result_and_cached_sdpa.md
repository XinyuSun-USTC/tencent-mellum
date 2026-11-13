# 29 KV旋转对照结果与同分页缓存SDPA decode候选

## 实测结果

原始运行`results/mellum_decode_validation/20261112T141141879416Z`。对比基线`20261112T134058133231Z`，四prefill仍精确，28个decode均未精确；部分位置改善，另有位置增大。

首步decode最大logits差：chinese_inventory 0.25→0.25，raw16 0.5625→0.131836，raw17 0.53125→0.96875，window1023 0.75→0.5。不能将Q/K旋转与K写回对齐当作完整decode修复。强制历史/position、加载审计仍通过，参考HF cached不变。

## 下一单因素候选

新增`mellum_attention.py:gather_reference_kv/sdpa_cached_decode/install_sdpa_decode`，只在显式`--reference-kv-rope --sdpa-decode`启用，保留本轮正确Q/K和K写回，再用同一实际分页缓存计算SDPA。没有引入第二份独立缓存或绕过调度。

- 读取block_table的真实物理block映射，K按5维[B,H,D/x,BS,x]取值。
- V按原rope_cache记录的_cache_format区分NHD四维和SHUFFLE视图[B,H,BS/x,D,x]，不把两者混读。
- 单请求、单token decode，核对context_lens与绝对position+1一致。
- full层读取完整历史；sliding层仅读最后window个token，包括当前token。
- 对齐HF DynamicSlidingWindowLayer：过去最多保留window-1，加入当前token后最多window。窗口阈值起使用显式全True mask及展开KV头，之前使用无mask GQA；单tokendecode is_causal=False。

依据：HF `cache_utils.py:DynamicSlidingWindowLayer.update/get_mask_sizes`、masking_utils及sdpa_attention_forward；ATOM `PagedAttentionImpl._gather_prefix_and_concat_kv`的布局说明及rope_cache的_cache_format赋值。候选尚未GPU核验，不把CPU读取算法视为实际缓存已正确证明。

## 同输入取证合并执行

新候选每请求首个decode的第0层额外调用一次原decode Attention方法，使用相同query和实际缓存，并保存`decode/sample_NNNNNN/first_decode_attention.pt`：Q、逻辑顺序K/V、SDPA/native输出、可见范围和block table。仅这一小点保留两路结果，不重复完整模型。后续可直接比较同Q/缓存输出，避免再次GPU取证。该额外计算没有用于性能评测。

CPU模型相关43项、运行器5项通过，新增NHD/SHUFFLE物理block重排、跨block读取、窗口裁剪及SDPA参考测试；取证代码加入后Attention7项再次通过。脚本语法/diff已检查。新GPU候选未执行，默认模型路径不变。

普通终端：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_decode_validation.py --reuse-hf-run /workspace/results/mellum_decode_validation/20261112T123830320864Z --reference-kv-rope --sdpa-decode
```

继续复用HF并保留心跳、失败检测和超时。先检查32个原始logits位置，再使用同缓存首层取证区分内核与取数；不以被强制的token相同作为通过，不提前宣称完整KV、服务或Step0完成。
