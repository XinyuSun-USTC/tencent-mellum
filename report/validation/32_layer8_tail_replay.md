# 32 窗口首decode首差定位至索引8层后半段

## 全层观察结果

原始运行`results/mellum_decode_validation/20261112T152156609360Z`，ATOM`results/atom_mellum_gpu/20261112T152209900643Z`。单窗口样例prefill所有观察精确，首decode embedding与索引0–7层输出精确，首个非精确状态是索引8（第9个Decoder层）的输出：max_abs0.0625、相对L2约0.00432935。

索引0–8层的Q、可见K/V和SDPA context全部精确（9层×4项共36项）。索引9开始query等出现差异，属于前层输出已不同后的传播，不能把索引9 query差异直接当作RoPE错误。60份prefill/decode层状态中21份非精确，均为首decode索引8–27及final norm；全112项Attention观察中76项非精确，首差在索引9。

因此当前最小定位范围为索引8的O投影、残差、post-norm、router或MoE，而不是缓存/Attention前半段。

## 轻量同输入后半段重放

新增`scripts/diagnose_mellum_layer_tail.py`，自动读取first_nonexact_state选择层，要求前层输入和该层Attention Q/K/V/context都精确后才执行。

仅加载原checkpoint该层的O、post-norm、gate及64个专家权重，不构造完整12B模型，不运行前8层或prefill/cache。两端复用同一canonical专家张量：

- HF分支：3D F.linear O投影、官方MellumRMSNorm、F.linear router及官方grouped_mm专家。
- ATOM分支：2D AITER tgemm O/gate、实际MellumRMSNorm类（从工作副本源码提取）、实际MellumTorchExperts；保留层参数布局和算术。
- 另用完全相同HF norm/router输入重放ATOM专家，分离上游因素与专家计算。

逐项比较O、残差、norm、router logits、专家ID/权重和MoE；将两路重建的最终层输出分别对照保存的真实HF/ATOM层输出。两端真实输出都能精确重建时才标记inspect_components，否则resolve_reconstruction_mismatch_first，不据不完整重建归因。

原始层权重数值仍以checkpoint读取为依据；本次不是从存活ATOM进程导出的参数全量审计。前prefill全层精确、真实输入/context精确提供额外证据，但不把它替代所有参数证明。

## 本轮检查

CPU prepare-only已验证源首差、前层状态和该层四项Attention输入/context精确，产物`results/mellum_layer_tail/20261112T154911393422Z`。实际MellumRMSNorm类提取/构造接口及BF16 2D/3D对官方CPU结果精确，未初始化CUDA。语法和CLI检查通过，GPU重放未执行；正常模型实现没有改变。

普通终端：

```bash
/workspace/venvs/hf-reference/bin/python /workspace/scripts/diagnose_mellum_layer_tail.py --run /workspace/results/mellum_decode_validation/20261112T152156609360Z
```

先观察两路真实层输出重建是否精确，再根据第一个不同组件决定局部修正。当前Step0仍未完成，不继续增加自由生成或性能实验。
