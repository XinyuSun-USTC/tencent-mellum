# 最终三项组合一致性回归：已知范围

核对运行 `/workspace/results/mellum_delivery_matrix/20261117T003519808987Z` 中step012与step0123的runtime及生成结果，未启动新GPU任务。

两行实际差异包括decode metadata once、SWA group4、前缀缓存、1024token分段预算、四轮decode配额与断连取消。③不是此前单独通过的SWA候选，当前不能仅凭回归将失败归因到SWA或通常浮点误差。

相对step012，step0123有103/200条实际序列不同；其中75条输入少于1024token、28条超过1024token，没有首个输出token分歧。短输入也不能排除chunk影响：1024是整个调度批的预算而非每请求预算，另有共享前缀后缀计算路径。

四条长输入均与step012分叉：8191/8192/8193/16384分别在0起算的第8、17、8、8个生成token首次不同。此前集成smoke仅生成8token，因此这些四条长输入的前8token一致与本次回归并不矛盾；短smoke恰好未覆盖后续分叉。此前SWA单项的200×64及长组一致，也不能继承给新增prefix/chunk组合。

核验实现：`mellum_batch.ragged_prefill`在reference prefix/chunk打开后调用 `mellum_chunked.cached_prefill`；当cached>0时，从分页KV读取历史、构造绝对位置mask、显式展开KV头并执行SDPA。该路径在数学mask之外还涉及位置、缓存状态、形状与阶段执行。公式正确不能证明实现或数值兼容；需要单变量重跑隔离，当前无证据确定具体错误算子。

集成流程的问题是：完整验证推迟到最终轮，此前仅凭8token集成smoke筛选，未及时覆盖组合64token回归。不能将该问题表述为参考实现不可靠、GPU随机性或题目门槛过严。下一步以step012为通过对照，分别测试仅SWA/metadata、仅prefix、仅chunk，必要时再交叉；先200×64与四条长组，定位后再决定是否值得重跑HumanEval及ABC。
