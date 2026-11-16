# 42 Systems诊断完成：参考控制全部一致

运行/workspace/results/mellum_greedy_recovery/20261116T142518071993Z_systems，diagnostic_complete、source_unchanged=true；92项产物哈希一致，5631项加载通过。

| 配置 | 200位置率 | 200整序列 | 长组 |
| --- | ---: | ---: | ---: |
| 优化基线 | 65.3594% | 71/200 | 36.3281% |
| 关闭prefix | 64.7891% | 69/200 | 24.6094% |
| Torch路由 | 66.6719% | 67/200 | 41.7969% |
| 分离QKV | 62.2656% | 63/200 | 33.2031% |
| 原优化profile batch1 | 64.1016% | 69/200 | 48.4375% |
| R参考数学控制，batch1 | **100%，12800/12800** | **200/200** | **100%，256/256，4/4序列** |

16k参考控制64/64。实际R控制为Torch grouped/batched专家、参考Norm、分离QKV、参考SDPA/参考KV、eager、单槽位、无prefix；groups=LE/shared_routing，batch1下不发生多请求拆分。它不是历史Step0 tag，不用于冒充其正式性能行。

## 证据意义

同一冻结环境、预置权重、HF参考和输入管道能够产生全部一致的输出，因此无需重装环境或重新生成HF作为默认动作。关闭prefix、减少batch、只恢复路由/投影不足以解决优化版差距；Norm控制也只到72.9%。这进一步支持优先检查原生专家和Attention管线。

不能由此断言优化版一定只是正常浮点误差，也不能彻底排除优化后端特有的权重shuffle/缓存布局或组合实现问题。参考控制证明检查管道可用，不自动证明所有优化后端的实现正确。

## 资源安排

GPU0已启动20条2×2专家/Attention短矩阵；只读看到core_nn首项完成，诊断子集69.84375%，不能外推为原200通过。等待native-Attention+Torch专家及reference-Attention+native专家结果，选最有希望的组合后利用空闲GPU1跑完整200，而非重复慢参考控制或直接全面回滚。

无需启动计划47快速HF Norm后备：agree_norms/agree_norms_route仍未到90%，R参考控制的100%不能替代该前置证据，因为它同时更换了专家/Attention等路径。
