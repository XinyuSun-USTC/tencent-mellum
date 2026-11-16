# 等待诊断期间的 P/D 可行性与解码候选

本页为CPU源码核对与后续决策，不启动GPU，不修改正在运行的 `step1/c-service-diagnosis`。纯解码和断连诊断未完成，不提前认定主因。

## 1. 优先考虑的兼容增量

| 候选 | 当前源码证据 | 可能收益与边界 |
|---|---|---|
| 解码元数据一次解析、各层复用 | `mellum_batch.ragged_decode()` 每层读取context_lens；`mellum_attention.sdpa_cached_decode()`对每个请求读取context/position标量 | 减少重复GPU到CPU取值及同步。作用域必须是一次forward；新步、请求重排、复用和长度更新均使旧值失效。保持Q/K/V、mask和SDPA调用不变，收益待测 |
| 按请求KV gather/attention执行粒度 | ragged_decode按请求循环，随后gather与SDPA | 若纯decode仍不足，应研究小组执行减少调用。需验证padding/mask、实际KV读取和数值行为，不能直接恢复此前失败的整条原生attention |
| 超时取消与资源回收 | 已定义ABORT但未接通处理，流生成器提前终止缺少清理保证 | 若真实断连后仍有新步骤，补finally和主线程取消。必须等待异步forward不再使用KV后释放，不能只删前端映射或在IO线程直接改调度队列 |
| 缓存+四轮配额 | 单独缓存已复用35.47%输入，四轮分段完成数较一轮更高 | 可保留一次有界组合筛查，但二者单项都未解决容量，不应成为无界参数网格 |

优先根据当前decode曲线确定优化粒度；不再将prefill减负当作可以自动提高decode容量的保证。SWA解码已经按窗口读取历史，物理缓存截断的额外收益需要另查，不能重复计算“避免读取窗口外KV”的贡献。

## 2. 双卡 P/D 的潜在收益

P卡处理prefill，D卡只处理decode，可以隔离长prefill的资源占用。它能改善干扰，但不直接提高单张D卡的模型计算能力，也不免费省掉KV传输、协调和排队。

题面按声明卡数扩大负载：单卡C为200请求/0.3req/s；双卡为400请求/0.6req/s、输出256。若一张P、一张D，D侧需求约153.6输出token/s。若当前纯decode在已测批量范围明显低于此值，不能指望仅靠1P1D解决题面双卡负载。单卡0.3负载上的1P1D试验只能是开发诊断，不能当正式双卡成绩。

## 3. 冻结源码和环境的可用基础

- `ATOM/atom/kv_transfer/disaggregation/factory.py:KVConnectorFactory` 提供后端注册，默认MoRI。
- `moriio/moriio_connector.py:register_kv_caches()`区分MLA与MHA，对MHA注册K及V缓冲区；这是接口支持证据，不是Mellum两种缓存布局已验证的证明。
- Mooncake连接器也有block-indexed传输路径；`ATOM/recipes/pd_disaggregation_guide.md`示例以多节点RDMA为前提，不能当作同机双卡零配置说明。
- 在`/opt/venv`用find_spec可找到mori，未找到mooncake。未导入/初始化GPU传输引擎，未核验同机P2P、RDMA设备、实际带宽或传输完整性。不得为此升级冻结推理依赖。

## 4. Mellum KV 传输量的结构估算

BF16、28层、4KV头、head_dim128、K/V各一份，原始逻辑数据为每token 57344字节。16k整段全层约896MiB；若7个全局层传全部16k、21个SWA层只传末1024，约266MiB。该计算不含布局填充、注册、临时缓冲、首token与传输协议开销，不是实测带宽或时延。

当前K是五维分页布局，V在全局/SWA路径有SHUFFLE/NHD差异。按层缩减传输需对照块表、绝对位置和D侧后续写入，不能只裁剪tensor就宣称正确；既有连接器按块传输的接口也不自动证明按层窗口截断已支持。

## 5. 若诊断支持 P/D，再启动的最小路线

1. 在冻结依赖下核验MoRI同机双卡最小缓冲区往返，测完整性与实际带宽。
2. 用单个短请求闭合P生成KV、D接收、首token衔接及继续decode；保持同一参考数学路径和布局。
3. 扩展全局/SWA、滑窗边界、长输入及多请求块复用，核对传输后的greedy输出。
4. 与同参数单卡比较后，再执行按两卡扩大后的完整ABC和正确性；所有双卡推理计一个优化点，需重新选择最多三个方向，不能在现有三点之外隐式增加。

当前不立即开工完整P/D，也不预估必然显著加速。等待纯decode是否有足够服务能力的证据，可以避免消耗大量时间搭建一个仍受D侧计算限制的系统。
