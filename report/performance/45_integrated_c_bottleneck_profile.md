# 单卡集成方案的稳态decode与受控混合负载profile

## 完整性与口径

原始结果：`/workspace/results/mellum_c_bottleneck_profile/20261117T000857196743Z`（decode）、`20261117T000905500998Z`（mixed）。各11个文件哈希校验全部一致；运行内源码检查未变化，GPU kernel与模块标签均存在。配置固定 `r_swa4_abort_prefix_chunk1024_d4`，单卡。该实验不是正式C泊松负载，不用于正确性晋级。

decode组16条输入并发，每条16输出，重复同批提示形成暖前缀；mixed组先4请求，实际至少4轮decode后加入4条新输入，每条16输出，各pass使用不同提示但允许共享前缀。两次无profiler测量仍有阶段审计及标签hook。

| 项目 | decode | mixed |
|---|---:|---:|
| 无profiler wall秒，两次 | 5.490 / 5.497 | 16.031 / 15.544 |
| trace wall秒 | 6.995 | 22.465 |
| trace实际prefill/decode step | 1 / 16 | 57 / 116 |
| GPU kernel数 | 172608 | 614900 |
| 模块归因覆盖 | 99.9% | 99.5% |
| kernel累计时间秒 | 4.774 | 12.942 |
| kernel时间跨度秒 | 6.937 | 22.446 |
| 时间跨度内kernel空隙比例 | 31.19% | 42.34% |

mixed每次提示、实际阶段数不同，不能将其wall比值解释为精确profiler开销；decode暖负载约1.27倍的wall扰动也不能外推。空隙不是已证明的CPU瓶颈，kernel累计时间与CPU范围不混加。

## 可定位的开销

decode trace的decode阶段累计kernel4151.753ms，其中：全局层SDPA1253.589ms（30.19%）、SWA SDPA296.304ms（7.14%）、专家1060.287ms（25.54%）、参考KV gather498.439ms（12.01%）、SWA分组范围内其余kernel464.593ms（11.19%）。按最内层范围归因，分组范围不再包含已分给sdpa的时间；与inclusive表不可相加。全局/SWA按模型每四层一层全局的层号区分。

mixed trace中prefill阶段9582.000ms，约占总kernel时间74.04%；prefill SDPA3883.884ms，专家3408.915ms，分别占该阶段40.53%与35.58%。其中全局SDPA2931.747ms，占prefill阶段30.60%。其decode阶段3348.525ms，全局SDPA663.717ms（19.82%）、专家765.557ms、参考gather394.947ms。

这支持同时关注全局注意力与专家路径；不支持“只改chunk或只修取消即可解决C”的判断。trace中的全局attention父范围包含投影、SDPA等子范围，不能把父子数值相加。原始分析JSON还包含memcpy及CPU事件，但copy_包含等待等因素，不能由其self CPU时间断言HBM搬运耗时。

## 建议与交付边界

若继续单项开发，优先调查全局层分页KV读取和逐请求调用：`mellum_batch.ragged_decode`目前仅SWA分组，全局仍逐请求调用`mellum_attention.sdpa_cached_decode`与`gather_reference_kv`。先尝试保持单请求SDPA形状和算术路径、减少KV索引/物化/调用的方案；改变全局SDPA批量形状则作为独立数值候选，不能预先保证一致性。KV读取完全消除也只对应当前decode kernel约12%的理想局部上限，不等于wall收益，更不保证C通过。

若没有剩余开发时间，保留该profile作为第三方向的瓶颈解释证据，交付已有改善及失败范围。C60仍43/60，完整C、最终组合完整正确性和正式累计消融的缺口不因profile完成而消失。
