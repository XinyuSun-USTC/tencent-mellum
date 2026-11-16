# 44 完整ABC结果与最终候选验证

核验full_campaign 20261116T131621651602Z、metadata_finish 20261116T131627296724Z：68+142项产物哈希全部一致，源码未变，各请求完成且无参数/长度审计错误。

## 完整规模已确认的结果

当前配置sprint_graph_qk_router，1GPU、BF16、Graph、Norm/Packed、QK融合及native route，prefix/chunk/metadata关闭。

| 指标 | 实测 | 官方baseline | 比较 |
| --- | ---: | ---: | --- |
| A512吞吐 tokens/s/GPU | 1129.3798 | 638.2827 | 1.7694倍，+76.94% |
| B50 TPOT p50 ms | 4.9299 | 未给独立B数值 | 报告实测，不虚构对比 |
| B50 TTFT p99 ms | 96.2116 | 未给独立B数值 | 报告实测 |
| C200 TPOT p99 ms | 13.6623 | 41.2650 | 下降66.89% |
| C200 TTFT p99 s | 1.48832 | 0.99355 | 增加49.80%，仍是薄弱项 |

A512/B50/C200均全部成功，真实decode最大batch分别32/1/6。C吞吐76.8156主要受固定到达率限制，不用它宣称大幅性能提升。本次是注明客户端适配/到达差异的本地一次复测，GPU1并行筛选，不冒称裁判空闲隐藏集三次中位数；target_g仍null，不虚构评分。

## 小规模最终筛选

| GPU1配置 | A64 tokens/s | C20 TPOT p99 ms | C20 TTFT p99 s |
| --- | ---: | ---: | ---: |
| QK+route | 1110.7978 | 16.4465 | 0.91776 |
| 加prefix | 1155.0187 | 12.6662 | 0.74052 |
| prefix+metadata | 1130.6071 | 12.4346 | 0.70070 |

metadata真实validated/reused回执存在。三项均服务成功、输入不变。prefix单项A略高，但单次小负载不能据此认定metadata稳定损失；考虑C的TTFT弱项，选择prefix+metadata进入完整测量，不把C20直接赋给C200。

## 接下来两卡命令

GPU0：prefix+metadata最终候选完整ABC。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-campaign/scripts/run_mellum_full_campaign.py --gpu 0 --profiles sprint_graph_qk_router_prefix_metadata --budget-minutes 90
```

GPU1：同一组合的一次性开发200/长组/HumanEval验证，复用HF。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-acceptance/scripts/run_mellum_candidate_acceptance.py --gpu 1 --profile sprint_graph_qk_router_prefix_metadata --budget-minutes 60
```

campaign代码22597b2与acceptance25350e6的ATOM/atom、serve_mellum、执行profile及runtime配置无差异；验证用原始IDs、一套引擎不重跑HF。GPU0端口8451/9426，GPU1原始engine9926，独立端口/工作区。前批已退出后执行；本轮无新模型代码修改，仅选择profile与CPU准备。

仍按原HE绝对2pp、位置90%及16k单列口径报告，不预判通过。若metadata或prefix完整负载退化，优先回退其开关；若功能门槛未全部通过，只定位不满足的任务指标，不对所有中间张量展开逐位排查。两项完成前暂停新优化；随后必要消融/重复与最终材料收敛。
