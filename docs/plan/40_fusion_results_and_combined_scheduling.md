# 40 路由/融合结果与组合调度筛选

基于计划38/39，不做逐组件HF验证。当前研发工作区/workspace/tencent-mellum-scheduling；正在运行的fusion代码保持不变。

## 已核验结果

routing 20261116T122514554696Z与fusion 20261116T122521482724Z共322项产物哈希通过，所有A64/C20请求成功、5631项加载完整，实际Graph32回放及路由/融合后端证据齐全。

| 同卡配置 | A64 tokens/s | C20 TPOT p99 ms | C20 TTFT p99 s |
| --- | ---: | ---: | ---: |
| GPU0 Graph+Torch，A1 | 930.1006 | 22.8602 | 0.94512 |
| GPU0 Graph+native，B1 | 1006.9646 | 20.0095 | 0.91949 |
| GPU0 Graph+native，B2 | 1010.9100 | 19.9583 | 0.93088 |
| GPU0 Graph+Torch，A2 | 948.8790 | 22.8092 | 0.95041 |
| GPU1 Graph基线 | 941.6311 | 22.8110 | 0.94516 |
| GPU1 仅RoPE融合 | 1032.7189 | 18.8762 | 0.93509 |
| GPU1 QK-Norm+RoPE融合 | 1062.4629 | 18.6394 | 0.94799 |

路由两次吞吐算术均值相对提升7.39%，不是裁判3次隐藏集中位数。QK融合较同卡Graph基线吞吐提升12.83%。数学冒烟多数输出精确，最差相对L2约1.1962e-5，仅证明小规模局部实现，不能转称整个模型通过。C20固定低到达率下吞吐接近，重点看延迟，不以接近的C吞吐否定优化。

## 下一批命令

GPU0，两项组合ABBA（QK融合/融合+native路由/后者重复/前者重复）：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-scheduling/scripts/run_mellum_scheduling_sprint.py --track combine --budget-minutes 50
```

GPU1，QK融合基线/加prefix/加chunk2048/加chunk4096；后三项均分别从基线派生，prefix与chunk不同时启用，Torch路由不变：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-scheduling/scripts/run_mellum_scheduling_sprint.py --track latency --budget-minutes 60
```

每项A64/并发32、C20/0.3req/s、256输出；客户端协议不变。两条线各4项，预计每卡15–30分钟，首次捕获/负载可能更长；执行有总预算，CPU准备另计。GPU0端口8251/8926，GPU1端口8151/8726，先确认前批均结束。

新profile通过profile_settings派生于已测sprint_graph_qk，仅覆盖声明的路由/prefix/chunk设置；不用未测组合假设代替实测。41项CPU检查通过，覆盖派生隔离、prefix direct写后gather、逻辑块顺序、调度参数及旧路径回归。实际GPU缓存/混合batch仍待检验，任何失败保留，不fallback。prefix/chunk实现及风险详见计划39。

继续推迟逐项完整HF/HumanEval；筛选后只固定最终组合集中验证。若chunk失败或退化，停止该候选，不影响组合与prefix结果；组合收益不按7.39%与12.83%简单叠乘。小规模结果不能替代正式ABC或宣称最终超过官方baseline。
