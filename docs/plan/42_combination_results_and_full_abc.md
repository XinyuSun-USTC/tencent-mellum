# 42 组合/调度结果与完整ABC收敛

核验combine 20261116T124059219299Z、latency 20261116T124105330881Z，两批369项产物哈希一致，全部请求成功，源码运行期间未改变。

## 已返回数据

| 同卡候选 | A64 tokens/s | C20 TPOT p99 ms | C20 TTFT p99 s |
| --- | ---: | ---: | ---: |
| GPU0 QK融合 A1 | 1044.9105 | 18.4813 | 0.93861 |
| GPU0 QK+native路由 B1 | 1124.9061 | 16.3717 | 0.93077 |
| GPU0 QK+native路由 B2 | 1127.3150 | 16.4533 | 0.91730 |
| GPU0 QK融合 A2 | 1035.4461 | 18.6599 | 0.94846 |
| GPU1 QK融合基线 | 1033.6255 | 18.6191 | 0.94707 |
| GPU1 加prefix | 1040.4009 | 15.9079 | 0.75063 |
| GPU1 chunk2048 | 933.4001 | 21.2276 | 2.35553 |
| GPU1 chunk4096 | 974.1759 | 19.7411 | 1.12753 |

原生路由在QK融合上的ABBA平均吞吐增益约8.26%。Prefix有19次cached后缀事件，合计72352token；chunk4096/2048的276342/569910为此前chunk历史token，不得冒称跨请求前缀命中。两档chunk吞吐及C尾延迟均退化，停止该方向本轮晋级，不无界调参。Prefix TTFT p99改善约20.7%、TPOT p99改善约14.6%；完整C200仍待实测。

## GPU0：已测最快组合完整ABC

```bash
/opt/venv/bin/python /workspace/tencent-mellum-campaign/scripts/run_mellum_full_campaign.py --gpu 0 --profiles sprint_graph_qk_router --budget-minutes 90
```

明确B50/A512/C200、256输出；不把A64替代A512。估计20–35分钟，C200到达过程本身约11分钟，执行预算90分钟。失败stage保留且继续，支持原目录resume；实际后端与Graph回放必须有回执，自动汇总有效完整阶段相对官方baseline比值。另一GPU并行开发实验可能争用主机资源，本地一次测量不宣称裁判空闲隐藏集三次中位数。

## GPU1：最后一次小规模组合筛选

```bash
/opt/venv/bin/python /workspace/tencent-mellum-campaign/scripts/run_mellum_metadata_sprint.py --track finish --budget-minutes 45
```

顺序QK+native路由→再加prefix→prefix+当次forward元数据复用，全部Graph。约10–20分钟，执行预算45分钟，CPU准备另计。每项A64/C20，旧chunk不再跑；派生开关通过CPU检查验证隔离。仅metadata候选必须看到实际reused=true，不以配置存在认定生效。完整HumanEval/HF仍集中在最终组合，不为本批每项单独运行。

两条线独立端口（完整ABC GPU0为8451/9426，小筛选GPU1为8151/8726），当前调度工作区ee3e238不改。代码位于独立campaign分支。40项相关CPU检查通过，所有新profile有独立开关；GPU收益未产生前不预判叠加结果。

## 下一步停步条件

GPU1若prefix/metadata退化，保留GPU0已测组合，停止该小支线；若改善明显，再补获胜组合完整ABC及统一正确性。优先完成正式规模性能与最终组合验收，不新增量化/MTP/双卡架构改造。最终最多3个优化点的选择与正式消融，以实际收益和题目定义整理，不能用本表替代。
