# 最终交付与累计消融执行方案

## 固定版本与累计行

独立分支 `release/final-matrix`，工作树 `/workspace/tencent-mellum-delivery-matrix`，冻结后不再修改代码。默认 `start_server.sh` 指向 `serve_delivery.py`，其默认行与矩阵最后一行均为step0123。历史入口另存，不混入结果。

| 行 | profile | 槽数 | 与上一行相比 |
|---|---|---:|---|
| Step0 | delivery_step0 | 1 | 最终源码关闭优化的参考计算路径；保留单请求服务桥接与审计 |
| +① | delivery_step01 | 32 | 批处理、共享routing、专家decode group4及逐请求兼容投影 |
| +①② | delivery_step012 | 32 | 参考RoPE / KV单次写入 |
| +①②③ | delivery_step0123 | 32 | SWA decode group4、host metadata once、prefix、chunk1024/d4、disconnect abort |

原始Step0标签不移动；本矩阵基线不是历史源码逐字节复现，是最终源码关闭优化的受控重跑。①包含从单槽到32槽服务能力，客户端A并发仍32，B仍1，C仍rate0.3，服务器槽数在各场景不另换。③为协同机制集合；这里不能分离其每个子机制的贡献，原有分项实验另外保留。

## 统一验证与性能

完整性能每行A512/B50/C200，按B→C→A执行，输出256、C180秒请求超时保持；已有本地客户端兼容调整保留，不等同裁判三次隐藏集。性能关闭torch profiler与phase-audit，保留现有batch审计开销，正式口径中披露。

完整正确性每行200条前64token、边界/16k单列、164题HumanEval关闭thinking/max2048，复用校验HF参考。Step0单槽，其余32槽，MTP/模型双卡并行关闭。前缀/chunk实际缓存调用以batch审计验证，不误用原生prefix审计文件。失败的正确性不阻断性能采集，但最终资格独立标记。

每轮先用实际start_server入口做HTTP非流式/流式及usage检查，保持对应行上下文与槽数，不用另一个缩小配置冒充最终启动。

两卡分工：GPU0行0/01；GPU1行012/0123。每张卡顺序性能campaign及每行正确性，减少人工介入。每项完整测试可能较长，预计总计数小时，不能用此前短profile的3–10分钟估算。预算按阶段而非整条命令；可续跑性能，已完成产物哈希必须一致。失败正确性需新建结果目录，不覆盖原失败。

`collect_delivery_matrix.py --runs <两组结果> --output <新目录>`校验原始产物、拒绝不同源码混合及重复成绩，合并四行展示。仍失败的C显示完成数，尾延迟记未完成；不能只展示成功请求的分位数。

## 报告更新与交付

性能43–45提供最新SWA/取消、C60和profile证据。最新trace稳态decode中全局SDPA30.19%、专家25.54%、KV读取12.01%；受控mixed有74.04%kernel时间在prefill。profile只证明定位与路径，不证明最终C通过。

最终报告保留四行矩阵，实验结束后只填对应版本测得值；缺失记待测，超时记失败。默认候选是否合格由最终正确性与ABC分别决定；若不合格，保留①②保守配置及其已验证范围，不宣称题目全部通过。正式提交需指定Git提交并保留完整历史，不移动旧标签或squash。
