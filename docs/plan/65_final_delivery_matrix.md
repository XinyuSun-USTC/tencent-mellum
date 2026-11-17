# 最终累计消融与执行口径

当前方案按[决议67](67_r32_baseline_revision.md)采用r32-v2基线，替代旧单槽累计方案。所有行固定单卡32槽。

| 行 | profile | 优化增量 |
|---|---|---|
| Step0 | delivery_step0（等价r） | 参考兼容批处理、逐请求投影与共享routing属于基线 |
| +① | delivery_step01 | decode专家group4 |
| +①② | delivery_step012 | RoPE/KV单次写入 |
| +①②③ | delivery_step0123 | SWA分组、metadata、prefix、chunk1024/d4及取消 |

默认start_server.sh仍指向step0123。该行当前token一致率75.9453%、长输入43/256、HumanEval147/164，尚未全部通过本地正确性。前三行本地正确性通过；C均有大量失败。

现有矩阵已收集完整负载结果，Step0复用最近历史r32完整测量，后三行复用本轮累计实验；不再补测单槽A，不重复运行整套ABC。[汇总](../../artifacts/submission/final_r32_matrix/matrix.md)及matrix.json保留每行来源与哈希。历史单槽A预算中断记录只作为初始参考，不填当前矩阵。

未来若重测，run_delivery_matrix.py使用新ROWS全行32槽；旧单槽manifest与新源码不兼容，不允许强行resume或直接改名。collect_delivery_matrix.py检查行定义；显式历史复用由assemble_r32_delivery_matrix.py完成。

性能范围保持A512/B50/C200、输出256、C单请求180秒。C分位数只针对成功样本，必须同时列完成数。两卡仅运行独立任务，主机竞争及客户端到达归一化按实际记录披露。

当前剩余实验集中在第三项精度隔离；使用旧冻结工作树执行，不能修改其源码以套用本次名称调整。选定最终候选后再决定必要的局部回归与测量，现有失败不覆盖。
