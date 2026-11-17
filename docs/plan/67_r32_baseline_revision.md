# r32消融基线统一决议

## 当前定义

全行单卡、32槽，A客户端并发32、B并发1、C速率0.3；模型为BF16/eager，无MTP。

| 行 | 当前profile等价对象 | 增量 |
|---|---|---|
| Step0 | r | 已含参考兼容批处理、逐请求投影与共享routing；专家group4关闭 |
| +① | r_expert_decode4 | 只增加decode专家group4 |
| +①② | r_expert_decode4_kv_once | 参考RoPE / KV单次写入 |
| +①②③ | r_swa4_abort_prefix_chunk1024_d4 | SWA、元数据、prefix、chunk及取消回收 |

代码中delivery_step0直接调用profile_settings('r')，所有ROWS槽数均32。初始单槽路径另存initial_single_reference，slots为1，不纳入四行矩阵。默认入口仍step0123，与最后一行一致；其当前正确性失败如实保留，不因重新定义基线而自动认定合格。

## 数据复用

Step0选用最近一份已完成的r32完整测量 `mellum_completion_benchmarks/20261116T165949897188Z_primary`，不是按较好分数择优：A512/512、41.3351token/s/GPU；B50/50、TPOT p50 61.253ms；C1/200。基线正确性来自该测量明确绑定的 `mellum_reference_acceptance/20261116T160221792307Z_batch32`。两份运行原始哈希校验，且启动记录与当前r profile开关一致。

后三行使用 `mellum_delivery_matrix/20261117T003512204779Z`、`20261117T003519808987Z` 的实际结果，跳过其中历史单槽step0行。不是同版本同轮完整重测；源码版本、原始路径及摘要哈希在matrix.json逐行保留，不能宣称逐字节等价或裁判三次中位数。

`assemble_r32_delivery_matrix.py`校验原产物、基线32槽与profile设置后生成 `artifacts/submission/final_r32_matrix/`。旧collect工具拒绝将定义不同的历史单槽矩阵直接按新ROWS重新标注。

## 取消与保留

取消单槽A完整/短规模补测；新交付树明确拒绝旧补测命令。原工作树、原始结果和历史标签不移动、不改写。正在旧工作树运行的精度隔离任务不修改，结果作为独立诊断证据。

技术报告、README、启动入口、配置定义、自动矩阵与当前交付审查均以本页口径为准。早期计划与日志中的单槽定义属于当时方案；保留历史记录并注明后续调整，不倒写时间线。
