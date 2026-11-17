# 当前交付核对

当前工作树 `/workspace/tencent-mellum-delivery-r32`，分支 `release/r32-baseline`。根目录start_server.sh默认step0123；README、TECHNICAL_REPORT、AI_USAGE、WORKLOG为交付材料，内部记录置于docs与report。

## 代码与基线

- Step0固定32槽r参考路径，包含批处理桥接、逐请求兼容投影及共享routing。
- ①仅decode专家group4；②参考RoPE/KV单次写入；③SWA分组及混合负载执行机制。
- 初始单槽路径为initial_single_reference，不作为当前消融Step0。历史标签保留不动，单槽A补测取消。
- 默认入口与累计末行配置相同；当前③精度失败，不将默认候选称为已合格主版本。

## 实测状态

[当前矩阵](../../artifacts/submission/final_r32_matrix/matrix.md)复用最近历史r32完整基线和本轮后三行结果，逐行保存测量身份。前三行本地token、长组、HumanEval检查通过；最后一行token一致率75.9453%、长组16.7969%，HumanEval差1.2195个百分点。C各行分别1/200、1/200、1/200、48/200完成。

基线A41.3351，后三行51.3440、53.3934、71.2464token/s/GPU；第三项性能对应未全部通过正确性的组合。C尾延迟仅为成功样本统计，不掩盖失败数量。

开发200由HumanEval164与A36构成，保留为已约定本地回归集合；不是仅凭文件行数确认的官方唯一集合。MTP及双卡模型并行未启用。原始结果、版本与哈希保留，旧单槽记录不改名。

后续工作：分析正在运行的第三项精度隔离结果，确定最终配置，再完成文档结果收敛、路径整理与Git提交。未取得的结果不补造。
