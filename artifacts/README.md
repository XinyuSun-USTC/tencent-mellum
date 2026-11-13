# 轻量实验归档

`results/`保留从原`/workspace/results`复制的聚合JSON和文本日志，目录层次与原运行ID不变。包括成功与失败记录，不将诊断完成状态改写为验证通过。

[index.json](index.json)记录每个文件的原路径、仓库路径、字节数与SHA256。本轮共509文件，约84.48MiB。JSON里的绝对路径和张量哈希保持原样，不回填为迁移后路径，不伪造历史元数据。

未提交逐步输入trace JSON和`.pt`张量；它们均保留在完整原始结果及本地tar备份中。库存见[raw inventory](../manifests/results_raw_inventory.json)，备份路径/哈希见[local backups](../manifests/local_backups.json)。只有轻量证据时可以阅读已完成比较结论，重新执行张量比较还需原始归档；模型权重另由`/data/models`提供。

离线校验Step0标签中的源码与当前仓库轻量证据（Step1工作树允许变化）：

```bash
python scripts/verify_project_archive.py
```

若需要在另一个机器完整复现，需另行获取原始结果归档和符合冻结要求的环境。本地同盘备份不提供跨机器灾难恢复保证。

如需严格检查当前工作树是否仍等于Step0，可加`--worktree`；Step1修改之后该模式出现源码差异是预期结果，不能将标签验证通过解释成当前优化代码已经验证。
