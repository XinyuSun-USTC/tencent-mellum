# 07 项目Git组织与迁移记录

## 当前结构

总仓库路径`/workspace/tencent-mellum`，主分支main，个人origin为`git@github.com:XinyuSun-USTC/tencent-mellum.git`。提交身份在仓库本地设为XinyuSun-USTC / sunnewrain@mail.ustc.edu.cn；不沿用原global.name残留的中文占位文字。

ATOM以不带--squash的Git subtree放在`ATOM/`，保留原始769个可达上游提交；新增上游来源remote `upstream-atom=https://github.com/ROCm/ATOM.git`。不再在ATOM内维护嵌套.git，普通clone可获取源代码与本项目修改。

现有`/workspace/.git`为环境只读目录，不作为本项目仓库、不修改或删除。后续Git操作统一：

```bash
cd /workspace/tencent-mellum
git status
git log --oneline -5
```

为兼容已有脚本和引用，`/workspace/ATOM`、`scripts`、`docs`、`report`已指向总仓库对应目录，旧目录完整移至`/workspace/project-backups/tencent-mellum-initial/working-directories/`。因此旧执行路径不变，修改的是总仓库工作树，不需要在两个副本之间同步。

完整`/workspace/results`、`venvs`和`packages`留在外部；仓库内results/venvs兼容链接被.gitignore排除。不得把这些链接当作远程包含了原始大文件。

## 历史与提交

1. 初始化总仓库入口和忽略规则。
2. 导入ATOM上游基线26f23a0b完整历史。
3. 提交当前真实Step0源码、测试与脚本快照，提交号`00a75c9`。
4. 提交文档、证据和仓库集成说明，并建立`step0-local-validated`标签。

这些是实际创建的迁移提交，不是回填的11月10–13日实验开发提交。旧报告中的HEAD/feature分支/未提交状态是迁移前快照；历史实验保持原始语义，当前版本身份以本仓库commit和标签为准。已有上游历史完整保留，后续优化按真实增量独立提交、不squash。

## 备份和产物

迁移前备份见`/workspace/project-backups/tencent-mellum-initial`：atom-all-refs.bundle保留全部Git引用；source.tar保留源码/脚本/文档和未提交文件；results.tar保留完整约13GB结果。原始目录另行保留，不删除。备份哈希见[清单](../../manifests/local_backups.json)，仍为同机备份，不是异地保存。

仓库artifacts保存509个聚合JSON/文本日志；完整51501文件库存含tensor与逐步trace见[raw inventory](../../manifests/results_raw_inventory.json)。可以运行`python scripts/verify_project_archive.py`校验65个Step0源文件与509个轻量证据的哈希。

引用`/workspace/results`的旧文档在原容器仍有效，普通克隆后需按artifacts路径阅读轻量镜像；没有外部tensor时不能重新计算完整logits比较。模板、权重和解释器仍由只读模型资产及独立环境提供，不纳入普通Git。

## 远端状态与推送

用户终端已完成SSH认证并确认空仓库ls-remote退出0。工具进程不能访问该终端SSH agent，且SSH连接无法解析github.com；这是工具侧连接限制，不推断用户凭据失效。本轮先完成本地提交和检查，首次远程上传需用户在已认证终端执行：

```bash
cd /workspace/tencent-mellum
git push -u origin main --follow-tags
```

不使用force。推送成功前，文档中的“仓库已整理”仅指本地，不表示GitHub已收到内容。推送后用`git status -sb`及`git ls-remote origin refs/heads/main refs/tags/step0-local-validated`核对。后续Step1按优化点分支/提交，实际测试与报告同步。

## 本地集成检查

迁移后65个Step0文件及509个轻量证据哈希匹配；181个当前文档链接有效；Mellum与sampler CPU回归50项通过（3.65s）。上游基线是总仓库HEAD的祖先，原769个提交可达。未启动GPU推理。大产物/venv链接由根.gitignore同时按目录与符号链接形式排除；常见私钥/PAT模式扫描未发现命中，不将此扫描当完整安全审计。
