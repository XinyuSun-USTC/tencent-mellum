# Mellum 2.1 on ATOM

冻结AMD ROCm环境中的Mellum2.1-12B-A2.5B-Thinking适配与优化项目。

**Step0本地验证已完成，当前开展Step1性能优化与交付收敛。** 最新实验分布在独立分支，当前目录尚不是最终交付版本；历史Step0结论见阶段总结，最终配置须绑定自身验证证据。提交缺口与收敛顺序见[36 提交与版本验收](docs/plan/36_submission_and_version_acceptance.md)。

## 阅读入口

- [项目文档导航](docs/README.md)
- [Step0完整阶段总结](report/validation/39_step0_summary.md)
- [Step0基线配置与复现](docs/plan/05_step0_baseline_and_reproduction.md)
- [Step1可选优化路线](docs/plan/06_step1_optimization_routes.md)
- [Git组织与本地路径](docs/plan/07_project_repository_management.md)
- [每日工作日志](report/worklog.md)
- [归档证据说明](artifacts/README.md)

## 仓库布局

| 目录 | 内容 |
| --- | --- |
| `ATOM/` | 以非squash subtree导入的上游源码与本项目适配/测试，保留上游LICENSE及历史 |
| `scripts/` | 参考环境检查、验证、诊断、运行和归档检查脚本 |
| `docs/` | 题目、环境约束、决策与当前规划 |
| `report/` | 研究、验证、阶段总结和唯一每日worklog |
| `artifacts/results/` | 所有已整理运行的聚合JSON和文本日志，保留原始运行目录结构 |
| `manifests/` | 原始大产物、外部资产及本地备份的来源/哈希清单 |

普通克隆即可获得ATOM源码，无需submodule递归初始化：

```bash
git clone git@github.com:XinyuSun-USTC/tencent-mellum.git
cd tencent-mellum
git status
```

环境和执行路径按[基线说明](docs/plan/05_step0_baseline_and_reproduction.md)配置。当前脚本含`/workspace`绝对路径，普通克隆不自动提供模型、ROCm或参考venv；当前容器通过兼容符号链接维持已有命令。不要把“克隆完成”等同“已配置完整运行环境”。

## 历史与大文件

`origin`指向本个人仓库；`upstream-atom`用于查看ROCm/ATOM来源。上游基线`26f23a0bdbf50b9010c07ee3a71171189723e767`通过subtree保留769个可达提交。本项目初始提交、上游导入和当前Step0快照分别提交；此前未保存的逐轮开发提交不补造，实验过程保留在报告与原始产物。

不将模型权重、虚拟环境、安装包、缓存、全量tensor/profiler产物或认证信息纳入普通Git。约13GB的完整results已另做本地tar备份，其库存和哈希见manifests；该备份仍在同一机器，不是异地备份，也没有随Git push上传。轻量JSON可能引用外部`.pt`路径，缺少原始张量时不能重新计算bitwise比较。

项目脚本/文档保留其来源说明；ATOM上游许可证见[ATOM/LICENSE](ATOM/LICENSE)。Step0标签`step0-local-validated`表示开发阶段快照，不代表出题方正式验收。
