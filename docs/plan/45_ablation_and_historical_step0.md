# 45 最终实验运行期间：消融与历史Step0复现准备

独立/workspace/tencent-mellum-ablation，分支step1/ablation-preparation，基于25350e6。当前campaign/acceptance工作区和已启动进程均不修改，本轮不使用GPU。

## 发现与必须区分的版本

step0-local-validated实际指向f382471。该tag含ATOM OpenAI server，但没有后来新增的serve_mellum批处理桥接脚本；mellum_attention.sdpa_single_prefill要求单请求，sdpa_cached_decode同样只支持1请求。因此历史Step0应按max_num_seqs=1运行，客户端A仍保持题目parallel32、number512，由服务排队，不能偷偷把客户端并发改小。

后来的R已经加入请求拆分/共享路由等额外机制，不得重命名成原始Step0。本次新建只读使用的detached worktree /workspace/tencent-mellum-step0-repro，未改历史源码。serve_mellum_step0_snapshot.py放在新工作区之外，先核验HEAD等于tag且工作区干净，再通过PYTHONPATH指定历史ATOM；沿用已验证Step0环境设置，BF16/eager/TP1/DP1、prefix/chunk关闭、1调度槽位。输出launch.json含源码hash、tag、实际命令和明确局限。

只做CPU启动准备，不导入GPU模型：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-ablation/scripts/serve_mellum_step0_snapshot.py --prepare-only --output /workspace/results/mellum_step0_snapshot_prepare
```

实际服务/GPU任务尚未启动。先短服务冒烟再考虑历史全量测量，避免把很慢的Step0 A512长作业无条件塞进当前时间预算；失败/超时如实报告，不能用R数值补成历史Step0。

## 内部组件消融准备

prepare_mellum_ablation_matrix.py只写matrix.json，不启动任何服务。暂以待验收的sprint_graph_qk_router_prefix_metadata为anchor；若本轮正确性未通过，必须换anchor，不能凭准备状态当最终方案。

| profile | 相对anchor的声明变化 |
| --- | --- |
| ablate_graph | 关闭Graph，保留全部数学后端 |
| ablate_qk | 关闭QK融合，保留独立AITER Norm与参考RoPE |
| ablate_route | 原生路由改回Torch路由 |
| ablate_prefix | 关闭跨请求前缀缓存 |
| ablate_metadata | 关闭本次forward内元数据复用 |
| ablate_packed | 改回分离QKV投影 |
| ablate_norms | Norm恢复参考实现，并关闭依赖其语义的QK融合；明确是关联消融，不伪装单开关 |

GPU常量驻留、direct-KV、reference专家/Attention不能在当前Graph+融合组合中任意单独切换：分别有设备、写入、动态形状/CPU控制依赖。矩阵把这些列作受约束项，不生成无效配置后静默fallback。

这份anchor+组件关闭表是内部因果证据，**不是原题要求的Step0/+点1/+点1+2/+点1+2+3正式消融表**，也不表示要把全部8行再跑一遍完整ABC。根据当前结果与已有同卡对照，选择解释主要收益的必要行；正式提交最多3个自包含优化点，不能仅以多个开关归到同一名字就认定符合题意。已有历史对照可引用，但不同基线/协议不能拼成虚假的累加表。

## 验证及后续顺序

47项相关CPU检查通过，包含anchor派生隔离、开关依赖、历史tag启动与单槽位限制、矩阵pending状态、现有正确性/缓存/Graph/全量批次。py_compile/diff检查与矩阵/历史启动CPU准备通过，无新GPU结论。

1. 等当前完整ABC与统一正确性返回，优先处理未通过的明确指标。
2. 固定最终anchor，仅补必要消融/重复；不自动开启所有历史实验，不扩展新量化/MTP/双卡改造。
3. 历史Step0与后来的R分开标注；完整性能与开发负载分开，保留超时和资源争用说明。
4. 最后统一整理复现与交付材料，避免方法未定时反复改写报告。
