# 当前交付核对表

本页与根目录正式材料共同维护当前状态，历史方案不作为默认执行指令。重构前版本保留在 `report/archive/submission_document_snapshots/20261116_before_current_scheme/`。

## 材料与当前状态

| 交付要求 | 位置 | 当前状态 |
|---|---|---|
| ATOM 适配源码与冻结环境 | `ATOM/`、`configs/submission/environment.json` | 已有参考适配；新候选分布于独立分支，尚待最终集成 |
| 一条命令启动服务 | `start_server.sh` | 现默认仍为旧 r；新候选入口已做 CPU 准备，GPU 启动待测 |
| 各优化独立开关 | `release/qualified-entrypoint` | 专家分组、KV 和分段开关已准备；分段当前依赖 direct KV，须披露组合限制 |
| ≤4 页技术报告 | `TECHNICAL_REPORT.md` | 已按当前方案重写，数据缺口留空，作者最终确认页数 |
| 一页 AI 使用说明 | `AI_USAGE.md` | 已更新方法、错误与责任，作者最终确认页数 |
| D1—D7 日志与完整 Git 历史 | `WORKLOG.md` | 已统一每日记录，历史详细记录归档；不 squash |
| Profiling 证据 | `artifacts/submission/reference_profiling/` | 已保存参考路径原始trace、哈希、phase统计和图；旧快速路径轨迹另存，不混作基线 |
| 同配置完整 ABC 与累计消融 | 正式报告第 3 节 | 尚缺最终方案；未以缩减或激进路线数据补表 |
| 来源和许可证 | `01_sources_and_licenses.md` | 已保留，候选最终集成后再核对增量 |

## 候选证据与不可替代关系

- 参考组合 `r_expert_decode4_kv_once`：developer200 96.8125%，长组 256/256，16k 64/64，HumanEval 148/164 对 HF145/164，绝对差1.8293个百分点，本地通过；C200仅1/200成功。归档为 `reference_combination_acceptance/` 与 `reference_combination_c200/`。
- 四轮配额分段：C60成功21–24/60，包含16k的小样本生成通过；没有完整候选正确性或C通过声明。见 `report/performance/38_c_chunked_prefill_screen.md`。
- 一轮配额与前缀缓存：正在测，结果待填。CPU检查不等于GPU验收。
- 旧 aggressive：已有完整ABC和缩减对照，但未通过正确性。其数据不放入新方案累计消融。
- 旧保守r：单槽和32槽本地正确性均通过；r32容量失败。旧标签不代表当前最终候选。

最终公开集与位置/序列匹配口径未由裁判确认，本地两种分母同时保留。MTP关闭、推理单卡，附加条件不适用。隐藏集、空闲机器三次中位数由裁判执行，不是本地已完成项目。

## 最终晋级与交付顺序

1. 核验已结束实验的状态、请求分母、实际功能调用和产物/源码哈希。
2. 选择有容量改善的候选，跑同配置 C200；完成最终完整本地正确性。
3. 集成自包含源码与默认预设，检查独立开关及兼容限制，运行真实入口的普通/SSE检查。
4. 补同配置 A512/B50 和规定累计消融，最后一行必须等于最终主配置；失败行保留失败，缺失数据留空。
5. 更新根目录报告、证据索引、提交和新的最终标签。旧标签不移动，不借用旧正确性或性能结论。

`verify_submission.py`可检查文件、部分链接与证据索引，但其通过不代表GPU或题目验收通过。打包前必须在选定的最终工作树运行检查并确认打包脚本选择正确分支；当前默认脚本仍面向旧release/submission，不能直接打包充当新方案。完整Git历史可另存bundle；模型权重、虚拟环境与大型张量通过外部资产清单引用。不要在候选仍测试时提前冻结“最终合格版本”。

## 面试重点备查

[06 Profiling证据](06_profiling_evidence.md)提供两份原始压缩trace与阅读顺序；[07 决策记录](07_optimization_decisions.md)保留有效与失败尝试及转向；[08 收益和消融台账](08_estimates_and_ablation.md)区分事前假设、结构估计、实测及待测；[09 AI案例](09_ai_collaboration_cases.md)说明本人如何质疑和纠正AI建议。它们是内部备查，不自动计入正式四页或一页报告正文。
