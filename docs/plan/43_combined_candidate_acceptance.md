# 43 完整ABC运行期间准备最终组合验证

独立工作区/workspace/tencent-mellum-acceptance，分支step1/combined-acceptance，基于22597b2。当前campaign实验工作区保持不变，未启动任何GPU。本轮不新增长周期优化或逐组件HF对照，提前准备最终选定组合的一次性验收入口。

## 已实现入口

run_mellum_candidate_acceptance.py --profile显式候选 --gpu空闲卡。一次创建ATOM引擎，按现有profile保留Norm/Packed、Graph、原生路由、融合、prefix/metadata/chunk开关；模型只加载一次，依次200条greedy64、长输入边界4条（含16383/16384）、完整164题HumanEval。原始input IDs直接进入同一ATOM引擎，避免HTTP文本拼接/再分词；模型执行配置与serve_mellum一致，runtime.json记录实际参数。它不能代替OpenAI server协议及完整性能验收。

参考复用：

- 开发200：20261113T104458011376Z；校验现有归档SHA、配置与HF版本、greedy64、全1mask/绝对位置，无强制历史。
- 长输入：20261113T103154043872Z的已通过Step0参考；保存引用SHA，单列≥16000token子组。结果明确为本地长度边界组，不冒称官方全部长输入覆盖。
- HE：20261115T213706098181Z完整恢复的HF164；核验completed_hf_sha256、inputs_sha256、题集、HF源文件、模型配置、版本、参考完成状态。固定关闭thinking、2048上限，与原协议一致，不重跑HF。
- 官方HumanEval evaluator沿用固定commit与沙箱；所有164题均保留分母，提取失败/截断如实统计。候选分数对照原HF145/164逐题verdict，核对官方pass@1与逐题计数。

## 判定与边界

分别输出开发200位置一致率≥90%、长组/16k子组位置一致率、HE绝对差≤2pp。保留原abs差口径；若候选分数高出2pp，也不静默改判，报告原始增失题及差值供解释。没有logits/中间tensor精确门槛；token差异不阻止HE评分。

实际检查加载5631项、Graph回放、原生路由、QK融合；选prefix/metadata时必须有真实cached-token/reused事件。记录实际batch。HF/候选Torch版本必须一致，ATOM仍为冻结Transformers5.2.0，不使用升级HF环境运行候选。

44项相关CPU检查通过：位置分母含缺失token、16k单列、重复/缺失/顺序错误拒绝、输出预算、最终profile运行参数及审计路径、既有调度/缓存/Graph入口。真实参考准备成功，核验全部输入及源码/外部参考指纹；未有最终GPU正确性结论。

准备示例（可执行但不占GPU）：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-acceptance/scripts/run_mellum_candidate_acceptance.py --profile sprint_graph_qk_router_prefix_metadata --prepare-only
```

实际运行必须等当前完整ABC/finish结果选择profile并有空闲卡，再去掉--prepare-only；默认预算60分钟，可1–120分钟。参考HE不重新生成，预期比早期HF重跑显著省时，但时间仍由实际输出长度决定。不提前交付固定profile的GPU命令，避免尚未筛选就验证错误候选。

## 下一步顺序

1. 当前finish结果选择prefix/metadata是否保留；chunk已退化不回头无界调参。
2. 对最终组合运行本入口一次，保留所有门槛原始结果；若失败先解释任务指标，再用开关定位，不直接回到逐位浮点排查。
3. 补获胜组合完整ABC及必要重复/消融。当前一次开发测量不冒称空闲隐藏集三次中位数。
4. 最后再集中整理提交材料，按最终实测配置写报告，避免反复重写。
