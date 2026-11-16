# 参考兼容组合的验收与消融收敛计划

当前 GPU 1 正在独立工作树 `tencent-mellum-qualified-combination` 运行组合筛查。此期间不修改其源码、配置和依赖，不启动额外 GPU 任务。实验版本为 `6ddec3c`；本文件与后续调度脚本位于交付工作树，不进入该实验的源码集合。

## 当前证据

分组 4 已通过 developer200 和长输入筛查，短负载 A 吞吐提高约 13.74%，C20 完成 20/20。KV 单次写入生成结果同参考批处理配置一致，短负载 B TPOT 降低约 7.49%，A 吞吐没有明确收益。二者组合尚在运行，HumanEval 与 C200 均不能提前标为通过。

## 组合筛查后的选择

1. 先核对完整状态、全部文件哈希、运行期间源码一致性、200 条与长输入结果、两种功能的实际调用记录。
2. 分别比较组合相对分组 4 的 A 吞吐与 B TPOT，使用两次重复和首尾基线。若结果方向不一致，应视为收益不确定，不能只选较好一次。
3. 若组合保持生成门槛且收益明确，选择组合进行一次完整验收；若 KV 收益不足或组合退化，则选择分组 4。组合筛查失败则不直接投入完整 HumanEval。
4. 最终候选通过完整 HumanEval、developer200 与长输入后运行 C200。沿用速率 0.3、输出 256 和超时 180 秒，不通过提高超时、减小请求数或降低速率冒充题面结果。
5. C200 完成后，对相同最终配置补 A512/B50；已有同源码同配置的 C200 可以复用，无须为拼接报告机械重跑。不同配置的成绩不能拼为同一版本。

C20 实际最高解码批量为 19，不覆盖满 32 槽持续积压。若 C200 仍大量超时，应保留失败数据，不能把这一候选描述为完整性能达标版本。

## 已准备的后续入口

筛查结束后将实际目录填写到 `--screen-run`，不要使用正在写入或旧版配置的结果目录。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-delivery-final/scripts/followup_screened_mellum.py \
  --source-root /workspace/tencent-mellum-qualified-combination \
  --profile r_expert_decode4_kv_once \
  --screen-run /workspace/results/mellum_reference_combination/实际完成编号 \
  --phase acceptance --gpu 0 --budget-minutes 90
```

完整验收通过后，再使用其实际目录：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-delivery-final/scripts/followup_screened_mellum.py \
  --source-root /workspace/tencent-mellum-qualified-combination \
  --profile r_expert_decode4_kv_once \
  --screen-run /workspace/results/mellum_reference_combination/实际完成编号 \
  --acceptance-run /workspace/results/mellum_qualified_followup/实际验收通过编号 \
  --phase capacity --count 200 --gpu 0 --budget-minutes 30
```

若最终选择分组 4，入口支持 `r_expert_decode4`，源工作树与筛查目录必须相应切回分组 4 实验。脚本会核验配置与源码身份，不允许跨工作树借用验收结论。

## 最终消融的配置对应关系

| 行 | 配置内容 | 当前证据与缺口 |
|---|---|---|
| Step0 | 原始已冻结适配版本 | 使用原标签；不能以原生快速路径替代 |
| 参考批处理 | 参考数学路径的多请求执行 | 正确性已通过；完整 C200 失败须保留 |
| 参考批处理 + 专家分组 4 | 仅合并解码专家计算边界 | 筛查与 C20 通过；完整验收待补 |
| 参考批处理 + 专家分组 4 + KV 单次写入 | 两个候选改动的组合 | 正在筛查；全部正式结论待定 |

最终交付默认配置必须与最终一行的实际开关、源码和测量完全相同。表中是可复现实验配置，不预先将每行认定为独立优化点；第三项贡献是否成立须依据相对 Step0 的实现和收益作实质说明。不得把激进原生路径的性能写到参考兼容组合名下。完整实验未结束前不改写根目录正式报告的最终成绩。

## 运行期间的交付接口核对

CPU 配置核验记录位于 `artifacts/submission/qualified_profile_cpu_audit/profile_differences.json`：KV 单次写入相对 r 只改 `MELLUM_DIRECT_KV_WRITE`；分组 4 只增加 `decode_expert_group_size=4`；组合严格为两项并集。未启用其他原生计算或 graph 功能。

当前 `start_server.sh` 转交 `serve_submission.py`，后者仍从旧 conservative/aggressive JSON 取 profile，不能直接选择新候选。候选通过后必须将已验收实现迁入独立最终集成版本，补齐候选配置选择及专家分组关闭开关；单次 KV 写入已有关闭开关。不能只修改 preset 名称而遗漏模型实现与 profile 注册。

集成完成后，先用 dry-run 核对有效环境变量、32 槽、模型长度 16640、BF16、单卡、固定题面输出参数及各关闭开关的差异，再运行最终入口的普通与流式请求启动检查。性能与正确性记录需绑定最终实际配置；如集成改变模型行为，旧验收不能自动迁移。原始标签和历史提交保持不变。

现有 `run_submission_ablation.py` 明确属于旧激进方案的小规模筛查，不能用于新候选正式消融；`run_mellum_completion_benchmarks.py` 的 primary 也绑定旧 r 配置。新矩阵须显式选择最终候选，避免误用旧脚本得到配置错配的结果。以上为待集成事项，不表示当前交付默认已经升级。
