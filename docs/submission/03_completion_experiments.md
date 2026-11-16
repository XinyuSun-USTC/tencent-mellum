# 主版本验收与完整补充实验

## 交付选择

主交付候选采用参考计算配置`r`，默认单槽；激进配置只作为附加性能实验版本。主候选须完成本地200条、长输入和完整HumanEval，实际达到门槛后才记录为通过。并行验证`r`的32槽配置，若通过，可据实选择服务容量更高的主配置；不得在未验证时替换主配置。

## 完整正确性补验

入口：`scripts/run_mellum_reference_acceptance.py`。复用经过哈希和协议检查的HF参考，重新生成候选全部输出；完整评分164道HumanEval，不缩减任务数、输出预算或评分分母。参考输入为关闭thinking、最多2048 token。验证后再进行200条与长输入比对，最终状态仅在所有本地检查均通过时为`local_candidate_checks_passed`。

```bash
/opt/venv/bin/python scripts/run_mellum_reference_acceptance.py --gpu 1 --batch-size 1 --budget-minutes 120
/opt/venv/bin/python scripts/run_mellum_reference_acceptance.py --gpu 0 --batch-size 32 --budget-minutes 120
```

实际已启动：卡1 `20261116T160211830830Z_batch1`；卡0 `20261116T160221792307Z_batch32`，位于`/workspace/results/mellum_reference_acceptance/`。运行期间保持模型和被manifest绑定的脚本不变。

## 同配置完整性能

入口：`scripts/run_mellum_completion_benchmarks.py --track primary`。要求输入的验收运行实际通过，校验结果哈希，读取已验证的调度槽数，再调用正式交付启动器测A512/B50/C200。若验收失败，性能任务不启动。

客户端A并发仍为32。单槽引擎内部decode批次为1不作为失败原因；其余请求、生成参数、长度和错误检查保留，吞吐与尾延迟均从真实客户端数据库读取。

## 历史Step0完整累计消融

`--track cumulative`依次测量历史Step0、原生快速配置、加入graph/QK融合及路由、加入前缀缓存和元数据复用。历史Step0采用未修改的`step0-local-validated`工作树，启动前验证提交和工作树状态；其余配置来自当前交付源码。每行均为完整A512/B50/C200，组件边界与技术报告三个方向对应。

单槽Step0可能较慢。超时、请求失败或未完成阶段均保存实际状态，不以部分成功数据作为合格性能行。原有局部实验仍保留，不覆盖为完整消融。

## 结果更新

实测完成后归档输入配置、实际执行记录、评分和客户端数据，并更新主版本配置、技术报告与发布标签。未通过时继续修复或选择已有实测合格配置，不以更改验收规则获得通过。页数由作者最终手工核对。
