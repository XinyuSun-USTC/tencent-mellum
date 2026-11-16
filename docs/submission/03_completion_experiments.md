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

## 排队执行

卡1在单槽完整验收通过后，自动测量同一批大小的主配置ABC：

```bash
/opt/venv/bin/python scripts/run_mellum_completion_benchmarks.py --track primary --gpu 1 --acceptance-run /workspace/results/mellum_reference_acceptance/20261116T160211830830Z_batch1 --wait-minutes 180 --budget-minutes 120
```

卡0等待当前32槽验收任务释放GPU，再测完整累计消融：

```bash
/opt/venv/bin/python scripts/run_mellum_completion_benchmarks.py --track cumulative --gpu 0 --after-run /workspace/results/mellum_reference_acceptance/20261116T160221792307Z_batch32 --wait-minutes 180 --budget-minutes 180
```

等待时间不计入性能执行预算。累计配置的报告顺序仍为Step0、+①、+①②、+①②③，实际执行先完成三个优化配置，再测慢速历史Step0。运行顺序在manifest中记录。正式客户端并发保持32；历史Step0与单槽主版本允许内部decode批大小为1，不使用开发工具额外的“实际decode必须32”条件拒绝该测量，其他请求完整性检查保持不变。

## 结果处理

完整验收返回后检查所有任务数、输入协议、评分分母及源文件哈希。通过的配置才能记录为主版本“本地验收通过”；若两种批大小都通过，结合性能结果选择主版本容量并同步启动预设。激进配置继续作为附加实验，不将其性能赋予主版本。性能与累计消融仅填入实际完成的数据；失败阶段可据原始日志修复并重跑，不用预测或不同规模结果补表。版面页数由作者最终确认。

## 32槽补验结果与后续优先级

运行`20261116T160221792307Z_batch32`已通过所有本地检查，20项产物SHA256一致：200条12392/12800（96.8125%）、174/200整序列，长组256/256、16k64/64，HumanEval148/164对HF145/164，绝对差1.829个百分点。主启动预设已选择32槽参考路径，并以单元检查核对实际MELLUM/ATOM设置与已测runtime一致。

优先在GPU0测该32槽主版本自己的完整ABC，替代先前单槽主性能安排；累计消融留待空闲卡运行。单槽补验继续作为对照，不阻碍已通过32槽主版本的性能测量。
