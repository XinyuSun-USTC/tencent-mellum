# 50 阶段诊断结果与最后完整200确认

两批20261116T150206760682Z/20261116T150214019384Z均完成，27+27项产物hash一致，源码未变、实际后端及canonical192分片回执通过。

| 配置（其余参考路径） | 20条位置率 | 整序列 |
| --- | ---: | ---: |
| AITER专家只prefill | 1112/1280=86.875% | 10/20 |
| AITER专家只decode | 1074/1280=83.90625% | 13/20 |
| 原生Attention只prefill | 943/1280=73.671875% | 11/20 |
| 原生Attention只decode | 1007/1280=78.671875% | 12/20 |

没有一项在诊断集到90%，但不能直接判原200必然失败：20条为12HE/8A（A占40%），原200为164HE/36A（A占18%）。expert_prefill在12HE为715/768=93.10%，8A为397/512=77.54%；这是最后一个值得直接全量确认的接近候选，不能把来源重加权估算当验收。

## 收敛，不再开发新算法

新独立/workspace/tencent-mellum-confirmation，基于b699065，所有模型源码与已测阶段诊断相同。run_mellum_final_greedy_confirmation.py只原200+长组，禁止将20条子集混入；不跑HF、HE、ABC。profile、batch及原始引用均冻结，真实decode最大batch必须与声明一致，专家阶段仍有192分片及backend审计。

若提交前仍留有约20–30分钟，最后两项并行：

```bash
# GPU0：最接近的部分优化候选，batch1
/opt/venv/bin/python /workspace/tencent-mellum-confirmation/scripts/run_mellum_final_greedy_confirmation.py --gpu 0 --profile expert_prefill --batch 1 --budget-minutes 30
# GPU1：参考数学保留batch32，作为可交付保守配置确认
/opt/venv/bin/python /workspace/tencent-mellum-confirmation/scripts/run_mellum_final_greedy_confirmation.py --gpu 1 --profile r --batch 32 --budget-minutes 30
```

预算每项最多30分钟，并不保证固定完成时间。GPU0预期约12–20分钟，GPU1约数分钟到15分钟，依实际调度。若真实截止不足以留出启动/提交时间，不启动本批：已验证R单槽位200/长组100%可作为保守基础，高速候选以未通过一致率的实验配置如实交付，不能拼接两者的正确性/性能冒称通过。

55项CPU检查通过，完整200/4条长组准备及source/input指纹核验通过；新脚本不会生成HumanEval，不能由本批结果自动宣布HE或完整题目验收通过。没有新模型改动、没有GPU任务由工具启动。

## 结果对应动作

- expert_prefill全量过90%：再看实际剩余时间，保留这一部分优化；需评估诊断双份布局内存与交付稳定性，旧高速版1139.78的性能不能继承给它。
- r batch32过90%：保留参考批处理作为保守候选；此前r batch1已经200/长组全100%，两者分别记录，不互相冒称。
- 两个都未过：不再盲改数值内核，冻结已验证保守配置与显式实验配置，按当前证据提交，说明尚未同时满足速度与精度。
- HumanEval绝对差、高速版62.85%等原始失败证据不删改；最后交付不能用A配置验证精度、B配置速度合称同一方案通过。
