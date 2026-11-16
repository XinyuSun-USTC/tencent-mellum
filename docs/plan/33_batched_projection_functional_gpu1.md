# 33 GPU1：整批投影高性能候选完整HumanEval

性能31已确认：相同ATOM模型源码、相同开发负载，取消campaign L请求拆分后A64吞吐121.50→439.67tokens/s（约3.62倍），C20 TPOT p99 134.78→80.37ms，所有请求成功。现在按性能优先流程补该配置完整功能评估，不重复HF或先跑完整ABC。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-batched-functional/scripts/run_mellum_humaneval_pair.py --attention native --batched-projections --reuse-hf-run /workspace/results/mellum_humaneval_recovery/20261115T213706098181Z --budget-minutes 35
```

独立batched-functional基于0555b76，功能入口基于e55cd7a；显式--batched-projections只把campaign groups从L改为空，与刚测profile相同。GPU1/engine8626、无HTTP，与GPU0 prefix-functional的9126隔离。此轮prefix关闭；不能把结果当作新投影与prefix的组合已验。

完整164题、关闭thinking、最多2048token、batch32、正常greedy/EOS，复用HF完整引用并校验输入/模型源码/完成hash；官方隔离评分与提取方式不变。必须实际FusedMoE、aiter_varlen与paged decode、5631加载完整、实际batch32。token位置与张量相等无前置门槛，记录截断/提取失败、逐题增减与2pp检查。

22项相关CPU检查通过，覆盖原功能评分/引用保护/加载诊断序列化、原生设置、投影控制flag、服务profile与所有旧路径；真实164输入prepare-only和参考引用检查通过。GPU尚未执行，不修改GPU0或既有运行工作区/冻结依赖。

功能达标则以整批投影配置取代原先较慢主线，再根据GPU0 prefix功能结果整合。若超标，先看实际差距/逐题变化，不无界追踪浮点，保留旧144/164配置作为可交付后备。最终配置重新做必要功能与完整ABC；开发小集性能不冒称正式规模成绩，公开200/16k仍独立报告。
