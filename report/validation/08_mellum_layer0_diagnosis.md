# 同前缀分层结果：定位第0层子算子

日期：2026-11-11。状态：已分析真实层dump；根因未定，子算子诊断待GPU执行。没有修改模型数学或放宽容差。

## 本轮已分析证据

[层比较](../../results/mellum_validation/20261111T114644582295Z/layers_comparison.json) · [logits比较](../../results/mellum_validation/20261111T114644582295Z/comparison.json)

四个同前缀探针embedding逐元素一致。第0层materialized hidden相对L2误差为1.10%–1.16%；final norm相对L2约9.02%–12.62%，最后位置约6.13%–10.20%。差异不只来自缓存decode，不是仅因自由生成前缀已经分叉。

第0层误差不能自动证明Attention或MoE中的哪一项有错；此前加载覆盖只证明回调覆盖，不是参数值全量核对。本轮依据debug-guide继续检查权重值和第一个有差异的层，而不猜测OPUS或直接替换fused算子。

## HF缓存与prefill自身差异

对照失败64token run中的原HF下一token，与本轮同前缀重新prefill：

| 探针 | 原HF cached下一token | HF整体prefill | ATOM整体prefill |
|---|---|---|---|
| python_add_prefix_0 | think | think | think |
| python_add_prefix_37 | 11993 ` Should` | 19368 ` Provide` | 11993 ` Should` |
| arithmetic_prefix_0 | think | think | think |
| arithmetic_prefix_6 | 359 ` is` | 359 ` is` | 6617 ` asked` |

HF代码探针prefill前两名为32.75/32.5，margin0.25；算术探针为4.59375/4.375，margin0.21875。不能将HF内部差异归咎于ATOM；也不能因此认定全部ATOM差异可接受。原始两个prompt的新旧HF prefill仍完全一致，此处是扩展前缀的cached与重新prefill比较。

后续分别报告HF内部cached/prefill、两端同前缀prefill及两端cached差异，不用混合参考判断单算子正确性。

## 新增定向诊断

`atom/model_config/mellum_diagnostics.py:install_stage_hooks`仅为观察器，不参与reference数学。HF通过文件导入该观察器，不导入ATOM引擎；共享的是导出布局，不是计算实现。

仅第0层捕获：input norm输入/输出、Q/K/V投影、Q/K norm、Attention context/O投影输出、post-attention残差与norm、MoE输入、router logits和MoE输出。将HF批维/ATOM token维归一化到相同二维布局，保留dtype；不改变kernel输入或输出。

同时保存该层input/post norm、Q/K norm、Q/K/V/O投影和router的实际权重（共9项，QKV拆分后与HF逐项比较）。不包含shuffle后的expert权重，不能冒称所有5631张量值都已验证。

观察器以环境/命令行显式开启，跳过dummy/decode；顺序样例沿用prefill索引。原始/app仓库、第三方依赖和已有compile模型均未改。普通运行不安装这些hooks。

新增/扩展：`--stages`诊断入口、`--dump-stages`双端导出、`stages_comparison.json`子算子与权重报告。CPU配置/模型24项、运行器/诊断5项测试通过，包括观察器不改输入、QKV导出拆分、采样开关及原有协议。

## 下一次GPU命令

```bash
/opt/venv/bin/python /workspace/scripts/diagnose_mellum_mismatch.py --run /workspace/results/mellum_validation/20261111T113407765642Z --stages
```

它重复相同4个输入、每条1token，保持失败配置256，以新增观察器收集第0层证据，不自动修改数值路径。结果目录中重点读取stages_comparison.json：

1. 实际权重是否精确一致；
2. embedding一致时input norm是否首先不同；
3. 相同norm输入下QKV是否异常；
4. Attention context差异；
5. post norm与router/MoE差异是否仅由上游输入传播。

必要时可用已经保存的输入/权重在CPU公式下重算norm及小规模linear，或用独立kernel脚本进一步隔离。只有证据支持后再改必要适配；目前不改norm、Attention、MoE，也不继续window/long。
