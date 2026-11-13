# 64 token分叉：离线分析与同前缀诊断

日期：2026-11-11。状态：已确认分叉和prefill数值差异；根因未确定；暂停window/long。新增诊断尚未执行GPU，不宣称已修复。

## 证据

[套件summary](../../results/mellum_validation/20261111T113407765642Z/summary.json) · [comparison](../../results/mellum_validation/20261111T113407765642Z/comparison.json) · [ATOM candidate](../../results/atom_mellum_gpu/20261111T113428522075Z/candidate.json)

| 样例 | 第一个不一致的新token（0-based） | prefill max abs | prefill RMS | top1 |
|---|---:|---:|---:|---|
| python_add | 37（第38个） | 3.203125 | 1.30151689 | 相同 |
| arithmetic | 6（第7个） | 2.0 | 0.67307222 | 相同 |

两端均生成64token，输入/模板/位置/生成设置一致。不是只有不同自回归前缀引起的后续变化：原始prompt的prefill logits已经不同。尚无依据把该差异直接判作允许的BF16误差。

已比较旧HF16token run与本次HF：同输入，前16token完全一致，保存的prefill logits逐元素一致。因此HF参考迁移没有解释本次差异。旧ATOM arithmetic前16通过，新运行第7token不同，说明还需审查运行配置/路径变化或非确定性；其中max_num_batched_tokens由512变256是已知配置变化，不能直接断言因果。

本次按ATOM debug-guide读取采样、加载和分层诊断建议。两轮均level0/eager、加载覆盖通过。OPUS警告不是根因证据，不改GPU_ARCHS、依赖、路由或数值容差来消除失败。

## 定向诊断

[diagnose_mellum_mismatch.py](../../scripts/diagnose_mellum_mismatch.py)从失败artifact生成4个探针：

- python_add原prompt（24token）。
- python_add + HF共同前37个新token（61token）。
- arithmetic原prompt（18token）。
- arithmetic + HF共同前6个新token（24token）。

每条只生成1token；将同一HF前缀整体prefill，记录原HF cached路径在该位置的下一个token，可区分同前缀prefill与缓存decode差异。此为分歧点探针，不是全部64步完整teacher-forcing覆盖。

HF测试进程hooks与新Mellum骨干可关闭诊断分别保存embedding、28层输出、final norm和末位置logits。ATOM层输出保存hidden+residual的materialized状态，对齐HF已相加状态，不误比ATOM物理tuple。只对单请求真实prefill开启，跳过dummy/decode，不修改已有compile模型或HF site-packages。

离线[层比较器](../../scripts/compare_mellum_layers.py)报告shape、finite、exact、max abs、RMS和relative L2。第一层不完全一致不自动代表该层有bug，需检查差异增长、权重值与具体子算子。

本轮保留失败配置max_num_batched_tokens256；新增可选参数--max-batched-tokens，供随后有依据地进行512对照，当前不同时改配置与宣称修复。

## CPU验证与运行

配置/模型测试23通过；顺序运行器和新诊断测试5通过。验证同前缀的长度/来源、分层materialization、诊断不改变CPU替身forward结果、dummy跳过、层比较统计。不能代替真实GPU算子验证。

在普通容器终端执行一次定向诊断：

```bash
/opt/venv/bin/python /workspace/scripts/diagnose_mellum_mismatch.py --run /workspace/results/mellum_validation/20261111T113407765642Z
```

将顺序运行HF和ATOM，保存独立logs、输入、prefill/层张量、comparison.json和layers_comparison.json。即使token失败仍进行已有artifact比较；不要自行重复运行window/long。工具本轮仅生成探针和做CPU测试，没有访问GPU。

## 后续判断

- embedding不同：先查输入与embedding加载值。
- 前几层首次显著增长：进一步拆QK-Norm/RoPE/Attention/MoE/残差，优先核对canonical权重和实际dtype。
- 同前缀prefill与HF一致但缓存步骤不同：检查decode cache、位置和kernel路径。
- 所有层只呈微小漂移但分歧点margin很小：需要实际margin与跨后端对照，不能仅凭语言文本相近定性。
- 重跑原prefix的logits相对失败run变化：检查max-batched配置与运行确定性，不能删除旧失败数据。
