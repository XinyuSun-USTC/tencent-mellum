# 24 四例完整prefill精确对齐与新增输入回归

## 已完成的闭环

原始产物`results/mellum_prefill_study/20261112T104449256883Z`，ATOM`results/atom_mellum_gpu/20261112T104450459535Z`。配置为separate QKV、torch MoE、sdpa_cpu_rope_diagnostic。

- 加载审计5631/5631、无missing/unexpected。
- 四例下一token及98304维prefill logits全部精确一致，max_abs和RMS均为0。
- embedding、28层输出、final norm共120份观察张量全部精确一致。
- 首层14个子算子观察×4例共56份全部精确一致；9项选定权重全部精确一致。

这是四个固定输入的完整prefill闭环，超出了此前“局部算子/下一token相同”的证据范围。CPU逆频率初始化修正与分开QKV、匹配专家及SDPA组合在这些输入上消除了已观察差异。但组合结果不提供各项修正对完整模型贡献的线性分解，也不意味着decode、KV、长输入、多请求或Step0正式验收完成。

## 不立即提升默认

当前候选只在单请求无缓存、输入不超过滑窗的prefill生效，实际KV写入/decode仍走原路径。完整生成后端必须统一旋转系数与缓存语义，不能直接依赖这个诊断入口。默认packed QKV+AITER Attention暂不切换。

下一步先用未参与上述修正的新内容检查prefill泛化，再进入强制同历史decode/KV；不立即重跑自由生成或性能评测。

## 新prefill开发集

`scripts/prepare_mellum_suite.py:build`新增prefill：六条新提示、两条raw code token边界、两条已有窗口边界，共10条，每条仅1个新token。六条新提示未用于这次四探针修正；窗口两条属于既有开发资产，不冒称整套都是真正独立holdout或正式题集。

| 输入 | 实际token数 |
|---|---:|
| rotate_list | 24 |
| stable_unique | 20 |
| missing_integer | 19 |
| parenthesized_math | 26 |
| chinese_inventory | 32 |
| identity_explanation | 19 |
| raw_16 / raw_17 | 16 / 17 |
| window_1023 / window_1024 | 1023 / 1024 |

raw项使用原始代码token、不套聊天模板；输入IDs是权威数据。窗口长度不超过当前诊断支持范围，但是否在1024边界精确对齐仍需实测，不能由短输入结果推断。这里的16/17仅测prefill形状，不代替跨KV block的decode验证。

已生成CPU prepare-only产物`results/mellum_validation/20261112T115300262558Z/inputs.json`。套件/运行器CPU8项通过，语法检查通过，GPU新集未执行。运行器新增layer_observation_summary，明确比较张量总数、非精确数和首差；tokens_and_logits_exact状态本身仍不保证所有层观察精确，要同时查看该汇总。

## 下一命令及判据

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite prefill --qkv-backend separate --prefill-backend sdpa_cpu_rope_diagnostic --dump-layers
```

不保存每层算子/大权重，保存每层hidden用于全前向定位。首先核查10例logits与各层是否精确；如果边界或新内容出现差异，固定最早层检查，不扩大自由生成。如果精确，则按统一协议建立强制同历史decode与KV测试，先补完整缓存/旋转一致路径，再做端到端回归。prefill通过不跳过后者。
