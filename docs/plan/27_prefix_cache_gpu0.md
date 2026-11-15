# 27 GPU0并行开发：Mellum前缀KV复用

GPU1继续运行next工作区的FusedMoE+SWA CK容量测试；本工作在独立tencent-mellum-prefix（基于9a3b3ca），不修改GPU1源码。GPU0用于新增prefix能力的定向试验，不重跑HF或HumanEval。

## 已实现

新增ATOM/atom/model_config/mellum_prefix.py：

- install_prefix_rope：保留原rope_cache的V与槽位写入、格式选择；cached场景仅对局部metadata副本推迟gather，参考K覆盖后读取完整KV；SHUFFLE调用原框架gather，NHD复用已有gather_reference_kv按当前请求读取，避免原实现对整个分配池permute.contiguous。共享metadata不修改，引用计数、页表、block ownership仍由引擎负责。
- cached_prefill：支持每请求q_len<=kv_len的ragged输入，验证query绝对位置恰为完整KV的后缀。矩形causal采用bottom-right绝对坐标；SWA额外应用窗口条件，显式mask与GQA repeat走Torch SDPA。
- MELLUM_PREFIX_COMPAT=1显式启用；关闭沿用原路径。cached-prefill暂不调用SWA CK（uncached路径保持既有选择）；没有实现物理SWA回收/FP8/调度/MTP。
- layer0记录真实cached/new token数。复用的是KV，不缓存答案或针对评测ID写分支。

## GPU0命令

```bash
/opt/venv/bin/python /workspace/tencent-mellum-prefix/scripts/run_mellum_prefix_probe.py
```

固定GPU0，engine8826，与GPU1 HTTP8151/engine8726分离；只启动离线引擎，不占HTTP端口。25分钟执行总预算（CPU准备另计，最多可调40）。按序启动cache off/on两个worker，均R参考配置、BF16、eager、TP/DP1、maxseq4、chunked off、16输出ignore_eos。

三条真实C prompt C-0000/C-0100/C-0199，实际长度7889/11039/4294，thinking=True；此小试验不声称覆盖16k。四组：首次冷请求、相同prompt复用、相关前缀三请求ragged、逆序复用。两套相同输入；记录实际命中/新token、输出差异、分组耗时、加载和源码hash。

这不是再次追求HF逐位对齐：无需HF模型；目的是检查新增缓存机制有没有错误复用KV、绝对位置/窗口偏移、是否真的命中。off/on输出分叉记review，不自动宣称功能不达标。FusedMoE此前已存在重复生成波动，首轮使用R可隔离该因素，不用它的不同输出错误归因prefix。

## 验证与后续

CPU检查覆盖参考K先写后gather、共享metadata不变、矩形绝对因果/窗口mask、ragged后缀与完整Attention一致、错误位置拒绝；相关旧batch/Attention/native回归同时运行。真实输入prepare-only通过；GPU尚未执行。

GPU试验若无命中则检查引擎block复用配置；有错误则按日志修一个明确缺口，不追加全模型数值搜索。命中且行为合理后，服务入口显式接入prefix开关，组合FusedMoE，按同一C20作缓存开/关吞吐/尾延迟消融；自然冷启动，不针对公开集特殊预热。之后根据C长prefill干扰决定是否继续chunked prefill。

功能门槛与最终ABC仍独立；本轮分组耗时不替代HTTP和正式性能成绩。最终提交从开发候选中收敛最多三个有收益的优化点，计点讨论不阻塞实现。
