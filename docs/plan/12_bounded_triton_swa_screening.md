# 12 优化点1：一轮有界Triton窗口实现筛选

## 目的与停止条件

用户同意：对当前Attention点做一次替代实现筛选，再推进多请求/服务基准；不继续扩大同类数值定位。当前CK保留可选候选，开发200位置93.11%、激活32条56.93%、16K TTFT约2.8×，见[性能07](../../report/performance/07_representative_correctness.md)。不因非bitwise输出直接否决，也不把局部误差小当最终验收。

本轮仅现有AITER Triton default local forward一条路线，只用已采集1K/4K首层真实Q/K/V。无模型加载，无HF重新生成，无依赖升级，无16K/ABBA/200条重复。若不能兼顾数值和速度，结束该筛选，保留CK，不再尝试大量配置或其它新内核。若有明确潜力，才接同一优化点的替代实现开关、复用HF参考做32条激活输入生成比较；改善明确再扩200与性能。

## 冻结源码核验与参数语义

- `/app/aiter-test/aiter/ops/triton/attention/mha.py:flash_attn_varlen_func`（805行起）：THD varlen、MQA/GQA、BF16路径可调用。返回无lse/probs时为单Tensor。
- 同文件`_flash_attn_forward`（约82行）：明确拒绝`window_size_right != -1`。不能照抄CK `(1023,0,0)`；适配器使用`(1023,-1)`及`causal=True`。default实现会调用`_attn_fwd`；dao_ai分支断言不支持local，因此不切换全局实现，只接受default，否则报错。
- `/app/aiter-test/aiter/ops/triton/_triton_kernels/attention/mha.py`：约185行causal上界、195行`k_pos >= q_adj - SLIDING_WINDOW`含左端，所以SLIDING_WINDOW=1023代表最多1024 key；约562行`off_k_head=off_q_head//grp_sz`为连续Q-head组映射KV，匹配32Q/4KV。
- 新[triton_swa_prefill](../../ATOM/atom/model_config/mellum_swa_triton.py)只作筛选适配，拒绝window<2（内部0关闭local mask）、不兼容shape/GQA和dao_ai状态；编译/执行错误传播，无静默回退。没有接入MellumAttention，也没有改默认/CK实现。
- gfx942上的实际编译/执行与kernel选路仍需GPU验证，不能只因源码存在就宣称支持已通过。脚本限定分配的gfx942并保存环境/两份冻结源码SHA。

## GPU筛选步骤

```bash
/opt/venv/bin/python /workspace/scripts/screen_mellum_swa_triton.py
```

源为`/workspace/results/mellum_swa_layer/20261114T145253662684Z`，校验两份原始QKV SHA及reference精确回放状态，不重采集。

先用n1025、window1024的零Q/K、首key脉冲V做可解释探针：第0位置精确检查GQA映射，第1023位置仍含首key，第1024位置必须完全排除首key。探针的小数值容差仅用于均值求和检查，不是HF/张量验收阈值；窗口外脉冲输出必须严格0。

随后在两份真实QKV上计算SDPA（显式mask+KV展开）、原CK、Triton。SDPA必须逐位重现原采集输出；各输出必须同shape且有限。报告CK/Triton vs SDPA relative L2/max_abs、最后位置误差。这里不再重复FP32 oracle，前轮已有其结果，避免扩大诊断。

每方法3次预热，三个方法轮转顺序各10次CUDA event计时，保存所有样本和中位数；reference计时含mask和KV展开，CK/Triton含各自封装与实际分配，不混入profiler。最后仅Triton单调用采短trace、保存kernel名与SHA，确认实际执行路径。

结果`/workspace/results/mellum_swa_triton_screen/<id>/`，包含plan/summary/screening.json、screen.log和两个短trace。15秒心跳、总900秒上限。预算通常数分钟，但首次Triton编译耗时未知；失败即保留日志停止，不自动切其它实现或全局配置。`screening_complete`不是准确率通过或生产采用。

## 复核与下一步

误差明确下降且仍有实用速度空间，才值得32条正常64token生成验证；不新增要求所有浮点逐位一致的门槛。若相对SDPA误差无改善、速度明显不适用或编译不支持，记录放弃原因，结束本轮筛选。此为优化点1不同实现候选，不计成一个额外优化点，也不把不同候选拼为免费基线。

之后推进多请求/服务，使A并发32与C到达负载可测；批量支持的数学/执行策略变化与性能贡献需显式披露，再根据批量profile选择第二算子优化点。

## 当前完成状态

适配器、筛选入口和边界/GQA/无回退CPU检查已完成；相关22项通过，py_compile、diff检查通过。prepare-only：`/workspace/results/mellum_swa_triton_screen/20261114T193745688088Z`。GPU设备在工具中仍不可见，本轮未执行GPU、未宣称筛选结果，需用户终端运行上面命令。数学生产路径与当前优化开关不变。

## 实测与有条件接入

筛选200217完成，见[性能08](../../report/performance/08_triton_swa_screen.md)。边界/GQA探针通过、Triton路径trace确认；1K/4K relative L2分别0.000576/0.0000161，4K局部12.06×对SDPA，但慢于CK。已新增实现选择ck/triton（默认ck且整体优化仍默认关闭），相关44项CPU通过。下一步命令：`/opt/venv/bin/python /workspace/scripts/run_mellum_swa_200.py --eligible-only --swa-prefill-impl triton`，仅32条且复用HF，预计2–3分钟；结果改善明确才补完整200与性能。
