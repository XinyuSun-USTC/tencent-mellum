# 08 首轮单请求性能Profile实验

## 范围与状态

本轮为Step1准备的单卡BF16/eager、bs1诊断采集，不是正式A/B/C跑分，也不是新的正确性验收。模型数学、权重与已验证后端保持Step0组合；不支持的bs8/32、多请求、HTTP负载尚不纳入本轮。

已开发并完成CPU检查，GPU采集能力待用户普通终端验证。现有Torch2.10 ROCm/Kineto、rocprofv3、rocprof-compute和SMI工具可用，无新工具安装或冻结依赖升级。本轮只用Torch Profiler，暂不运行硬件counter或evalscope。

## 新增与修改

| 位置 | 职责 |
| --- | --- |
| `scripts/mellum_runtime_config.py` | 可复用的显式候选环境和引擎参数；CPU测试与200条真实运行settings对照；清除继承的Mellum dump/forced/replay设置 |
| `scripts/profile_mellum.py` | 父进程准备固定输入/指纹，启动受监控worker；一次加载、预热、无profile计时、短trace、等待回执、校验和分析 |
| `scripts/analyze_mellum_profile.py` | 离线解析Chrome trace/gzip，汇总kernel/CPU算子、phase和模块归因，输出JSON及Markdown |
| `ATOM/atom/model_config/mellum_profile.py` | 仅观察的module标签、原子worker回执，不改tensor输出 |
| `ATOM/atom/model_engine/model_runner.py` | opt-in profiler标签安装/移除及实际启动/停止回执；开关未启用时保持既有路径 |
| `scripts/verify_project_archive.py` | 默认核对Step0标签源码与轻量证据，允许Step1工作树变化；`--worktree`可严格对比当前文件 |

已有`check_atom_mellum.py`验证入口未重构，以避免在此改变Step0测试路径；新增统一配置供本轮和后续服务复用，由测试核对其与真实通过配置一致。正常Step0生成不设置`MELLUM_PROFILE_LABELS/CONTROL_DIR`，不会安装新hook或写profile回执。

## 采集协议

默认输入长度1024、4096、16384，均单请求。1024/16384来自已归档原始固定输入；4096由8192输入截取prefix并保留最后64源token，保证精确长度和稳定尾部。输入ID、来源文件哈希、ID哈希和模型config哈希写入plan；是合成诊断输入，不冒称真实场景A/C代表全部分布。

一次模型加载，max_model_len按最大输入加64向16对齐，至少256，max_num_seqs=1、util=0.35、TP1/DP1、BF16 KV、eager、prefix/chunked关闭。每组依次：

1. 预热请求，8输出token。
2. 两次无profiler请求，各64输出token，记录客户端wall time和引擎request latency/TTFT/TPOT，不声称p99或正式吞吐分数。
3. 启动profiler并等待rank0 started回执，采集一次16输出token普通greedy请求。
4. 停止并等待stopped回执及trace导出，解析并检查GPU kernel、prefill/decode区间及Mellum模块标签。

本轮统一temperature=0、ignore_eos=True以固定预算；不是Step0正确性测试的normal EOS协议，也不是题目正式256输出性能负载。只有完成对应预算才记该请求成功。profile期间采集的请求耗时明确标为不可当benchmark，基线计时来自独立无profiler请求。

模块hook只在实际start_profile时安装，在stop_profile后移除；不让无profiler计时承担hook开销。收集attention_swa/full、qkv、output_projection、norm、router、moe_experts，便于初步区分MoE、Attention和投影。label嵌套不相加当总耗时。

ATOM的start/stop utility是异步发送，原API返回不能当作worker实际开始/停止。新增opt-in worker回执，在真实__enter__/__exit__及导出之后原子写入；启动/停止边界在worker设备上同步。启动等待上限60秒、导出等待240秒，整个worker默认900秒；父进程保留15秒心跳、死亡检查和进程组清理。

不保存完整logits/逐层tensor，不加载HF。默认关闭record_shapes/stack/memory；只有需要进一步诊断小样例才使用`--detailed`，不默认扩大trace。

## 首轮命令

在有GPU的普通终端执行，原路径兼容链接可用：

```bash
/opt/venv/bin/python /workspace/scripts/profile_mellum.py
```

先1024再4096和16384，首组失败即停止；不会为了“收集完整”继续跑后续组。一般可能数分钟，900秒是保护上限而非耗时承诺。

如仅希望先验证最小GPU采集链路：

```bash
/opt/venv/bin/python /workspace/scripts/profile_mellum.py --lengths 1024
```

只准备配置、不初始化GPU：

```bash
/opt/venv/bin/python /workspace/scripts/profile_mellum.py --prepare-only
```

本次prepare-only：`results/mellum_profile/20261113T172037168802Z`。实际GPU运行会创建新目录并记录当时commit、工作树状态及脚本/worker指纹，不复用prepare-only状态冒充执行。

## 结果与判读

输出根为`/workspace/results/mellum_profile/<run>/`：plan.json、summary.json、atom.log、worker_results.json、loading_audit.json、control回执、traces压缩Chrome trace、每组ctx_N_analysis.json及.md。只有真实GPU kernel非空、prefill/decode均存在且Mellum标签存在才给`profile_collected`；CPU-only trace、缺回执或导出失败均不算完成。

summary给采集数量和模块归因覆盖；详细计时在worker_results，kernel/模块表在analysis。分析器用External id/correlation将GPU kernel归到CPU launch所在的module/phase，不按GPU时间戳落入CPU范围来猜。CPU范围与GPU异步执行不重合时仍可正确归因。无法匹配保留unattributed，低覆盖时先检查真实trace schema，不强行给瓶颈结论。

`exclusive_module_kernel_us`按最内层标签归因；inclusive含父子重叠。kernel累计时间可跨stream重叠，CPU op时长包含嵌套，不能简单加总当wall time。另给kernel区间union和span/gaps；空隙本身不证明CPU瓶颈，须结合时间线。首次GPU数据后还要核对schema、归因和是否有异步末尾额外forward，才选择优化点。

trace在results外部目录，默认不纳入Git。后续保留精选汇总与原始trace哈希。真实GPU采集完成后才建立report/performance实测报告，不预写性能收益。

## 已执行检查与限制

CPU测试14项通过：候选配置与既有GPU记录一致、固定输入长度、原子回执/超时、异步时间线correlation归因/区间重叠、实际CPU Profiler下hook输出不变和可移除、提取真实ModelRunner start/stop方法的生命周期测试、进程监控及原运行器回归。Profiler单周期事件清理警告不表示测试失败；没有实际GPU证据时不声称ROCm采集已通过。

语法检查、prepare-only及差异检查通过。GPU仍需普通终端执行，不启用性能优化开关、不安装evalscope/TensorBoard、不将无profile两次样例当正式统计。后续依据本轮trace选择MoE或Attention先做，再补批量与服务场景；不把“较全面profile”扩大为长期全组合扫描。
