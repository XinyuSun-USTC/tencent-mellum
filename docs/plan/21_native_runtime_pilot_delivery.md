# 21 原生MoE与Attention两卡候选交付

这是[计划20](20_native_moe_and_attention_routes.md)的首轮实现，依据[研究05](../../report/research/05_optimization_implementation_sources.md)。没有GPU正确性或性能结果，不宣称候选已通过。独立worktree不修改原R基线目录。

## 已实现

### GPU0：整批BF16 FusedMoE

新增`ATOM/atom/model_config/mellum_native.py:MellumFusedMoeBlock/hf_route`，开关`MELLUM_MOE_BACKEND=fused_torch_route`。保留Mellum gate的F.linear、FP32 softmax/top-k/renormalize和BF16路由权重舍入；AITER排序接口要求FP32存储，因此在BF16舍入后转回FP32。专家ID int32。这一接口要求已核对冻结AITER `csrc/py_itfs_ck/moe_sorting_kernels.cu`和`py_itfs_cu/moe_sorting_opus_kernels.cu`。

专家计算为冻结`FusedMoE`，参数E64/K8/H2304/I896、SiLU、无bias/共享专家；权重仍由现有loader和quant_method做一次加载后shuffle。没有自行换权重格式或增加量化。不再安装E逐请求专家wrapper，仅保留L控制；Attention保持R参考路径。

layer0审计记录实际整批行数/请求数、FusedMoE和UnquantizedFusedMoEMethod、两份权重shuffle标记。该审计确认类/调用路径，尚不是某个具体GPU kernel的trace证明。

### GPU1：原生批量Attention

开关`MELLUM_ATTENTION_EXECUTION=native`。保留已验证的分层RoPE和BF16 KV修正写入，关闭旧SDPA decode和参考ragged桥；取消不再使用的raw Q/K clone。这样先替换Attention compute，避免同时更改RoPE语义。

`native_prefill`一次调用AITER flash_attn_varlen_func处理真实cu_seqlens、原生GQA和窗口；窗口left=W-1，全局层无窗口。`install_native_attention`的decode使用原PagedAttentionImpl.dispatch_backend所选的Triton/ASM分页内核，一次处理整个批次。若不是paged backend就失败，不静默SDPA回退。

layer0记录prefill varlen和decode实际backend、请求数；沿用batch元数据审计。此版只支持完整未缓存prefill，不启用prefix/chunk。MoE保持R。

## 两个终端分别运行

GPU0，worktree分支`step1/native-runtime`：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-native/scripts/run_mellum_native_campaign.py --track moe
```

GPU1，固定同一代码快照的独立worktree `step1/native-attention`：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-native-attention/scripts/run_mellum_native_campaign.py --track attention
```

每个track自动绑定对应物理GPU并拒绝错误GPU；引擎端口分别8226/8326，不开HTTP服务。通过各自绝对路径启动，不依赖当前cd；子进程PYTHONPATH明确指向所在worktree。两边运行期间不再修改这两个工作区源码。

预计每条约20–40分钟，可并行；单条累计预算60分钟、每worker上限30分钟、最多3次worker启动。每15秒打印日志心跳，progress.json按请求完成落盘；长batch在整组完成前日志可能不增长，不等同挂死。

## 自动阶段与门槛

1. **当前R对照**：S1、S32、L32，每条固定256输出、ignore_eos=true，各两次、预热。S1真实并发1，其余32。R与候选用同一组原始token ID，不改变prompt长度。
2. **候选性能**：同负载和预算，记录实际256token、wall、输出、真实batch和native路径。若batch组加速几何均值<1.10×或任一组<0.95×，结束为no_material_speedup。这是性能筛选，不是题目附加正确性标准。
3. **有收益才做D200**：正常greedy64、ignore_eos=false、固定HF缓存；200全部请求完成且实际batch32。总体位置>=90%标记candidate_eligible_for_combination，整序列/长短/来源全部单列；不要求内部张量精确，不施加相对R下降2pp限制。

性能256token明确不计算HF正确性分数，避免拿64token参考误判；D200使用原有比较器，保留提前EOS分母。两个阶段参数不同且如实记录。

这是独立加载进程、每配置两次的探索性测量，**不是同worker ABBA或官方ABC成绩**。候选性能阶段包含layer0 native审计，R没有该额外native审计；均有相同batch审计。正收益用于晋级，不直接包装为最终精确加速比。两卡共享主机资源，最终消融/计分需空闲环境。

## 预算、恢复和交付边界

输出分别`/workspace/results/mellum_native_moe/<UTC>`、`mellum_native_attention/<UTC>`。运行成功或失败都回传各自最后RESULT和目录。`--resume <目录>`需相同track、worktree、源码和输入hash；已完成job产物校验后跳过，预算和最多3次启动次数不重置，失败尝试归档。

底层worker新增白名单runtime_overrides、可选独立端口、256性能记录；默认旧实验行为不变。按当前输入+输出计算max_model_len，避免长prompt+256越界。仍为BF16/eager、TP/DP1、prefix/chunk关闭；不修改被冻结的依赖和已有compile装饰模型。

本轮未运行GPU：工具无/dev/kfd和/dev/dri。CPU包括路由权重/ID、窗口含当前位置、请求隔离、decode整批单次调用、Fused整批接口、dummy审计、速度先筛/正确性后筛、恢复及256预算分类测试。CPU只能验证语义与接线，GPU kernel适用性和真实速度由以上实验决定。

本轮不自动组合两个候选、不启动完整ABC/16k/HumanEval。单点结果具备条件后，下一轮固定一个组合，集中完成这些验收，失败方案不展开无限数值修复。

交付CPU核验：原有回归与新语义/调度测试共117项通过（116项组合回归＋新增shuffle审计用例），py_compile和git diff --check通过；GPU0 prepare-only `20261115T182107624040Z`通过。GPU1在独立快照建立后单独prepare-only。仍无GPU运行结论。
