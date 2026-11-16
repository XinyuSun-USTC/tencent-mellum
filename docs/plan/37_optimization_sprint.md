# 37 连续优化与批量性能筛选

以fastpaths提交68a4086为基础建立独立step1/optimization-sprint工作区。用户明确要求优先持续优化，完整正确性与最终消融集中进行；本批不跑HF、HumanEval或公开200，不安装或升级依赖。

## 已返回的基础快路径结果

20261116T112501628142Z_resident与20261116T112507993869Z_kernels全部完成，190+342项产物哈希核验一致，各5631项权重加载完整。GPU1同卡A64：基线507.7752、Norm565.0037、Packed509.5110、组合625.6160 tokens/s；组合C20 TPOT/TTFT p99为57.912ms/1.02807s。GPU0驻留前后511.4879→532.6822。均为开发缩减负载，不是正式ABC，也不能赋予组合未做过的功能验收结论。

## 本批候选及源码依据

- fast_combined：上一批最快配置，保留空groups观察器作为本批对照。
- sprint_clean：数学与组合相同，不安装空Campaign。mellum.py仅在MELLUM_BATCH_CAMPAIGN_CONFIG非空时安装；原空groups仍会进入Campaign.begin并产生额外Python控制/设备到CPU同步，且model_runner.run_model明确拒绝非eager。关闭空观察器同时消除该Graph障碍，无需修改现有guard。
- sprint_graph：在clean上使用ATOM既有capture_cudagraph/run_model replay，prefill仍走eager，level0、TP/DP1；捕获1/2/4/8/16/32批次。捕获期间跳过native审计与context_lens.tolist，真实运行的批大小审计移到runner replay外侧，与eager一样保留可核验批证据。graph_replays.jsonl在实际replay后按key首次记录；没有回执判失败，不能以去掉enforce-eager冒称Graph成功。
- sprint_unpacked：clean上关闭Packed，检验Norm+驻留能否优于625.62组合。
- sprint_router：clean上使用FusedMoE内置select_experts路径（custom_routing_function=None），替代mellum_native.hf_route的softmax/topk/归一化/低精度往返；gate投影、专家、Norm与Attention不改。属于有意允许数值变化的性能候选，最终才按题目验证。

改动只在独立工作区；不改冻结依赖、既有带support_torch_compile装饰器的模型。runner新增回执仅在设置审计路径时开启。未新增自定义Graph引擎，不自动降级掩盖失败。

## 一次启动两条实验线

GPU0（最多45分钟执行预算）：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-sprint/scripts/run_mellum_optimization_sprint.py --track graph --budget-minutes 45
```

顺序fast_combined→sprint_clean→sprint_graph。

GPU1（最多45分钟执行预算）：

```bash
/opt/venv/bin/python /workspace/tencent-mellum-sprint/scripts/run_mellum_optimization_sprint.py --track kernels --budget-minutes 45
```

顺序sprint_clean→sprint_unpacked→sprint_router。GPU0端口8251/8926，GPU1端口8151/8726。每候选A64/并发32、C20/0.3req/s/256输出；去掉重复B2/A8/C2启动，A64本身承担首次运行门槛。输入每track一次准备并只读复用。准备CPU耗时不包含执行预算，子阶段也受剩余预算限制；通常预计每卡约10–20分钟，Graph首次编译可能延长，预算非时间保证。

所有候选检查加载、请求完成、长度、实际batch、后端及来源/输入哈希，失败保留并继续独立候选。未运行完整数值检测，不将服务完成等同模型正确；同样不以token差异阻断后续优化。脚本默认不启动完整ABC或评分。

## 后续决策

1. Graph成功且有收益则在最快合格性能组合上继续；捕获失败只分析首个明确阻碍并做有界修复，不启动漫长数值排查。
2. 原生路由若有收益，下一批组合路由+Graph；无收益即停止该支线。
3. 接着评估QK-Norm/RoPE融合、原生prefix组合、有限chunked prefill参数，依源码条件实现，不能把现有不支持缓存的direct-KV路径直接打开chunked开关。
4. 最终集中进行题目正确性、完整ABC及正式消融。微小收益先重复对照，大收益优先组合；不为每个组件跑全套HF。

CPU检查27项通过，py_compile/diff检查通过；GPU环境/dev/kfd和/dev/dri在工具侧不可见，需用户终端执行。GPU结果未产生，不预先宣称收益。
