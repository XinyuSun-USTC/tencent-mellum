# 10 首轮ragged多请求与HTTP服务GPU检查

## 结论与范围

B0参考桥接实际运行到prefill/decode batch4，错峰结束、反序/块复用与迟到请求均完成；四条bs1全部与HF前缀精确。多请求小集出现少量生成差异，按位置统计仍超过95%，不把全token逐位一致加为新的容量测试门槛。后续扩大8→16→32容量，并核验HTTP batch4，不再为本轮少量差异立即启动大规模数值追踪。

HTTP bs1的completion非流/stream、usage、DONE及两个并发客户端请求均完成；chat原始reasoning相差一个开头换行，属于既有展示规范差异，不能据此称推理错误。发现正常完成非流式请求前端登记未清理，已修复并做CPU回归，尚待下一轮HTTP日志核验。

GPU代码对应`142a929`，offline运行`20261114T214224031219Z`、HTTP `20261114T214721936495Z`。[原始轻量记录](../../artifacts/performance/10_batch_and_http_smoke/manifest.json)，[计划13](../../docs/plan/13_serving_batching_and_workload_baseline.md)。本轮分析后仅修复前端完成记账、测试分类及启动指纹记录，模型数学未改，没有新的GPU执行。

## Offline真实batch结果

reference、MELLUM_REQUEST_BATCHING=1，max_num_seqs4、eager、TP/DP1、BF16、无prefix/chunk、无点1优化。加载5631/5631，无missing/unexpected；审计来自真实layer0调用，并非客户端并发数推算。

| 阶段 | prefill请求数 | decode请求数 | token位置匹配 | 完整序列 |
| --- | --- | --- | --- | --- |
| 单请求四例合计 | 1 | 1 | 64/64 | 4/4 |
| pair | 2 | 1/2 | 46/48 | 1/2 |
| ragged4 | 4 | 1/2/4 | 62/64 | 3/4 |
| reverse_reuse | 4 | 1/2/4 | 64/64 | 4/4 |
| late_arrival | 2 | 1/2/3 | 62/64 | 3/4 |

所有阶段304位置有298相同（98.03%），仅多请求阶段234/240（97.5%）。这些是同四条样本多次运行，不是独立公开集；不将该统计称为正式≥90%验收。

差异均在humaneval_0的16token检查中出现：pair/ragged4/late各14/16；reverse中该请求排末尾则16/16。说明批量/排列可影响生成，但本轮未定位到投影、MoE或其它具体算子，不能直接宣称只因舍入，也不能仅凭CPU隔离测试声称GPU不存在任何缓存错误。相同位置/长度的其余三请求在当前检查预算内匹配。

迟到请求插入时前端仍有1条原请求pending；随后实际decode batch3，证实运行中加入请求。完成后batch收缩到2/1，反序轮完整匹配；输入包含1023/1025边界和跨16-token分页，使用实际KV池。审计可能包含引擎末尾额外forward，不把forward次数视为生成token数。

## HTTP bs1结果

server max_num_seqs=1、无点1、无batch桥接；两个客户端并发只是排队服务验证，不能声称HTTP GPU batch2通过。

- 两条样本各非流/流，共4次，输出文本均与HF参考skip_special_tokens=True展示一致。
- usage完成token分别16/32准确；SSE均收到DONE；并发两请求也分别文本、usage一致。
- chat非流reasoning为`Okay,...`，stream为`\nOkay,...`，content均空，16token usage准确，收到DONE。`reasoning.py:separate_reasoning`对未结束think内容使用strip，stream parser保留开头换行，因此这是原始rendering差异。
- owned server正常关闭，所有EngineCore退出；未验证异常断连/engine abort，不对此承诺。

验证脚本保留raw stream_matches=False，同时新增只对reasoning外围空白归一的比较；最终答案/code content仍严格比较，不修改API输出或生成token，也不悄悄改原GPU产物。归一比较通过只能解释这一展示差异，不等同原始文本逐字相同。

## 正常完成请求的前端记账修复

原日志顺序请求后pending数升至5/6，源码发现非流`generate_async`/`generate_async_multimodal`/`generate_async_fanout`绕过InputOutputProcessor.postprocess，完成后没有pop前端requests；流式cleanup已有pop。

本轮在三个非流路径收到finished后、返回结果前，删除自己的已完成seq登记。保留其它活跃请求，不更改EngineCore调度、KV释放或采样；不是取消GPU计算的abort功能，异常断开等路径仍需后续验证。新增对实际async函数体的CPU测试，覆盖文本、多模态及fanout，检查只清理本次完成ID；测试仅将executor替为inline，未声称CPU测试验证了真实线程调度。

相关32项CPU检查通过（完成清理、reasoning、SSE/服务配置、ragged），编译/diff检查通过。HTTP验证增加pending日志上界观察（本小集最多同时2请求），启动manifest增加API及共享配置源码指纹。修复后尚无新GPU结果。

## 下一步命令

已有bs4实际执行证据，下一条一次加载按8→16→32阶梯检查；任意执行/内存错误中止，生成差异保留review，不发明bitwise门槛：

```bash
/opt/venv/bin/python /workspace/scripts/validate_mellum_batch.py --max-seqs 32
```

再顺序运行HTTP batch4（会自动启动/关闭服务，端口8051/8026不能同时占用）：

```bash
/opt/venv/bin/python /workspace/scripts/validate_mellum_http.py --max-seqs 4
```

预计各为数分钟级小集，实际取决批量临时显存/首轮预热；前者1200秒上限、至少80GiB free，后者延用启动/请求超时。结果用于容量/接入核验，仍非A512×256或正式HTTP尾延迟。完成后再准备实际A/B/C与批量profile客户端，不提前宣布吞吐收益。
