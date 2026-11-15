# 28 GPU0：FusedMoE+prefix真实HTTP容量

前置：prefix probe 20261115T225622969152Z通过，8次/128输出token全部off/on一致，累计命中46592token；共享前缀ragged5.608→3.571秒，相同prompt2.112→1.269秒。GPU1正在独立next工作区跑FusedMoE+原生Attention，不修改它的源码。

## 命令

```bash
/opt/venv/bin/python /workspace/tencent-mellum-prefix/scripts/run_mellum_candidate_capacity.py --gpu 0 --profile fused_moe --prefix-caching
```

保持fused_moe baseline的Attention/SWA配置（SWA优化off），只增加prefix兼容路径与引擎缓存；不把未经验证的cached SWA CK/native Attention混入本轮。HTTP8251/engine8926，与GPU1的8151/8726隔离。35分钟执行预算，smoke B2/A8/C2→A64并发32→长度分层C20率0.3，stream/ignore_eos/temperature0/min-max256、thinkingTrue与前几轮保持一致。阶段重启服务，所以C20无额外预热，靠自然到达复用，不针对数据集特殊初始化。

服务新增--prefix-caching显式开关，只接受r/fused_moe＋batching；默认prefix关闭不变。worker逐case记录命中token数、命中请求数及cached批次新增token，原始prefix日志按case切片，继续核验真实batch、输入输出长度、stream、加载完整性与客户端审计。

## 分析

与性能23 FusedMoE相同小负载协议比较，实际机器从GPU1变为GPU0、另一卡负载不同，观察增益不作为最终受控速度比。可同时参考性能24的SWA组合，决定后续哪种组合更好。看C20命中与TTFT/TPOT/完成率，不以离线warm重复请求的2.64倍作为真实收益。

若命中为0，先判断引擎是否只能在请求结束后复用以及真实到达关系；若命中但性能回退，检查rectangular SDPA及gather成本，不直接宣称prefix无价值。默认不自动跑完整ABC/新HF；有收益后才做组合与必要功能回归。
