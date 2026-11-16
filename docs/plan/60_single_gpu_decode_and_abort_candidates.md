# 单卡主线：SWA分组解码与断连取消修复

## 已获得的诊断结果

运行 `mellum_c_service_diagnosis/20261116T221216213089Z_decode` 在C真实输入、无新增请求的条件下，批1/4/8/16/32的最后prefill后实际交付速率分别为18.67/32.19/38.42/42.09/43.68 token/s。32批完成4096输出token，但末次prefill前已有部分交付，后段统计为4066token/93.092秒。不能用调度工作4096冒充该窗口的交付数。

该组即使隔离持续prefill仍明显低于单卡C的76.8token/s需求；不是全数据的理论上界，但足以说明不能只调prefill。断连运行`20261116T221223452847Z_disconnect`中，正常完成后0个新步骤；role/首内容帧后断连，分别仍启动236/253步，证明生命周期缺口真实存在。主机step累计时间13.46/13.75秒不是GPU浪费时间。

依用户选择，单卡为当前主线，P/D暂缓；两个候选在独立工作树，均不修改已测诊断版本。

## GPU0：参考SWA分组解码

工作树 `/workspace/tencent-mellum-swa-decode-groups`，分支`step1/swa-decode-groups`。

- `r_decode_meta`：每次forward只解析一次context lengths和positions，各层复用，消除重复标量取值；`Campaign.begin()`在每次真实forward重置，不跨步/请求重排复用旧值。
- `r_swa_decode2/4`：在上述基础上，对SWA层按2/4请求向量化分页KV读取并一次调用Torch SDPA。只对每个请求均达到窗口、NHD缓存的组启用；短请求、单请求及其他布局走参考分支，全局注意力仍按请求执行。
- 保留BF16、参考RoPE、独立QKV、专家组4与显式全真SWA mask。GPU批量形状仍可能影响数值，CPU精确对照不等于GPU验收。

`screen_mellum_swa_decode_groups.py`自动执行：先未改动组合R1的隔离decode计时；依次对metadata/group2/group4做完整developer200及长输入筛查，未到90%的版本不追加性能；通过者测相同batch8/32、64输出的隔离decode；最后测R2控制漂移。没有重复HF、HumanEval或C200。实际分组事件必须存在，不能以回退作为优化结果。

跨轮128输出结果不直接与本轮64输出比增益，使用本轮R1/R2。阶段审计和功能记录有扰动，这轮是有界筛选，不是正式ABC或裁判中位数。

## GPU1：默认关闭的安全取消候选

工作树 `/workspace/tencent-mellum-disconnect-abort`，分支`step1/single-card-disconnect-abort`。

`r_abort`只在参考组合上打开`MELLUM_ABORT_ON_DISCONNECT`。当前仅覆盖单样本chat流接口；fanout与text completion不作已修复声明。限制TP1/DP1、eager、无MTP和KV传输。

链路：流生成器finally→前端发送ABORT→IO线程只入队→EngineCore主线程聚合取消→对活跃请求先调用ModelRunner设备同步→Scheduler移除waiting/running请求并释放块。前端映射及callback清理，不向已关闭客户端伪造成功输出。重复或已完成ID不重复释放，部分prefill计数同时维护；默认关闭时保留旧入口行为。

实际HTTP再次测正常完成、role后关闭、首内容后关闭。要求观察到真实abort移除回执，且移除后该ID不再被调度；同时记录KV空闲块变化与已在途步骤，不要求关闭瞬间取消已经开始的GPU工作。每个用例独立服务，由父进程清理拥有的进程组。修复可能减少浪费，但不能凭此承诺弥补decode服务能力缺口。

## 后续选择

单卡计算和生命周期问题分别测增量。若SWA候选兼顾生成与速率，再与已验证取消修复组合，先C60后C200，并只对最终有价值方案补完整HumanEval。若SWA数值未达门槛，保留metadata-only收益或回退；不恢复整条未通过的原生路径。新版本均不自动替换交付默认，也不把C60、隔离decode或断连测试写成正式通过。
