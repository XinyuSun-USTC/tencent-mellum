# 04 正确性回归协议

状态：Step0本地开发验证已通过，进入Step1。当前结果见[报告39](../../report/validation/39_step0_summary.md)，配置和命令见[05基线复现](05_step0_baseline_and_reproduction.md)。本协议维护后续仍有效的比较规则；早期逐轮执行计划已[保存快照](../../report/archive/step0_document_snapshots/docs/plan/04_correctness_protocol.md)。

## 固定参考与输入

- ATOM使用冻结`/opt/venv/bin/python`，HF使用独立5.10.1参考解释器，不混用依赖。
- 共用只读BF16权重，检查model config SHA256；固定输入ID、mask、position、模板/thinking、生成预算和EOS/pad，不通过重新分词或改停止规则隐藏失败。
- HF Attention为SDPA。正常generate实际prefill为grouped_mm、GPU decode临时batched_mm；手动cached诊断若未进入该context仍是grouped_mm。记录实际执行值，不能只记结束后的config。
- Step0实际后端与默认启动不同，按05选择；每轮核对worker实际设置和加载审计。

## 比较与判读

1. 加载完整性：成功回调槽5631/5631、missing/unexpected为空；不是全权重逐元素审计。
2. 普通生成：无forced计划、正常EOS，逐例报告实际输出长度、完整序列一致数和token位置一致率。题目≥90%分母未完全明确时两者同时报告，不自行选更宽松口径。
3. 长输入：涉及SWA、YaRN、KV、Attention或调度的改动，补1024附近、8192附近和16k；按实测覆盖说明，不泛化到所有长度。
4. 任务质量：HumanEval需完整答案、统一代码提取、测试/执行及有效参考值；64token精确不能代替pass@1。被Rejected参考不使用。
5. logits/层状态：被动观察不得修改输出；只比较相同历史下采样前向量，含首分叉处，分叉后停止同输入比较。非零BF16差异不能自动判定题目失败，需结合token/任务指标。
6. 强制历史：仅定位decode，核对真实输入/position与实际分页KV；修改前raw logits才可评分。forced token相同不作为普通生成通过证据。

## Step1回归安排

每次改动先执行影响范围内的CPU检查和短greedy回归；窗口/缓存/长输入相关改动补相应边界。候选组合形成后跑固定200条开发集合与长输入，记录题目正式指标和未确认口径。无需每轮重跑全部诊断、重新安装参考环境或全量dump中间层。

出现token/任务退化时先检查保存产物、配置和输入；若logits精确但token分叉，先查并列最大值及采样规则。若同历史数值不同，先检查实际后端/shape/缓存，再用固定输入首差定位，不盲目逐个切换后端或放宽阈值。不得为提高通过率删样本或替换参考语义。

## 证据与运行生命周期

每次保存唯一结果目录、输入/参考/候选、实际配置、加载审计、日志与来源哈希；不覆盖旧结果。当前manifest只是整理时源码指纹，不能回填为历史测试commit。原始结果路径和代码版本必须区分。

阶段有15秒心跳、worker死亡检测、可配置超时及自有进程组清理。工具故障、timeout、blocked_no_gpu与数值失败分开记录，优先复用已完成且哈希/协议一致的参考产物。运行命令和超时上限统一维护于05。
