# 候选交付入口的 CPU 集成准备

分支 `release/qualified-entrypoint` 基于分段实验提交 `6730362`，独立于正在运行的 `tencent-mellum-qualified-chunked`。不修改其源码，不升级依赖，不占 GPU。尚未将实验候选设为交付默认。

## 已实现

- `start_server.sh` / `serve_submission.py` 新增显式 `--profile`，可选择参考组合与 chunk512/1024/2048，旧 conservative/aggressive 默认保留。
- 显式 profile 的状态不继承旧 preset 的验收标签；启动记录保存真实 profile、token 预算、全部 Mellum 实现和调度器哈希。
- `--disable-expert-grouping` 只取消解码专家分组，保留参考批处理和共享路由。
- `--disable-chunked-prefill` 同时取消分段桥接、分段预算和解码配额，回到未分段参考组合路径。
- `--disable-direct-kv` 沿用现有关闭方式；当前分段桥接依赖 direct KV，因此不能在分段仍启用时单独关闭它。入口显式拒绝，而不静默回退；先关闭分段后可继续关闭 KV 单次写入。
- `smoke_submission.py` 增加候选 profile，使用真实最终入口、16640 上下文、32 槽及一条 C 输入检查普通和流式请求。该脚本已准备，但 GPU 启动检查未运行。

## CPU 证据

七项入口与流协议检查通过。五种实际 dry-run 覆盖参考组合、分段、关闭专家分组、关闭分段及同时关闭分段/KV。25 个模型、调度和 profile 文件逐字节等于当前分段实验工作树。证据为 `artifacts/submission/qualified_entrypoint_cpu/`。这些证据不构成 GPU 运行或正式验收。

## 选定候选后再运行

以下只是模板，不应在当前双卡实验未结束时执行；profile 必须替换为实际选定配置。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-qualified-entrypoint/scripts/smoke_submission.py \
  --gpu 0 --modes conservative --profile r_chunk1024_d4
```

同一入口可用于最终测量。进入最终交付前仍需用实际有效配置复核完整正确性和 C 容量，并使累计消融最后一行与默认配置一致。不能把未分段组合的正确性和分段方案的性能直接拼接。GPU 实验失败时，保留候选入口为实验功能，不据 dry-run 晋级。
