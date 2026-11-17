# 双模式交付入口集成

本轮交付工作树为`/workspace/tencent-mellum-delivery-dual`，分支`release/dual-mode`。基于最新精度优先实验分支，保留原始报告及实验历史；导入性能优先轻量证据。ATOM功能代码与两条已测路线一致，本轮仅集成配置、入口与检查工具。

入口`start_server.sh`调用`serve_delivery.py`；`mellum_submission_modes.py`统一定义两种模式、累计行与子集。默认performance＋step0123，precision＋step0123对应已完整验证的`r_swa_decode4`，不再使用旧第三项prefix/chunk/abort组合。历史delivery_step配置保持不变，避免改写实验身份。旧`run_delivery_matrix.py`主入口停止执行并提示在历史分支复现，防止其旧行定义与新默认入口混用。

## 开关

`--mode performance|precision`独立于优化编号；`--row step0|step01|step012|step0123`控制累计消融；`--disable-opt 1|2|3`可重复。

精度模式支持八种子集；性能模式支持空集、①、①②、①③、①②③。性能②需要原生投影/Attention/MoE路径，③也按原生前缀兼容路径使用，故两者均要求①。启动前拒绝不兼容子集，不自动改写用户选项。非累计子集不引用累计测量成绩。

`launch.json`的submission字段记录mode和实际points，enabled_optimization_points同步填写；settings与引擎命令记录实际环境、Graph、prefix及32槽设置。记录新配置定义与入口源码哈希。

## 已完成验证

112项CPU测试通过。全部13种合法组合从真实根入口以prepare-only执行，核对profile、环境字典、模式与实际优化点。累计四行与此前测量profile逐字段相等。全部ATOM源码与性能实验分支无差异。CPU准备证据：`artifacts/submission/dual_mode/cpu_preparation.json`。

## GPU入口烟测待执行

两卡分别执行，覆盖各路线四行，每行检查权重5631张量加载审计、health/models、非流式与SSE八token回复、usage、端口回收和模式身份。预计每卡约3–8分钟，取决于加载与Graph捕获。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-delivery-dual/scripts/smoke_dual_delivery.py --gpu 0 --mode performance --all-rows
/opt/venv/bin/python /workspace/tencent-mellum-delivery-dual/scripts/smoke_dual_delivery.py --gpu 1 --mode precision --all-rows
```

此为入口集成烟测，不重新定义正确性与性能门槛。结果返回后归档，再统一更新报告中的两条路线及完整矩阵。
