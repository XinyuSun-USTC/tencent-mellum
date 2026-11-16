# 取消修复的多请求交错验证

已有单请求真实断连检查通过，但不能证明取消一条请求不会影响另一条。GPU0的SWA筛查源码保持不动；本轮只在交付工作树增加外部工具 `scripts/check_mellum_abort_interleaving.py`，运行已测工作树 `/workspace/tencent-mellum-disconnect-abort`，不修改其e15b691实现。

工具要求已完成的r_abort运行`20261116T223755052494Z_disconnect`，核验全部产物与模型/服务源码指纹，防止换代码借用通过结论。

## 两个独立服务用例

1. 同一参考组合单独生成一条真实C请求，记录128token控制输出。
2. 再提交同一提示的存活请求，收到首内容后提交第二条真实C输入、请求256token。
3. 分别在第二条的role帧或首内容帧后关闭连接。
4. 存活请求须实际完整输出128token并收到DONE/usage；ABORT发生时须仍有请求运行；取消ID移除后不再调度，存活ID须继续调度并完成。
5. 再提交8token短请求验证新分配与生成，等待最终回执并核对waiting/running为零；正常请求不得误发ABORT。

两种取消时点分别启动新服务，避免遗留请求污染。存活输出与单独输出的文本一致性记录但不设额外严格门槛，因为交错批形状可改变浮点路径；不因此跳过最终正式正确性。流间隔与文本长度不冒充token数，完成长度来自usage。

运行由run_stage管理拥有的进程组，正常、失败或超时均清理其后代，不影响GPU0。默认20分钟预算，预计约3–8分钟，无完整HumanEval或C200。两个CPU检查与prepare-only已通过；实际GPU结果待测。通过也只证明本轮请求生命周期，不代表容量达标。

```bash
/opt/venv/bin/python /workspace/tencent-mellum-delivery-final/scripts/check_mellum_abort_interleaving.py --gpu 1 --budget-minutes 20
```

两项计算与生命周期候选都验证后，再在独立集成版本组合；不修改正在使用的源树或移动旧标签。
