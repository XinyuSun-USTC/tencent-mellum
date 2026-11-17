# 来源与许可证

- ATOM：仓库 `ATOM/`，保留上游源码与 LICENSE，通过非squash导入保留上游历史。本项目在冻结容器版本上修改；未替换框架、Torch或AITER依赖。
- AITER：容器 `/app/aiter-test`，本项目调用已有 attention、MoE、Norm、routing 与 RoPE 相关接口，未把第三方内核宣称为原创。实际调用记录和本项目适配/融合见 `ATOM/atom/model_config/mellum_native.py`、`mellum_fastpaths.py`、`mellum_fused_rope.py` 等。
- Hugging Face Transformers：正确性参考为5.10.1原生 `MellumForCausalLM`；推理环境仍为题给5.2.0。参考 MoE 的 grouped_mm、激活/归约精度、RoPE 和生成语义进行对齐；模型文件保留对应来源说明。上游：https://github.com/huggingface/transformers 。
- HumanEval： https://github.com/openai/human-eval ，评估使用 commit `b9453e8f1865df0a0057a1383767caa400acfbcf`，下载到独立工具目录，不作为启动服务依赖。未修改题目测试来提高分数。
- EvalScope：独立客户端1.15，使用题目ABC命令所对应的场景协议，兼容当前release参数/数据字段；不声称客户端版本或隐藏集与裁判完全一致。https://github.com/modelscope/evalscope 。
- 参考阅读涉及 vLLM/SGLang/新版ATOM 的范围、路径与修订信息，见 `report/research/05_optimization_implementation_sources.md`；不是将这些框架替换为提交运行时。
- 题目原件在 `docs/requirements/`。模型与数据由容器 `/data` 提供，不在本仓库重新分发模型权重。

本项目实现范围：Mellum在冻结ATOM的配置/权重映射与加载审计、参考数值兼容路径、适配到原生批处理/分页KV/graph、专用融合与元数据兼容、实验/验收工具及文档。调用的上游算法和内核按其来源归属，优化效果由实际测量给出。

精度优先路线复用HF数值语义、ATOM分页块管理与调度，以及AITER已有缓存写入接口；专家分组、参考缓存桥接与SWA请求分组为本项目适配。性能优先路线调用框架原生Attention、FusedMoE、归一化与路由，结合本项目的投影组织、QK融合、Graph及前缀元数据兼容。上游SDPA、MoE底层内核及块哈希算法按原来源归属。两条路线的实测正确性与性能分别见根目录技术报告，历史分段及取消机制保留为探索记录。
