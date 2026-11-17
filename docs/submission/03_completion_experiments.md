# 交付补充实验状态

Step0基线已统一为32槽r参考配置，直接复用已完成的A512/B50/C200及本地正确性证据，不再安排单槽A完整或A16补测。见[决议67](../plan/67_r32_baseline_revision.md)。

现有四行结果见[矩阵](../../artifacts/submission/final_r32_matrix/matrix.md)。①与②无需因基线调整重测；它们相对r的开关增量关系已核对。基线与后三行来自不同测量轮次，保留出处，不表述为同轮配对。

剩余实验为第三项精度隔离：①②对照、SWA/metadata、prefix、chunk、prefix+chunk及完整③，运行在原冻结工作树 `/workspace/tencent-mellum-final-followup`。本次交付修改不触碰该树，不中断其任务。按实际结果选择下一步，不重跑无关的完整HumanEval或ABC。

所有失败及旧运行目录保留。旧补测脚本在新交付树已标为退役，不应继续按历史单槽定义运行。
