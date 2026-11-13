# 19 Prefill内核同输入证据与SDPA受控验证

用户结果`results/mellum_attention_math/20261111T203010271900Z`确认：HF旋转+HF SDPA四例精确复现参考；固定HF旋转Q/K后改用AITER prefill，context相对HF误差约0.000592–0.000759。使用实际融合旋转+AITER prefill，相对旧ATOM输出一例精确、另三例约1.05e-5/1.55e-4/1.52e-4。旧ATOM还包含少量上游projection差异，因此不要求四例全部复现。

同输入跨内核差异得到直接证据，但不能把BF16实现差异自动称为内核bug，也不能断言已解释多层/全部生成分叉。下一步通过单变量替换检查端到端影响。

## 受控SDPA prefill

新增`ATOM/atom/model_config/mellum_attention.py:sdpa_single_prefill/install_sdpa_prefill`。仅为Mellum每个Attention实例绑定prefill_attention方法，复用实际融合旋转后的Q/K/V，用冻结PyTorch SDPA causal+GQA计算；没有修改共享PagedAttentionImpl。调度、KV写入和decode路径均沿用原实现。

`ATOM/atom/models/mellum.py:MellumAttention.__init__`通过显式`MELLUM_PREFILL_BACKEND=sdpa_diagnostic`启用，默认aiter保持原样。诊断入口检查单请求cu_seqlens=[0,n]、Q/K/V长度一致且滑窗输入不超过窗口；拒绝cached/chunked、多请求或跨窗prefill。它不是完整的SDPA适配后端，也不能拿去跑expanded长输入。

`scripts/check_atom_mellum.py`及`run_mellum_validation.py`增加`--prefill-backend`并记录实际设置。正常短生成使用同样输入和HF比较协议，不替换隐藏状态。此次结果只用于评价prefill替换影响，不证明decode、mask边界或服务验收通过。

模型/配置/专家/Attention CPU32项通过；运行器3项通过；ATOM diff检查通过。新增测试用显式重复KV头的SDPA对照GQA结果，并验证cached/跨窗拒绝。GPU尚未执行。

普通终端：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite short --prefill-backend sdpa_diagnostic --dump-layers --dump-stages
```

比较正常short64与原AITER prefill结果`20261111T172731825963Z`的prefill logits、逐层误差及128token。即使改善，仍需隔离全注意力层旋转及decode剩余差异，不能提前标记Step0完成。

## SDPA短回归结果与下一观察层

真实套件`results/mellum_validation/20261111T204036708995Z`，ATOM`results/atom_mellum_gpu/20261111T204057372892Z`。arithmetic64token匹配，python_add第38新token分叉（公共前缀37）。prefill logits最大差为1.5/1.75，未闭合。

算术题首层Attention context、MoE output、layer_00均逐元素精确一致，但后续仍有误差。算术题layer_08相对L2为0.00105645、layer_09为0.00937592，final norm为0.0514511；代码题final norm为0.0757580。初始首层精确不能排除深层路由对小误差敏感，也不能证明后续Attention/decode正确。替换prefill并未使正常生成全面改善，暂不提升为默认后端。

新增`--stage-layer`（0–27，默认0）同时传给HF和ATOM观察器，实际选择写入运行元数据；模型仅切换stage hooks，不修改该层数学路径。既有MoE固定输入重放只接受layer0，以免误用第0层参考。下一步在相同SDPA实验条件下观察第9层的输入/norm/QKV/Attention/router/MoE，检查已观察到的放大点：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite short --prefill-backend sdpa_diagnostic --dump-layers --dump-stages --stage-layer 9
```

该层为sliding_attention（0起始索引9）。只看自然输入下输出变化不能断定该层自身有错；先比较其输入、选定权重和路由，再考虑固定同输入。运行器CPU3项、脚本语法检查通过，GPU第9层观察尚未执行。

## 第9层证据与分步旋转实验

真实套件`results/mellum_validation/20261111T204551273938Z`，ATOM`results/atom_mellum_gpu/20261111T204612856623Z`。结果与前次SDPA实验一致（算术64token匹配、代码公共前缀37）。第9层9项选定权重精确一致，但进入input norm的相对L2已为代码0.00188746、算术0.00105645；norm后分别0.0241600、0.0119471。因此不能把norm放大当成相同输入下的norm错误。

CPU读取保存router logits，按同一FP32 softmax/topk公式复核：代码第18号token集合最后专家由1变为46，HF边界logits为-3.375/-3.390625，ATOM边界并列-3.390625；算术第12号token最后专家由23变为42，HF边界并列-3.59375，ATOM边界-3.59375/-3.609375。每例仅这一行集合改变；路由差异发生在两端不同输入下，不能据此判定路由函数错误。算术MoE输出相对L2达0.0289170，还包含其他行变化，不能全部归因于该一行。

准备显式`sdpa_rope_diagnostic`：在MellumAttention Q/K norm后保存旋转前Q/K，SDPA prefill观察入口使用`mellum_attention.py:stepwise_rope`按原缓存选取cos/sin，转Q/K dtype，分别执行两次BF16乘法再加法，替代该prefill计算中融合旋转后的Q/K。此实验对齐HF分步算术，仍复用现有AITER缓存，尚不保证缓存与HF全部层/位置逐元素一致。

实际KV写入和decode仍用原融合/HIP路径，故这是有意受限的prefill变量实验，不是生产修复，不能根据生成通过宣称cache一致。原始Q/K克隆保存在实例中且每次forward覆盖，避免原位旋转破坏观察输入。默认aiter及已有sdpa_diagnostic不改变。

模型相关CPU33项、运行器3项通过，新增测试验证非零position及BF16分步旋转；语法/diff通过。GPU未执行。下一命令与上一实验保持同输入、同第9层观察，只改变prefill旋转算术：

```bash
/opt/venv/bin/python /workspace/scripts/run_mellum_validation.py --suite short --prefill-backend sdpa_rope_diagnostic --dump-layers --dump-stages --stage-layer 9
```

优先比较初始prefill logits及逐层误差；decode未改，不能用后续分叉单独否定prefill改动。
