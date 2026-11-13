# Step 0 正确性验证资产审计

日期：2026-11-10。只读检查容器文件与 Python 安装元数据；未启动 GPU、服务、评测或加载权重，未安装/升级依赖。

## 结论

容器中确有题目相关数据与历史结果，但本轮未找到可直接执行的完整正式正确性验证套件。不能把“未找到”扩大为全机绝对不存在，也不能把通用工具或历史 rejected 结果当作正式验收依据。

## 已确认资产

| 路径 | 证据 | 可用性 |
|---|---|---|
| `/data/datasets/HumanEval.jsonl`、`.gz` | 164题，字段 task_id/prompt/entry_point/canonical_solution/test | 有题目、标准解和测试，可作为后续评测输入；没有候选输出及完整评测协议 |
| `/data/datasets/humaneval_bf16.json` | count=164，passed=3，pass_at_1=0.018292682926829267；每题仅 task_id/passed/error | 历史执行结果，无生成文本、token、logits；不是可用参考输出 |
| `/data/datasets/baseline.json` | humaneval_pass1_bf16、humaneval_pass1_fp8 均 null；humaneval_note 明确 Rejected: raw /v1/completions without the chat template or thinking mode | 明确否定旧 3/164 作为有效基准；提供 A/C 性能值，不提供 HF token 参考 |
| `/data/datasets/scenario_a_prompts.jsonl` | 512条，id/prompt/intended_words | 性能负载 |
| `/data/datasets/scenario_c_prompts.jsonl` | 200条，另有 shared_prefix_group | C 性能负载；无参考 token、不能仅因200条认定为正式正确性集合 |
| `/data/models/Mellum2.1-12B-A2.5B-Thinking/chat_template.jinja` | ChatML、reasoning_content、think 等处理 | 后续参考与服务端必须统一模板；本身不是验证器 |
| 同目录 README.md:279–285 | JetBrains 自报 thinking mode 的 HumanEval+ 91.5 | HumanEval+ 与当前 HumanEval 不是同一个验收协议，不能代替题目参考值 |

历史错误统计：humaneval_bf16.json 中147题错误含缺失 pytest，153题含 ModuleNotFoundError。这只能说明历史运行存在问题；当前安装元数据显示 pytest=9.0.3，不能把旧错误当成当前环境缺包诊断。

## 容器工具及仓库脚本

通过 importlib.metadata 读取版本，未导入模型/推理框架：Transformers 5.2.0、lm_eval 0.4.12、evaluate 0.4.6、pytest 9.0.3；evalscope、human-eval、evalplus 未找到对应安装分发元数据。`/opt/venv/bin` 也没有 evalscope 命令。此结论针对当前 Python 环境，不断言所有可能环境不存在这些工具。

- `/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/humaneval/humaneval.yaml`：通用 HumanEval 任务，dataset_path=openai/openai_humaneval，原始 prompt，max_gen_toks=1024，带 stop 序列和 create_test filter。不是 Mellum 专用 harness；直接照搬可能重现模板/生成/代码提取口径问题，且默认数据路径可能触发远端取数。
- `/app/ATOM/scripts/run_gsm8k_eval.sh`：lm_eval local-completions、GSM8K；不是本题 HumanEval/200 prompt 验证器。
- `/app/ATOM/tests/test_prefix_cache_accuracy.py`：共享数学前缀、已知算术答案与重复请求一致性；不是 HF 逐 token 对齐。
- `/app/ATOM/tests/entrypoints/`：服务/API/解析等通用测试；未发现本题 Mellum 的 HF reference 比对入口。
- `/app/ATOM/atom/benchmarks/benchmark_serving.py`：通用服务性能 benchmark，不能替代模型数值验收。
- 当前 Transformers models 文件目录中未发现 Mellum 路径；Mellum checkpoint 目录没有 modeling/configuration Python 文件。冻结环境不能直接假设 AutoModel 能运行题目所述 HF Mellum。

## 搜索范围与边界

检查 `/app`、`/data`、`/workspace` 的项目/脚本/数据文件，`/opt/venv` 的入口与相关包源码，`/root/.cache`、`/usr/local/bin`、`/usr/local/share`、`/usr/local/lib`、`/srv`、`/mnt`、`/media`、`/var/tmp` 等的相关文件名。关键词包括 Mellum、HumanEval、baseline、correctness、reference、evalscope、prompt，以及题目数据文件名、前64token匹配相关文本。

大型第三方源码、ROCm 文档、编辑器/代理内部文件不作完整逐文件内容扫描；未读取凭据，未查询代理会话记录来充当外部证据。`/home/ubuntu` 无读取权限。模型仅核查文件清单/文本，未读权重 payload。本轮没有穷举 Git 历史对象或缓存内所有无扩展二进制，因此结论限定为上述已检查范围。

题目 PDF（临时文本 `/tmp/csig_assessment_text.txt`）中，正确性条款描述200条prompt、HF前64token参考、HumanEval和16k子集；D0条款列有“评测脚本、参考输出”。原题确实预期这些资产存在，但没有提供本机路径。本轮未找到对应完整资产。

## 后续路径

用户已确认只有题目和容器，不继续要求用户提供未知附件。适配继续以最快正确复用为主，不再刻意建立出题人朴素 Baseline，不预定 Step1 优化目标。

1. 保存当前数据、模板和版本证据，Step0开发继续进行。
2. 建立可复现的本地验证：固定精确 input token IDs、模板、greedy/EOS/长度设置；覆盖短输入、SWA边界、逐步KV decode和16k。记录token级与整序列级指标，保留完整输出，避免“90%”统计歧义。
3. 需要独立于 ATOM 的可信模型参考。下一步可核查官方 Mellum 源码与当前冻结库的兼容性，在 workspace 内引入最小必要参考代码，不升级依赖；来源、版本、兼容改动明确记录。此方案尚未实施，也不能自动宣称等价于出题人的正式HF参考。
4. HumanEval使用现有164题，明确thinking模板、生成预算、答案代码提取和执行协议，在同一协议下比较参考与ATOM。不能将旧3/164或模型卡HumanEval+91.5直接当阈值分母。
5. 若最终仍缺正式资产，分别报告内部验证结果与正式验收缺项，不能把自建200条数据冒充题目公开集合或宣称正式门槛已通过。

## 文件指纹 SHA256

- baseline.json: `14674cbbc641709358e197bd5ad71c3cac8f41ed3932b3091266f5636b0dd6c8`
- humaneval_bf16.json: `399354335c6da2f1e57ca0f14133a509b60b8e99d7588b4bd97011dad2b9287b`
- HumanEval.jsonl: `1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2`
- scenario_a_prompts.jsonl: `f5cb156bb468757841a321142c8813d281c0732b494d70d3c59d5b415bf7489d`
- scenario_c_prompts.jsonl: `38fc340a76fd48c025bf5806376bdabbc7762f6b5c48d4509bbb109af2cfe265`
