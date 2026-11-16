"""Build the four-page technical report and one-page AI disclosure without installs."""
import os,textwrap
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mellum-submission-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report/submission'
PAGES=[('1. Delivery, architecture and profiling', '''
Mellum 2.1 / ATOM | XinyuSun-USTC | 2026-11-16 | BF16, one GPU, MTP off

DELIVERY STATUS
The adapter, full weight mapping and OpenAI server run in the supplied frozen ROCm environment. Conservative and aggressive runtime policies are separate presets, not an extra optimization point. The aggressive configuration DOES NOT pass the local 90% token agreement gate. Its performance is experimental, not an accepted assessment score. Conservative results and aggressive speed must not be combined.

IMPLEMENTATION
ATOM/atom/models/mellum.py implements MellumAttention, MellumDecoderLayer and MellumForCausalLM. validate_loaded_weights audits all 5,631 source tensors. Config registration and checkpoint mappings use the existing ATOM interfaces. Reference-compatible Torch experts, Norm, SDPA and KV/RoPE paths coexist with native AITER-backed paths. Inference retains the supplied Transformers 5.2.0; HF 5.10.1 runs only in an isolated reference environment.

ENTRYPOINT
./start_server.sh --mode conservative --gpu 0
./start_server.sh --mode aggressive --gpu 0 --audit
Default HTTP port 8000, engine port 8026; model is the supplied local BF16 checkpoint. No reference venv or other worktree is needed for serving. scripts/serve_submission.py records effective configuration, source hashes and package versions. Conservative defaults to one scheduler slot; aggressive to 32, both context limit 16640. Runtime mode is independent of component disable switches.

PROFILE EVIDENCE
Torch profiler plus labelled operators, fixed-input replay, microbenchmarks, AB/ABBA screens and full ABC were used. A bs1 trace contains 18,627 kernels, 98.3% attributed: 170.569ms active kernel time over 641.084ms span; 470.515ms gaps are NOT all proven CPU overhead. MoE 58.372ms, SWA 41.228ms, QKV 20.007ms, two Norm sites 25.183ms total.
For bs32: 1573.961ms active / 2027.532ms span; MoE 714.275ms, QKV 331.458ms, SWA 258.752ms. These traces are largely prefill work, not a pure decode breakdown. They motivated native batched computation, launch reduction and data/cache reuse.

PROVENANCE
Frozen ATOM/AITER provide existing model patterns and GPU kernels; HF native Mellum provides reference numerical semantics. vLLM/SGLang/newer ATOM were reading references, not replacement runtimes. Upstream kernels are not claimed as original work. SOURCES.md and report/research/05_optimization_implementation_sources.md retain attribution. Full Git history and Step0 tag are preserved.
'''),('2. Optimization experiments and decisions', '''
The following three bottleneck directions summarize the work. They do not certify three independently accepted optimization points or replace the required Step0 cumulative ablation. Multiple mechanisms were explored; all gains below retain their own experiment scope.

1 / NATIVE BATCHED COMPUTE AND DATA PATH
AITER-backed MoE/attention, batched projection, packed QKV and direct paged-KV writes reduce per-request execution and intermediate layout operations. A64 native batching reached 439.67 token/s versus about 121.5 for the preceding native-L path (3.62x). Direct KV then reached 499.64 (+13.64%). Numerical differences in experts and attention remain an accuracy blocker.
Files: ATOM/atom/model_config/mellum_native.py, mellum_projection.py, mellum_direct_kv.py, mellum_fastpaths.py.
Disable: --disable-native-moe, --disable-native-attention, --disable-direct-kv, --separate-qkv, --reference-norm.

2 / DECODE GRAPH, QK FUSION AND ROUTING
Small kernels and launch gaps motivated decode graph replay and Q/K Norm+RoPE fusion. A64 graph screen: 620.995 -> 943.26 (+51.9%). Same-card QK fusion screen: 941.63 -> 1062.46 (+12.83%). Native-routing ABBA means: 939.49 -> 1008.94 (+7.39%). These are separate controlled comparisons; their percentages cannot be multiplied as one ablation chain.
Files: ATOM/atom/model_engine/model_runner.py, model_config/mellum_fused_rope.py, mellum_fastpaths.py, mellum_native.py.
Disable: --disable-graph, --disable-qk-fusion, --disable-native-routing, --disable-resident-rope.

3 / PREFIX COMPATIBILITY AND METADATA REUSE
Reuse paged KV prefixes and prefill metadata rather than generated outputs. Full ABC without prefix/metadata versus final combination: C TTFTp99 1.4883 -> 1.0260s (-31.06%); A 1129.38 -> 1139.78 (+0.92%). Both mechanisms changed together in this comparison, so individual attribution is unavailable. Actual execution includes 199 cached-suffix events and 757,792 cached tokens.
Files: ATOM/atom/model_config/mellum_prefix.py, mellum_prefill_metadata.py.
Disable: --disable-prefix, --disable-metadata-reuse.

REJECTED OR INCOMPLETE ROUTES
Chunk budgets 2048/4096 regressed throughput/TTFT. Indexed experts and prefill-only alternatives did not produce a satisfactory selected result. CK/Triton SWA accelerated kernels but changed generation; reference Norm and phase hybrids improved some token comparisons without reaching the final gate. Duplicate expert weights are diagnostic-only and disabled in delivery presets. No answer caching or prompt-specific output hardcoding was introduced.

EXPECTED VERSUS OBSERVED
The profile predicted compute-path, launch and reuse opportunities; screens confirmed useful gains. However, assumed numerical interchangeability was not established by hardware-specific implementation alone. Functional HumanEval improvements do not waive the independent token agreement requirement.
'''),('3. Full ABC and ablation accounting', '''
FINAL MEASURED EXPERIMENTAL PROFILE
sprint_graph_qk_router_prefix_metadata, BF16, one GPU, 32 scheduler slots.
Run: mellum_full_campaign/20261116T134727613186Z.
A512, B50 and C200 all completed without request failures. Source hashes, loading audit and actual native/graph/prefix events are retained, not just configuration flags.

METRIC                    PROVIDED BASELINE       LOCAL AGGRESSIVE
A output token/s/GPU      638.282724              1139.7781 (+78.57%)
B TPOT p50                not provided            4.865ms
C TPOT p99                41.265ms                11.502ms (-72.13%)
C TTFT p99                993.549ms               1025.998ms (+3.27%)

C throughput 76.8348 token/s is arrival-limited, not a saturation-speed claim. B's baseline is unavailable, so no improvement is invented. Another GPU may have run validation concurrently. These are NOT the judge's three idle-machine hidden-set runs or their median.

CLIENT / PROTOCOL DISCLOSURE
An isolated EvalScope 1.15 client follows the assessment A/B/C workload with explicit release compatibility changes: prompt-field adapter, native A filtering/cycling to 512, stream flag syntax, deterministic seed42 and output paths. Open-loop exponentials are normalized to total n/r, unlike unconditioned Poisson. Literal commands and actual commands are both retained. Exact judge client version/public-set equivalence is not asserted.

REQUIRED CUMULATIVE ABLATION: UNFINISHED
Historical Step0 (one card): full ABC not measured for the exact tagged snapshot.
Step0 + point1: comparable full A/B/C row unavailable.
Step0 + point1 + point2: comparable full A/B/C row unavailable.
Step0 + point1 + point2 + point3: the final experimental profile has the values above, but it is not a certified three-point cumulative row and fails token agreement.
The conservative reference-control preset is NOT substituted for historical Step0. Microbenchmarks and different-size screens do not fill the missing formal rows. Submission therefore retains the gap rather than fabricating measurements.

DELIVERY REVISION RECHECK
scripts/smoke_submission.py launched the actual delivery entrypoint in both modes, loaded all 5,631 tensors, and passed non-stream HTTP plus streaming/usage checks (20261116T152441526820Z). Scope: context512, slot1, eight output tokens; not full production acceptance. A final-revision full ABC and reduced cumulative screen may be appended as a separate evidence update after completion; until then they are not reported as measured.

REPRODUCE
/opt/venv/bin/python scripts/run_mellum_full_campaign.py --gpu 0 --profiles sprint_graph_qk_router_prefix_metadata --budget-minutes 25
Evidence: artifacts/submission/index.json; compact JSON and original paths/hashes are included. Large tensors, model weights and virtual environments are external. Full stdout/logs remain in the supplied container.
'''),('4. Correctness, limitations and next work', '''
CORRECTNESS METHOD
Use identical rendered/tokenized inputs and HF 5.10.1 reference greedy outputs. Inspect first64 tokens, same-prefix logits, teacher-forced decode, layer hooks, and identical-QKV / identical-expert-input replay. Full loading audit prevents silent missing weights. Reference and inference environments remain separate.

LOCAL RESULTS                     CONSERVATIVE r/1 SLOT     AGGRESSIVE FINAL
Developer200 positions             12800/12800 = 100%       8045/12800 = 62.85%
Exactly matched sequences          200/200                 65/200
Long-boundary positions            256/256                 138/256
16k sample, separately reported    64/64                   63/64
HumanEval for exact preset         not separately measured 150/164; HF145/164
Full ABC for exact preset          not measured            page3

Long inputs: 8191,8192,8193,16384 tokens; aggressive matches 55,9,11,63 of64. An overall90% long-group gate is an internal extra check, not a separately invented official rule. Official200 identity remains to be confirmed. Conservative here is a later reference control; historical Step0 tag remains separately preserved.

HUMANEVAL
Use official openai/human-eval commit b9453e8f1865df0a0057a1383767caa400acfbcf, thinking off, max2048. Candidate150/164=91.46%, HF145/164=88.41%: seven gains, two losses, net+5 tasks / +3.05pp. Under a literal absolute-gap BF16 threshold of2pp, this is not claimed passed even though the candidate scores higher. Do not silently change the rule to allow arbitrary positive differences. Functional quality does not replace the failed90% token gate.
MTP is off; inference is single-card. Two GPUs ran independent experiments, not tensor-parallel inference, so no MTP or dual-card correctness support is claimed.

DIAGNOSIS / REMAINING RISK
Expert replay located grouped_mm, activation and FP32 reduction differences; a repaired reference-compatible path replayed exactly. Projection packing, RoPE coefficient construction and HF generation semantics were also diagnosed. Final20-case core matrix: native/native69.84%, native/Torch71.17%, reference/native77.03%, reference/reference100%. Best phase hybrid86.875% still offers no accepted high-speed final configuration. A small diagnostic subset is not a universal failure proof.

SUBMISSION BOUNDARY
Deliver runnable source with a default conservative policy, explicit experimental aggressive policy, independent component switches, source/version receipts, preserved Git history, daily WORKLOG and this report plus AI disclosure. The GPU smoke is passed; the overall assessment is NOT declared passed. Missing formal ablation, conservative-preset HumanEval/full ABC, public-set confirmation and qualified optimized correctness are disclosed.

NEXT PRIORITY
Restore the90% gate for the same configuration whose speed is measured; then complete the required Step0 cumulative full ABC and judge protocol. Do not merge conservative accuracy and aggressive performance. Preserve reproduction evidence and upstream attribution throughout.
''')]
AI='''AI assistance disclosure | XinyuSun-USTC | 2026-11-16

TOOLS AND ROLES
OpenAI Codex assisted with source searches, terminal/Python/Git workflows, architecture analysis, implementation and report preparation. Torch profiler and custom scripts measured execution. The user ran GPU commands in the supplied container and supplied results; the restricted agent environment did not run GPU inference.

ASSISTED WORK
ATOM/Mellum configuration and checkpoint mapping; numerical reference compatibility; loading audit and same-prefix/layer/expert/attention diagnostics; batched campaigns and HTTP/SSE verification; native backend, graph, fusion, prefix and metadata experiments; result aggregation, documentation and preserved Git history.

HUMAN DECISIONS
The user set Step0/Step1 priorities, kept the inference environment frozen while allowing a separate HF reference environment, executed GPU jobs, challenged overly narrow token diagnostics and prompted HumanEval evaluation. Later review restored emphasis on the independent assessment token gate. The user selected an explicit conservative/aggressive delivery policy under the deadline.

ERRORS DISCOVERED AND CORRECTED
Actual tracebacks exposed a non-JSON-serializable set, AITER import-path issues and an incompatible MellumRotaryEmbedding constructor argument. A new schema field once broke comparison and was handled explicitly. Short greedy agreement did not establish long-generation correctness; expanded developer200 and length-boundary tests exposed divergence. Good HumanEval performance was insufficient to waive the separate90% token requirement. Backend event audits checked that performance switches actually executed.

VALIDATION AND LIMITS
AI suggestions were not accepted as measurements. Reports cite saved results, counts, commands and hashes. CPU tests do not substitute for GPU execution. The actual delivery launcher passed both conservative/aggressive load and HTTP/SSE smoke checks on 2026-11-16. Aggressive developer200 remains62.85% and fails the stated gate. Exact conservative-preset HumanEval/full ABC and formal three-point cumulative ablation remain incomplete. Missing evidence is not filled by inferred values.

ATTRIBUTION AND INTEGRITY
Existing ATOM/AITER kernels and HF numerical semantics are credited in SOURCES.md; upstream licenses and Git history are retained. Official HumanEval and isolated EvalScope tools support evaluation. No generated-answer caching or public-prompt answer hardcoding was used. Runtime mode is not counted as Step0 or an optimization point. Conservative accuracy and aggressive speed are never combined into one passing configuration.
'''

def render(path,pages):
    with PdfPages(path,metadata={'Title':path.stem,'Author':'XinyuSun-USTC','Subject':'Mellum ATOM submission; limitations explicitly disclosed'}) as pdf:
        for i,(title,body) in enumerate(pages,1):
            lines=[]
            for paragraph in body.strip().split('\n'):
                lines.extend(textwrap.wrap(paragraph,84,replace_whitespace=False,drop_whitespace=True) or [''])
            if len(lines)>60:raise ValueError(f'Text overflow {title}: {len(lines)} lines')
            fig=plt.figure(figsize=(8.27,11.69));fig.text(.075,.95,title,fontsize=15,fontweight='bold',va='top')
            fig.text(.075,.91,'\n'.join(lines),family='DejaVu Sans Mono',fontsize=10,linespacing=1.22,va='top')
            fig.text(.075,.035,f'Mellum / ATOM | release/submission | {i}/{len(pages)}',fontsize=8,color='#555555')
            pdf.savefig(fig);plt.close(fig)
OUT.mkdir(parents=True,exist_ok=True)
render(OUT/'01_technical_report.pdf',PAGES)
render(OUT/'02_ai_usage.pdf',[('AI usage disclosure',AI)])
print('Built technical report:4 pages; AI disclosure:1 page')
