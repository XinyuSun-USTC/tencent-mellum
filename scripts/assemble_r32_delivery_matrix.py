"""Explicit provenance merge: historical r32 baseline plus measured optimized rows."""
import json,hashlib
from pathlib import Path
from mellum_runtime_config import PROJECT
from mellum_execution_profiles import profile_settings
from mellum_delivery_matrix import ROWS
from mellum_perf_protocol import sha

BASE=Path('/workspace/results/mellum_completion_benchmarks/20261116T165949897188Z_primary')
RUNS=[Path('/workspace/results/mellum_delivery_matrix')/n for n in ('20261117T003512204779Z','20261117T003519808987Z')]

def verified(root):
 s=json.loads((root/'summary.json').read_text())
 if not s.get('source_unchanged'):raise ValueError('Missing source check')
 for f,h in s['artifact_sha256'].items():
  if sha(root/f)!=h:raise ValueError('Changed original artifact '+str(root/f))
 return s

def main():
 verified(BASE)
 for root in RUNS:verified(root)
 launch=json.loads((BASE/'primary/A/server/launch.json').read_text())
 cmd=launch['command'];assert cmd[cmd.index('--max-num-seqs')+1]=='32'
 expected=profile_settings('r')
 for k,v in expected.items():
  if launch['settings'].get(k)!=v:raise ValueError('Historical r baseline differs on '+k)
 source_accept=Path(json.loads((BASE/'summary.json').read_text())['acceptance_run']);verified(source_accept)
 baseline={stage:next(iter(json.loads((BASE/'primary'/stage/'results.json').read_text())['cases'].values()))['metrics'] for stage in ('A','B','C')}
 rows={'step0':{'metrics':baseline,'acceptance':json.loads((source_accept/'summary.json').read_text()),'performance_run':str(BASE),'acceptance_run':str(source_accept),'measurement':'historical r32 reuse; not same-source rerun'}}
 for root in RUNS:
  perf=json.loads((root/'performance/summary.json').read_text())
  plan=json.loads((root/'manifest.json').read_text())
  for row in plan['rows']:
   if row=='step0':continue
   rows[row]={'metrics':perf['metrics'][ROWS[row]['profile']],'acceptance':json.loads((root/row/'summary.json').read_text()),'performance_run':str(root),'acceptance_run':str(root/row),'measurement':'final first-pass measurement'}
 out=PROJECT/'artifacts/submission/final_r32_matrix';out.mkdir(parents=True,exist_ok=True)
 payload={'baseline_policy':'r32-v2','rows':rows,'definitions':ROWS,'original_summary_sha256':{str(root/'summary.json'):sha(root/'summary.json') for root in [BASE,source_accept,*RUNS]},'note':'Baseline includes compatible batching and shared routing; point1 only expert group4. Cross-version measurements retain origin. All C percentiles are success-only.'}
 (out/'matrix.json').write_text(json.dumps(payload,indent=2)+'\n')
 lines=['# 最终累计消融：r32基线','','Step0复用历史完整r32测量；后三行取本轮测量。全行32槽，①仅计专家group4。C分位数仅覆盖成功请求。','',
 '| 配置 | A token/s/GPU | B TPOT p50 ms | C TPOT p99 / TTFT p99 ms（成功样本） | A/B/C完成数 | 开发200位置一致率 | HumanEval |',
 '|---|---:|---:|---|---|---:|---|']
 for row in ROWS:
  d=rows[row];m=d['metrics'];a=d['acceptance']
  counts='；'.join(f"{s}:{m[s]['successes']}/{m[s]['requests']}" for s in ('A','B','C'))
  lines.append(f"| {row} | {m['A']['output_tokens_per_s_gpu']:.4f} | {m['B']['tpot_p50_s']*1000:.3f} | {m['C']['tpot_p99_s']*1000:.3f} / {m['C']['ttft_p99_s']*1000:.3f} | {counts} | {a['tokens']['all']['position_match_fraction']*100:.4f}% | {a['humaneval']['passed_tasks']['candidate']}/164 |")
 lines+=['','HF HumanEval145/164。前三行本地正确性通过；step0123的token与长上下文检查未通过。历史单槽结果不改名，不填入本表。来源及哈希见matrix.json。']
 (out/'matrix.md').write_text('\n'.join(lines)+'\n');print(out)
if __name__=='__main__':main()
