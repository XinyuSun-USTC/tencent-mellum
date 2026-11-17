"""Bounded A16 Step0 supplement; not A512 or resumption of truncated request records."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
from run_final_followup import prepare,execute,write
from mellum_perf_protocol import load_rows,sha


def main():
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=[0,1],default=0);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 out=Path('/workspace/results/mellum_step0_a_short')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
 plan=prepare(out,'step0-a',a.gpu);job=json.loads((out/'A/plan.json').read_text());spec=job['cases'][0]
 spec['expected']=spec['expected'][:16];spec['number']=16
 rows={r['id']:r for r in load_rows(spec['source'])};source=out/'inputs_a16.jsonl'
 source.write_text(''.join(json.dumps(rows[r['id']])+'\n' for r in spec['expected']))
 spec['source']=str(source);job['timeout']=900;write(out/'A/plan.json',job)
 plan['scope']='A16 supplemental Step0 measurement, client parallel32, one server slot, output256; NOT A512 score, NOT appended remainder of old run'
 plan['sha256'][str(Path(__file__).resolve())]=sha(__file__);plan['external_sha256'][str(source)]=sha(source)
 write(out/'manifest.json',plan)
 if a.prepare_only:print('RESULT short_A_prepared output',out);return
 # The generic runner permits four hours; this worker retains its own 900s limit.
 result=execute(out,plan);result['scope']=plan['scope'];write(out/'summary.json',result)
 print('RESULT short_A_supplement_measured output',out)
if __name__=='__main__':main()
