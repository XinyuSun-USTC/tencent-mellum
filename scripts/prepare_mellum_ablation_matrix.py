"""CPU-only component evidence matrix; not the mandated three-point final table."""
import argparse,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from mellum_execution_profiles import ABLATION_BASE,ABLATIONS,profile_settings
from mellum_perf_protocol import sha
from run_mellum_batch_campaign import write
from serve_mellum_step0_snapshot import describe


def prepare(out):
    snapshot=describe(Path('/workspace/tencent-mellum-step0-repro'),0,'127.0.0.1',8751,9726,out/'step0')
    anchor=profile_settings(ABLATION_BASE);rows=[]
    for name in [ABLATION_BASE,*ABLATIONS]:
        settings=profile_settings(name)
        rows.append({'profile':name,'changes_from_anchor':{k:v for k,v in settings.items() if anchor.get(k)!=v},
            'full_command':['/opt/venv/bin/python',str(Path(__file__).with_name('run_mellum_full_campaign.py')),
                '--gpu','0','--profiles',name,'--budget-minutes','90'],
            'status':'not_run','historical_step0':False})
    plan={'status':'prepared_only','anchor':ABLATION_BASE,'anchor_acceptance':'pending; change anchor if final candidate fails',
        'scope':'Internal component ablations, not claimed as the mandatory <=3 optimization point table',
        'rows':rows,'excluded_single_switches':{
            'resident_rope':'fused kernel requires inverse on GPU; disabling residency alone is invalid',
            'direct_kv':'fused QK and cached suffixes depend on this write bridge',
            'torch_experts':'reference expert route has dynamic CPU metadata, unsupported in this graph path',
            'reference_attention':'ragged reference route has dynamic shapes/host operations; requires eager'},
        'historical_step0':{'tag':'step0-local-validated','commit':snapshot['snapshot_commit'],
            'snapshot':'/workspace/tencent-mellum-step0-repro','max_num_seqs':1,'source_unchanged':True,
            'note':'No batching bridge in historical tag. R includes later work and must not be relabeled Step0.'},
        'source_sha256':{str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('mellum_execution_profiles.py'),Path(__file__).with_name('serve_mellum_step0_snapshot.py')]}}
    write(out/'matrix.json',plan);return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    out=a.output or Path('/workspace/results/mellum_ablation_preparation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=True);prepare(out);print('RESULT ablation_prepared output',out)
if __name__=='__main__':main()
