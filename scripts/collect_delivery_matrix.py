"""Read-only verified merge of split-card final delivery matrix results."""
import argparse,json
from pathlib import Path
from mellum_perf_protocol import sha
from mellum_delivery_matrix import ROWS
from run_delivery_matrix import render,write

def collect(runs,out):
    metrics={};tasks={};rows=[];identity=None;acceptances={}
    for root in runs:
        summary=json.loads((root/'summary.json').read_text());plan=json.loads((root/'manifest.json').read_text())
        if plan.get('definitions')!=ROWS:raise ValueError('Historical row definitions differ; use explicit r32 provenance assembler')
        if not summary.get('source_unchanged'):raise ValueError('Unverified matrix source')
        for f,h in summary['artifact_sha256'].items():
            if sha(root/f)!=h:raise ValueError('Changed artifact: '+f)
        if identity is None:identity=plan['source_sha256']
        elif identity!=plan['source_sha256']:raise ValueError('Cannot merge different source configurations')
        perf=root/'performance/summary.json'
        if perf.exists():
            for profile,m in json.loads(perf.read_text()).get('metrics',{}).items():
                if profile in metrics:raise ValueError('Duplicate performance profile')
                metrics[profile]=m
        for row in plan['rows']:
            if row not in rows:rows.append(row)
            f=root/row/'summary.json'
            if f.exists():
                if row in acceptances:raise ValueError('Duplicate acceptance row')
                acceptances[row]=json.loads(f.read_text())
        tasks[str(root)]=summary['status']
    out.mkdir(parents=True,exist_ok=False);(out/'performance').mkdir()
    write(out/'performance/summary.json',{'metrics':metrics})
    for row,value in acceptances.items():(out/row).mkdir();write(out/row/'summary.json',value)
    render(out,{'rows':[r for r in ROWS if r in rows]})
    write(out/'evidence.json',{'runs':[str(r) for r in runs],'tasks':tasks,'source_sha256':identity,
                              'note':'Merged display only. Original per-row source and raw failure counts retained; no cross-run median.'})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.runs,a.output)
    print('RESULT matrix_collected output',a.output)
if __name__=='__main__':main()
