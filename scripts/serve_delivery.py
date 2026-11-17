"""Final cumulative-row launcher; default equals ablation last row, qualification pending."""
import argparse,subprocess,sys,importlib.metadata,json
from pathlib import Path
from mellum_delivery_matrix import ROWS,DEFAULT_ROW

def command(row,gpu,port,engine_port,output,prepare=False):
    item=ROWS[row]
    result=[sys.executable,str(Path(__file__).with_name('serve_mellum.py')),'--gpu',str(gpu),
        '--host','0.0.0.0','--server-port',str(port),'--engine-port',str(engine_port),'--batching',
        '--profile',item['profile'],'--max-num-seqs',str(item['slots']),'--output',str(output)]
    if prepare:result.append('--prepare-only')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--row',choices=ROWS,default=DEFAULT_ROW)
    p.add_argument('--gpu',type=int,choices=[0,1],default=0);p.add_argument('--port',type=int,default=8000)
    p.add_argument('--engine-port',type=int,default=8026);p.add_argument('--output',type=Path,default=Path('results/delivery_server'))
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    expected=json.loads((Path(__file__).resolve().parents[1]/'configs/submission/environment.json').read_text())
    if {k:importlib.metadata.version(k) for k in expected}!=expected:raise RuntimeError('Frozen inference package versions differ')
    print('ROW',a.row,'Full final-row qualification pending; C60=43/60 is not a pass',flush=True)
    raise SystemExit(subprocess.call(command(a.row,a.gpu,a.port,a.engine_port,a.output,a.prepare_only)))
if __name__=='__main__':main()
