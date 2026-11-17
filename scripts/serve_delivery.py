"""Start the BF16 single-GPU Mellum server with cumulative or independent optimization switches."""
import argparse,sys,importlib.metadata,json,os
from datetime import datetime,timezone
from pathlib import Path
from mellum_delivery_matrix import ROWS,DEFAULT_ROW

def command(row,gpu,port,engine_port,output,prepare=False,disabled=()):
    item=ROWS[row]
    result=[sys.executable,str(Path(__file__).with_name('serve_mellum.py')),'--gpu',str(gpu),
        '--host','0.0.0.0','--server-port',str(port),'--engine-port',str(engine_port),'--batching',
        '--profile',item['profile'],'--max-num-seqs',str(item['slots']),'--output',str(output)]
    if disabled:result+=['--optimization-points',*[str(p) for p in item['points'] if p not in disabled]]
    if prepare:result.append('--prepare-only')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--row',choices=ROWS,default=DEFAULT_ROW)
    p.add_argument('--gpu',type=int,choices=[0,1],default=0);p.add_argument('--port',type=int,default=8000)
    p.add_argument('--engine-port',type=int,default=8026);p.add_argument('--output',type=Path)
    p.add_argument('--disable-opt',type=int,choices=[1,2,3],action='append',default=[],help='Disable a selected optimization point; repeat to disable several')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if not 1<=a.port<=65535 or not 1<=a.engine_port<=65535 or a.port==a.engine_port:p.error('Distinct valid server and engine ports required')
    if set(a.disable_opt)-set(ROWS[a.row]['points']):p.error('Cannot disable a point absent from the selected row')
    a.output=a.output or Path(__file__).resolve().parents[1]/'results/server'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    expected=json.loads((Path(__file__).resolve().parents[1]/'configs/submission/environment.json').read_text())
    if {k:importlib.metadata.version(k) for k in expected}!=expected:raise RuntimeError('Frozen inference package versions differ')
    print('ENABLED_OPTIMIZATIONS',[p for p in ROWS[a.row]['points'] if p not in a.disable_opt],flush=True)
    print('ROW',a.row,'LOG_DIR',str(a.output),flush=True)
    cmd=command(a.row,a.gpu,a.port,a.engine_port,a.output,a.prepare_only,a.disable_opt)
    os.execv(cmd[0],cmd)
if __name__=='__main__':main()
