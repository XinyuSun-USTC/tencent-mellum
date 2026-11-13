"""Compare HF CPU/GPU inverse-frequency initialization with AITER GPU caches."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import traceback
from diagnose_mellum_attention_math import import_native_rope
from diagnose_mellum_projection_math import digest

ROOT=Path('/workspace')

def coefficient_errors(actual, expected, positions):
    import torch
    from diagnose_mellum_expert_math import error
    result=error(actual,expected)
    different=(actual!=expected)
    result['different_elements']=int(different.sum().item())
    rows=different.reshape(len(positions),-1).any(-1)
    result['different_positions']=positions[rows].cpu().tolist()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    os.environ['HIP_VISIBLE_DEVICES']=str(a.gpu)
    out=ROOT/'results/mellum_rope_coefficients'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    report={'status':'started','script_sha256':digest(__file__),'records':[],
            'scope':'RoPE coefficients only, no layer, KV or full model correctness claim'}
    def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        import torch,transformers
        if transformers.__version__!='5.10.1':raise RuntimeError('Use dedicated HF 5.10.1')
        if not torch.cuda.is_available():report['status']='blocked_no_gpu';save();return 2
        from transformers import MellumConfig
        from transformers.models.mellum.modeling_mellum import MellumRotaryEmbedding
        get_rope=import_native_rope()
        spec=importlib.util.spec_from_file_location('bridge',ROOT/'ATOM/atom/model_config/mellum_utils.py')
        bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
        config_path=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking/config.json')
        cfg=MellumConfig.from_pretrained(config_path.parent,local_files_only=True)
        # Match reference model's original construction and explicit alternative.
        with torch.device('cpu'):cpu_init=MellumRotaryEmbedding(cfg)
        cpu_init=cpu_init.to('cuda')
        with torch.device('cuda'):gpu_init=MellumRotaryEmbedding(cfg)
        positions=torch.tensor(list(range(166))+[1023,1024,8191,8192,8193,16383],device='cuda')
        x=torch.zeros(1,len(positions),128,device='cuda',dtype=torch.bfloat16)
        report.update(model_config_sha256=digest(config_path),transformers=transformers.__version__,torch=torch.__version__)
        with torch.inference_mode():
            for kind in ['sliding_attention','full_attention']:
                params=bridge.aiter_rope_parameters(cfg.rope_parameters[kind])
                with torch.device('cuda'):
                    native=get_rope(128,rotary_dim=128,max_position=16384,base=params['rope_theta'],rope_scaling=params,dtype=torch.bfloat16)
                hc,hs=cpu_init(x,positions[None],layer_type=kind)
                gc,gs=gpu_init(x,positions[None],layer_type=kind)
                ac=native.cos_cache.index_select(0,positions).reshape(len(positions),64).repeat(1,2)[None]
                ass=native.sin_cache.index_select(0,positions).reshape(len(positions),64).repeat(1,2)[None]
                ci=getattr(cpu_init,kind+'_inv_freq');gi=getattr(gpu_init,kind+'_inv_freq')
                rows={'native_vs_reference_cos':coefficient_errors(ac[0],hc[0],positions),
                      'native_vs_reference_sin':coefficient_errors(ass[0],hs[0],positions),
                      'native_vs_gpu_init_cos':coefficient_errors(ac[0],gc[0],positions),
                      'native_vs_gpu_init_sin':coefficient_errors(ass[0],gs[0],positions),
                      'gpu_init_vs_reference_cos':coefficient_errors(gc[0],hc[0],positions),
                      'gpu_init_vs_reference_sin':coefficient_errors(gs[0],hs[0],positions)}
                row={'layer_type':kind,'inv_freq_exact':torch.equal(ci,gi),
                     'inv_freq_max_abs':(ci-gi).abs().max().item(),'comparisons':rows}
                report['records'].append(row);save();print(kind,json.dumps(row),flush=True)
                torch.save({'positions':positions.cpu(),'reference_inv_freq':ci.cpu(),'gpu_inv_freq':gi.cpu(),
                    'reference_cos':hc.cpu(),'reference_sin':hs.cpu(),'gpu_cos':gc.cpu(),'gpu_sin':gs.cpu(),
                    'native_cos':ac.cpu(),'native_sin':ass.cpu()},out/(kind+'.pt'))
        report['status']='diagnostic_complete';save();print('RESULT diagnostic_complete output',out,flush=True);return 0
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());save();print(report['traceback']);print('RESULT failed output',out,flush=True);return 1
if __name__=='__main__':raise SystemExit(main())
