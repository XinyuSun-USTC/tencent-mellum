"""CPU-only evidence packaging from existing reference Torch-profiler results."""
import hashlib,json,shutil,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    os.environ.setdefault('MPLCONFIGDIR','/tmp/mellum-profiling-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=ROOT/'artifacts/performance/02_expanded_profile'
    out=ROOT/'artifacts/submission/reference_profiling';out.mkdir(parents=True,exist_ok=True)
    worker=json.loads((source/'trace/worker_results.json').read_text());records=[]
    for n in [1024,16384]:
        job=next(j for j in worker['jobs'] if j['name']==f'ctx_{n}')
        path=Path(job['trace'])
        if sha(path)!=job['trace_sha256']:raise ValueError('Trace differs from original receipt')
        target=out/f'ctx_{n}.pt.trace.json.gz';shutil.copyfile(path,target)
        analysis=source/f'trace/ctx_{n}_analysis.json';shutil.copyfile(analysis,out/analysis.name)
        data=json.loads(analysis.read_text());phase=data['phase_exclusive_module_kernel_us'];totals=data['phase_kernel_us']
        metrics={}
        for label in ['prefill','decode']:
            components=phase[label];metrics[label]={
                'total_kernel_ms':totals[label]/1000,
                'experts_ms':components.get('moe_experts',0)/1000,
                'sdpa_ms':components.get('sdpa',0)/1000,
                'rope_and_key_overwrite_ms':(components.get('reference_position_rope',0)+components.get('overwrite_reference_keys',0))/1000,
                'kv_gather_ms':components.get('gather_reference_kv',0)/1000}
        paired=job['paired_unprofiled']['token_ids']==job['profiled_request']['token_ids']
        records.append({'name':job['name'],'input_tokens':n,'output_tokens':job['output_tokens'],'trace':target.name,
            'original_trace':str(path),'trace_sha256':sha(target),'trace_bytes':target.stat().st_size,
            'analysis':analysis.name,'analysis_sha256':sha(analysis),'profile_wall_ratio':job['profile_wall_ratio'],
            'profiled_unprofiled_tokens_exact':paired,'metrics':metrics,'attribution_fraction':data['module_attribution_fraction']})
    labels=['Experts','SDPA','RoPE + K overwrite','KV gather','Other'];colors=['#41658a','#6e9f9b','#d49d49','#af7d91','#c9cbd0']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,phase in zip(axes,['prefill','decode']):
        left=[0.,0.]
        for i,(label,key) in enumerate(zip(labels,['experts_ms','sdpa_ms','rope_and_key_overwrite_ms','kv_gather_ms',None])):
            values=[]
            for r in records:
                m=r['metrics'][phase];value=m[key] if key else m['total_kernel_ms']-sum(m[k] for k in ['experts_ms','sdpa_ms','rope_and_key_overwrite_ms','kv_gather_ms'])
                values.append(value/m['total_kernel_ms']*100)
            ax.barh(['1K input','16K input'],values,left=left,color=colors[i],label=label)
            left=[a+b for a,b in zip(left,values)]
        ax.set_xlim(0,100);ax.set_xlabel('Share of phase cumulative GPU kernel duration (%)')
        ax.set_title('Prefill' if phase=='prefill' else 'Decode (16 captured forwards)');ax.invert_yaxis()
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=5,fontsize=8)
    fig.suptitle('Reference-path Torch profiler: phase attribution (diagnostic, not benchmark)',fontsize=10)
    fig.tight_layout(rect=[0,.1,1,.92]);fig.savefig(out/'phase_attribution.png',dpi=180);fig.savefig(out/'phase_attribution.svg');plt.close(fig)
    manifest={'source_run':'/workspace/results/mellum_profile_study/20261113T190525160560Z','source_commit':'63851f02fa53ee5e5ecd4122f091dac24cea31de',
        'scope':'Existing reference path bs1; one prefill and16 decode forwards per short trace. Profiler time is not throughput. No new GPU collection.',
        'settings':worker['settings'],'engine_options':worker['engine_options'],'records':records,
        'generator':str(Path(__file__).relative_to(ROOT)),'generator_sha256':sha(__file__)}
    manifest['artifact_sha256']={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('RESULT evidence_packaged output',out)
    for r in records:print(r['name'],r['metrics'])
if __name__=='__main__':main()
