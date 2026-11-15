"""No-GPU validation: Triton CPU interpreter and offline AMD target compilation."""
import argparse,importlib.util,json,os,re,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def load():
    spec=importlib.util.spec_from_file_location('mellum_indexed_check',ROOT/'ATOM/atom/model_config/mellum_indexed_gemv.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def interpret():
    os.environ['TRITON_INTERPRET']='1'
    import torch,triton
    module=load();records=[]
    for repeat in (1,2):
        for k,n in ((11,7),(896,23),(2304,13)):
            torch.manual_seed(41)
            ids=torch.tensor([[2,0],[1,2]])
            # Strided row input, masked K/N tails, multiple expert selection.
            storage=torch.randn((4//repeat,k+3));x=storage[:,:k]
            weight=torch.randn(3,n,k);out=torch.empty(4,n)
            module._indexed_expert_gemv[(triton.cdiv(n,4),4)](x,ids,weight,out,x.stride(0),weight.stride(0),weight.stride(1),
                            k,n,repeat,triton.next_power_of_2(k),4,num_warps=4,enable_fp_fusion=False)
            expected=torch.stack([weight[int(e)]@x[i//repeat] for i,e in enumerate(ids.flatten())])
            torch.testing.assert_close(out,expected,atol=1e-4,rtol=1e-4)
            records.append({'repeat':repeat,'k':k,'n':n,'max_abs':float((out-expected).abs().max())})
    print(json.dumps({'cpu_interpreter':records}));return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--interpret',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.interpret:interpret();return
    env=dict(os.environ);env['TRITON_INTERPRET']='1'
    result=json.loads(subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'--interpret'],env=env,text=True).splitlines()[-1])
    os.environ['TRITON_CACHE_DIR']='/tmp/mellum_moe_offline_compile'
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    module=load();records=[]
    for k,n,repeat in ((2304,1792,8),(896,2304,1),(11,7,2)):
        source=ASTSource(module._indexed_expert_gemv,signature={'X':'*bf16','IDS':'*i64','W':'*bf16','Y':'*bf16'},
                        constexprs={'SX':k,'WE':n*k,'WN':k,'K':k,'N':n,'REPEAT':repeat,
                                    'BLOCK_K':triton.next_power_of_2(k),'BLOCK_N':4})
        compiled=triton.compile(source,target=GPUTarget('hip','gfx942',64),options={'num_warps':4,'enable_fp_fusion':False})
        asm=compiled.asm['amdgcn']
        fields={name:re.findall(r'\.amdhsa_'+name+r'\s+(\d+)',asm) for name in ('next_free_vgpr','private_segment_fixed_size')}
        records.append({'k':k,'n':n,'repeat':repeat,'target':'gfx942','kernel_hash':compiled.metadata.hash,
                        'shared_bytes':compiled.metadata.shared,'assembly_resources':fields})
    result.update(offline_compilation=records,note='No GPU execution; CPU interpreter uses FP32 to validate addressing/reduction wiring, not AMD BF16 accuracy')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
