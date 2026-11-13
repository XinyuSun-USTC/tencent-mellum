"""Compare equivalent materialized residual states, not ATOM storage tuples."""
import torch

def compare_layers(hf_dir,atom_dir):
    rows=[]
    order=['embedding','input_norm_input','input_norm','q_proj','k_proj','v_proj','q_norm','k_norm','attention_context','attention_output','post_norm_input','post_norm','moe_input','router_logits','moe_output']
    for sample in sorted(hf_dir.glob('sample_*')):
        for file in sorted(sample.glob('*.pt'),key=lambda p: (0,order.index(p.stem)) if p.stem in order else (2,0) if p.stem=='final_norm' else (1,p.name)):
            other=atom_dir/sample.name/file.name
            if not other.exists():raise FileNotFoundError(other)
            a=torch.load(file,map_location='cpu',weights_only=True).float()
            b=torch.load(other,map_location='cpu',weights_only=True).float()
            row={'sample':sample.name,'stage':file.stem,'shape_hf':list(a.shape),'shape_atom':list(b.shape)}
            if a.shape!=b.shape:row['status']='shape_mismatch'
            elif not torch.isfinite(a).all() or not torch.isfinite(b).all():row['status']='nonfinite'
            else:
                diff=a-b
                row.update(status='exact' if torch.equal(a,b) else 'different',
                    max_abs=diff.abs().max().item(),rms=diff.square().mean().sqrt().item(),
                    relative_l2=(diff.norm()/a.norm().clamp_min(1e-12)).item(),
                    last_position_relative_l2=(diff[-1].norm()/a[-1].norm().clamp_min(1e-12)).item())
            rows.append(row)
    if not rows:raise ValueError('No HF layer files found')
    return {'scope':'Nonzero BF16 differences alone do not identify a bug; inspect error growth and suboperators','records':rows}


def compare_weights(hf_dir,atom_dir):
    rows=[]
    for file in sorted(hf_dir.glob('*.pt')):
        other=atom_dir/file.name
        if not other.exists():raise FileNotFoundError(other)
        a=torch.load(file,map_location='cpu',weights_only=True)
        b=torch.load(other,map_location='cpu',weights_only=True)
        same=a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
        rows.append({'name':file.stem,'exact':same,'shape_hf':list(a.shape),'shape_atom':list(b.shape)})
    if not rows:raise ValueError('No weights dumped')
    return rows
