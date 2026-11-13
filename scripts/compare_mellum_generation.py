"""Compare passive cached logits only while the two generated histories agree."""
import json
from pathlib import Path
from diagnose_mellum_projection_math import digest


def compare_generation(reference_path,candidate_path):
    import torch
    left=json.loads(reference_path.read_text());right=json.loads(candidate_path.read_text())
    candidates={r['name']:r for r in right['records']};rows=[]
    for ref in left['records']:
        actual=candidates[ref['name']];wanted=ref['generated_ids'][0];got=actual['generated_ids'][0]
        for step in range(min(len(wanted),len(got))):
            if wanted[:step]!=got[:step]:break
            values=[]
            for record,base in [(ref,reference_path.parent),(actual,candidate_path.parent)]:
                file=record['generation_logits'][step];path=base/Path(file['path'])
                if digest(path)!=file['sha256']:raise ValueError('Generation logits hash mismatch')
                values.append(torch.load(path,weights_only=True,map_location='cpu').float())
            h,a=values
            if h.shape!=a.shape or h.ndim!=1 or not torch.isfinite(h).all() or not torch.isfinite(a).all():
                raise ValueError('Invalid generation logits')
            rows.append({'name':ref['name'],'step':step,'logits_exact':torch.equal(h,a),
                'max_abs':(h-a).abs().max().item(),'hf_selected':wanted[step],'atom_selected':got[step],
                'both_selected_are_argmax':bool(h[wanted[step]]==h.max() and a[got[step]]==a.max())})
    nonexact=[r for r in rows if not r['logits_exact']]
    return {'scope':'Same-history pre-sampling logits; no comparison after the token fork',
            'compared_positions':len(rows),'nonexact_positions':len(nonexact),
            'first_nonexact':nonexact[:1],'records':rows}
