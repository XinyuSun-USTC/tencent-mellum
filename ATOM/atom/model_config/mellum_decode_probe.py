# SPDX-License-Identifier: MIT
"""Explicit teacher-forcing observer. Forced sampler logits are never scored."""
import json
from pathlib import Path
import torch

class FreeGenerationProbe:
    """Passive observer: no plan, token override or stopping intervention."""
    def __init__(self,output):
        self.output=Path(output);self.sample=-1;self.step=-1;self.pending=False
    def observe(self,ids,positions,context):
        if context.is_dummy_run:return
        if context.batch_size!=1 or self.pending:raise ValueError('Invalid passive generation observation')
        if context.is_prefill:self.sample+=1;self.step=0
        else:self.step+=1
        target=self.output/f'sample_{self.sample:06d}';target.mkdir(parents=True,exist_ok=True)
        (target/f'step_{self.step:03d}_inputs.json').write_text(json.dumps({
            'input_ids':ids.flatten().detach().cpu().tolist(),'positions':positions.flatten().detach().cpu().tolist(),
            'phase':'prefill' if context.is_prefill else 'decode'})+'\n')
        self.pending=True
    def save(self,logits,context):
        if context.is_dummy_run:return
        if not self.pending or logits.ndim!=2 or logits.shape[0]!=1:raise ValueError('Passive logits observation mismatch')
        torch.save(logits[0].detach().cpu(),self.output/f'sample_{self.sample:06d}'/f'step_{self.step:03d}_logits.pt')
        self.pending=False

class ForcedDecodeProbe:
    def __init__(self, plan, output):
        self.records=json.loads(Path(plan).read_text())['records']
        self.output=Path(output);self.sample=-1;self.step=-1;self.pending=False

    def install_layer_hooks(self,backbone):
        handles=[]
        def hook(name,materialize=False):
            def save(module,args,output):
                if not self.pending or self.sample<0 or self.step not in (0,1):return
                if isinstance(output,tuple):
                    value=output[0]
                    if materialize and len(output)>1 and output[1] is not None:value=value+output[1]
                else:value=output
                target=self.output/f'sample_{self.sample:06d}'
                target.mkdir(parents=True,exist_ok=True)
                torch.save(value.detach().reshape(-1,value.shape[-1]).cpu(),target/f'step_{self.step:03d}_{name}.pt')
            return save
        handles.append(backbone.embed_tokens.register_forward_hook(hook('embedding')))
        handles.append(backbone.norm.register_forward_hook(hook('final_norm')))
        for i,layer in enumerate(backbone.layers):handles.append(layer.register_forward_hook(hook(f'layer_{i:02d}',True)))
        return handles

    def observe(self, ids, positions, context):
        if context.is_dummy_run:return
        if context.batch_size!=1:raise ValueError('Forced decode requires one request')
        if self.pending:raise ValueError('Forward had no matching logits observation')
        if context.is_prefill:
            if self.sample>=0 and self.step+1<len(self.records[self.sample]['generated_ids'][0]):
                raise ValueError('Previous forced request did not complete')
            self.sample+=1;self.step=0
        else:self.step+=1
        record=self.records[self.sample];forced=record['generated_ids'][0];prompt=record['input_ids'][0]
        # Async scheduling can launch one final forward before the last
        # sampled token's max_tokens completion reaches the scheduler.
        if self.step>len(forced):raise ValueError('More than one post-budget forward')
        expected=prompt if self.step==0 else [forced[self.step-1]]
        expected_positions=list(range(len(prompt))) if self.step==0 else [len(prompt)+self.step-1]
        actual_ids=ids.flatten().detach().cpu().tolist();actual_pos=positions.flatten().detach().cpu().tolist()
        if actual_ids!=expected or actual_pos!=expected_positions:
            raise ValueError('Actual forward token/position differs from forced history')
        path=self.output/f'sample_{self.sample:06d}';path.mkdir(parents=True,exist_ok=True)
        label='terminal' if self.step==len(forced) else f'step_{self.step:03d}'
        (path/f'{label}_inputs.json').write_text(json.dumps({
            'name':record['name'],'step':self.step,'input_ids':actual_ids,'positions':actual_pos,
            'logical_context_tokens':len(prompt)+self.step,'phase':'prefill' if self.step==0 else 'decode',
            'unscored_terminal':self.step==len(forced)})+'\n')
        self.pending=True

    def force(self, logits, context):
        if context.is_dummy_run:return logits
        if not self.pending or logits.ndim!=2 or logits.shape[0]!=1:
            raise ValueError('Invalid forced decode logits observation')
        terminal=self.step==len(self.records[self.sample]['generated_ids'][0])
        label='terminal' if terminal else f'step_{self.step:03d}'
        path=self.output/f'sample_{self.sample:06d}'/f'{label}_logits.pt'
        torch.save(logits[0].detach().cpu(),path)
        if terminal:
            self.pending=False
            return logits
        token=self.records[self.sample]['generated_ids'][0][self.step]
        if not 0<=token<logits.shape[-1]:raise ValueError('Forced token outside vocabulary')
        forced=torch.full_like(logits,float('-inf'));forced[0,token]=0
        self.pending=False
        return forced
