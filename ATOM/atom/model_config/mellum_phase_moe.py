# SPDX-License-Identifier: MIT
"""Diagnostic-only phase split; a canonical reference copy doubles expert memory."""
import os
import torch
from atom.model_config.mellum_native import MellumFusedMoeBlock,record
from atom.model_config.mellum_experts import MellumTorchExperts


def dual_loader(native_loader,reference,target,loaded):
    def load(parameter,weight,*args,**kwargs):
        result=native_loader(parameter,weight,*args,**kwargs)
        name=kwargs.get('weight_name',args[0] if args else '')
        shard=kwargs.get('shard_id',args[1] if len(args)>1 else None)
        expert=kwargs.get('expert_id',args[2] if len(args)>2 else None)
        reference.weight_loader(target,weight,weight_name=name,shard_id=shard,expert_id=expert)
        loaded.add((shard,expert))
        return result
    return load


class MellumPhaseMoeBlock(MellumFusedMoeBlock):
    def __init__(self,config,quant_config=None,prefix=''):
        super().__init__(config,quant_config,prefix)
        self.phase=os.getenv('MELLUM_NATIVE_EXPERT_PHASE')
        if self.phase not in ('prefill','decode') or os.getenv('MELLUM_DECODE_GRAPH')=='1':
            raise ValueError('Expert phase split is eager diagnostic only')
        if self.experts.w13_weight.dtype!=torch.bfloat16:raise ValueError('Phase diagnostic requires BF16')
        ref=MellumTorchExperts(config).to(device=self.experts.w13_weight.device,dtype=torch.bfloat16)
        # Keep primary ATOM parameter names/audit unchanged. Reference tensors
        # are an explicit diagnostic copy, not another registered checkpoint target.
        object.__setattr__(self,'_reference_experts',ref)
        self._reference_loaded=set();self._reference_expected=3*config.num_experts
        for name in ('w13_weight','w2_weight'):
            p=getattr(self.experts,name)
            p.weight_loader=dual_loader(p.weight_loader,ref,getattr(ref,name),self._reference_loaded)

    def forward(self,hidden_states):
        if len(self._reference_loaded)!=self._reference_expected:raise ValueError('Incomplete canonical reference expert copy')
        from atom.utils.forward_context import get_forward_context
        ctx=get_forward_context();prefill=bool(ctx.context.is_prefill)
        native=prefill if self.phase=='prefill' else not prefill
        logits=self.gate(hidden_states)
        result=self.experts(hidden_states=hidden_states,router_logits=logits) if native else self._reference_experts(hidden_states,logits,batched=not prefill)
        record('moe_phase',self.layer_num,ctx,len(hidden_states),'FusedMoE' if native else 'MellumTorchExperts',
               reference_loaded_parts=len(self._reference_loaded),phase_split=self.phase)
        return result
