# SPDX-License-Identifier: MIT
"""Opt-in observer for first-layer prefill; never changes returned tensors."""
import types
from pathlib import Path
import torch


def install_swa_capture(impl, root):
    if impl.layer_num != 0:
        return
    original=impl.prefill_attention
    saved=set()
    def observe(self,q,k,v,*args):
        output=original(q,k,v,*args)
        n=q.shape[0]
        if n in (1024,4096) and n not in saved:
            from atom.model_config.mellum_attention import reference_position_rope
            raw_q,raw_k,positions=self._mellum_raw_qk
            inv,factor=self._mellum_cpu_rope
            rq,rk=reference_position_rope(raw_q,raw_k,positions,inv,factor)
            path=Path(root);path.mkdir(parents=True,exist_ok=True)
            torch.save({'q':rq.detach().cpu(),'k':rk.detach().cpu(),'v':v.detach().cpu(),
                'reference_output':output.detach().cpu(),'window':self.sliding_window,
                'scale':self.scale,'layer':0,'input_tokens':n},path/f'ctx_{n}.pt')
            saved.add(n)
        return output
    impl.prefill_attention=types.MethodType(observe,impl)
