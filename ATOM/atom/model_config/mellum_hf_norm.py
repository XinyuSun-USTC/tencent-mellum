# SPDX-License-Identifier: MIT
"""Default-off BF16 RMSNorm fusion with HF residual and rounding boundaries."""
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _hf_norm(X,R,W,Y,RO,D:tl.constexpr,B:tl.constexpr,EPS:tl.constexpr,HAS_RES:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,B);mask=i<D
    x=tl.load(X+row*D+i,mask,0).to(tl.float32)
    if HAS_RES:
        x=(x+tl.load(R+row*D+i,mask,0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        tl.store(RO+row*D+i,x,mask)
    normalized=(x*tl.rsqrt(tl.sum(x*x,0)/D+EPS)).to(tl.bfloat16).to(tl.float32)
    y=normalized*tl.load(W+i,mask,0).to(tl.float32)
    tl.store(Y+row*D+i,y,mask)


def launch(x,weight,residual,eps):
    shape=x.shape;dim=shape[-1];xx=x.contiguous();rr=residual.contiguous() if residual is not None else xx
    output=torch.empty(shape,dtype=x.dtype,device=x.device)
    residual_out=torch.empty(shape,dtype=x.dtype,device=x.device) if residual is not None else output
    _hf_norm[(x.numel()//dim,)](xx,rr,weight,output,residual_out,dim,triton.next_power_of_2(dim),float(eps),residual is not None,
                               num_warps=4,enable_fp_fusion=False)
    return output if residual is None else (output,residual_out)


class HFCompatibleRMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6,fused_allreduce=False):
        super().__init__()
        if fused_allreduce:raise ValueError('HF Norm first version is TP1 only')
        self.weight=nn.Parameter(torch.ones(dim));self.eps=eps;self.dim=dim

    def forward(self,x,residual=None):
        if x.dtype!=torch.bfloat16 or self.weight.dtype!=x.dtype or x.shape[-1]!=self.dim:
            raise ValueError('HF fusion requires BF16 and matching norm dimension')
        if residual is not None and (residual.shape!=x.shape or residual.dtype!=x.dtype or residual.device!=x.device):
            raise ValueError('HF fusion residual geometry differs')
        if not x.is_cuda or self.weight.device!=x.device:raise ValueError('HF fusion requires a CUDA/ROCm device')
        return launch(x,self.weight,residual,self.eps)
