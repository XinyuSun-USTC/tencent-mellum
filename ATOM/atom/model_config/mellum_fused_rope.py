# SPDX-License-Identifier: MIT
"""Default-off THD QK RMSNorm/RoPE fusion, retaining BF16 stage boundaries."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _norm_rope(Q,K,QW,KW,INV,POS,OQ,OK,
               QS0:tl.constexpr,QS1:tl.constexpr,KS0:tl.constexpr,KS1:tl.constexpr,
               QH:tl.constexpr,KH:tl.constexpr,D:tl.constexpr,
               EPS:tl.constexpr,FACTOR:tl.constexpr,NORM:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1);d=tl.arange(0,D)
    if h<QH:
        x=tl.load(Q+t*QS0+h*QS1+d).to(tl.float32)
        if NORM:
            w=tl.load(QW+d).to(tl.float32)
            x=(x*tl.rsqrt(tl.sum(x*x,0)/D+EPS)*w).to(tl.bfloat16).to(tl.float32)
    else:
        x=tl.load(K+t*KS0+(h-QH)*KS1+d).to(tl.float32)
        if NORM:
            w=tl.load(KW+d).to(tl.float32)
            x=(x*tl.rsqrt(tl.sum(x*x,0)/D+EPS)*w).to(tl.bfloat16).to(tl.float32)
    angle=tl.load(POS+t).to(tl.float32)*tl.load(INV+d%(D//2))
    c=(libdevice.cos(angle)*FACTOR).to(tl.bfloat16).to(tl.float32)
    s=(libdevice.sin(angle)*FACTOR).to(tl.bfloat16).to(tl.float32)
    rotated=tl.gather(x,(d+D//2)%D,0)*tl.where(d<D//2,-1.,1.)
    # Reference applies two BF16 multiplies, followed by BF16 add.
    y=(x*c).to(tl.bfloat16).to(tl.float32)+(rotated*s).to(tl.bfloat16).to(tl.float32)
    if h<QH:tl.store(OQ+(t*QH+h)*D+d,y)
    else:tl.store(OK+(t*KH+h-QH)*D+d,y)


def fused_qk_rope(q,k,positions,inverse,factor,norms=None):
    if (q.ndim!=3 or k.ndim!=3 or q.shape[0]!=k.shape[0] or q.shape[-1]!=128
            or k.shape[-1]!=128 or q.dtype!=torch.bfloat16 or k.dtype!=q.dtype
            or q.device!=k.device or not q.is_cuda or q.stride(-1)!=1 or k.stride(-1)!=1):
        raise ValueError('Fused QK RoPE requires CUDA BF16 THD head_dim128')
    if positions.numel()!=len(q) or inverse.numel()!=64 or inverse.device!=q.device or positions.device!=q.device:
        raise ValueError('Position/inverse geometry or device mismatch')
    eps=1e-6;qw=kw=inverse
    if norms is not None:
        qn,kn=norms;qw,kw=qn.weight,kn.weight;eps=qn.eps
        if kn.eps!=eps or qw.numel()!=128 or kw.numel()!=128:raise ValueError('Invalid QK norm parameters')
    oq=torch.empty(q.shape,device=q.device,dtype=q.dtype);ok=torch.empty(k.shape,device=k.device,dtype=k.dtype)
    _norm_rope[(len(q),q.shape[1]+k.shape[1])](q,k,qw,kw,inverse,positions.flatten(),oq,ok,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),q.shape[1],k.shape[1],128,eps,float(factor),norms is not None,
        num_warps=4,enable_fp_fusion=False)
    return oq,ok
