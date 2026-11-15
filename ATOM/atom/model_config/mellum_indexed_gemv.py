# SPDX-License-Identifier: MIT
"""BF16 indexed expert GEMV: direct weights, FP32 reduction, no weight gather.

Experimental decode only. FP32 reduction ordering differs from native BMM; normal
HF generation validation is required. This is not a bitwise replacement promise.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _indexed_expert_gemv(X, IDS, W, Y, SX:tl.constexpr, WE:tl.constexpr, WN:tl.constexpr,
                         K:tl.constexpr, N:tl.constexpr, REPEAT:tl.constexpr,
                         BLOCK_K:tl.constexpr, BLOCK_N:tl.constexpr):
    row=tl.program_id(1)
    columns=tl.program_id(0)*BLOCK_N+tl.arange(0,BLOCK_N)
    k=tl.arange(0,BLOCK_K)
    expert=tl.load(IDS+row)
    x=tl.load(X+(row//REPEAT)*SX+k,k<K,0).to(tl.float32)
    w=tl.load(W+expert*WE+columns[:,None]*WN+k[None,:],
              (columns[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
    value=tl.sum(w*x[None,:],axis=1)
    tl.store(Y+row*N+columns,value,columns<N)


def indexed_linear(x,ids,weight,repeat):
    if x.device.type!='cuda' or x.dtype!=torch.bfloat16 or weight.dtype!=x.dtype:
        raise ValueError('Indexed expert candidate requires GPU BF16')
    if weight.ndim!=3 or x.ndim!=2 or ids.ndim!=2 or repeat not in (1,ids.shape[1]):
        raise ValueError('Invalid indexed expert geometry')
    rows=ids.numel();experts,n,k=weight.shape
    if (x.shape!=(rows//repeat,k) or not ids.is_contiguous() or ids.dtype!=torch.int64
            or weight.stride(-1)!=1 or x.stride(-1)!=1 or x.device!=weight.device or ids.device!=x.device):
        raise ValueError('Invalid indexed expert strides, row alignment or device')
    if k>4096:raise ValueError('Indexed candidate is bounded to input width <=4096')
    out=torch.empty((rows,n),dtype=x.dtype,device=x.device)
    _indexed_expert_gemv[(triton.cdiv(n,4),rows)](x,ids,weight,out,x.stride(0),weight.stride(0),weight.stride(1),
        k,n,repeat,triton.next_power_of_2(k),4,num_warps=4,enable_fp_fusion=False)
    return out
