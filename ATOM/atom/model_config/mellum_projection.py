# SPDX-License-Identifier: MIT
"""Opt-in TP1 BF16 QKV shape intervention; checkpoint layout is unchanged."""
import types
import torch
from torch.nn import functional as F


def separate_qkv_forward(self, x, x_scale=None, otype=torch.bfloat16):
    if x_scale is not None or x.dtype != torch.bfloat16 or otype != torch.bfloat16:
        raise ValueError('Separate Mellum QKV diagnostic requires unquantized BF16')
    weights = self.weight.split(self._mellum_projection_sizes, dim=0)
    biases = self.bias.split(self._mellum_projection_sizes) if self.bias is not None else [None]*3
    return torch.cat([F.linear(x, weight, bias) for weight,bias in zip(weights,biases)], dim=-1)


def install_separate_qkv(module, q_size, kv_size):
    sizes = (q_size, kv_size, kv_size)
    if module.weight.ndim != 2 or module.weight.shape[0] != sum(sizes) or module.weight.dtype != torch.bfloat16:
        raise ValueError('Separate Mellum QKV requires canonical BF16 packed weights')
    module._mellum_projection_sizes = sizes
    module.forward = types.MethodType(separate_qkv_forward, module)


def reference_router_forward(self,x,x_scale=None,otype=torch.bfloat16):
    if x_scale is not None or x.dtype!=torch.bfloat16 or otype!=torch.bfloat16:
        raise ValueError('Reference Mellum router requires unquantized BF16')
    return F.linear(x,self.weight,self.bias)


def install_reference_router(module):
    if module.weight.ndim!=2 or module.weight.dtype!=torch.bfloat16:
        raise ValueError('Reference Mellum router requires canonical BF16 weights')
    module.forward=types.MethodType(reference_router_forward,module)
