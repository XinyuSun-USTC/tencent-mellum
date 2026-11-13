# SPDX-License-Identifier: MIT
"""Mellum Step0 expert arithmetic using the frozen PyTorch grouped-mm API.

Semantics follow Transformers 5.10.1's grouped_mm expert implementation.
Checkpoint parameter names remain compatible with ATOM's expert loader mapping.
"""
import torch
from torch import nn
from torch.nn import functional as F


def grouped_linear(x, weight, offsets):
    if x.device.type == 'cuda':
        return F.grouped_mm(x, weight.transpose(-2, -1), offs=offsets)
    # CPU reference for wiring and numerical tests, not the GPU execution path.
    result = x.new_empty((x.shape[0], weight.shape[1]))
    start = 0
    for expert, end in enumerate(offsets.tolist()):
        if end > start:
            result[start:end] = F.linear(x[start:end], weight[expert])
        start = end
    return result


class MellumTorchExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.custom_routing_function = None
        self.w13_weight = nn.Parameter(torch.empty(config.num_experts,
            2 * self.intermediate_size, config.hidden_size), requires_grad=False)
        self.w2_weight = nn.Parameter(torch.empty(config.num_experts,
            config.hidden_size, self.intermediate_size), requires_grad=False)
        self.w13_weight.weight_loader = self.weight_loader
        self.w2_weight.weight_loader = self.weight_loader

    def weight_loader(self, param, weight, weight_name='', shard_id=None, expert_id=None):
        if expert_id is None or not 0 <= expert_id < self.num_experts:
            raise ValueError('Invalid Mellum expert id')
        if param is self.w13_weight and shard_id in ('w1', 'w3'):
            start = 0 if shard_id == 'w1' else self.intermediate_size
            target = param.data[expert_id, start:start+self.intermediate_size]
        elif param is self.w2_weight and shard_id == 'w2':
            target = param.data[expert_id]
        else:
            raise ValueError('Invalid Mellum expert parameter/shard')
        if target.shape != weight.shape:
            raise ValueError('Mellum expert checkpoint shape mismatch')
        target.copy_(weight)

    def forward(self, hidden_states, router_logits, batched=False):
        if self.custom_routing_function is None:
            weights, ids = router_logits.float().softmax(-1).topk(self.top_k, dim=-1)
            if self.renormalize:
                weights = weights / weights.sum(-1, keepdim=True)
            weights = weights.to(router_logits.dtype)
        else:
            weights, ids = self.custom_routing_function(hidden_states=hidden_states,
                gating_output=router_logits, topk=self.top_k, renormalize=self.renormalize)
            weights = weights.to(router_logits.dtype)
        if batched:
            # HF 5.10.1 generate switches grouped_mm to batched_mm for decode.
            selected = hidden_states.repeat_interleave(self.top_k, dim=0)
            expert_ids = ids.reshape(-1).clamp(0, self.num_experts-1)
            gate, up = torch.bmm(self.w13_weight[expert_ids], selected.unsqueeze(-1)).squeeze(-1).chunk(2, dim=-1)
            activated = F.silu(gate) * up
            projected = torch.bmm(self.w2_weight[expert_ids], activated.unsqueeze(-1)).squeeze(-1)
            weighted = projected * weights.reshape(-1, 1)
            return weighted.view(hidden_states.shape[0], self.top_k, -1).sum(1).to(hidden_states.dtype)
        expert_ids, perm = ids.reshape(-1).sort()
        selected = hidden_states[perm // self.top_k]
        counts = torch.histc(expert_ids.int() if selected.is_cuda else expert_ids.float(),
                            bins=self.num_experts, min=0, max=self.num_experts-1)
        offsets = counts.cumsum(0, dtype=torch.int32)
        gate, up = grouped_linear(selected, self.w13_weight, offsets).chunk(2, dim=-1)
        projected = grouped_linear(F.silu(gate) * up, self.w2_weight, offsets)
        weighted = projected * weights.reshape(-1)[perm, None]
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(perm.numel(), device=perm.device)
        return weighted[inverse].view(hidden_states.shape[0], self.top_k, -1).sum(1)
