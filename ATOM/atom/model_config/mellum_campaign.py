# SPDX-License-Identifier: MIT
"""Default-off request-shape controls and bounded natural-history observation.

Request identity comes from ScheduledBatch, never from prompt content or reference
outputs. This module is only installed by an explicit campaign configuration.
"""
import hashlib
import json
import os
from pathlib import Path
import torch


def split_call(call, args, kwargs, bounds):
    """Split row-aligned arguments; preserve kwargs and tuple/residual structure."""
    total = bounds[-1]
    def sliced(value, a, b):
        return value[a:b] if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == total else value
    outputs = [call(*(sliced(v, a, b) for v in args),
                    **{k: sliced(v, a, b) for k, v in kwargs.items()})
               for a, b in zip(bounds, bounds[1:])]
    def combine(values):
        if isinstance(values[0], torch.Tensor): return torch.cat(values, dim=0)
        if isinstance(values[0], tuple): return tuple(combine(list(v)) for v in zip(*values))
        if all(v is None for v in values): return None
        raise TypeError('Unsupported requestwise output')
    return combine(outputs)


class Campaign:
    def __init__(self, model, config):
        self.config = config
        self.phase = None
        self.active = True
        self.moe_variant = config.get("moe_optimization", "cpd")
        self.bounds = None
        self.rows = []
        self.history = {}
        self.current = {}
        self.used_bytes = 0
        self.root = Path(config['observe']) if config.get('observe') else None
        if self.root: self.root.mkdir(parents=True, exist_ok=True)
        self.handles = []
        for i, layer in enumerate(model.model.layers):
            modules = {'input_norm': layer.input_layernorm,
                       'qkv': layer.self_attn.qkv_proj, 'attention': layer.self_attn,
                       'o': layer.self_attn.o_proj, 'post_norm': layer.post_attention_layernorm,
                       'router': layer.mlp.gate, 'experts': layer.mlp.experts,
                       'layer': layer, 'q_norm': layer.self_attn.q_norm,
                       'k_norm': layer.self_attn.k_norm}
            for name, module in modules.items():
                group = 'L' if name in ('qkv', 'o', 'router') else 'E' if name == 'experts' else 'N' if name in ('input_norm','post_norm','q_norm','k_norm') else None
                if group and group in config.get('groups', ''):
                    if group=='E' and config.get('moe_optimization'):
                        from atom.model_config.mellum_moe_optimization import install_experts
                        install_experts(self,module,i)
                    else:self.wrap(module, group)
                if self.root:
                    self.handles.append(module.register_forward_hook(self.hook(f'{i:02d}.{name}')))
                    if name=='o':
                        self.handles.append(module.register_forward_pre_hook(self.prehook(f'{i:02d}.attention_context')))
            if self.root and config.get('cache_layers') and i in config['cache_layers']:
                self.install_cache(layer.self_attn.attn.impl, i)
        if 'N' in config.get('groups', ''): self.wrap(model.model.norm, 'N')
        if 'H' in config.get('groups', ''): self.wrap_head(model.lm_head)
        if self.root:
            self.handles.append(model.model.norm.register_forward_hook(self.hook('28.final_norm')))

    def wrap(self, module, group):
        original = module.forward
        def forward(*args, **kwargs):
            if self.active and self.bounds and self.phase in self.config.get('phases', []) and len(self.bounds) > 2:
                return split_call(original, args, kwargs, self.bounds)
            return original(*args, **kwargs)
        module.forward = forward

    def wrap_head(self, module):
        # ParallelLMHead itself selects the last prefill row using global cu.
        # Slicing its input and calling the original would reuse invalid global
        # indices. Select once, then use the same TP1 tuned GEMM per request.
        original = module.forward
        def forward(x):
            if not (self.active and self.bounds and self.phase in self.config.get('phases', []) and len(self.bounds)>2):
                return original(x)
            from aiter.tuned_gemm import tgemm
            selected=x[[b-1 for b in self.bounds[1:]]] if self.phase=='prefill' else x
            return torch.cat([tgemm.mm(v, module.weight, module.bias) for v in selected.split(1)], dim=0)
        module.forward=forward

    def begin(self, batch, ctx, input_ids):
        self.bounds = None; self.rows = []; self.current = {}
        if self.config.get('runtime_switch'):
            self.active=json.loads(Path(self.config['runtime_switch']).read_text())['enabled']
        if self.config.get('moe_switch'):
            mode=json.loads(Path(self.config['moe_switch']).read_text())
            if mode['variant'] not in ('cpd','shared_routing','indexed_decode','prefill_shared_elementwise'):raise ValueError('Unknown switched MoE variant')
            self.moe_variant=mode['variant']
        if batch is None or getattr(ctx.context, 'is_dummy_run', False): return
        self.phase = 'prefill' if ctx.context.is_prefill else 'decode'
        lengths = [int(n) for n in batch.num_scheduled_tokens]
        ids = [int(n) for n in batch.req_ids]
        if len(ids) != len(lengths) or any(n <= 0 for n in lengths) or sum(lengths) != len(input_ids):
            raise ValueError('Campaign requires real eager rows without graph padding')
        bounds = [0]
        for n in lengths: bounds.append(bounds[-1]+n)
        self.bounds = bounds
        if not self.root: return
        control = json.loads((self.root.parent/'observe_control.json').read_text())
        positions = ctx.context.positions.flatten().tolist()
        tokens = input_ids.flatten().tolist()
        for j, (seq, a, b) in enumerate(zip(ids,bounds,bounds[1:])):
            target = control.get(str(seq))
            if not target: continue
            if self.phase == 'prefill': self.history[seq] = tokens[a:b]
            else:
                if seq not in self.history: raise ValueError('Observed decode missing actual prefill history')
                self.history[seq].extend(tokens[a:b])
            step = len(self.history[seq])-target['input_tokens']
            if not 0 <= step < 8: continue
            key = f"{target['stage']}__{target['name']}__{step:02d}"
            meta = {'key':key,'name':target['name'],'stage':target['stage'],'step':step,
                    'request_id':seq,'batch_request_ids':ids,'bounds':bounds,'row':j,
                    'phase':self.phase,'position':positions[b-1],
                    'history_sha256':hashlib.sha256(json.dumps(self.history[seq]).encode()).hexdigest()}
            attn = ctx.attn_metadata
            slots=getattr(attn,'slot_mapping',None)
            if slots is not None:meta['last_slot']=int(slots.flatten()[b-1])
            if not ctx.context.is_prefill:
                meta['block_table'] = attn.block_tables[j].tolist()
                meta['context_length'] = int(attn.context_lens[j])
            self.rows.append((key,b-1,j,meta))
            self.current[key] = {}

    def prehook(self,name):
        def observe(module,args):
            if self.rows: self.hook(name)(module,(),args[0])
        return observe

    def install_cache(self,impl,layer):
        original=impl.dispatch_backend
        def dispatch(ctx):
            backend=original(ctx)
            def observed(q,k,v,kcache,vcache,kscale,vscale,fctx):
                output=backend(q,k,v,kcache,vcache,kscale,vscale,fctx)
                if self.rows and not fctx.context.is_prefill:
                    from atom.model_config.mellum_attention import gather_reference_kv
                    for key,_,j,meta in self.rows:
                        n=meta['context_length'];start=max(0,n-impl.sliding_window) if impl.sliding_window else 0
                        keys,values=gather_reference_kv(kcache,vcache,fctx.attn_metadata.block_tables[j],start,n,impl._cache_format)
                        self.capture(key,f'{layer:02d}.cache_k',keys)
                        self.capture(key,f'{layer:02d}.cache_v',values)
                return output
            return observed
        impl.dispatch_backend=dispatch

    def hook(self, name):
        def observe(module, args, output):
            if not self.rows: return
            values = output if isinstance(output,tuple) else (output,)
            for key,row,_,meta in self.rows:
                for k,value in enumerate(values):
                    if isinstance(value,torch.Tensor) and value.ndim and value.shape[0] == self.bounds[-1]:
                        self.capture(key, f'{name}.{k}', value[row])
                if name.endswith('.layer') and len(values)==2:
                    self.capture(key,name+'.materialized',values[0][row]+values[1][row])
        return observe

    def capture(self,key,name,value):
        size=value.numel()*value.element_size()
        if self.used_bytes+size > int(self.config.get('capture_bytes',3*1024**3)):
            (self.root/'capture_truncated').touch(); return
        self.current[key][name]=value.detach().cpu().clone();self.used_bytes+=size

    def end(self, logits):
        for key,row,j,meta in self.rows:
            if logits.shape[0] == self.bounds[-1]: index=row
            elif logits.shape[0] == len(self.bounds)-1: index=j
            else: raise ValueError('Unrecognized campaign logits row layout')
            self.capture(key,'29.logits',logits[index])
            if self.current[key]:
                torch.save(self.current[key],self.root/(key+'.pt'))
                (self.root/(key+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
        self.rows=[];self.bounds=None;self.current={}


def install(model):
    raw=os.getenv('MELLUM_BATCH_CAMPAIGN_CONFIG')
    if not raw: return
    config=json.loads(raw)
    if set(config.get('groups',''))-set('LENH') or set(config.get('phases',[]))-{'prefill','decode'}:
        raise ValueError('Unknown campaign control')
    if config.get('moe_optimization') and (config['moe_optimization'] not in ('cpd','shared_routing','indexed_decode','prefill_shared_elementwise')
            or not all(c in config.get('groups','') for c in 'LE')
            or set(config.get('phases',[]))!={'prefill','decode'}):
        raise ValueError('MoE candidates require full CPD control')
    model._mellum_campaign=Campaign(model,config)
