import importlib.util,sys,types
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[2]

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
prefix=load('prefix_math','ATOM/atom/model_config/mellum_prefix.py')
attention=load('reference_math','ATOM/atom/model_config/mellum_attention.py')
NS=types.SimpleNamespace

@pytest.mark.parametrize('window',[0,1,3,1024])
def test_suffix_mask_matches_absolute_causal_and_window(window):
 mask=prefix.suffix_mask(3,8,window,'cpu')[0,0]
 assert mask.tolist()==[[j<=i and (window<=0 or j>i-window) for j in range(8)] for i in range(5,8)]


def test_reference_keys_are_written_before_gather_without_shared_metadata_mutation(monkeypatch):
 monkeypatch.setitem(sys.modules,'atom.model_config.mellum_attention',attention)
 cache=torch.zeros((1,1,1,8,2),dtype=torch.bfloat16);v=torch.zeros((1,1,2),dtype=torch.bfloat16)
 meta=NS(has_cached=True,slot_mapping=torch.tensor([3]));ctx=NS(attn_metadata=meta)
 q=torch.tensor([[[1.,2.]]],dtype=torch.bfloat16);k=q.clone()
 events=[]
 def original(q,k,v,qkv,pos,local):
  assert local is not ctx and local.attn_metadata is not meta
  assert not local.attn_metadata.has_cached and meta.has_cached
  return q,k,v,cache,None,None,None
 def gather(q,k,v,kc,vc,ks,vs,metadata):
  assert metadata is meta and meta.has_cached
  assert torch.equal(kc[0,:,:,3,:].reshape_as(k),k)
  events.append('corrected');return q,k,v,kc,vc,ks,vs
 impl=NS(use_flash_layout=False,_gather_prefix_and_concat_kv=gather)
 prefix.install_prefix_rope(impl,original,torch.tensor([1.]),1.)
 result=impl.rope_cache(q,k,v,None,torch.tensor([3]),ctx)
 assert events==['corrected'] and meta.has_cached and len(result)==7


def test_ragged_cached_outputs_match_full_attention_suffix_and_validate_positions(monkeypatch):
 monkeypatch.setitem(sys.modules,'atom.model_config.mellum_batch',NS(audit=lambda *a:None))
 torch.manual_seed(3);q=torch.randn(3,4,8);k=torch.randn(10,2,8);v=torch.randn_like(k)
 meta=NS(cu_seqlens_q=torch.tensor([0,2,3]),cu_seqlens_k=torch.tensor([0,6,10]))
 ctx=NS(attn_metadata=meta,context=NS(positions=torch.tensor([4,5,3])))
 impl=NS(sliding_window=3,scale=8**-.5,layer_num=-1)
 got=prefix.cached_prefill(impl,q,k,v,None,None,None,None,ctx)
 for a,b,c,d in [(0,2,0,6),(2,3,6,10)]:
  nq=b-a;nk=d-c
  fullq=torch.cat([torch.zeros(nk-nq,4,8),q[a:b]])
  mask=prefix.suffix_mask(nk,nk,3,'cpu')
  ref=torch.nn.functional.scaled_dot_product_attention(fullq.transpose(0,1)[None],
   k[c:d].repeat_interleave(2,dim=1).transpose(0,1)[None],v[c:d].repeat_interleave(2,dim=1).transpose(0,1)[None],
   attn_mask=mask,scale=impl.scale)[0].transpose(0,1)[-nq:]
  assert torch.allclose(got[a:b],ref,atol=1e-6)
 ctx.context.positions[0]=0
 with pytest.raises(ValueError,match='positions'):prefix.cached_prefill(impl,q,k,v,None,None,None,None,ctx)


def test_nhd_cached_gather_reads_only_logical_request_after_reference_write(monkeypatch):
 monkeypatch.setitem(sys.modules,'atom.model_config.mellum_attention',attention)
 cache=torch.zeros((3,1,1,8,2),dtype=torch.bfloat16)
 vc=torch.zeros((3,1,2,8),dtype=torch.bfloat16)
 # Request uses physical block 2, not block 0; cached prefix keys have known data.
 cache[2,:,:,0,:]=7;cache[2,:,:,1,:]=8
 vc[2,:,:,0]=9;vc[2,:,:,1]=10;vc[2,:,:,2]=11
 ctx=NS(attn_metadata=NS(has_cached=True,slot_mapping=torch.tensor([18]),
     context_lens=torch.tensor([3]),block_tables=torch.tensor([[2]])))
 def original(q,k,v,qkv,pos,local):return q,k,v,cache,vc,None,None
 impl=NS(use_flash_layout=False,_cache_format='NHD')
 prefix.install_prefix_rope(impl,original,torch.tensor([1.]),1.)
 q=torch.tensor([[[1.,2.]]],dtype=torch.bfloat16)
 got=impl.rope_cache(q,q.clone(),q.clone(),None,torch.tensor([2]),ctx)
 assert got[1].shape==(3,1,2) and got[2].shape==(3,1,2)
 assert got[1][:2].flatten().tolist()==[7,7,8,8]
 assert got[2].flatten().tolist()==[9,9,10,10,11,11]
 assert torch.equal(got[1][-1],cache[2,:,:,2,:].reshape(1,2))
