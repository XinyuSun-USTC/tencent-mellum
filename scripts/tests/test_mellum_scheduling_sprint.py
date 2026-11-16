import sys,json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import profile_settings
import serve_mellum
from atom.model_config.mellum_prefix import native_prefix_geometry,gather_direct_prefix


def test_cached_suffix_geometry_and_invalid_positions():
    m=NS(cu_seqlens_q=torch.tensor([0,2,3]),cu_seqlens_k=torch.tensor([0,5,9]),max_seqlen_q=2,max_seqlen_k=5)
    assert native_prefix_geometry(m,torch.tensor([3,4,3]),3,9)==([2,1],[5,4])
    with pytest.raises(ValueError):native_prefix_geometry(m,torch.tensor([0,1,3]),3,9)


def test_ndh_gather_respects_noncontiguous_blocks_without_pool_conversion():
    block=4;heads=2;dim=8;x=2
    kc=torch.arange(5*heads*(dim//x)*block*x).reshape(5,heads,dim//x,block,x).bfloat16()
    vc=torch.arange(5*heads*dim*block).reshape(5,heads,dim,block).bfloat16()
    table=torch.tensor([[3,1],[4,0]])
    ctx=NS(attn_metadata=NS(block_tables=table,context_lens=torch.tensor([5,3])))
    q=torch.zeros(3,4,dim);k=torch.zeros(3,heads,dim);v=k.clone()
    result=gather_direct_prefix(NS(_cache_format='NHD'),q,k,v,kc,vc,None,None,ctx)
    expected_k=[];expected_v=[]
    for req,n in enumerate([5,3]):
        for t in range(n):
            physical=table[req,t//block];offset=t%block
            expected_k.append(kc[physical,:,:,offset,:].reshape(heads,dim))
            expected_v.append(vc[physical,:,:,offset])
    assert torch.equal(result[1],torch.stack(expected_k))
    assert torch.equal(result[2],torch.stack(expected_v))
    assert result[3] is kc and result[4] is vc


def test_scheduling_flags_do_not_change_client_protocol(tmp_path,monkeypatch):
    monkeypatch.setattr(serve_mellum.os,'environ',dict(serve_mellum.os.environ))
    for profile in ['sprint_graph','sprint_graph_prefix','sprint_graph_chunk2048','sprint_graph_chunk4096']:
        out=tmp_path/profile
        monkeypatch.setattr(sys,'argv',['serve','--profile',profile,'--batching','--prepare-only','--output',str(out)])
        serve_mellum.main();d=json.loads((out/'launch.json').read_text());cmd=d['command']
        assert '--enforce-eager' not in cmd and '--level' in cmd
        assert ('--enable_prefix_caching' in cmd)==profile.endswith('_prefix')
        assert ('--enable_chunked_prefill' in cmd)==('chunk' in profile)
        assert cmd[cmd.index('--max-model-len')+1]=='16640'
        wanted='2048' if profile.endswith('2048') else '4096' if profile.endswith('4096') else '16640'
        assert cmd[cmd.index('--max-num-batched-tokens')+1]==wanted
        assert profile_settings(profile)['MELLUM_NATIVE_ROUTING']=='0'


def test_cached_direct_writes_new_key_before_gather(monkeypatch):
    from atom.model_config.mellum_direct_kv import install_direct_kv_rope
    from atom.model_config.mellum_attention import reference_position_rope
    monkeypatch.setenv('MELLUM_PREFIX_COMPAT','1');monkeypatch.setenv('MELLUM_RESIDENT_ROPE','1')
    monkeypatch.setenv('MELLUM_FUSED_ROPE','0')
    kc=torch.zeros(2,1,16,16,8,dtype=torch.bfloat16);vc=torch.zeros(2,1,128,16,dtype=torch.bfloat16)
    q=torch.ones(1,2,128,dtype=torch.bfloat16);k=torch.ones(1,1,128,dtype=torch.bfloat16);v=k*3
    positions=torch.tensor([1]);inv=torch.ones(64);calls=[]
    def write(key,value,key_cache,value_cache,slots,**kwargs):
        calls.append('write');assert not kwargs['asm_layout']
        key_cache[1,:,:,1,:]=key[0].reshape(1,16,8);value_cache[1,:,:,1]=value[0]
    monkeypatch.setitem(sys.modules,'aiter',NS(reshape_and_cache=write))
    impl=NS(layer_num=1,use_flash_layout=False,kv_cache_dtype='bf16',q_norm=None,k_norm=None,
            sliding_window=1024,head_dim=128,num_heads=2,num_kv_heads=1)
    ctx=NS(attn_metadata=NS(has_cached=True,slot_mapping=torch.tensor([17]),block_tables=torch.tensor([[1]]),context_lens=torch.tensor([2])),
        kv_cache_data={'layer_1':NS(k_cache=kc,v_cache=vc,k_scale=None,v_scale=None)},context=NS(is_prefill=True,is_dummy_run=True))
    install_direct_kv_rope(impl,inv,1.)
    out=impl.rope_cache(q,k,v,None,positions,ctx)
    expected=reference_position_rope(q,k,positions,inv,1.)[1]
    assert calls==['write'] and out[1].shape==(2,1,128)
    assert torch.equal(out[1][1],expected[0]) and torch.equal(out[2][1],v[0])


def test_native_cached_rectangular_bounds_reach_kernel(monkeypatch):
    from atom.model_config.mellum_native import native_prefill
    monkeypatch.setenv('MELLUM_PREFIX_COMPAT','1');seen=[]
    def flash(**kwargs):seen.append(kwargs);return kwargs['q'].clone()
    monkeypatch.setitem(sys.modules,'aiter',NS(flash_attn_varlen_func=flash))
    q=torch.zeros(3,4,128);k=torch.zeros(9,1,128)
    meta=NS(has_cached=True,cu_seqlens_q=torch.tensor([0,2,3]),cu_seqlens_k=torch.tensor([0,5,9]),max_seqlen_q=2,max_seqlen_k=5)
    ctx=NS(attn_metadata=meta,context=NS(is_prefill=True,positions=torch.tensor([3,4,3])))
    out=native_prefill(NS(layer_num=1,sliding_window=1024,scale=128**-.5),q,k,k,None,None,None,None,ctx)
    assert out.shape==q.shape and seen[0]['causal'] and seen[0]['window_size']==(1023,0,0)
    assert seen[0]['max_seqlen_q']==2 and seen[0]['max_seqlen_k']==5


def test_derived_candidates_change_only_declared_mechanisms():
    base=profile_settings('sprint_graph_qk')
    expected={
        'sprint_graph_qk_router_prefix':{'MELLUM_NATIVE_ROUTING':'1','MELLUM_PREFIX_COMPAT':'1','MELLUM_PREFIX_ENABLED':'1'},
        'sprint_graph_qk_router_prefix_metadata':{'MELLUM_PREFILL_METADATA_REUSE':'1','MELLUM_NATIVE_ROUTING':'1','MELLUM_PREFIX_COMPAT':'1','MELLUM_PREFIX_ENABLED':'1'},
        'sprint_graph_qk_router':{'MELLUM_NATIVE_ROUTING':'1'},
        'sprint_graph_qk_prefix':{'MELLUM_PREFIX_COMPAT':'1','MELLUM_PREFIX_ENABLED':'1'},
        'sprint_graph_qk_chunk2048':{'MELLUM_PREFIX_COMPAT':'1','MELLUM_CHUNK_BUDGET':'2048'},
        'sprint_graph_qk_chunk4096':{'MELLUM_PREFIX_COMPAT':'1','MELLUM_CHUNK_BUDGET':'4096'}}
    for name,changes in expected.items():
        assert profile_settings(name)=={**base,**changes}
