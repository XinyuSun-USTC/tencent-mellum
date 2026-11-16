import sys
from types import SimpleNamespace as NS
import pytest
import torch
from atom.model_config.mellum_attention import install_reference_kv_rope
from atom.model_config.mellum_attention import reference_position_rope,gather_reference_kv


@pytest.mark.parametrize('window',[-1,1024])
def test_direct_write_keeps_reference_rotation_and_slot_layout(monkeypatch,window):
    torch.manual_seed(71);n=4;h=2;d=128;bs=16;x=8
    kc=torch.full((3,h,d//x,bs,x),-7,dtype=torch.bfloat16)
    vc=torch.full((3,h,d,bs),-9,dtype=torch.bfloat16)
    q=torch.randn(n,4,d).bfloat16();k=torch.randn(n,h,d).bfloat16();v=torch.randn_like(k)
    slots=torch.tensor([15,16,-1,33]);positions=torch.tensor([15,16,17,33]);inv=1/(500000**(torch.arange(0,d,2).float()/d));calls=[]
    def write(key,value,key_cache,value_cache,slot_mapping,**kw):
        calls.append(kw)
        for i,slot in enumerate(slot_mapping.tolist()):
            if slot<0:continue
            block,off=divmod(slot,bs);key_cache[block,:,:,off,:]=key[i].reshape(h,d//x,x)
            if kw['asm_layout']:value_cache.view(3,h,bs//x,d,x)[block,:,off//x,:,off%x]=value[i]
            else:value_cache[block,:,:,off]=value[i]
    monkeypatch.setitem(sys.modules,'aiter',NS(reshape_and_cache=write))
    impl=NS(layer_num=1,use_flash_layout=False,kv_cache_dtype='bf16',q_norm=None,k_norm=None,
            sliding_window=window,head_dim=d,num_heads=4,num_kv_heads=h)
    ctx=NS(attn_metadata=NS(has_cached=False,slot_mapping=slots),kv_cache_data={'layer_1':NS(k_cache=kc,v_cache=vc,k_scale=None,v_scale=None)},
           context=NS(is_prefill=True,is_dummy_run=True))
    impl.rotary_emb=NS(base=500000.,_compute_inv_freq=lambda base:inv)
    impl.rope_cache=lambda *a: (_ for _ in ()).throw(AssertionError('Legacy rotation/cache path must not execute'))
    monkeypatch.setenv('MELLUM_DIRECT_KV_WRITE','1');install_reference_kv_rope(impl)
    actual=impl.rope_cache(q,k,v,None,positions,ctx)
    expected=reference_position_rope(q,k,positions,inv,1.)
    torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
    torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0)
    assert len(calls)==1 and calls[0]['asm_layout']==(window==-1)
    keys,values=gather_reference_kv(kc,vc,torch.tensor([0,1,2]),0,48,impl._cache_format)
    for i,slot in enumerate(slots.tolist()):
        if slot<0:continue
        torch.testing.assert_close(keys[slot],expected[1][i],rtol=0,atol=0)
        torch.testing.assert_close(values[slot],v[i],rtol=0,atol=0)
    valid=torch.ones(48,dtype=torch.bool);valid[torch.tensor([15,16,33])]=False
    assert (keys[valid]==-7).all() and (values[valid]==-9).all()
    ctx.attn_metadata.slot_mapping=slots[:3]
    with pytest.raises(ValueError,match='shape mismatch'):impl.rope_cache(q,k,v,None,positions,ctx)
    ctx.attn_metadata.slot_mapping=slots
    ctx.attn_metadata.has_cached=True
    with pytest.raises(ValueError,match='cached/chunked'):impl.rope_cache(q,k,v,None,positions,ctx)
