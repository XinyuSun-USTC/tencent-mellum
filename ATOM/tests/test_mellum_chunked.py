import torch
import pytest
from atom.model_config.mellum_chunked import suffix_geometry,suffix_mask
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceType
from conftest import MockConfig


def test_suffix_geometry_ragged_absolute_offsets():
    assert suffix_geometry([0,2,5],[0,9,12],[7,8,0,1,2])==[(0,0,2,7,9),(1,2,5,0,3)]
    with pytest.raises(ValueError):suffix_geometry([0,2],[0,9],[0,1])


@pytest.mark.parametrize('window',[-1,4,8])
@pytest.mark.parametrize('cached,end',[(0,3),(3,9),(8,13),(15,17)])
def test_suffix_mask_is_full_attention_mask_slice(window,cached,end):
    start,mask=suffix_mask(cached,end,window,'cpu')
    full=torch.ones(end,end,dtype=torch.bool).tril()
    if window>0:full=full.triu(1-window)
    assert torch.equal(mask[0,0],full[cached:end,start:end])
    # Omitted historical keys are invisible to every query in this suffix.
    assert not full[cached:end,:start].any()


def test_chunked_sdpa_matches_full_context_cpu():
    torch.manual_seed(7)
    q=torch.randn(1,4,17,8);k=torch.randn(1,4,17,8);v=torch.randn_like(k)
    for window in [-1,4,8]:
        _,fullmask=suffix_mask(0,17,window,'cpu')
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=fullmask)
        parts=[]
        for cached,end in [(0,3),(3,9),(9,17)]:
            start,mask=suffix_mask(cached,end,window,'cpu')
            parts.append(torch.nn.functional.scaled_dot_product_attention(q[:,:,cached:end],k[:,:,start:end],v[:,:,start:end],attn_mask=mask))
        torch.testing.assert_close(torch.cat(parts,2),expected,rtol=1e-5,atol=1e-6)


def test_decode_quota_serves_ready_request_without_decoding_partial(monkeypatch,seq_factory):
    monkeypatch.setenv('MELLUM_C_DECODE_STEPS','4')
    sched=Scheduler(MockConfig(max_num_seqs=4,max_num_batched_tokens=4,num_kvcache_blocks=100,enable_chunked_prefill=True))
    ready=seq_factory([1,2]);partial=seq_factory(list(range(12)))
    sched.add(ready);sched.schedule()
    ready.num_cached_tokens=ready.num_prompt_tokens;ready.append_token(3)
    sched.add(partial)
    # Force next prefill, then emulate its KV progress as postprocess would.
    sched._mellum_decode_remaining=0
    batch,_=sched.schedule()
    assert batch.total_tokens_num_prefill==4
    partial.num_cached_tokens=4;partial.is_partial_prefill=True;sched._partial_prefill_count=1
    for _ in range(4):
        batch,seqs=sched.schedule()
        assert list(seqs)==[ready.id]
        assert ready.type==SequenceType.DECODE
        assert partial.is_partial_prefill and partial in sched.running
    batch,seqs=sched.schedule()
    assert batch.total_tokens_num_prefill==4 and list(seqs)==[partial.id]


@pytest.mark.parametrize('layout',['NHD','SHUFFLE'])
@pytest.mark.parametrize('window',[-1,4])
def test_cached_prefill_reads_physical_blocks_and_absolute_windows(layout,window):
    from types import SimpleNamespace as NS
    from atom.model_config.mellum_chunked import cached_prefill
    torch.manual_seed(19)
    cached,end=15,19
    q=torch.randn(4,4,8,dtype=torch.bfloat16)
    k=torch.randn(end,2,8,dtype=torch.bfloat16);v=torch.randn_like(k)
    kc=torch.zeros(3,2,1,16,8,dtype=torch.bfloat16)
    vc=torch.zeros(3,2,8,16,dtype=torch.bfloat16)
    table=torch.tensor([2,0])
    for t in range(end):
        block=int(table[t//16]);offset=t%16
        kc[block,:,:,offset,:]=k[t].reshape(2,1,8)
        if layout=='NHD':vc[block,:,:,offset]=v[t]
        else:vc.view(3,2,2,8,8)[block,:,offset//8,:,offset%8]=v[t]
    meta=NS(cu_seqlens_q=torch.tensor([0,4]),cu_seqlens_k=torch.tensor([0,end]),block_tables=table[None])
    ctx=NS(attn_metadata=meta,context=NS(positions=torch.arange(cached,end)))
    impl=NS(sliding_window=window,scale=8**-0.5,_cache_format=layout)
    got=cached_prefill(impl,q,k[cached:],v[cached:],kc,vc,None,None,ctx)
    _,mask=suffix_mask(cached,end,window,'cpu')
    start=max(0,cached-window+1) if window>0 else 0
    expected=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],k[start:].transpose(0,1)[None].repeat_interleave(2,1),v[start:].transpose(0,1)[None].repeat_interleave(2,1),attn_mask=mask,scale=impl.scale)[0].transpose(0,1)
    assert torch.equal(got,expected)



def test_prefix_hashes_publish_only_completed_blocks(seq_factory):
    from atom.model_engine.block_manager import BlockManager
    bm=BlockManager(MockConfig(num_kvcache_blocks=100,kv_cache_block_size=4,enable_prefix_caching=True))
    cold=seq_factory(list(range(14)));bm.allocate(cold)
    probe=seq_factory(list(range(8))+[101,102,103,104])
    assert bm.can_allocate(probe)==0
    bm.hash_blocks(cold,5);cold.num_cached_tokens=5
    assert bm.can_allocate(probe)==1
    bm.hash_blocks(cold,3);cold.num_cached_tokens=8
    assert bm.can_allocate(probe)==2
    bm.deallocate(cold)
    assert bm.can_allocate(probe)==2
    bm.allocate(probe,2)
    assert probe.num_cached_tokens==8
