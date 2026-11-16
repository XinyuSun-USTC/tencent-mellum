import torch,pytest
from types import SimpleNamespace as NS
from atom.model_config.mellum_grouped_decode import grouped_swa,host_geometry
from atom.model_config.mellum_attention import gather_reference_kv
from atom.model_config.mellum_campaign import Campaign


@pytest.mark.parametrize('batch',[2,3,4])
@pytest.mark.parametrize('window',[4,16])
def test_grouped_swa_matches_requestwise_sdpa_with_noncontiguous_blocks(batch,window):
    torch.manual_seed(103)
    kc=torch.randn(9,2,1,16,8,dtype=torch.bfloat16);vc=torch.randn(9,2,8,16,dtype=torch.bfloat16)
    q=torch.randn(batch,4,8,dtype=torch.bfloat16)
    tables=torch.tensor([[8,1,5],[2,7,0],[6,3,4],[5,0,7]])[:batch]
    lengths=torch.tensor([33,31,39,35])[:batch]
    got=grouped_swa(q,kc,vc,tables,lengths,window,8**-.5)
    expected=[]
    for i,n in enumerate(lengths.tolist()):
        k,v=gather_reference_kv(kc,vc,tables[i],n-window,n,'NHD')
        k=k.transpose(0,1)[None].repeat_interleave(2,1);v=v.transpose(0,1)[None].repeat_interleave(2,1)
        out=torch.nn.functional.scaled_dot_product_attention(q[i:i+1].transpose(0,1)[None],k,v,attn_mask=torch.ones(1,1,1,window,dtype=torch.bool),scale=8**-.5)
        expected.append(out[0].transpose(0,1))
    assert torch.equal(got,torch.cat(expected))


def test_host_geometry_is_reset_on_each_actual_model_forward(monkeypatch):
    monkeypatch.setenv('MELLUM_DECODE_META_ONCE','1')
    ctx=NS(attn_metadata=NS(context_lens=torch.tensor([10,20])),context=NS(positions=torch.tensor([9,19]),is_prefill=False))
    fake=NS(config={},root=None);batch=NS(num_scheduled_tokens=[1,1],req_ids=[8,7])
    Campaign.begin(fake,batch,ctx,torch.tensor([1,2]));assert host_geometry(ctx)[0]==[10,20]
    ctx.attn_metadata.context_lens=torch.tensor([30,40]);ctx.context.positions=torch.tensor([29,39])
    Campaign.begin(fake,batch,ctx,torch.tensor([3,4]));assert host_geometry(ctx)[0]==[30,40]
    ctx.context.positions=torch.tensor([1,2]);Campaign.begin(fake,batch,ctx,torch.tensor([3,4]))
    with pytest.raises(ValueError):host_geometry(ctx)
