import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import serve_submission as s
from smoke_submission import parse_sse


def args(mode):
    d=dict(mode=mode,model=s.MODEL,gpu=0,host='127.0.0.1',port=8000,engine_port=8026,max_num_seqs=None,max_model_len=None,max_batched_tokens=None,audit=True)
    d.update({k:False for k in ['disable_graph','disable_qk_fusion','disable_native_routing','disable_prefix','disable_metadata_reuse','separate_qkv']})
    return argparse.Namespace(**d)


def test_modes_and_independent_overrides(tmp_path,monkeypatch):
    monkeypatch.setattr(s.os,'environ',dict(s.os.environ))
    c=s.build(args('conservative'),tmp_path)
    a=s.build(args('aggressive'),tmp_path)
    assert c['effective_max_num_seqs']==1 and '--enforce-eager' in c['command']
    assert a['effective_max_num_seqs']==32 and '--enforce-eager' not in a['command']
    assert c['settings']['MELLUM_MOE_BACKEND']=='torch' and a['settings']['MELLUM_FUSED_QK_ROPE']=='1'
    x=args('aggressive');x.disable_graph=True;x.disable_prefix=True
    z=s.build(x,tmp_path)
    assert '--enforce-eager' in z['command'] and '--no-enable_prefix_caching' in z['command']
    assert z['settings']['MELLUM_MOE_BACKEND']==a['settings']['MELLUM_MOE_BACKEND']
    assert not a['preset']['mode_is_optimization_point']


def test_sse_reasoning_and_usage_are_preserved():
    r=parse_sse(['data: '+json.dumps({'choices':[{'delta':{'reasoning_content':'hello'}}]}),
        'data: '+json.dumps({'choices':[],'usage':{'completion_tokens':8}}),'data: [DONE]'])
    assert r['done'] and r['text']=='hello' and r['usage']['completion_tokens']==8
