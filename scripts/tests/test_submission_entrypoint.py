import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import serve_submission as s
from smoke_submission import parse_sse


def args(mode):
    d=dict(mode=mode,model=s.MODEL,gpu=0,host='127.0.0.1',port=8000,engine_port=8026,max_num_seqs=None,max_model_len=None,max_batched_tokens=None,audit=True)
    d.update({k:False for k in ['disable_graph','disable_qk_fusion','disable_native_routing','disable_prefix','disable_metadata_reuse','separate_qkv','disable_native_moe','disable_native_attention','reference_norm','disable_direct_kv','disable_resident_rope']})
    return argparse.Namespace(**d)


def test_modes_and_independent_overrides(tmp_path,monkeypatch):
    monkeypatch.setattr(s.os,'environ',dict(s.os.environ))
    c=s.build(args('conservative'),tmp_path)
    a=s.build(args('aggressive'),tmp_path)
    assert c['effective_max_num_seqs']==32 and '--enforce-eager' in c['command']
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


def test_native_fallbacks_do_not_implicitly_switch_mode(tmp_path,monkeypatch):
    monkeypatch.setattr(s.os,'environ',dict(s.os.environ))
    x=args('aggressive');x.disable_native_moe=True;x.disable_native_attention=True;x.reference_norm=True
    d=s.build(x,tmp_path)
    assert d['mode']=='aggressive' and d['settings']['MELLUM_MOE_BACKEND']=='torch'
    assert d['settings']['MELLUM_ATTENTION_EXECUTION']=='reference'
    assert d['settings']['MELLUM_SDPA_DECODE']=='1' and d['settings']['MELLUM_REQUEST_BATCHING']=='1'
    assert d['settings']['MELLUM_DIRECT_KV_WRITE']=='0' and d['settings']['MELLUM_NORM_BACKEND']=='reference'
    assert d['settings']['MELLUM_DECODE_GRAPH']=='1'


def test_short_ablation_preserves_counts_and_hashes(tmp_path):
    from run_submission_ablation import shorten
    template=tmp_path/'template';inputs=template/'inputs';inputs.mkdir(parents=True)
    source=inputs/'original.jsonl'
    source.write_text(''.join(json.dumps({'id':str(i),'prompt':'prompt '+str(i)})+'\n' for i in range(20)))
    rows=[{'id':str(i),'tokens':i+1} for i in range(20)]
    plan={'template':{'stages':{'A':[{'number':512,'expected':rows*26}],
        'B':[{'number':50}],'C':[{'number':200,'source':str(source),'expected':rows}]},'input_sha256':{}}}
    shorten(plan,tmp_path)
    stages=plan['template']['stages']
    assert [stages[k][0]['number'] for k in 'ABC']==[32,6,8]
    assert len(stages['A'][0]['expected'])==32 and len(stages['C'][0]['expected'])==8
    new=Path(stages['C'][0]['source'])
    assert len(new.read_text().splitlines())==8 and str(new) in plan['template']['input_sha256']
    assert 'NOT historical Step0' in plan['scope']


def test_primary_preset_matches_measured_acceptance(tmp_path,monkeypatch):
    monkeypatch.setattr(s.os,'environ',dict(s.os.environ))
    launch=s.build(args('conservative'),tmp_path)
    saved=json.loads((s.PROJECT/'artifacts/submission/primary_acceptance_batch32/runtime.json').read_text())
    for key,value in saved['settings'].items():
        if key.endswith('_PATH'):continue
        assert launch['settings'].get(key)==value,key
    assert launch['effective_max_num_seqs']==saved['engine_options']['max_num_seqs']==32
    verdict=json.loads((s.PROJECT/launch['preset']['acceptance_evidence']).read_text())
    assert verdict['status']=='local_candidate_checks_passed' and all(verdict['local_checks'].values())
