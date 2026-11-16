import io,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_mellum_abort_interleaving import Stream,verify_lifetime,lookup_seq


def test_sse_can_pause_and_resume_without_losing_frames():
    values=[{'id':'x','choices':[{'delta':{'role':'assistant'}}]},
            {'id':'x','choices':[{'delta':{'reasoning_content':'hello'}}]},
            {'choices':[],'usage':{'completion_tokens':8}}]
    raw=''.join('data: '+json.dumps(x)+'\n\n' for x in values)+'data: [DONE]\n\n'
    stream=Stream(io.BytesIO(raw.encode()));stream.advance(stop_after_token=True)
    assert stream.request_id=='x' and stream.text=='hello' and not stream.done
    stream.advance();assert stream.finished(8)['done']
    with pytest.raises(ValueError):stream.finished(64)


def test_lifetime_requires_survivor_active_after_abort_and_no_zombie_step():
    rows=[{'kind':'abort','removed':[2],'time_ns':10,'running':1},
          {'kind':'engine_step','start_ns':11,'request_ids':[1],'finished_request_ids':[1]}]
    assert verify_lifetime(rows,2,1,8)['post_abort_survivor_steps']==1
    rows.append({'kind':'engine_step','start_ns':12,'request_ids':[2],'finished_request_ids':[]})
    with pytest.raises(ValueError,match='Canceled request'):verify_lifetime(rows,2,1,8)
    assert lookup_seq('API: Created request_id=chatcmpl-a, seq_id=7','chatcmpl-a')==7
