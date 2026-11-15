"""CPU contracts for workload preservation and evaluation-client preparation."""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mellum_perf_protocol as protocol
from audit_mellum_perf_client import query_summary,judge_stream


class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(range(len(text.split())))
    def apply_chat_template(self,messages,tokenize,add_generation_prompt,enable_thinking):
        assert tokenize is False  # never len(BatchEncoding) as token length
        return 'START '+messages[0]['content']+' END'


def test_chat_length_renders_then_encodes():
    assert len(protocol.chat_ids(Tokenizer(),[{'role':'user','content':'hello world'}]))==4


def test_derived_messages_preserve_newlines_unicode_and_source_order(tmp_path,monkeypatch):
    source=tmp_path/'data';source.mkdir();monkeypatch.setattr(protocol,'DATA',source)
    for scenario,count in [('a',512),('c',200)]:
        rows=[{'id':f'{scenario}{i}','prompt':f'line {i}\n中文\n\nlast','shared_prefix_group':'g'} for i in range(count)]
        (source/f'scenario_{scenario}_prompts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    result=protocol.prepare_datasets(tmp_path/'out',Tokenizer())
    rows=[json.loads(x) for x in Path(result['A']['proposed_messages_file']).read_text().splitlines()]
    assert len(rows)==512 and rows[17]==[{'role':'user','content':'line 17\n中文\n\nlast'}]
    assert result['C']['records']==200 and result['C']['rows'][0]['newlines']==3
    assert result['A']['chat_min_max']==[6,6] and result['A']['raw_min_max']==[4,4]
    assert result['A']['A_range_valid_chat']==0


def test_duplicate_ids_rejected(tmp_path):
    path=tmp_path/'x.jsonl';path.write_text('{"id":"1","prompt":"x"}\n'*2)
    with pytest.raises(ValueError,match='Duplicate'):protocol.load_rows(path)


@pytest.mark.parametrize('scenario,count',[('A','512'),('B','50'),('C','200')])
def test_literal_assessment_commands_are_not_silently_normalized(scenario,count):
    cmd=protocol.literal_command(scenario,'http://localhost/v1/chat/completions',Path('/tmp/out'))
    assert cmd[cmd.index('--number')+1]==count
    assert cmd[cmd.index('--max-tokens')+1]=='256' and cmd[cmd.index('--min-tokens')+1]=='256'
    assert json.loads(cmd[cmd.index('--extra-args')+1])=={'ignore_eos':True,'temperature':0.0}
    if scenario=='A':assert cmd[cmd.index('--dataset')+1]=='/data/datasets/scenario_a_prompts.jsonl'
    if scenario=='C':assert '--open-loop' in cmd and cmd[cmd.index('--rate')+1]=='0.3' and cmd[cmd.index('--total-timeout')+1]=='180'


def test_json_metadata_is_not_mistaken_for_a_prompt():
    assert query_summary({'id':'a','prompt':'hello'},'hello')['source_prompt_preserved'] is False
    value=query_summary({'messages':[{'role':'user','content':'hello'}],'stream':True,'ignore_eos':True},'hello')
    assert value['source_prompt_preserved'] and value['valid_chat_messages']


def test_stream_check_rejects_content_only_ttft_and_wrong_usage():
    observed={'success':True,'generated_text':'thinking continuesanswer','prompt_tokens':7,'completion_tokens':3,
              'first_chunk_latency':.05,'query_latency':.35}
    good=judge_stream(observed)
    assert all(good[k] for k in ('success','reasoning_text_preserved','usage_counts_preserved','first_observable_output_timed','last_content_observed'))
    assert not judge_stream({**observed,'first_chunk_latency':.35})['first_observable_output_timed']
    assert not judge_stream({**observed,'first_chunk_latency':0})['first_observable_output_timed']
    assert not judge_stream({**observed,'completion_tokens':1})['usage_counts_preserved']
