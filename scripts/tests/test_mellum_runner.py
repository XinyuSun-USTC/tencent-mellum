import importlib.util
import json
from pathlib import Path
import sys
import pytest

scripts=Path(__file__).parents[1]
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('suite_runner',scripts/'run_mellum_validation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize('kind',['exact','floating_difference','token_difference'])
def test_runner_retains_comparison_on_token_failure(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['runner','--suite','short'])
    atom=tmp_path/'atom_run';atom.mkdir();(atom/'candidate.json').write_text('{}')
    def stage(name,command,log_path,**kwargs):
        is_atom='check_atom_mellum.py' in command[1]
        content=f'RESULT token_mismatch output {atom}\n' if is_atom else 'done\n'
        log_path.write_text(content)
        return (1 if is_atom and kind=='token_difference' else 0),content
    monkeypatch.setattr(m,'run_stage',stage)
    mismatches=[] if kind=='exact' else ['logits_values'] if kind=='floating_difference' else ['generated_ids']
    monkeypatch.setattr(m,'compare',lambda *args:{'status':'passed' if not mismatches else 'failed',
        'mismatches':[] if not mismatches else ['sample'],
        'records':[{'name':'sample','mismatches':mismatches,'max_abs_error':0 if not mismatches else 1}]})
    code=m.main()
    output=next((tmp_path/'results/mellum_validation').iterdir())
    summary=json.loads((output/'summary.json').read_text())
    assert (output/'comparison.json').exists()
    assert code==(1 if kind=='token_difference' else 0)
    assert summary['status']=={'exact':'tokens_and_logits_exact','floating_difference':'tokens_exact_logits_review_required','token_difference':'comparison_failed'}[kind]

def test_candidate_prepare_only_is_explicit_and_nonforcing(tmp_path,monkeypatch):
    data=json.loads(Path('/workspace/results/atom_step0_batch_ab/hf_reference/inputs.json').read_text())
    source=tmp_path/'inputs.json';source.write_text(json.dumps(data))
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['runner','--suite','short','--inputs',str(source),'--prepare-only','--correctness-candidate'])
    assert m.main()==0
    summary=json.loads(next((tmp_path/'results/mellum_validation').glob('*/summary.json')).read_text())
    assert summary['correctness_candidate'] and summary['sampling_mode']=='normal_no_forcing'
    assert summary['qkv_backend']=='separate' and summary['prefill_backend']=='sdpa_cpu_rope_diagnostic'

def test_candidate_rejects_unsupported_long_prompt_before_gpu(tmp_path,monkeypatch):
    data=json.loads(Path('/workspace/results/atom_step0_batch_ab/hf_reference/inputs.json').read_text())
    data['records'][0]['input_ids']=[[1]*16385]
    source=tmp_path/'inputs.json';source.write_text(json.dumps(data))
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['runner','--inputs',str(source),'--prepare-only','--correctness-candidate'])
    with pytest.raises(ValueError,match='16384'):m.main()
