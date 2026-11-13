import hashlib
import json
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]))
from compare_mellum_generation import compare_generation

def test_same_history_comparison_stops_after_token_fork_and_records_ties(tmp_path):
    data=[]
    for name,ids,dtype in [('hf',[1,0],torch.float32),('atom',[2,0],torch.bfloat16)]:
        files=[]
        for step in range(2):
            f=tmp_path/f'{name}_{step}.pt';torch.save(torch.tensor([0,2,2],dtype=dtype),f)
            files.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
        payload={'records':[{'name':'x','generated_ids':[ids],'generation_logits':files}]}
        path=tmp_path/(name+'.json');path.write_text(json.dumps(payload));data.append(path)
    result=compare_generation(*data)
    assert result['compared_positions']==1 and result['nonexact_positions']==0
    assert result['records'][0]['both_selected_are_argmax']
    candidate=json.loads(data[1].read_text());candidate['records'][0]['generation_logits'][0]['sha256']='bad'
    data[1].write_text(json.dumps(candidate))
    with pytest.raises(ValueError,match='hash mismatch'):compare_generation(*data)
