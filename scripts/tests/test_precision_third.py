import sys,json,hashlib
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_execution_profiles import profile_settings
from run_precision_third import CONTROL,CANDIDATES,qualify,verify_result,render


def test_precision_candidates_add_only_grouping_metadata():
 b=profile_settings(CONTROL);m=profile_settings(CANDIDATES[0]);s=profile_settings(CANDIDATES[1])
 assert m=={**b,'MELLUM_DECODE_META_ONCE':'1'}
 assert s=={**m,'MELLUM_SWA_DECODE_GROUP':'4'}
 for x in [b,m,s]:
  assert x['MELLUM_ATTENTION_EXECUTION']=='reference' and x['MELLUM_MOE_BACKEND']=='torch'
  assert x['MELLUM_DIRECT_KV_WRITE']=='1'
  for key in ['MELLUM_REFERENCE_CHUNKED','MELLUM_PREFIX_ENABLED','MELLUM_DECODE_GRAPH','MELLUM_ABORT_ON_DISCONNECT']:
   assert x.get(key,'0')=='0'


def test_full_performance_rejects_failed_correctness(tmp_path):
 (tmp_path/'summary.json').write_text(json.dumps({'source_unchanged':True,'artifact_sha256':{},'profile':CANDIDATES[0],'status':'local_candidate_checks_not_all_passed'}))
 with pytest.raises(ValueError,match='pass token'):qualify(tmp_path,CANDIDATES[0])


def test_changed_archived_output_is_rejected(tmp_path):
 f=tmp_path/'output.json';f.write_text('first')
 (tmp_path/'summary.json').write_text(json.dumps({'source_unchanged':True,'artifact_sha256':{'output.json':hashlib.sha256(f.read_bytes()).hexdigest()}}))
 verify_result(tmp_path);f.write_text('second')
 with pytest.raises(ValueError,match='changed'):verify_result(tmp_path)
