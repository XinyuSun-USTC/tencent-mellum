import sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mellum_submission_modes import settings,profile_name,validate,ROWS,MODES
from mellum_execution_profiles import profile_settings
from serve_delivery import command

@pytest.mark.parametrize('row,parent',[('step0','r'),('step01','r_expert_decode4'),('step012','r_expert_decode4_kv_once'),('step0123','r_swa_decode4')])
def test_precision_matches_measured(row,parent):
 assert settings('precision',ROWS[row])==profile_settings(parent)

@pytest.mark.parametrize('row,parent',[('step0','r'),('step01','sprint_clean'),('step012','sprint_graph_qk_router'),('step0123','sprint_graph_qk_router_prefix_metadata')])
def test_performance_matches_measured(row,parent):
 expected=profile_settings(parent)
 if row=='step01':expected.update(MELLUM_DIRECT_KV_WRITE='0',MELLUM_RESIDENT_ROPE='0')
 assert settings('performance',ROWS[row])==expected

def test_default_launcher_and_precision_independent_switch():
 c=command('step0123',0,8000,8026,Path('/tmp/test'))
 assert c[c.index('--profile')+1]=='submission_performance_123'
 c=command('step0123',1,8000,8026,Path('/tmp/test'),disabled=[1,3],mode='precision')
 assert c[c.index('--profile')+1]=='submission_precision_2'
 with pytest.raises(ValueError,match='require'):command('step0123',0,8000,8026,Path('/tmp/test'),disabled=[1])

@pytest.mark.parametrize('mode',MODES)
def test_profile_registration_and_disable_three(mode):
 assert profile_settings(profile_name(mode,[1,2]))==settings(mode,[1,2])
 c=command('step0123',0,8000,8026,Path('/tmp/test'),disabled=[3],mode=mode)
 assert c[c.index('--profile')+1]==profile_name(mode,[1,2])
