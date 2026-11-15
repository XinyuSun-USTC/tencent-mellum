"""Fake-worker tests for the bounded MoE optimization state machine."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mellum_moe_optimization as campaign
from run_mellum_batch_campaign import write


def group(value):
    return {'records':32,'token_match_fraction':value,'exact_sequence_fraction':value,
            'matching_positions':int(512*value),'compared_positions':512,'exact_sequences':int(32*value)}


def result(value=.98,cohorts=('S32','L32')):
    return {'status':'complete','stable':True,'batch_verified':True,
            'cohorts':{c:{'groups':{k:group(value) for k in ('all','below_1024','at_least_1024')},
                          'wall_seconds':[1,1],'repeat_files':[c+'_r0.json']} for c in cohorts}}


def test_accuracy_selection_and_performance_tradeoff_are_predefined():
    assert campaign.screen_ok(result(.965),result(.98))
    assert not campaign.screen_ok(result(.95),result(.98))
    assert campaign.d200_ok(result(.977,('D200',)),result(.98,('D200',)))
    assert not campaign.d200_ok(result(.97,('D200',)),result(.98,('D200',)))
    assert not campaign.screen_ok({**result(),'stable':False},result())


def test_abba_order_and_analysis_uses_four_runs_not_vacuous_stability(tmp_path):
    data={'S32':[],'L32':[]};cohorts,modes=campaign.abba_plan(data,'shared_routing')
    assert len(cohorts)==16
    assert [modes[k] for k in cohorts if k.startswith('S32')]==['cpd','shared_routing','shared_routing','cpd']*2
    r=result(cohorts=cohorts)
    for key,row in r['cohorts'].items():
        row['wall_seconds']=[2 if modes[key]=='cpd' else 1]
        write(tmp_path/(key+'_r0.json'),{'records':[{'actual_ids':[1,2]}]})
    report=campaign.analyze_timing(tmp_path,r)
    assert report['geometric_mean_speedup']==2 and report['performance_eligible']
    key=next(k for k in cohorts if modes[k]=='shared_routing')
    write(tmp_path/(key+'_r0.json'),{'records':[{'actual_ids':[1,3]}]})
    assert not campaign.analyze_timing(tmp_path,r)['performance_eligible']


def fake_data():return {c:[{'name':str(i)} for i in range(32)] for c in ('K4','S32','L32','D200')}


def test_complete_twelve_worker_path(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'historical_match',lambda *a:{'checked':True})
    def fake(stage,command,log,timeout):
        log.write_text('ok');directory=log.parent;plan=json.loads((directory/'plan.json').read_text())
        if stage=='micro':
            r={'status':'complete','variants':{'indexed_decode':{'valid':True}},'records':[{'name':'decode_32',
               'median_ms':{'cpd':2,'indexed_decode':1},'peak_additional_allocated_bytes':{'cpd':100,'indexed_decode':10}}]}
        elif stage.startswith('HTTP'):
            write(directory/'http/summary.json',{'status':'http_smoke_completed','review_required':False});return
        else:
            r=result(cohorts=plan['cohorts'])
            if stage.startswith('profile'):r['trace']='trace.gz'
            if stage.startswith('timing'):
                for key,row in r['cohorts'].items():
                    row['wall_seconds']=[2 if plan['cohort_variants'][key]=='cpd' else 1]
                    write(directory/(key+'_r0.json'),{'records':[{'actual_ids':[1]}]})
        write(directory/'results.json',r)
    manifest={'gpu':0,'provenance':{},'micro_inputs':[],'jobs':{},'max_workers':12}
    r=campaign.execute(tmp_path,manifest,fake_data(),NS(budget_minutes=2,job_timeout=30),fake)
    assert len(r['jobs'])==12 and r['status']=='complete'
    assert r['decisions']['best'] in campaign.VARIANTS and r['decisions']['service_passed']


def test_micro_compile_failure_does_not_skip_independent_shared_routing(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'historical_match',lambda *a:{})
    def fake(stage,command,log,timeout):
        log.write_text('compiler unsupported')
        if stage=='micro':raise RuntimeError('compile failure')
        plan=json.loads((log.parent/'plan.json').read_text())
        r=result(.85 if stage=='screen_CPD' else .98,plan['cohorts'])
        write(log.parent/'results.json',r)
    manifest={'gpu':0,'provenance':{},'micro_inputs':[],'jobs':{},'max_workers':12}
    r=campaign.execute(tmp_path,manifest,fake_data(),NS(budget_minutes=2,job_timeout=30),fake)
    assert 'screen_shared_routing' in r['jobs'] and 'screen_indexed_decode' not in r['jobs']
    assert not any(n.startswith('D200') for n in r['jobs'])
    assert r['status']=='complete_with_unresolved'
