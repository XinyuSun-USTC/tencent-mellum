import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_precision_c_completion import aggregate

def test_no_median_from_one_run_or_failed_run():
 good={'metrics':{'complete_without_failures':True,'tpot_p99_s':2,'ttft_p99_s':4}}
 assert aggregate([good])['three_run_median'] is None
 assert aggregate([good,good,{'metrics':{'complete_without_failures':False}}])['three_run_median'] is None

def test_independent_run_median_not_pooled_percentile():
 runs=[{'metrics':{'complete_without_failures':True,'tpot_p99_s':x,'ttft_p99_s':x*2}} for x in [9,1,3]]
 assert aggregate(runs)['three_run_median']=={'tpot_p99_s':3,'ttft_p99_s':6}
