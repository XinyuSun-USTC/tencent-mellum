"""Dataset-only compatibility adapter; run the installed EvalScope CLI unchanged.

No vendor files, model arithmetic, timing metrics or arrival scheduler are patched.
"""
import argparse,asyncio,hashlib,itertools,json,os,sys
from pathlib import Path


def install_dataset():
    from evalscope.perf.plugin.datasets.base import DatasetPluginBase
    from evalscope.perf.plugin.registry import register_dataset
    @register_dataset('mellum_assessment_jsonl')
    class AssessmentJSONL(DatasetPluginBase):
        def build_messages(self):
            with open(self.query_parameters.dataset_path) as stream:
                for line in stream:
                    if not line.strip():continue
                    row=json.loads(line)
                    if not isinstance(row.get('prompt'),str):raise ValueError('Assessment row has no prompt')
                    # Native chat wrapping and native min/max filtering, including
                    # full newline content. Native benchmark.py performs cycling.
                    messages=self.prepare_messages(row['prompt'])
                    if messages is not None:yield messages
    return AssessmentJSONL


def prepare(output):
    from importlib.metadata import version
    from evalscope.perf.arguments import Arguments
    from evalscope.perf.plugin import ApiRegistry,DatasetRegistry
    from evalscope.perf.plugin.datasets.utils import tokenize_chat_messages
    from mellum_runtime_config import MODEL
    from mellum_perf_protocol import DATA,sha,load_rows
    install_dataset();output.mkdir(parents=True,exist_ok=True)
    report={'evalscope':version('evalscope'),'transformers':version('transformers'),'scenarios':{},'gpu_started':False}
    for scenario,count in [('A',512),('C',200)]:
        source=DATA/f'scenario_{scenario.lower()}_prompts.jsonl';rows=load_rows(source)
        args=Arguments(model=str(MODEL),tokenizer_path=str(MODEL),dataset='mellum_assessment_jsonl',dataset_path=str(source),
                       number=count,parallel=32,max_tokens=256,min_tokens=256,stream=True,
                       min_prompt_length=512 if scenario=='A' else 0,max_prompt_length=4096 if scenario=='A' else 131072,
                       extra_args={'ignore_eos':True,'temperature':0.0})
        plugin=DatasetRegistry.get_class(args.dataset)(args);api=ApiRegistry.get_class('openai')(args)
        messages=list(plugin.build_messages());by_content={r['prompt']:r for r in rows}
        if len(by_content)!=len(rows):raise ValueError('Duplicate prompt contents need explicit ID mapping')
        eligible=[]
        for item in messages:
            content=item[0]['content'];row=by_content[content]
            query=api.build_request(item)
            if query['messages'][0]['content']!=content:raise ValueError('Client changed source prompt')
            eligible.append({'id':row['id'],'prompt_sha256':hashlib.sha256(content.encode()).hexdigest(),
                             'tokens':len(tokenize_chat_messages(plugin.tokenizer,item))})
        if not eligible:raise ValueError('No eligible requests')
        planned=list(itertools.islice(itertools.cycle(eligible),count))
        from evalscope.perf.benchmark import get_requests
        async def generated():
            return [request async for request,warmup in get_requests(args,api)]
        actual=asyncio.run(generated())
        actual_hashes=[hashlib.sha256(q['messages'][0]['content'].encode()).hexdigest() for q in actual]
        if actual_hashes!=[r['prompt_sha256'] for r in planned]:raise ValueError('Native request generation/cycling differs from prepared schedule')
        report['scenarios'][scenario]={'native_request_generation_verified':True,'source':str(source),'source_sha256':sha(source),'source_records':len(rows),
            'native_eligible_records':len(eligible),'filtered_ids':[r['id'] for r in rows if r['id'] not in {x['id'] for x in eligible}],
            'planned_requests':planned,'number':count,'cycling_requests':max(0,count-len(eligible)),
            'native_filter':'chat-template tokens in [512,4096]' if scenario=='A' else 'native default bounds; no A filter'}
    from evalscope.perf.benchmark import get_requests
    from evalscope.utils.model_utils import seed_everything
    bargs=Arguments(model=str(MODEL),tokenizer_path=str(MODEL),dataset='random',number=50,parallel=1,
                    min_prompt_length=1024,max_prompt_length=1024,max_tokens=256,min_tokens=256,stream=True,
                    seed=42,extra_args={'ignore_eos':True,'temperature':0.0})
    seed_everything(42)
    bapi=ApiRegistry.get_class('openai')(bargs)
    async def b_requests():return [request async for request,warmup in get_requests(bargs,bapi)]
    bqueries=asyncio.run(b_requests())
    lengths=[len(tokenize_chat_messages(bapi.tokenizer,q['messages'])) for q in bqueries]
    if len(lengths)!=50 or any(n!=1024 for n in lengths):raise ValueError('Native B random dataset does not produce 50 x 1024 tokens')
    report['scenarios']['B']={'native_request_generation_verified':True,'number':50,'seed':42,'input_tokens':lengths,
                              'note':'CPU validation of native random generation; GPU server usage rechecked during benchmark'}
    import evalscope
    package=Path(evalscope.__file__).parent
    report['client_source_sha256']={str(p):sha(p) for p in sorted((package/'perf').rglob('*.py'))}
    (output/'adapter_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT adapter_prepared output',output)


def main():
    os.environ.update(HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if len(sys.argv)>1 and sys.argv[1]=='--prepare':
        p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);a=p.parse_args();prepare(a.prepare);return
    install_dataset()
    from evalscope.cli.cli import run_cmd
    run_cmd()
if __name__=='__main__':main()
