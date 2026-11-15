"""Audit actual pinned EvalScope code and synthetic SSE without starting inference."""
import argparse,asyncio,hashlib,inspect,itertools,json,os,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from importlib.metadata import version
from mellum_runtime_config import MODEL
from mellum_perf_protocol import prepare,sha,chat_ids,load_rows

EXPECTED='1.15.0'


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def chat_content(query):
    if not isinstance(query,dict):return None
    messages=query.get('messages')
    if not isinstance(messages,list) or not messages:return None
    if len(messages)==1 and messages[0].get('role')=='user' and isinstance(messages[0].get('content'),str):
        return messages[0]['content']
    return None


def query_summary(query,expected=None):
    content=chat_content(query)
    return {'keys':sorted(query) if isinstance(query,dict) else [],'valid_chat_messages':content is not None,
            'source_prompt_preserved':content==expected if expected is not None else None,
            'stream':query.get('stream') if isinstance(query,dict) else None,
            'max_tokens':query.get('max_tokens') if isinstance(query,dict) else None,
            'ignore_eos':query.get('ignore_eos') if isinstance(query,dict) else None,
            'temperature':query.get('temperature') if isinstance(query,dict) else None}


def judge_stream(value):
    return {'success':bool(value.get('success')),'reasoning_text_preserved':'thinking' in value.get('generated_text',''),
            'usage_counts_preserved':value.get('prompt_tokens')==7 and value.get('completion_tokens')==3,
            'first_observable_output_timed':.03<=value.get('first_chunk_latency',0)<.20,
            'last_content_observed':value.get('query_latency',0)>=.30,
            'note':'Synthetic local SSE, three output chunks; not a GPU or model correctness test'}


def stream_probe(api,args):
    import aiohttp
    captured=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*unused):pass
        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            def send(delta,usage=None):
                item={'id':'synthetic','object':'chat.completion.chunk','created':0,'model':str(MODEL),
                      'choices':[{'index':0,'delta':delta,'finish_reason':None}] if usage is None else [],'usage':usage}
                self.wfile.write(('data: '+json.dumps(item)+'\n\n').encode());self.wfile.flush()
            send({'role':'assistant'});time.sleep(.05)
            send({'reasoning_content':'thinking'});time.sleep(.05)
            send({'reasoning_content':' continues'});time.sleep(.25)
            send({'content':'answer'})
            send({}, {'prompt_tokens':7,'completion_tokens':3,'total_tokens':10})
            self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush();self.close_connection=True
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        body=api.build_request([{'role':'user','content':'hello'}])
        async def call():
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5),trust_env=False) as session:
                result=await api.process_request(session,f'http://127.0.0.1:{server.server_port}/v1/chat/completions',{},body)
                if hasattr(result,'finalize'):result.finalize(api)
                fields=('success','error','prompt_tokens','completion_tokens','generated_text','first_chunk_latency','query_latency','time_per_output_token')
                return {k:getattr(result,k,None) for k in fields}
        observed=asyncio.run(call())
        return {'observed':observed,'checks':judge_stream(observed),'request':captured}
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    os.environ.update(HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if Path(sys.prefix).resolve()!=Path('/workspace/venvs/evalscope-perf').resolve():raise ValueError('Run in dedicated client environment')
    if version('evalscope')!=EXPECTED or version('transformers')!='5.10.1':raise ValueError('Pinned client version mismatch')
    report={'status':'audit_started','python':sys.executable,'prefix':sys.prefix,
            'versions':{n:version(n) for n in ['evalscope','transformers','aiohttp','pydantic','numpy']},
            'gpu_inference_started':False,'issues':[],'protocol_ready_for_full_gpu':False}
    try:
        import evalscope
        from transformers import AutoTokenizer
        package=Path(evalscope.__file__).parent;report['package']=str(package)
        sources={}
        for file in sorted((package/'perf').rglob('*.py')):
            rel=file.relative_to(package);sources[str(rel)]={'path':str(file),'sha256':sha(file)}
            if any(part in str(rel) for part in ('arguments.py','plugin/api/','plugin/datasets/','core/strateg','http_client.py','benchmark_util.py')):
                target=out/'installed_source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
        report['source_sha256']=sources
        result=subprocess.run([str(Path(sys.prefix)/'bin/evalscope'),'perf','--help'],capture_output=True,text=True,timeout=120)
        (out/'perf_help.txt').write_text(result.stdout+result.stderr)
        report['help_exit_code']=result.returncode
        report['required_flags']={flag:flag in result.stdout for flag in ['--open-loop','--rate','--total-timeout','--dataset-path','--stream','--min-tokens']}
        if result.returncode or not all(report['required_flags'].values()):report['issues'].append('Installed CLI does not advertise all assessment flags')
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        protocol=prepare(out,tokenizer);write(out/'protocol_manifest.json',protocol)
        report['dataset_summary']={k:{n:v for n,v in row.items() if n!='rows'} for k,row in protocol['datasets'].items()}
        from evalscope.perf.arguments import Arguments
        from evalscope.perf.plugin import ApiRegistry,DatasetRegistry
        def parameters(**extra):
            return Arguments(model=str(MODEL),tokenizer_path=str(MODEL),number=3,parallel=1,
                max_tokens=3,min_tokens=3,stream=True,extra_args={'ignore_eos':True,'temperature':0.0},**extra)
        try:
            literal_path=str(Path('/data/datasets/scenario_a_prompts.jsonl'))
            cls=DatasetRegistry.get_class(literal_path)
            literal_params=parameters(dataset=literal_path,min_prompt_length=512,max_prompt_length=4096)
            literal_plugin=cls(literal_params)
            item=next(iter(literal_plugin.build_messages()))
            query=ApiRegistry.get_class('openai')(literal_params).build_request(item)
            write(out/'request_examples/literal_A.json',query)
            report['literal_A_dataset_registered']=True
            report['literal_A_probe']=query_summary(query)
            report['literal_A_dataset_class']=str(cls)
        except Exception as exc:
            report['literal_A_dataset_registered']=False;report['literal_A_dataset_error']=str(exc)
        if not report['literal_A_dataset_registered']:report['issues'].append('A literal --dataset filepath is not a registered dataset in this release')
        probes={}
        for scenario in ('A','C'):
            for variant,path in [('original_jsonl',protocol['datasets'][scenario]['source']),('proposed_messages',protocol['datasets'][scenario]['proposed_messages_file'])]:
                key=scenario+'_'+variant
                try:
                    params=parameters(dataset='line_by_line',dataset_path=path,
                                      min_prompt_length=512 if scenario=='A' else 0,max_prompt_length=4096 if scenario=='A' else 32768)
                    api=ApiRegistry.get_class('openai')(params);plugin=DatasetRegistry.get_class('line_by_line')(params)
                    items=list(itertools.islice(plugin.build_messages(),3));expected=load_rows(protocol['datasets'][scenario]['source'])[:3]
                    outputs=[]
                    for i,item in enumerate(items):
                        query=api.build_request(item);write(out/'request_examples'/f'{key}_{i}.json',query)
                        outputs.append(query_summary(query,expected[i]['prompt']))
                    probes[key]={'sample_count':len(items),'queries':outputs,'apply_chat_template':getattr(params,'apply_chat_template',None)}
                    if variant=='original_jsonl' and not all(q['source_prompt_preserved'] for q in outputs):
                        report['issues'].append(scenario+' original JSONL is not transmitted as its prompt field')
                except Exception:probes[key]={'error':traceback.format_exc()};report['issues'].append(key+' probe failed')
        try:
            params=parameters(dataset='random',min_prompt_length=1024,max_prompt_length=1024)
            api=ApiRegistry.get_class('openai')(params);plugin=DatasetRegistry.get_class('random')(params)
            result=[]
            for i,item in enumerate(itertools.islice(plugin.build_messages(),3)):
                query=api.build_request(item);write(out/'request_examples'/f'B_random_{i}.json',query)
                content=chat_content(query);result.append({'query':query_summary(query),'chat_tokens':len(chat_ids(tokenizer,query['messages'])) if content is not None else None})
            probes['B_random']={'queries':result,'apply_chat_template':getattr(params,'apply_chat_template',None)}
        except Exception:probes['B_random']={'error':traceback.format_exc()};report['issues'].append('B random probe failed')
        report['request_probes']=probes
        for scenario in ('A','C'):
            row=protocol['datasets'][scenario]
            if scenario=='A' and row['A_range_valid_chat']!=row['records']:
                report['issues'].append('A has prompts outside length bounds after chat rendering; filter/cycle behavior must be explicit')
        try:
            params=parameters(dataset='random');api=ApiRegistry.get_class('openai')(params)
            report['stream_probe']=stream_probe(api,params)
            checks=report['stream_probe']['checks']
            if not all(checks[k] for k in ['success','reasoning_text_preserved','usage_counts_preserved','first_observable_output_timed','last_content_observed']):
                report['issues'].append('Installed SSE parser timing/counting failed the reasoning-stream probe')
        except Exception:report['stream_probe']={'error':traceback.format_exc()};report['issues'].append('Synthetic SSE probe failed')
        hits=[]
        for rel,item in sources.items():
            if not any(part in rel for part in ['arguments','http_client','strateg','pipeline']):continue
            for line,text in enumerate(Path(item['path']).read_text().splitlines(),1):
                if any(key in text for key in ['total_timeout','ClientTimeout','exponential','poisson','open_loop']):
                    hits.append({'file':rel,'line':line,'text':text.strip()[:240]})
        write(out/'scheduling_timeout_source_hits.json',hits)
        report['status']='audit_complete'
        report['issues'].append('Pinned release is a local client choice; assessment did not specify evaluator version. Full GPU command translation awaits source review.')
    except Exception:
        report.update(status='audit_failed',traceback=traceback.format_exc());raise
    finally:
        report['torch_cuda_initialized']=bool('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized())
        write(out/'summary.json',report)
    print('RESULT perf_client_audit_complete output',out)
if __name__=='__main__':main()
