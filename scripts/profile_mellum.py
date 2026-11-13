"""Bounded, acknowledged bs1 profiling of the validated Mellum reference backend."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from mellum_runtime_config import PROJECT, MODEL, apply_candidate, engine_options
from mellum_process_runner import run_stage
from analyze_mellum_profile import analyze_file, render_markdown


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare_inputs(lengths):
    paths = [PROJECT/'artifacts/results/mellum_validation/20261113T085534271528Z/inputs.json',
             PROJECT/'artifacts/results/mellum_validation/20261113T103154043872Z/inputs.json']
    records = {}
    for path in paths:
        data = json.loads(path.read_text())
        if data['model_config_sha256'] != hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest():
            raise ValueError('Fixed inputs and local model config differ')
        for record in data['records']:
            records[len(record['input_ids'][0])] = (record, path)
    cases = []
    for length in lengths:
        source_length = length if length in records else 8192
        if length not in (1024, 4096, 16384):
            raise ValueError('First profiling suite supports 1024,4096,16384 only')
        record, path = records[source_length]
        original = record['input_ids'][0]
        ids = original if length == source_length else original[:length-64]+original[-64:]
        if len(ids) != length:
            raise ValueError('Fixed input length mismatch')
        cases.append(dict(name=f'ctx_{length}', input_ids=ids, input_tokens=length,
                          source=str(path.relative_to(PROJECT)),
                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                          derivation='unchanged' if length == source_length else 'prefix slice with last 64 source tokens retained'))
    return cases


def wait_receipt(root, kind, sequence, timeout):
    path = root / f'rank_0_{kind}_{sequence:03d}.json'
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            record = json.loads(path.read_text())
            if record['kind'] != kind or record['sequence'] != sequence or record['rank'] != 0:
                raise ValueError('Profiler acknowledgement mismatch')
            return record
        time.sleep(0.1)
    raise TimeoutError(f'Profiler {kind} acknowledgement missing: {path}')


def worker(output):
    plan = json.loads((output/'plan.json').read_text())
    settings = apply_candidate(plan['gpu'], output, plan['detailed'])
    sys.path.insert(0, str(PROJECT/'ATOM'))
    result = dict(status='started', scope='bs1 diagnostic profile, not official performance or correctness acceptance',
                  settings=settings, cases=[])
    save(output/'worker_results.json', result)
    engine = None
    try:
        import torch
        import transformers
        result['environment'] = dict(torch=torch.__version__, hip=torch.version.hip,
                                     transformers=transformers.__version__, python=sys.executable)
        if transformers.__version__ != '5.2.0':
            raise RuntimeError('Use frozen /opt/venv interpreter')
        if not torch.cuda.is_available():
            result['status'] = 'blocked_no_gpu'
            save(output/'worker_results.json', result)
            return 2
        props = torch.cuda.get_device_properties(0)
        result['gpu'] = dict(name=props.name, total_memory_bytes=props.total_memory,
                             free_memory_bytes=torch.cuda.mem_get_info(0)[0],
                             arch=getattr(props, 'gcnArchName', 'unknown'))
        if torch.cuda.mem_get_info(0)[0] < 80*1024**3:
            raise RuntimeError('Require 80GiB free before loading reference backend')
        from atom.model_engine.arg_utils import EngineArgs
        from atom import SamplingParams
        budget = max(plan['timing_tokens'], plan['trace_tokens'], plan['warmup_tokens'])
        options = engine_options(max(c['input_tokens'] for c in plan['cases'])+budget, output/'traces')
        result['engine_options'] = options
        save(output/'worker_results.json', result)
        print('PROFILE phase engine_initialization', flush=True)
        engine = EngineArgs(**options).create_engine()
        audit = json.loads((output/'loading_audit.json').read_text())
        if audit['source_tensors'] != audit['successful_parts'] or audit['source_tensors'] != 5631 or audit['missing'] or audit['unexpected']:
            raise RuntimeError('Loading audit failed')
        result['loading_audit'] = audit

        def generate(case, tokens):
            start = time.perf_counter()
            rows = engine.generate([case['input_ids']], SamplingParams(temperature=0, max_tokens=tokens, ignore_eos=True))
            elapsed = time.perf_counter()-start
            if len(rows) != 1 or len(rows[0]['token_ids']) != tokens:
                raise RuntimeError('Unexpected completed request/output token count')
            row = rows[0]
            return dict(output_tokens=tokens, wall_seconds=elapsed,
                        request_ttft_seconds=row['ttft'], request_tpot_seconds=row['tpot'],
                        request_latency_seconds=row['latency'], token_ids=row['token_ids'],
                        output_tokens_per_wall_second=tokens/elapsed)

        for sequence, case in enumerate(plan['cases'], 1):
            print('PROFILE phase warmup', case['name'], flush=True)
            generate(case, plan['warmup_tokens'])
            timings = []
            for repeat in range(plan['timing_repeats']):
                print('PROFILE phase unprofiled_timing', case['name'], repeat, flush=True)
                timings.append(generate(case, plan['timing_tokens']))
            before = set((output/'traces').rglob('*.pt.trace.json.gz'))
            print('PROFILE phase start_wait', case['name'], flush=True)
            engine.start_profile()
            started = wait_receipt(output/'control', 'started', sequence, 60)
            print('PROFILE phase trace', case['name'], flush=True)
            observed = generate(case, plan['trace_tokens'])
            engine.stop_profile()
            print('PROFILE phase export_wait', case['name'], flush=True)
            stopped = wait_receipt(output/'control', 'stopped', sequence, 240)
            traces = sorted(set((output/'traces').rglob('*.pt.trace.json.gz'))-before)
            if len(traces) != 1:
                raise RuntimeError(f'Expected exactly one rank0 trace, got {len(traces)}')
            analysis = analyze_file(traces[0])
            if analysis['module_annotation_count'] == 0:
                raise RuntimeError('No Mellum module labels found')
            if analysis['gpu_kernel_count'] == 0:
                raise RuntimeError('No GPU kernels found; CPU-only trace is not accepted')
            phases = analysis['phase_cpu_ranges']
            if not phases.get('prefill', {}).get('calls') or not phases.get('decode', {}).get('calls'):
                raise RuntimeError('Trace is missing prefill or decode CPU ranges')
            save(output/(case['name']+'_analysis.json'), analysis)
            (output/(case['name']+'_analysis.md')).write_text(render_markdown(analysis))
            result['cases'].append(dict(name=case['name'], input_tokens=case['input_tokens'],
                unprofiled_timings=timings, profiled_request=observed,
                profiled_timing_is_not_benchmark=True, trace=str(traces[0]),
                trace_sha256=hashlib.sha256(traces[0].read_bytes()).hexdigest(),
                started=started, stopped=stopped,
                gpu_kernel_count=analysis['gpu_kernel_count'],
                module_attribution_fraction=analysis['module_attribution_fraction']))
            save(output/'worker_results.json', result)
            print('PROFILE case_complete', case['name'], flush=True)
        result['status'] = 'profile_collected'
        save(output/'worker_results.json', result)
        return 0
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc())
        save(output/'worker_results.json', result)
        print(result['traceback'], flush=True)
        return 1
    finally:
        if engine is not None:
            engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lengths', nargs='+', type=int, default=[1024,4096,16384])
    parser.add_argument('--trace-tokens', type=int, default=16)
    parser.add_argument('--timing-tokens', type=int, default=64)
    parser.add_argument('--timing-repeats', type=int, default=2)
    parser.add_argument('--warmup-tokens', type=int, default=8)
    parser.add_argument('--stage-timeout', type=float, default=900)
    parser.add_argument('--detailed', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_output:
        return worker(args.worker_output)
    if not 2 <= args.trace_tokens <= 64 or not 1 <= args.timing_tokens <= 256 or not 1 <= args.warmup_tokens <= 64 or args.timing_repeats < 1 or args.stage_timeout <= 0:
        parser.error('Invalid profiling budget')
    if len(set(args.lengths)) != len(args.lengths):
        parser.error('Duplicate lengths')
    output = Path('/workspace/results/mellum_profile')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    cases = prepare_inputs(args.lengths)
    plan = dict(gpu=args.gpu, cases=cases, trace_tokens=args.trace_tokens,
                timing_tokens=args.timing_tokens, timing_repeats=args.timing_repeats,
                warmup_tokens=args.warmup_tokens, detailed=args.detailed,
                model_config_sha256=hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest())
    save(output/'plan.json', plan)
    fingerprints = {}
    for p in sorted((PROJECT/'scripts').glob('*profile*.py')) + [Path(__file__), PROJECT/'scripts/mellum_runtime_config.py', PROJECT/'ATOM/atom/model_engine/model_runner.py', PROJECT/'ATOM/atom/model_config/mellum_profile.py']:
        fingerprints[str(p.relative_to(PROJECT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(status='prepared', output=str(output), source_sha256=fingerprints,
                  git_head=subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
                  git_status=subprocess.check_output(['git','-C',str(PROJECT),'status','--short'],text=True))
    save(output/'summary.json', report)
    if args.prepare_only:
        print('RESULT prepared output', output, flush=True)
        return 0
    try:
        code, _ = run_stage('atom_profile', ['/opt/venv/bin/python',str(Path(__file__).resolve()),'--worker-output',str(output)],
                            output/'atom.log', timeout=args.stage_timeout, allowed_exit_codes=(0,2), with_code=True)
        observed = json.loads((output/'worker_results.json').read_text())
        report['status'] = observed['status']
        report['cases'] = [{k:r[k] for k in ['name','input_tokens','gpu_kernel_count','module_attribution_fraction']} for r in observed.get('cases',[])]
        save(output/'summary.json', report)
        print('RESULT', report['status'], 'output', output, flush=True)
        return code
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc())
        save(output/'summary.json', report)
        print(report['traceback'], flush=True)
        print('RESULT failed output', output, flush=True)
        return 1


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    raise SystemExit(main())
