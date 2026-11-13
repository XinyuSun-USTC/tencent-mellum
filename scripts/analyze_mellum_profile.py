"""Analyze Kineto Chrome traces without PyTorch or GPU access."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def external(event):
    for key, value in event.get('args', {}).items():
        if key.lower().replace(' ', '_') == 'external_id':
            return str(value)
    return None


def gpu_kind(event):
    category = event.get('cat', '').lower()
    if category == 'kernel' or category == 'gpu_kernel':
        return 'kernel'
    if 'gpu_memcpy' in category or 'gpu_memset' in category:
        return 'copy_or_memset'
    return None


def union_duration(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    start, end = intervals[0]
    total = 0.0
    for left, right in intervals[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def analyze_events(events):
    spans = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    gpu = [e for e in spans if gpu_kind(e)]
    kernels = [e for e in gpu if gpu_kind(e) == 'kernel']
    # ROCm emits GPU mirrors of user ranges with identical names/external IDs.
    # They are not CPU launch ranges and must not overwrite CPU ownership.
    cpu = [e for e in spans if not gpu_kind(e) and e.get('cat') != 'gpu_user_annotation']
    annotations = defaultdict(list)
    phases = defaultdict(lambda: {'calls': 0, 'cpu_range_us': 0.0})
    for e in cpu:
        name = e.get('name', '')
        phase = 'prefill' if name.startswith('prefill[') else 'decode' if name.startswith(('eager_decode[', 'cudagraph_decode[')) else None
        if name.startswith('mellum/') or phase:
            annotations[(e.get('pid'), e.get('tid'))].append((e['ts'], e['ts'] + e['dur'], name, phase))
        if phase:
            phases[phase]['calls'] += 1
            phases[phase]['cpu_range_us'] += e['dur']
    threads = defaultdict(list)
    for e in cpu:
        if external(e) is not None:
            threads[(e.get('pid'), e.get('tid'))].append(e)
    owners = {}
    correlations = {}
    instances = {}
    for thread, points in threads.items():
        ranges = sorted(annotations[thread], key=lambda x: (x[0], -x[1]))
        active = []
        index = 0
        for point in sorted(points, key=lambda e: e['ts']):
            ts = point['ts']
            while index < len(ranges) and ranges[index][0] <= ts:
                active.append(ranges[index]); index += 1
            active = [r for r in active if r[1] >= ts]
            names = [r[2] for r in active if r[2].startswith('mellum/')]
            containing = [r for r in active if r[3]]
            phase = min(containing, key=lambda r: r[1] - r[0])[3] if containing else 'unattributed'
            module_ranges = [r for r in active if r[2].startswith('mellum/')]
            deepest = min(module_ranges, key=lambda r: r[1] - r[0])[2].split('/')[1] if module_ranges else 'unattributed'
            layer_names = [n.split('/')[-1] for n in names if n.split('/')[-1].startswith('layer_')]
            layer = layer_names[-1] if layer_names else 'unattributed'
            phase_name = min(containing, key=lambda r:r[1]-r[0])[2] if containing else 'unattributed'
            owner = (phase, deepest, {name.split('/')[1] for name in names}, point.get('name', '<unnamed>'), layer, phase_name)
            # Runtime launches can share External id with an enclosing CPU op.
            # Prefer the semantic CPU op instead of overwriting it with hipLaunchKernel.
            if external(point) not in owners or point.get('cat') == 'cpu_op':
                owners[external(point)] = owner
            corr = point.get('args', {}).get('correlation')
            if corr is not None:
                correlations[str(corr)] = owner
    by_kernel = defaultdict(lambda: {'calls': 0, 'summed_us': 0.0})
    by_operator = defaultdict(lambda: {'calls': 0, 'summed_cpu_event_us': 0.0, 'self_cpu_event_us': 0.0})
    exclusive = defaultdict(float)
    inclusive = defaultdict(float)
    phase_gpu = defaultdict(float)
    phase_modules = defaultdict(lambda: defaultdict(float))
    launch_operators = defaultdict(lambda: {'calls': 0, 'summed_kernel_us': 0.0})
    layer_modules = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    copy_stats = defaultdict(lambda: {'calls': 0, 'summed_us': 0.0})
    attributed = 0.0
    for e in kernels:
        duration = float(e['dur'])
        entry = by_kernel[e.get('name', '<unnamed>')]
        entry['calls'] += 1; entry['summed_us'] += duration
        owner = owners.get(external(e))
        if owner is None:
            owner = correlations.get(str(e.get('args', {}).get('correlation')))
        phase, module, parents, operator, layer, instance = owner or ('unattributed', 'unattributed', set(), 'unattributed', 'unattributed', 'unattributed')
        exclusive[module] += duration
        phase_gpu[phase] += duration
        phase_modules[phase][module] += duration
        layer_modules[phase][layer][module] += duration
        inst = instances.setdefault(instance, dict(kernel_count=0, summed_kernel_us=0.0))
        inst['kernel_count'] += 1; inst['summed_kernel_us'] += duration
        launch_operators[(phase, module, operator)]['calls'] += 1
        launch_operators[(phase, module, operator)]['summed_kernel_us'] += duration
        for parent in parents:
            inclusive[parent] += duration
        if module != 'unattributed':
            attributed += duration
    op_threads = defaultdict(list)
    for e in cpu:
        if e.get('cat') == 'cpu_op':
            op_threads[(e.get('pid'), e.get('tid'))].append(e)
    for points in op_threads.values():
        stack = []
        children = defaultdict(list)
        ordered = sorted(points, key=lambda e:(e['ts'], -e['dur']))
        for i,e in enumerate(ordered):
            while stack and ordered[stack[-1]]['ts']+ordered[stack[-1]]['dur'] <= e['ts']:
                stack.pop()
            if stack:
                parent = ordered[stack[-1]]
                children[stack[-1]].append((e['ts'], min(e['ts']+e['dur'],parent['ts']+parent['dur'])))
            stack.append(i)
        for i,e in enumerate(ordered):
            row = by_operator[e.get('name','<unnamed>')]
            row['calls'] += 1
            row['summed_cpu_event_us'] += float(e['dur'])
            row['self_cpu_event_us'] += max(0, float(e['dur'])-union_duration(children[i]))
    for e in gpu:
        if gpu_kind(e) != 'kernel':
            row = copy_stats[e.get('name','<unnamed>')]
            row['calls'] += 1; row['summed_us'] += float(e['dur'])
    intervals = [(e['ts'], e['ts'] + e['dur']) for e in kernels]
    summed = sum(float(e['dur']) for e in kernels)
    busy = union_duration(intervals)
    span = max((end for _, end in intervals), default=0) - min((start for start, _ in intervals), default=0)
    return dict(module_annotation_count=sum(e.get('name','').startswith('mellum/') for e in cpu), gpu_kernel_count=len(kernels), gpu_copy_or_memset_count=len(gpu)-len(kernels),
                summed_gpu_kernel_us=summed, kernel_active_union_us=busy,
                kernel_span_us=span, kernel_gaps_within_span_us=max(0, span-busy),
                module_attribution_fraction=attributed/summed if summed else 0,
                exclusive_module_kernel_us=dict(exclusive), inclusive_module_kernel_us=dict(inclusive),
                phase_kernel_us=dict(phase_gpu), phase_cpu_ranges=dict(phases),
                phase_exclusive_module_kernel_us={k:dict(v) for k,v in phase_modules.items()},
                phase_layer_module_kernel_us={phase:{layer:dict(modules) for layer,modules in layers.items()} for phase,layers in layer_modules.items()},
                phase_instances=instances, gpu_copy_stats=dict(copy_stats),
                top_launch_operators=sorted((dict(phase=k[0],module=k[1],name=k[2],**v) for k,v in launch_operators.items()),key=lambda r:-r['summed_kernel_us'])[:40],
                top_kernels=sorted((dict(name=k, **v) for k,v in by_kernel.items()), key=lambda r:-r['summed_us'])[:25],
                top_cpu_operators=sorted((dict(name=k, **v) for k,v in by_operator.items()), key=lambda r:-r['summed_cpu_event_us'])[:25],
                top_self_cpu_operators=sorted((dict(name=k, **v) for k,v in by_operator.items()), key=lambda r:-r['self_cpu_event_us'])[:25],
                interpretation=[
                    'GPU attribution uses external-id/correlation to CPU launch ranges, not GPU timestamp containment.',
                    'Inclusive module totals overlap; kernel sums can overlap across streams and are not wall time.',
                    'CPU self duration subtracts nested CPU-op ranges but can still include waits; CPU and GPU time are not additive. Kernel gaps alone do not prove CPU bottlenecks.',
                    'Unmatched correlations remain unattributed; low coverage requires inspecting the actual trace schema.',
                    'These traces are diagnostics, not official A/B/C scores.'])


def analyze_file(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        data = json.load(stream)
    return analyze_events(data['traceEvents'])


def render_markdown(result):
    lines = ['# Profile摘要', '',
             '仅为诊断trace，不是正式A/B/C性能分数。单位为毫秒。', '',
             f"GPU kernel数：{result['gpu_kernel_count']}；模块归因覆盖：{result['module_attribution_fraction']:.1%}。", '',
             '| 项目 | ms |', '| --- | --- |']
    for key in ['summed_gpu_kernel_us', 'kernel_active_union_us', 'kernel_span_us', 'kernel_gaps_within_span_us']:
        lines.append(f"| {key} | {result[key]/1000:.3f} |")
    lines += ['', 'kernel累计时长可重叠；空隙不自动等于CPU瓶颈。', '',
              '## 按最内层模块归因', '', '| 模块 | 累计kernel ms |', '| --- | --- |']
    for name, duration in sorted(result['exclusive_module_kernel_us'].items(), key=lambda r:-r[1]):
        lines.append(f'| {name} | {duration/1000:.3f} |')
    lines += ['', '## 分阶段模块kernel累计ms', '', '| 阶段 | 模块 | 累计ms |', '| --- | --- | --- |']
    for phase, modules in result['phase_exclusive_module_kernel_us'].items():
        for name,duration in sorted(modules.items(),key=lambda r:-r[1]):
            lines.append(f'| {phase} | {name} | {duration/1000:.3f} |')
    lines += ['', '## 热点kernel', '', '| 名称 | 次数 | 累计ms |', '| --- | --- | --- |']
    for row in result['top_kernels'][:15]:
        name = row['name'].replace('|', '/').replace('\n', ' ')[:180]
        lines.append(f"| {name} | {row['calls']} | {row['summed_us']/1000:.3f} |")
    lines += ['', '详细算子、phase和重叠说明见同名JSON；inclusive模块总量包含父子重复，不相加作为总耗时。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze_file(args.trace)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    args.output.with_suffix('.md').write_text(render_markdown(result))
    print('GPU kernels', result['gpu_kernel_count'], 'attribution', result['module_attribution_fraction'])
    return 0 if result['gpu_kernel_count'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
