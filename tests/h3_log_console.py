#!/usr/bin/env python3
"""User-facing H3 log summaries; delegates sampler interpretation to h3_perf_compare.

No GPU, ComfyUI, third-party packages or network needed. The source logs remain
untouched. Stores portable JSON and text reports separately.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import h3_perf_compare as perf

H3_EVENT = re.compile(r'\[H3PERF\]\s*(\{.*\})')
MODEL_PREP = re.compile(r'Model\s+(MiniMaxH3(?:TEModel_|VideoVAE|AudioVAE)?)\s+prepared for dynamic VRAM loading\.\s+(\d+)MB Staged')
PROMPT_TIME = re.compile(r'Prompt executed in\s+([\d.]+)\s+seconds')
COMPILER_BREAKS = re.compile(r'Comfy model compiler graph breaks:\s+(\d+)')
GENERATED = re.compile(r'Generated (\d+) images?\b|Generated an image\b', re.IGNORECASE)
BAR = '=' * 76


def _read_log(path):
    return Path(path).read_text(encoding='utf-8-sig', errors='replace')


def legacy_scan(path):
    raw = _read_log(path)
    prepared = [{'model': n, 'staged_mb': int(mb)} for n, mb in MODEL_PREP.findall(raw)]
    times = [float(n) for n in PROMPT_TIME.findall(raw)]
    graph_breaks = [int(n) for n in COMPILER_BREAKS.findall(raw)]
    outputs = [int(n) if n else 1 for n in GENERATED.findall(raw)]
    return {'preparations': prepared, 'prompt_times_s': times,
            'compiler_graph_breaks': graph_breaks, 'generated_output_counts': outputs}


def split_runs(runs):
    """Preserve the order and sampler stages of each prompt, without inferring video count."""
    grouped = []
    for run in runs:
        pid = str(run['start'].get('prompt_id'))
        if not grouped or grouped[-1]['prompt_id'] != pid:
            grouped.append({'prompt_id': pid, 'samplers': []})
        grouped[-1]['samplers'].append(perf.describe(run))
    return grouped


def _details(first, second, third=None):
    report = perf.compare(first, second)
    if third is not None:
        final = perf.describe(third)
        report['one_frame_followup'] = final
        common = ('latent_shapes', 'ref_blocks', 'keyframes', 'cfg', 'sampler',
                  'scheduler', 'previews', 'disable_cfg1_optimization',
                  'weights_resident_mb_per_step')
        original = report['image_to_video']
        report['long_i2v_sampler_differences'] = {
            key: {'first_two_stage_run': original[key], 'one_frame_followup': final[key]}
            for key in common if original[key] != final[key]
        }
    return report


def build_report(mode, logs):
    required = {'scan': 1, 'two-in-one': 1, 'three-in-one': 1,
                'two-files': 2, 'three-files': 3, 'extract': 1}
    if mode not in required or len(logs) != required[mode]:
        raise ValueError(f'{mode}: expected {required.get(mode, "?")} log file(s), received {len(logs)}')
    for p in logs:
        if not Path(p).is_file():
            raise ValueError(f'Log file not found: {p}')
    result = {'mode': mode, 'sources': [str(Path(p).resolve()) for p in logs],
              'generated_at': datetime.now().astimezone().isoformat(timespec='seconds')}
    if mode == 'scan':
        runs = perf.parse_trace(logs[0])
        result['trace_found'] = bool(runs)
        result['prompts'] = split_runs(runs)
        result['legacy'] = legacy_scan(logs[0])
        return result
    if mode == 'extract':
        lines = []
        for raw in _read_log(logs[0]).splitlines():
            if '[H3PERF]' not in raw:
                continue
            found = H3_EVENT.search(raw)
            if not found:
                raise ValueError('Found a truncated or malformed [H3PERF] record')
            try:
                lines.append(json.loads(found.group(1)))
            except json.JSONDecodeError:
                raise ValueError('Found a truncated or malformed [H3PERF] record') from None
        if not lines:
            raise ValueError('No [H3PERF] events found. Launch SwarmUI with SWARM_H3_TRACE=1 and retry.')
        result['events'] = lines
        return result

    first_runs = perf.parse_trace(logs[0])
    if mode in ('two-in-one', 'three-in-one'):
        ordered = perf.split_sequential_prompts(first_runs, count=3 if mode == 'three-in-one' else 2)
        run_groups = ordered
    else:
        run_groups = [first_runs] + [perf.parse_trace(p) for p in logs[1:]]
    selected = [perf.select_longest_video(group) for group in run_groups]
    result['comparison'] = _details(*selected)
    result['sampler_counts'] = [len(group) for group in run_groups]
    return result


def _fmt(value, suffix='', ndigits=1):
    if value is None:
        return 'n/a'
    if isinstance(value, (int, float)):
        return f'{value:,.{ndigits}f}{suffix}'
    return str(value)


def _line(label, value, width=76):
    return f'  {label:<23}{value}'[:width]


def _sampler_block(name, sampler):
    lines = [f'  [{name}]']
    lines.append(_line('Prompt / node', f'{sampler.get("prompt_id") or "?"} / {sampler.get("sampler_node") or "?"}'))
    lines.append(_line('Sampler / schedule', f'{sampler.get("sampler") or "?"} / {sampler.get("scheduler") or "?"}'))
    lines.append(_line('Steps / CFG', f'{sampler.get("steps") or "?"} / {sampler.get("cfg") if sampler.get("cfg") is not None else "?"}'))
    lines.append(_line('Model / conditioning', f'{sampler.get("model_type") or "?"} / {len(sampler.get("ref_blocks") or [])} refs, {len(sampler.get("keyframes") or [])} keyframes'))
    lines.append(_line('Video/audio latents', str(sampler.get('latent_shapes') or 'n/a')))
    lines.append(_line('Average step', _fmt(sampler.get('average_step_seconds'), ' s', 2)))
    lines.append(_line('Sampler elapsed', _fmt(sampler.get('sampler_seconds'), ' s', 2)))
    lines.append(_line('Resident weights', _fmt(sampler.get('weights_resident_mb_start'), ' MB')))
    lines.append(_line('Offloaded weights', _fmt((sampler.get('weights_offloaded_mb_per_step') or [None])[0], ' MB')))
    if sampler.get('sampler_success') is False:
        lines.append('  ! Sampler failed; duration is not a completed-generation benchmark.')
    elif sampler.get('sampler_success') is None:
        lines.append('  ! No finish event captured; result may be incomplete.')
    return lines


def render(report):
    lines = ['', BAR,
             '  SWARM / H3 PERFORMANCE STUDIO'.ljust(76),
             BAR, '  MODE : ' + report['mode'].upper(),
             '  TIME : ' + report['generated_at'],
             '  LOGS : ' + str(len(report['sources'])), '-' * 76]
    mode = report['mode']
    if mode == 'scan':
        prompts = report['prompts']
        lines.extend([_line('Captured H3 prompts', str(len(prompts))),
                      _line('Detailed [H3PERF]', 'FOUND' if report['trace_found'] else 'NOT FOUND')])
        for i, p in enumerate(prompts, 1):
            lines.append('-' * 76)
            lines.append(f'  PROMPT {i} / {p["prompt_id"]}    ({len(p["samplers"])} samplers)')
            for j, sampler in enumerate(p['samplers'], 1):
                lines += _sampler_block(f'SAMPLER {j}', sampler)
        legacy = report['legacy']
        if not report['trace_found']:
            lines.extend(['', '  No detailed trace in this log. Legacy timing only.',
                          '  To capture sampler inputs: SWARM_H3_TRACE=1 on the ComfyUI process.'])
        lines.append('-' * 76)
        lines.append('  LEGACY LOG COUNTERS (not per-sampler measurements)')
        lines.append(_line('Prompt times (s)', str(legacy['prompt_times_s'][-8:] or 'none')))
        lines.append(_line('Compiler graph breaks', str(legacy['compiler_graph_breaks'][-8:] or 'none')))
        lines.append(_line('Generated outputs', str(legacy['generated_output_counts'][-8:] or 'none')))
        for model in legacy['preparations'][-8:]:
            lines.append(_line('Staged weights', f'{model["model"]}: {model["staged_mb"]} MB'))
    elif mode == 'extract':
        lines.append(_line('Valid [H3PERF] events', str(len(report['events']))))
        lines.append('  Raw events are saved as a JSONL file without prompt text or pixels.')
    else:
        cmp = report['comparison']
        lines.extend(_sampler_block('A / IMAGE TO VIDEO', cmp['image_to_video']))
        lines.append('-' * 76)
        lines.extend(_sampler_block('B / MAIN VIDEO', cmp['main_video']))
        if 'one_frame_followup' in cmp:
            lines.append('-' * 76)
            lines.extend(_sampler_block('C / ONE-FRAME FOLLOWUP', cmp['one_frame_followup']))
        a = cmp['image_to_video']['average_step_seconds']
        b = cmp['main_video']['average_step_seconds']
        lines.append('-' * 76)
        if a is not None and b is not None and a > 0:
            lines.append(_line('B / A step-time ratio', f'{b/a:.2f}x'))
            lines.append(_line('B - A per-step', f'{b-a:+.2f} s'))
        lines.append(_line('Sampler counts per run', str(report.get('sampler_counts', []))))
        changed = list(cmp['different'])
        lines.append(_line('Different recorded fields', str(len(changed))))
        for key in changed[:18]:
            lines.append('    * ' + key)
        if len(changed) > 18:
            lines.append(f'    ... plus {len(changed) - 18} more (see JSON)')
        lines.extend(['', '  NOTE: Differences are measurements, not a causal GPU diagnosis.'])
    lines.extend(['-' * 76, '  Files are written under H3-Reports. Original logs are never modified.', BAR, ''])
    return '\n'.join(lines)


def save_report(report, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    prefix = outdir / f'H3-{stamp}-{report["mode"]}'
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    text = render(report)
    prefix.with_suffix('.txt').write_text(text, encoding='utf-8')
    saved = {'json': str(prefix.with_suffix('.json')), 'summary': str(prefix.with_suffix('.txt'))}
    if report['mode'] == 'extract':
        dest = prefix.with_suffix('.jsonl')
        with dest.open('w', encoding='utf-8') as f:
            for event in report['events']:
                f.write(json.dumps(event, sort_keys=True) + '\n')
        saved['events'] = str(dest)
    return text, saved


def main(argv=None):
    parser = argparse.ArgumentParser(description='Human-readable MiniMax H3 sampler diagnostics')
    parser.add_argument('--mode', required=True,
                        choices=['scan', 'two-in-one', 'three-in-one', 'two-files', 'three-files', 'extract'])
    parser.add_argument('--log', action='append', required=True, help='Path to input log (repeat for separate logs)')
    parser.add_argument('--outdir', default='H3-Reports', help='Destination for JSON and TXT reports')
    args = parser.parse_args(argv)
    try:
        result = build_report(args.mode, args.log)
        text, saved = save_report(result, args.outdir)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f'[H3 ERROR] {exc}', file=sys.stderr)
        return 2
    print(text)
    for label, path in saved.items():
        print(f'  {label.upper():>8}: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
