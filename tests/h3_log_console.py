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
H3_VAE_EVENT = re.compile(r'\[H3VAE\]\s*(\{.*\})')
MODEL_PREP = re.compile(r'Model\s+(MiniMaxH3(?:TEModel_|VideoVAE|AudioVAE)?)\s+prepared for dynamic VRAM loading\.\s+(\d+)MB Staged')
PROMPT_TIME = re.compile(r'Prompt executed in\s+([\d.]+)\s+seconds')
COMPILER_BREAKS = re.compile(r'Comfy model compiler graph breaks:\s+(\d+)')
GENERATED = re.compile(r'Generated (\d+) images?\b|Generated an image\b', re.IGNORECASE)
BAR = '=' * 76


def _read_log(path):
    return Path(path).read_text(encoding='utf-8-sig', errors='replace')


def vae_events(path):
    """Read only opt-in decoder events, without inferring decoder time from a log gap."""
    events = []
    for line in _read_log(path).splitlines():
        if "[H3VAE]" not in line:
            continue
        match = H3_VAE_EVENT.search(line)
        if match is None:
            raise ValueError("Malformed [H3VAE] record")
        try:
            event = json.loads(match.group(1))
        except json.JSONDecodeError:
            raise ValueError("Malformed [H3VAE] record") from None
        events.append(event)
    return events


def summarize_vae(events, prompt_id=None):
    finished = [e for e in events if e.get("event") == "finish" and
                (prompt_id is None or str(e.get("prompt_id")) == str(prompt_id))]
    return {
        "events": len([e for e in events if prompt_id is None or str(e.get("prompt_id")) == str(prompt_id)]),
        "video_decode_s": round(sum(e.get("elapsed_s", 0) for e in finished if e.get("stage") == "video"), 3),
        "audio_decode_s": round(sum(e.get("elapsed_s", 0) for e in finished if e.get("stage") == "audio"), 3),
        "image_decode_s": round(sum(e.get("elapsed_s", 0) for e in finished if e.get("stage") == "image"), 3),
        "finish_count": len(finished),
        "incomplete": bool([
            e for e in events if e.get("event") == "start" and
            (prompt_id is None or str(e.get("prompt_id")) == str(prompt_id))
        ]) and len(finished) < len([
            e for e in events if e.get("event") == "start" and
            (prompt_id is None or str(e.get("prompt_id")) == str(prompt_id))
        ]),
    }


def trace_health(path):
    """Evidence-based startup/dispatch diagnosis; never equates absent logs to an absent hook."""
    lines = _read_log(path).splitlines()
    host = [line for line in lines if "[H3TRACE-BOOT] host=SwarmUI" in line]
    plugin_import = [line for line in lines
                     if "[H3TRACE-BOOT] source=SwarmComfyCommon import_start" in line]
    plugin_state = [line for line in lines
                    if "[H3TRACE-BOOT] source=SwarmComfyCommon status=" in line]
    csharp_routes = [line for line in lines
                     if "[H3TRACE-ROUTE] stage=" in line]
    python_routes = [line for line in lines
                     if "[H3TRACE-ROUTE] source=SwarmKSampler" in line]
    sampler_records = sum("[H3PERF] " in line for line in lines)
    vae_records = sum("[H3VAE] " in line for line in lines)

    if not host:
        verdict = "No host bootstrap evidence: this log may omit startup, predate tracing, or come from another installation."
    elif "traceEnabled=False" in host[-1]:
        verdict = "SwarmUI started without SWARM_H3_TRACE=1; use the trace-enabled updater and restart."
    elif not plugin_import:
        verdict = "Host enabled tracing, but custom-node import was not observed; inspect ComfyUI custom-node loading or log completeness."
    elif plugin_state and "status=disabled" in plugin_state[-1]:
        verdict = "ComfyUI plugin saw tracing disabled despite the SwarmUI host setting; inspect process environment."
    elif not plugin_state:
        verdict = "Custom node began importing, but did not confirm that its VAE hook installed."
    elif not csharp_routes:
        verdict = "Trace package loaded; no H3 sampler route was recorded in this log."
    elif "sampler=KSamplerAdvanced" in csharp_routes[-1]:
        verdict = "Native KSamplerAdvanced was selected; custom [H3PERF] step events are not expected."
    elif not python_routes:
        verdict = "SwarmKSampler was selected by workflow generation but Python sampler entry was not observed."
    elif "active=False" in python_routes[-1]:
        verdict = "The Python sampler ran, but its effective model class did not satisfy the H3 tracing guard."
    elif not sampler_records:
        verdict = "The Python sampler entered with H3 tracing enabled, but [H3PERF] records did not arrive."
    elif not vae_records:
        verdict = "H3 sampling was traced; no VAE decoder records were captured in this log."
    else:
        verdict = "Host, plugin, sampler and VAE trace events are all observable."

    return {
        "host_boot_seen": bool(host),
        "host_trace_enabled": bool(host and "traceEnabled=True" in host[-1]),
        "plugin_import_seen": bool(plugin_import),
        "plugin_hook_status_seen": bool(plugin_state),
        "sampler_route_seen": bool(csharp_routes),
        "python_sampler_entry_seen": bool(python_routes),
        "sampler_event_count": sampler_records,
        "vae_event_count": vae_records,
        "verdict": verdict,
    }


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
        result['trace_health'] = trace_health(logs[0])
        result['vae_events'] = vae_events(logs[0])
        result['vae_summary'] = summarize_vae(result['vae_events'])
        return result
    if mode == 'extract':
        lines = []
        for raw in _read_log(logs[0]).splitlines():
            kind = 'H3PERF' if '[H3PERF]' in raw else 'H3VAE' if '[H3VAE]' in raw else None
            if kind is None:
                continue
            found = (H3_EVENT if kind == 'H3PERF' else H3_VAE_EVENT).search(raw)
            if not found:
                raise ValueError(f'Found a truncated or malformed [{kind}] record')
            try:
                event = json.loads(found.group(1))
                lines.append({'trace_kind': kind, **event})
            except json.JSONDecodeError:
                raise ValueError(f'Found a truncated or malformed [{kind}] record') from None
        if not lines:
            raise ValueError('No [H3PERF] or [H3VAE] events found. Launch SwarmUI with SWARM_H3_TRACE=1 and retry.')
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
    traces_by_file = [vae_events(path) for path in logs]
    stages = ["image_to_video", "main_video", "one_frame_followup"]
    result['vae_by_run'] = {}
    for i, chosen in enumerate(selected):
        events = traces_by_file[0] if mode in ('two-in-one', 'three-in-one') else traces_by_file[i]
        result['vae_by_run'][stages[i]] = summarize_vae(events, chosen['start'].get('prompt_id'))
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
        health = report['trace_health']
        lines.append('-' * 76)
        lines.append('  TRACE DELIVERY HEALTH')
        lines.append(_line('Host startup seen', 'YES' if health['host_boot_seen'] else 'NO'))
        lines.append(_line('Child plugin import', 'YES' if health['plugin_import_seen'] else 'NO'))
        lines.append(_line('Sampler route / entry',
                           f"{int(health['sampler_route_seen'])} / {int(health['python_sampler_entry_seen'])}"))
        lines.append(_line('Sampler / VAE events',
                           f"{health['sampler_event_count']} / {health['vae_event_count']}"))
        lines.append('  ' + health['verdict'])
        vae = report['vae_summary']
        if vae['events']:
            lines.append('-' * 76)
            lines.append('  H3 VAE DECODE TIMELINE (measured, not estimated)')
            lines.append(_line('VAE telemetry events', str(vae['events'])))
            lines.append(_line('Video VAE decode', _fmt(vae['video_decode_s'], ' s', 2)))
            lines.append(_line('Audio VAE decode', _fmt(vae['audio_decode_s'], ' s', 2)))
            lines.append(_line('Image VAE decode', _fmt(vae['image_decode_s'], ' s', 2)))
            if vae['incomplete']:
                lines.append('  ! An H3 VAE decode started but had no finish event.')
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
        lines.append(_line('Valid H3 trace events', str(len(report['events']))))
        lines.append(_line('H3 sampler events', str(sum(e.get('trace_kind') == 'H3PERF' for e in report['events']))))
        lines.append(_line('H3 VAE events', str(sum(e.get('trace_kind') == 'H3VAE' for e in report['events']))))
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
        if any(v['events'] for v in report.get('vae_by_run', {}).values()):
            lines.append('-' * 76)
            lines.append('  POST-SAMPLER VAE DECODING')
            for stage, vae in report['vae_by_run'].items():
                if vae['events']:
                    lines.append(_line(stage, f"video {_fmt(vae['video_decode_s'], ' s', 2)} / audio {_fmt(vae['audio_decode_s'], ' s', 2)}"))
                    if vae['incomplete']:
                        lines.append('  ! Incomplete VAE timing in ' + stage)
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
