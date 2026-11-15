#!/usr/bin/env python3
"""Compare opt-in [H3PERF] sampler traces from two or three SwarmUI/ComfyUI runs.

Only tagged JSON records are read. Image data, prompts, and untagged logs are
neither needed nor echoed. Selects the largest H3 video latent in each log,
which distinguishes the long video from the extra short video in a two-stage run.
"""
import argparse
import json
import re
from pathlib import Path

MARKER = re.compile(r"\[H3PERF\]\s*(\{.*\})")


def parse_trace(path):
    runs = []
    active = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = MARKER.search(line)
        if match is None:
            continue
        data = json.loads(match.group(1))
        ident = (str(data.get("prompt_id")), str(data.get("node_id")))
        if data.get("event") == "start":
            run = {"start": data, "steps": [], "finish": None}
            active[ident] = run
            runs.append(run)
        elif ident in active and data.get("event") == "step":
            active[ident]["steps"].append(data)
        elif ident in active and data.get("event") == "finish":
            active[ident]["finish"] = data
    return runs


def latent_video_volume(run):
    shapes = run["start"].get("latent_shapes", [])
    video = next((shape for shape in shapes if len(shape) == 5), None)
    return video[2] * video[3] * video[4] if video else 0


def select_longest_video(runs):
    if not runs:
        raise ValueError("No [H3PERF] sampler start records; set SWARM_H3_TRACE=1 on the ComfyUI process.")
    return max(runs, key=lambda run: (latent_video_volume(run), len(run["steps"])))


def split_sequential_prompts(runs, count=2):
    """Controlled generations in one warm ComfyUI process, in requested order."""
    groups = []
    for run in runs:
        pid = run["start"].get("prompt_id")
        if not groups or groups[-1][0] != pid:
            groups.append((pid, []))
        groups[-1][1].append(run)
    if len(groups) != count:
        raise ValueError("Expected exactly %d distinct sequential H3 prompts in one log; found %d" % (count, len(groups)))
    return tuple(group[1] for group in groups)


def describe(run):
    s = run["start"]
    steps = run["steps"]
    average = sum(v["elapsed_s"] for v in steps) / len(steps) if steps else None
    return {
        "prompt_id": s.get("prompt_id"), "sampler_node": s.get("node_id"),
        "model_type": s.get("model_type"),
        "latent_shapes": s.get("latent_shapes"),
        "ref_blocks": s.get("refs", []), "keyframes": s.get("keyframes", []),
        "text_tokens": s.get("text_tokens", []),
        "steps": s.get("steps"), "cfg": s.get("cfg"),
        "sampler": s.get("sampler"), "scheduler": s.get("scheduler"),
        "previews": s.get("previews"),
        "disable_cfg1_optimization": s.get("disable_cfg1_optimization"),
        "patch_keys": s.get("patch_keys"),
        "cuda_reserved_mb_start": s.get("cuda_reserved_mb"),
        "cuda_reserved_mb_finish": (run["finish"] or {}).get("cuda_reserved_mb"),
        "weights_total_mb": s.get("weights_total_mb"),
        "weights_resident_mb_start": s.get("weights_resident_mb"),
        "weights_resident_mb_per_step": [step.get("weights_resident_mb") for step in steps],
        "weights_resident_mb_finish": (run["finish"] or {}).get("weights_resident_mb"),
        "weights_offloaded_mb_per_step": [step.get("weights_offloaded_mb") for step in steps],
        "dynamic_model": s.get("dynamic_model"),
        "step_seconds": [step["elapsed_s"] for step in steps],
        "preview_callback_seconds": [step.get("callback_s") for step in steps],
        "average_step_seconds": round(average, 2) if average is not None else None,
        "sampler_seconds": (run["finish"] or {}).get("elapsed_s"),
        "sampler_success": (run["finish"] or {}).get("success"),
    }


def compare(first, second):
    one, two = describe(first), describe(second)
    diff = {key: {"image_to_video": one[key], "main_video": two[key]}
            for key in one if key not in ("prompt_id", "sampler_node")
            and one[key] != two[key]}
    return {"image_to_video": one, "main_video": two, "different": diff}


def main():
    parser = argparse.ArgumentParser(description="Compare real H3 sampler inputs and timings from two or three runs.")
    parser.add_argument("image_to_video_log", help="Log from two-output follow-up run")
    parser.add_argument("main_video_log", nargs="?", help="Optional separate log from main-model run; omit for a shared log.")
    parser.add_argument("--one-frame-followup-log", help="Optional third log where main H3 has 1 frame and I2V has 324 frames.")
    parser.add_argument("--three-in-one-log", action="store_true",
                        help="One log has three sequential requests: 124-frame I2V, 324-frame main, then 1-frame I2V.")
    args = parser.parse_args()
    if args.three_in_one_log and (args.main_video_log or args.one_frame_followup_log):
        parser.error("--three-in-one-log expects a single log path.")
    first_runs = parse_trace(args.image_to_video_log)
    third_runs = None
    if args.three_in_one_log:
        first_runs, second_runs, third_runs = split_sequential_prompts(first_runs, count=3)
    elif args.main_video_log:
        second_runs = parse_trace(args.main_video_log)
    else:
        first_runs, second_runs = split_sequential_prompts(first_runs)
    if args.one_frame_followup_log:
        third_runs = parse_trace(args.one_frame_followup_log)
    first = select_longest_video(first_runs)
    second = select_longest_video(second_runs)
    report = compare(first, second)
    report["sampler_counts"] = {"image_to_video": len(first_runs), "main_video": len(second_runs)}
    if third_runs is not None:
        third = select_longest_video(third_runs)
        report["one_frame_followup"] = describe(third)
        report["sampler_counts"]["one_frame_followup"] = len(third_runs)
        a, b = report["image_to_video"], report["one_frame_followup"]
        keys = ("latent_shapes", "ref_blocks", "keyframes", "cfg", "sampler", "scheduler",
                "previews", "disable_cfg1_optimization")
        report["long_i2v_sampler_differences"] = {
            key: {"124_frame_first_stage": a[key], "1_frame_first_stage": b[key]}
            for key in keys if a[key] != b[key]
        }
        if a["sampler_seconds"] is not None and b["sampler_seconds"] is not None:
            report["long_i2v_sampler_delta_seconds"] = round(
                a["sampler_seconds"] - b["sampler_seconds"], 2)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
