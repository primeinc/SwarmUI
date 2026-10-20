"""Real Qwen 2.1 *text-only* segment replacement reproduction.

Reproduction reported failing 10/10:
  <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes

This is independent of confined LoRAs. Requires a real, exported SwarmUI
Comfy API graph and running Comfy backend; no simulated images or graph stubs.
It queues real inference only when explicitly invoked.

Usage:
 python tests/integration/qwen_segment_text_replacement.py \
   --workflow /path/to/actual-swarm-comfy-api-graph.json \
   --comfy-url http://127.0.0.1:8188 --outdir /path/to/evidence --attempts 1

A positive pixel-change result is not semantic acceptance. The output remains
NOT_CLOSED until reviewed against the requested subject transformation.
"""
from __future__ import annotations

import argparse
import copy
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import uuid

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
import qwen_segment_effect as actual_comfy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "launchtools"))
import qwen_segment_capture as capture


PROMPT = "Edit the image, replace the girl with a black boy with blue eyes"
TAG = "<segment:girl,0.8,0.1>"
NODE_IDS = {
    "before": "__swarm_text_probe_before",
    "crop_before": "__swarm_text_probe_crop_before",
    "crop_after": "__swarm_text_probe_crop_after",
    "mask_image": "__swarm_text_probe_mask_image",
    "mask": "__swarm_text_probe_mask",
    "after": "__swarm_text_probe_after",
}
OUTPUT_KINDS = ("before", "crop_before", "crop_after", "mask", "after")


def require_single(nodes, what):
    if len(nodes) != 1:
        raise AssertionError(f"Expected exactly one {what} in the real exported workflow, found {len(nodes)}")
    return nodes[0]


def inspect_and_instrument(original):
    graph = copy.deepcopy(original)
    if not capture.graph_or_none(graph):
        raise AssertionError("Not an actual Comfy API graph")
    detector_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "SwarmClipSeg"],
        "SwarmClipSeg detector",
    )
    detection = graph[detector_id]["inputs"]
    if detection.get("match_text", "").strip().casefold() != "girl":
        raise AssertionError(f"Expected CLIPSeg match_text='girl', got {detection.get('match_text')!r}")
    if abs(float(detection.get("threshold", -1)) - 0.1) > 1e-4:
        raise AssertionError(f"Expected exact reported mask threshold 0.1, got {detection.get('threshold')!r}")
    enc_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "SwarmQwenImage21SegmentEncode"],
        "Qwen 2.1 segment encoder",
    )
    encoder = graph[enc_id]["inputs"]
    if encoder.get("prompt", "").strip() != PROMPT:
        raise AssertionError(f"Expected exact segment instruction {PROMPT!r}, got {encoder.get('prompt')!r}")
    diagnostics_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "SwarmQwen21SegmentDiagnostics"],
        "Qwen segment diagnostics node",
    )
    compositor_id = actual_comfy.choose_compositor(graph, None)
    compositor = graph[compositor_id]["inputs"]
    if "destination" not in compositor:
        raise AssertionError("Segment compositor has no pre-segmentation destination image")
    report = capture.analyze(graph)
    segment_samplers = [
        seg for seg in report["segments"]
        if any(c["id"] == compositor_id for c in seg["compositors"])
    ]
    segment = require_single(segment_samplers, "segment sampler feeding the final composite")
    sampler_id = segment["sampler"]
    sampler = graph[sampler_id]["inputs"]
    steps = int(sampler.get("steps", 0))
    actual_start = int(sampler.get("start_at_step", -1))
    if not steps or abs(actual_start - round(steps * 0.2)) > 1:
        raise AssertionError(
            f"0.8 creativity should preserve about 80% of steps: steps={steps}, start={actual_start}"
        )
    for node_id in NODE_IDS.values():
        if node_id in graph:
            raise AssertionError(f"Probe node id collides with real workflow: {node_id}")
    def preview(kind, image_ref):
        graph[NODE_IDS[kind]] = {
            "class_type": "PreviewImage",
            "inputs": {"images": image_ref},
        }
    preview("before", compositor["destination"])
    preview("crop_before", graph[diagnostics_id]["inputs"]["source"])
    preview("crop_after", [diagnostics_id, 0])
    graph[NODE_IDS["mask_image"]] = {
        "class_type": "MaskToImage",
        "inputs": {"mask": [detector_id, 0]},
    }
    preview("mask", [NODE_IDS["mask_image"], 0])
    preview("after", [compositor_id, 0])
    return graph, {"detector": detector_id, "encoder": enc_id,
                   "sampler": sampler_id, "compositor": compositor_id,
                   "steps": steps, "start_at_step": actual_start,
                   "effective_steps": max(0, steps - actual_start),
                   "creativity": 0.8, "clipseg_threshold": 0.1}


def read_result(base, prompt_id, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        record = actual_comfy.fetch_json(base, "history/" + prompt_id).get(prompt_id)
        if record is None:
            time.sleep(1)
            continue
        status = record.get("status", {})
        if status.get("status_str") == "error":
            raise AssertionError(f"Actual Comfy execution failed: {status}")
        outputs = record.get("outputs", {})
        if all(outputs.get(NODE_IDS[kind], {}).get("images") for kind in OUTPUT_KINDS):
            result = {}
            for kind in OUTPUT_KINDS:
                images = outputs[NODE_IDS[kind]]["images"]
                if len(images) != 1:
                    raise AssertionError(f"{kind}: expected one real image, got {len(images)}")
                media = images[0]
                url = "view?" + urlencode({
                    "filename": media["filename"],
                    "subfolder": media.get("subfolder", ""),
                    "type": media.get("type", "temp"),
                })
                with urlopen(base.rstrip("/") + "/" + url, timeout=30) as reply:
                    data = reply.read()
                with Image.open(BytesIO(data)) as img:
                    result[kind] = img.convert("RGB").copy()
            return result
        if status.get("completed"):
            raise AssertionError(f"Comfy finished without all real diagnostic outputs: {list(outputs)}")
        time.sleep(1)
    raise TimeoutError(f"Actual Qwen segment inference timed out after {timeout_seconds}s")


def measure(a, b):
    if a.size != b.size:
        raise AssertionError(f"Size mismatch in real image comparison: {a.size} != {b.size}")
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    return float(np.abs(left - right).mean())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True,
                        help="Genuine SwarmUI-exported Comfy API workflow for the reported failure")
    parser.add_argument("--comfy-url", required=True, help="Real Comfy test backend, explicitly selected")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=1,
                        help="Number of real inference attempts with different segment seeds (1-10)")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if not 1 <= args.attempts <= 10 or args.timeout < 1:
        parser.error("--attempts must be 1-10 and timeout must be positive")
    original = json.loads(args.workflow.read_text(encoding="utf-8-sig"))
    if isinstance(original, dict) and "prompt" in original and isinstance(original["prompt"], dict):
        original = original["prompt"]
    graph, spec = inspect_and_instrument(original)
    actual_comfy.fetch_json(args.comfy_url, "system_stats")
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(args.attempts):
        instance = copy.deepcopy(graph)
        sampler = instance[spec["sampler"]]["inputs"]
        original_seed = sampler.get("seed")
        if not isinstance(original_seed, int):
            raise AssertionError(f"Segment sampler does not contain a real integer seed: {original_seed!r}")
        sampler["seed"] = original_seed + i
        queued = actual_comfy.fetch_json(
            args.comfy_url, "prompt",
            {"prompt": instance, "client_id": str(uuid.uuid4())},
        )
        if queued.get("node_errors"):
            raise AssertionError(f"Comfy rejected the real graph: {queued['node_errors']}")
        pid = queued.get("prompt_id")
        if not isinstance(pid, str) or not pid:
            raise AssertionError(f"Real Comfy backend refused to queue this workflow: {queued}")
        images = read_result(args.comfy_url, pid, args.timeout)
        for kind, image in images.items():
            image.save(args.outdir / f"run-{i+1:02d}-{kind}.png")
        mask = np.asarray(images["mask"].convert("L"), dtype=np.float32) / 255
        mask_coverage = float(np.mean(mask > 0.05))
        crop_change = measure(images["crop_before"], images["crop_after"])
        composite_change = measure(images["before"], images["after"])
        failure = ("EMPTY_DETECTION" if mask_coverage < 0.005 else
                   "BROAD_DETECTION" if mask_coverage > 0.90 else
                   "SAMPLER_CROP_UNCHANGED" if crop_change < 1.0 else
                   "FINAL_COMPOSITE_UNCHANGED" if composite_change < 1.0 else
                   "IMAGE_CHANGED_SEMANTIC_REVIEW_REQUIRED")
        result = {
            "attempt": i+1, "segment_seed": sampler["seed"], "prompt_id": pid,
            "mask_fraction_above_5_percent": mask_coverage,
            "crop_mean_absolute_rgb_difference": crop_change,
            "final_mean_absolute_rgb_difference": composite_change,
            "diagnostic": failure,
            "identity_replacement_verified": False,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = {
        "reproduction": TAG + " " + PROMPT,
        "graph": spec, "runs": results,
        "verdict": "NOT_CLOSED",
        "reason": "Real crops and final images saved for semantic review. "
                  "A pixel change alone cannot verify that the girl was replaced "
                  "with a black boy with blue eyes.",
    }
    (args.outdir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 2 if any(r["diagnostic"] != "IMAGE_CHANGED_SEMANTIC_REVIEW_REQUIRED" for r in results) else 3


if __name__ == "__main__":
    raise SystemExit(main())
