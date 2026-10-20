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
from urllib.request import Request, build_opener, ProxyHandler
import uuid

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "launchtools"))
import qwen_segment_capture as capture


PROMPT = "Edit the image, replace the girl with a black boy with blue eyes"
TAG = "<segment:girl,0.8,0.1>"
NODE_IDS = {
    "before": "__swarm_text_probe_before",
    "native_crop": "__swarm_text_probe_native_crop",
    "crop_before": "__swarm_text_probe_crop_before",
    "crop_after": "__swarm_text_probe_crop_after",
    "mask_image": "__swarm_text_probe_mask_image",
    "mask": "__swarm_text_probe_mask",
    "mask_used_image": "__swarm_text_probe_mask_used_image",
    "mask_used": "__swarm_text_probe_mask_used",
    "after": "__swarm_text_probe_after",
}
OUTPUT_KINDS = ("before", "native_crop", "crop_before", "crop_after", "mask", "mask_used", "after")
_HTTP = build_opener(ProxyHandler({}))


def fetch_bytes(base, route, data=None):
    # Comfy-Org/ComfyUI script_examples/basic_api_example.py protocol.
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    req = Request(
        base.rstrip("/") + "/" + route.lstrip("/"),
        data=payload, headers={"Content-Type": "application/json"} if payload else {},
    )
    with _HTTP.open(req, timeout=30) as response:
        return response.read()


def fetch_json(base, route, data=None):
    return json.loads(fetch_bytes(base, route, data))


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
        [k for k, v in graph.items() if v["class_type"] == "TextEncodeQwenImage21"],
        "native TextEncodeQwenImage21 encoder",
    )
    encoder = graph[enc_id]["inputs"]
    if "image_1" not in encoder:
        raise AssertionError("Native Qwen segment edit is missing the cropped original in image_1")
    if any(v["class_type"] == "SwarmQwenImage21SegmentEncode" for v in graph.values()):
        raise AssertionError("Legacy custom segment encoder must not be used")
    if encoder.get("prompt", "").strip() != PROMPT:
        raise AssertionError(f"Expected exact segment instruction {PROMPT!r}, got {encoder.get('prompt')!r}")
    diagnostics_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "SwarmQwen21SegmentDiagnostics"],
        "Qwen segment diagnostics node",
    )
    compositor_id = require_single(
        [key for key, node in graph.items()
         if node["class_type"] in capture.COMPOSITES
         and diagnostics_id in capture.upstream(
             graph, capture.roots(node["inputs"].get("source"), graph))],
        "segment compositor whose source is the real Qwen diagnostics output",
    )
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
    # This case is specifically the user's FOUR-step Turbo configuration.
    # The old bug incorrectly executed only 3/4 iterations: steps=4, start=1.
    # ComfyUI's partial-denoise contract now schedules five and starts at one,
    # retaining all four requested model evaluations.
    expected_iterations = 4
    effective_iterations = steps - actual_start
    if effective_iterations != expected_iterations:
        raise AssertionError(
            f"QWEN SEGMENT STEP REGRESSION: base requested {expected_iterations} "
            f"denoising iterations but segment will execute {effective_iterations} "
            f"(scheduled_steps={steps}, start_at_step={actual_start}). "
            "Do not run costly GPU inference against a truncated 4-step LoRA schedule."
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
    scaled_ref = graph[diagnostics_id]["inputs"]["source"]
    if not isinstance(scaled_ref, list) or len(scaled_ref) != 2:
        raise AssertionError("Qwen segment diagnostics lacks its linked scaled crop")
    scaler = graph.get(str(scaled_ref[0]), {})
    if scaler.get("class_type") != "SwarmImageScaleForMP":
        raise AssertionError("Qwen segment source did not pass through SwarmImageScaleForMP")
    unscaled_ref = scaler["inputs"]["image"]
    if not isinstance(unscaled_ref, list) or len(unscaled_ref) != 2:
        raise AssertionError("Qwen scaler has no original-size crop")
    crop_node = graph.get(str(unscaled_ref[0]), {})
    if crop_node.get("class_type") != "SwarmImageCrop":
        raise AssertionError("Qwen edit is not using a real SwarmImageCrop")
    preview("native_crop", unscaled_ref)
    preview("crop_before", scaled_ref)
    preview("crop_after", [diagnostics_id, 0])
    graph[NODE_IDS["mask_image"]] = {
        "class_type": "MaskToImage",
        "inputs": {"mask": [detector_id, 0]},
    }
    preview("mask", [NODE_IDS["mask_image"], 0])
    # The detector output and the mask actually fed to the final compositor
    # can differ after thresholding and cropping. Capture BOTH from real nodes.
    graph[NODE_IDS["mask_used_image"]] = {
        "class_type": "MaskToImage",
        "inputs": {"mask": compositor["mask"]},
    }
    preview("mask_used", [NODE_IDS["mask_used_image"], 0])
    preview("after", [compositor_id, 0])
    return graph, {"detector": detector_id, "encoder": enc_id,
                   "sampler": sampler_id, "compositor": compositor_id,
                   "steps": steps, "start_at_step": actual_start,
                   "effective_steps": effective_iterations,
                   "expected_iterations": expected_iterations,
                   "creativity": 0.8, "clipseg_threshold": 0.1}


def read_result(base, prompt_id, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        record = fetch_json(base, "history/" + prompt_id).get(prompt_id)
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
                data = fetch_bytes(base, url)
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
    stats = fetch_json(args.comfy_url, "system_stats")
    if not isinstance(stats, dict) or not isinstance(stats.get("system"), dict):
        raise AssertionError("The requested URL is not a real Comfy backend")
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(args.attempts):
        instance = copy.deepcopy(graph)
        sampler = instance[spec["sampler"]]["inputs"]
        # SwarmKSampler.INPUT_TYPES names the seed noise_seed; native
        # KSampler uses seed. Inspect the real node, never invent a value.
        seed_key = "noise_seed" if instance[spec["sampler"]]["class_type"] == "SwarmKSampler" else "seed"
        original_seed = sampler.get(seed_key)
        if not isinstance(original_seed, int):
            raise AssertionError(f"Segment sampler does not contain integer {seed_key}: {original_seed!r}")
        sampler[seed_key] = original_seed + i
        queued = fetch_json(
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
        used_mask = np.asarray(images["mask_used"].convert("L"), dtype=np.float32) / 255
        used_mask_coverage = float(np.mean(used_mask > 0.05))
        crop_change = measure(images["crop_before"], images["crop_after"])
        composite_change = measure(images["before"], images["after"])
        original_w, original_h = images["before"].size
        crop_w, crop_h = images["native_crop"].size
        native_crop_fraction = (crop_w * crop_h) / (original_w * original_h)
        failure = ("EMPTY_DETECTION" if mask_coverage < 0.005 else
                   "BROAD_DETECTION" if mask_coverage > 0.90 else
                   "FULL_FRAME_CROP" if native_crop_fraction >= 0.90 else
                   "SAMPLER_CROP_UNCHANGED" if crop_change < 1.0 else
                   "FINAL_COMPOSITE_UNCHANGED" if composite_change < 1.0 else
                   "IMAGE_CHANGED_SEMANTIC_REVIEW_REQUIRED")
        result = {
            "attempt": i+1, "segment_seed": sampler[seed_key], "prompt_id": pid,
            "mask_fraction_above_5_percent": mask_coverage,
            "compositor_mask_fraction_above_5_percent": used_mask_coverage,
            "native_crop_fraction_of_original": native_crop_fraction,
            "native_crop_dimensions": [crop_w, crop_h],
            "source_dimensions": [original_w, original_h],
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
