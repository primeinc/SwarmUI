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

Graph preflight also works without Comfy, Qwen weights, or a GPU:

 python tests/integration/qwen_segment_text_replacement.py \
   --workflow /path/to/actual-swarm-comfy-api-graph.json --preflight-only

A passing preflight is graph-contract evidence, not a model-backed image
test. A positive pixel-change result is not semantic acceptance. The output
remains NOT_CLOSED until reviewed against the requested subject transformation.
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
    "crop_before": "__swarm_text_probe_crop_before",
    "crop_after": "__swarm_text_probe_crop_after",
    "mask_image": "__swarm_text_probe_mask_image",
    "mask": "__swarm_text_probe_mask",
    "mask_used_image": "__swarm_text_probe_mask_used_image",
    "mask_used": "__swarm_text_probe_mask_used",
    "after": "__swarm_text_probe_after",
}
OUTPUT_KINDS = ("before", "crop_before", "crop_after", "mask", "mask_used", "after")
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
        raise AssertionError("Not an actual SwarmUI-exported Comfy API workflow")

    # A production export without optional diagnostics must have exactly
    # one user-visible saved result, not a gallery of internal crops/masks.
    # Check before adding the intentional evidence PreviewImage probes below.
    final_outputs = [
        k for k, v in graph.items()
        if v["class_type"] in ("SwarmSaveImageWS", "SaveImage", "PreviewImage")
    ]
    if len(final_outputs) != 1:
        raise AssertionError(
            f"Expected exactly one user-visible output in the original graph, "
            f"got {len(final_outputs)}: {final_outputs}"
        )

    detector_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "SwarmClipSeg"],
        "real SwarmClipSeg detector",
    )
    detected = graph[detector_id]["inputs"]
    if detected.get("match_text", "").strip().casefold() != "girl":
        raise AssertionError(f"Wrong segment target {detected.get('match_text')!r}")
    if abs(float(detected.get("threshold", -1)) - 0.1) > 1e-4:
        raise AssertionError(f"Wrong CLIPSeg detection threshold: {detected.get('threshold')!r}")

    if any(n["class_type"] == "SwarmQwenImage21SegmentEncode" for n in graph.values()):
        raise AssertionError("Obsolete homemade Qwen encoder remains in actual graph")
    crop_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "InpaintCropImproved"],
        "upstream InpaintCropImproved",
    )
    crop = graph[crop_id]["inputs"]
    # This reproduction is segment-only. The cropped subject must be the
    # ORIGINAL photograph; a base sampler on the crop input path would mean
    # Qwen changed the entire image before segmentation even started.
    # Check the real exported graph, not a fixture or a code string.
    source_nodes = capture.upstream(graph, capture.roots(crop.get("image"), graph))
    pre_segment_samplers = [
        node for node in sorted(source_nodes)
        if graph[node]["class_type"] in {
            "KSampler", "KSamplerAdvanced", "SwarmKSampler",
            "LanPaint_KSampler", "LanPaint_KSamplerAdvanced",
            "SamplerCustomAdvanced",
        }
    ]
    if pre_segment_samplers:
        raise AssertionError(
            "Segment-only crop source depends on a base sampler instead of the "
            f"original photograph: {pre_segment_samplers}"
        )
    guard_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "SwarmRequireNonEmptyMask"
         and v["inputs"].get("mask") == [detector_id, 0]],
        "nonempty-detector-mask guard",
    )
    solid_mask_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "ThresholdMask"
         and v["inputs"].get("mask") == [guard_id, 0]
         and abs(float(v["inputs"].get("value", -1)) - 0.5) < 1e-6],
        "native binary subject-mask threshold at crop cutoff",
    )
    if crop.get("mask") != [solid_mask_id, 0]:
        raise AssertionError("Crop adapter must consume the validated BINARY subject mask, not fractional detector confidence")
    if abs(float(crop.get("mask_hipass_filter", -1)) - 0.5) > 1e-6:
        raise AssertionError("Crop adapter must exclude low-confidence detector background")
    if not crop.get("output_resize_to_target_size"):
        raise AssertionError("Crop adapter must generate the target-resolution Qwen reference")

    enc_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "TextEncodeQwenImage21"],
        "native TextEncodeQwenImage21",
    )
    encoder = graph[enc_id]["inputs"]
    if encoder.get("prompt", "").strip() != PROMPT:
        raise AssertionError(f"Wrong exact segment instruction {encoder.get('prompt')!r}")
    # The official Qwen example joins the source image with the SOURCE's own
    # transparency, and passes the INPAINT mask separately to LanPaint.
    # Swarm's RGB crop preserves ordinary RGB image pixels without manufacturing
    # alpha from the region targeted for replacement.
    masked_reference = [crop_id, 1]
    if encoder.get("images.image_1") != masked_reference:
        raise AssertionError("Native Qwen must receive the exact CropAndStitch image")
    if any(v["class_type"] == "JoinImageWithAlpha" and
           v["inputs"].get("image") == [crop_id, 1] and
           v["inputs"].get("alpha") == [crop_id, 2]
           for v in graph.values()):
        raise AssertionError("Inpainting mask incorrectly reused as source transparency")
    if "images.image_2" in encoder:
        raise AssertionError("Original whole frame duplicated as an extra Qwen reference")

    # Segment default is actual native Qwen masked inpainting. The upstream
    # adapter's cropped IMAGE and MASK must both reach LanPaint_ImageEncode,
    # and its output must drive the sampler's latent input. Simply masking
    # the final stitch is not an inpaint sampler.
    masked_encoder_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "LanPaint_ImageEncode"
         and v["inputs"].get("image") == masked_reference
         and v["inputs"].get("mask") == [crop_id, 2]],
        "upstream LanPaint masked encoder of the exact Qwen reference",
    )
    sampler_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "LanPaint_KSampler"
         and v["inputs"].get("latent_image") == [masked_encoder_id, 0]],
        "upstream LanPaint_KSampler on the MASKED latent",
    )
    sampler = graph[sampler_id]["inputs"]
    steps = int(sampler.get("steps", -1))
    denoise = float(sampler.get("denoise", -1))
    cfg = float(sampler.get("cfg", -1))
    if steps != 4 or abs(denoise - 0.8) > 1e-4 or "start_at_step" in sampler:
        raise AssertionError(f"Native masked Qwen schedule broken: steps={steps}, denoise={denoise}")
    if abs(cfg - 1.0) > 1e-4:
        raise AssertionError(f"Native Qwen CFG should be 1.0, got {cfg}")
    if int(sampler.get("LanPaint_NumSteps", 0)) != 5:
        raise AssertionError("LanPaint's upstream inner update schedule is missing")
    if sampler.get("LanPaint_PromptMode") != "Image First":
        raise AssertionError("LanPaint's required prompt mode is not wired")
    if sampler.get("LanPaint_Info") != "LanPaint KSampler.":
        raise AssertionError("LanPaint's required information input is missing")
    if sampler.get("Inpainting_mode") != "🖼️ Image Inpainting":
        raise AssertionError("LanPaint's required native inpainting mode is missing")
    if any(v["class_type"] == "KSampler" and
           v["inputs"].get("latent_image") == [enc_id, 2]
           for v in graph.values()):
        raise AssertionError("Unmasked empty-latent sampler remains reachable in masked segment edit")

    stitch_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "InpaintStitchImproved"],
        "upstream InpaintStitchImproved",
    )
    stitch = graph[stitch_id]["inputs"]
    if stitch.get("stitcher") != [crop_id, 0]:
        raise AssertionError("Stitcher must be the EXACT metadata returned by the upstream crop adapter")
    stitch_input = stitch.get("inpainted_image")
    stitch_input_id = capture.link(stitch_input, graph)
    if stitch_input_id is None or graph[stitch_input_id]["class_type"] != "SplitImageWithAlpha":
        raise AssertionError("Stitch must receive the decoded sampled crop without unexpected alpha")
    decoder_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "LanPaint_ImageDecode"
         and v["inputs"].get("samples") == [sampler_id, 0]
         and v["inputs"].get("image") == masked_reference
         and "mask" not in v["inputs"]],
        "upstream LanPaint decoder without a second mask blend, fed by the masked Qwen sampler",
    )
    if decoder_id not in capture.upstream(graph, capture.roots(stitch_input, graph)):
        raise AssertionError("Stitch image must originate from the unmasked native decoder")

    for node_id in NODE_IDS.values():
        if node_id in graph:
            raise AssertionError(f"Probe ID collision: {node_id}")

    def preview(kind, image_ref):
        graph[NODE_IDS[kind]] = {"class_type": "PreviewImage",
                                 "inputs": {"images": image_ref}}

    # Images captured from the ACTUAL path, not a shadow simulation:
    #  * crop output 1 is the source seen by native Qwen
    #  * stitch metadata output 0 controls the exact final placement
    #  * stitch output 0 is the actual final frame
    preview("before", crop["image"])
    preview("crop_before", [crop_id, 1])
    preview("crop_after", stitch_input)
    graph[NODE_IDS["mask_image"]] = {
        "class_type": "MaskToImage",
        "inputs": {"mask": [detector_id, 0]},
    }
    preview("mask", [NODE_IDS["mask_image"], 0])
    graph[NODE_IDS["mask_used_image"]] = {
        "class_type": "MaskToImage",
        "inputs": {"mask": [crop_id, 2]},
    }
    preview("mask_used", [NODE_IDS["mask_used_image"], 0])
    preview("after", [stitch_id, 0])
    return graph, {
        "detector": detector_id, "validated_mask": guard_id,
        "solid_subject_mask": solid_mask_id, "crop_adapter": crop_id,
        "source_reference": crop_id,
        "encoder": enc_id, "sampler": sampler_id, "stitch_adapter": stitch_id,
        "mode": "Masked Img2Img", "sampler_class": "LanPaint_KSampler",
        "sampler_latent_source": [masked_encoder_id, 0],
        "steps": steps, "native_denoise": denoise, "effective_steps": steps,
        "expected_iterations": 4, "requested_creativity": 0.8, "cfg": cfg,
        "clipseg_threshold": 0.1,
        "mask_used_definition": "adapter output 2 (cropped mask), not inaccessible private stitcher blend",
    }

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
    parser.add_argument("--comfy-url", help="Real Comfy test backend, required for inference only")
    parser.add_argument("--outdir", type=Path, help="Image evidence directory (required for inference only)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate the actual exported graph offline; no backend, models, network, or GPU required")
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
    if args.preflight_only:
        print(json.dumps({
            "verdict": "GRAPH_PREFLIGHT_PASSED_NOT_IMAGE_ACCEPTANCE",
            "mode": "offline",
            "node_count": len(original),
            "structural_invariants": spec,
            "model_inference_runs": 0,
            "semantic_subject_replacement_verified": False,
            "image_space_lora_effect_verified": False,
            "runtime_node_registration_verified": False,
        }, indent=2))
        return 0
    if not args.comfy_url or args.outdir is None:
        parser.error("--comfy-url and --outdir are required for model-backed inference")
    stats = fetch_json(args.comfy_url, "system_stats")
    if not isinstance(stats, dict) or not isinstance(stats.get("system"), dict):
        raise AssertionError("The requested URL is not a real Comfy backend")
    # Probe EVERY load-bearing runtime node before scheduling model-backed inference.
    # A checked-in plugin does not prove Comfy loaded or registered it.
    for expected_node in (
        "TextEncodeQwenImage21", "InpaintCropImproved", "InpaintStitchImproved",
        "SwarmClipSeg", "SwarmRequireNonEmptyMask", "ThresholdMask", "LanPaint_ImageEncode",
        "LanPaint_KSampler", "LanPaint_ImageDecode",
    ):
        registered = fetch_json(args.comfy_url, "object_info/" + expected_node)
        if expected_node not in registered:
            raise AssertionError(f"Actual Comfy backend does not expose {expected_node}; "
                                 "a checked-in plugin is not proof that the runtime loaded it.")
        # Comfy's live object_info is the actual input contract, not Python
        # function defaults or C# compilation. Fail before expensive inference
        # if any graph node omits a required native field.
        node_schema = registered[expected_node]
        required = set(node_schema.get("input", {}).get("required", {}))
        for node_id, node in graph.items():
            if node["class_type"] != expected_node:
                continue
            missing = sorted(required - set(node["inputs"]))
            if missing:
                raise AssertionError(
                    f"Native {expected_node} node {node_id} is missing required "
                    f"Comfy inputs: {missing}"
                )
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
        failure = ("EMPTY_DETECTION" if mask_coverage < 0.005 else
                   "BROAD_DETECTION" if mask_coverage > 0.90 else
                   "SAMPLER_CROP_UNCHANGED" if crop_change < 1.0 else
                   "FINAL_COMPOSITE_UNCHANGED" if composite_change < 1.0 else
                   "IMAGE_CHANGED_SEMANTIC_REVIEW_REQUIRED")
        result = {
            "attempt": i+1, "segment_seed": sampler[seed_key], "prompt_id": pid,
            "mask_fraction_above_5_percent": mask_coverage,
            "adapter_cropped_mask_fraction_above_5_percent": used_mask_coverage,
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
