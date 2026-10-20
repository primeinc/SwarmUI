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
    "final_input": "__swarm_text_probe_final_input",
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

    # A production export must have one final output SINK. This alone says
    # nothing about IMAGE batch size; verify actual image count from history
    # during inference before claiming a single final result.
    final_outputs = [
        k for k, v in graph.items()
        if v["class_type"] in ("SwarmSaveImageWS", "SaveImage", "PreviewImage")
    ]
    if len(final_outputs) != 1:
        raise AssertionError(
            f"Expected exactly one final output sink in the original graph, "
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
    # CLIPSeg already applied the author's <segment:...threshold>. Its
    # normalized/interpolated output still carries valid positive pixels
    # below 0.5; a second 0.5 cutoff would silently replace the user's choice.
    # Binarize all detector-selected support BEFORE crop/stitch hipass(0.5).
    if abs(float(graph[guard_id]["inputs"].get("minimum_confidence", -1))) > 1e-6:
        raise AssertionError("Segment detector guard must not add a second threshold")
    # A normalized weak background response can leave positive scores over
    # almost the whole frame. ThresholdMask(0) would turn all of them opaque.
    # Require a finite cutoff at the actual detector-mask guard, not a new
    # implicit CLIPSeg score threshold.
    max_coverage = graph[guard_id]["inputs"].get("maximum_coverage")
    if not isinstance(max_coverage, (float, int)) or not (0 < max_coverage <= 0.99):
        raise AssertionError(
            "Qwen mask guard must reject near-whole-canvas detection without "
            "changing the authored CLIPSeg confidence threshold")
    solid_mask_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "ThresholdMask"
         and v["inputs"].get("mask") == [guard_id, 0]
         and abs(float(v["inputs"].get("value", -1))) < 1e-6],
        "native binary subject-mask threshold after the authored CLIPSeg threshold",
    )
    if crop.get("mask") != [solid_mask_id, 0]:
        raise AssertionError("Crop adapter must consume the validated BINARY subject mask, not fractional detector confidence")
    if abs(float(crop.get("mask_hipass_filter", -1)) - 0.5) > 1e-6:
        raise AssertionError("Crop adapter must exclude low-confidence detector background")
    if not crop.get("output_resize_to_target_size"):
        raise AssertionError("Crop adapter must generate the target-resolution Qwen reference")

    # Swarm's production Qwen 2.1 segment path now uses the standard
    # SwarmTextEncodeAdvanced parser for weight/schedule/LoRA tags and Comfy's
    # ReferenceLatent for VAE references. Requiring TextEncodeQwenImage21
    # here would reject the production graph while blessing an obsolete fixture.
    if any(v["class_type"] == "TextEncodeQwenImage21" for v in graph.values()):
        raise AssertionError("Unexpected old Qwen encoder: Swarm segment tags must use the native Swarm parser")
    enc_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "SwarmTextEncodeAdvanced"
         and v["inputs"].get("prompt", "").strip() == PROMPT],
        "authored positive SwarmTextEncodeAdvanced instruction",
    )
    encoder = graph[enc_id]["inputs"]
    masked_reference = [crop_id, 1]
    # A full Swarm export legitimately includes additional empty global/base
    # negative encoders. Identify the SEGMENT negative by its exact source
    # crop, not by treating every empty prompt in the graph as the same node.
    neg_id = require_single(
        [k for k, v in graph.items() if v["class_type"] == "SwarmTextEncodeAdvanced"
         and v["inputs"].get("prompt") == ""
         and (v["inputs"].get("images") or v["inputs"].get("image_1")) == masked_reference],
        "native segment negative conditioning on the original subject crop",
    )
    if ((encoder.get("images") or encoder.get("image_1")) != masked_reference
            or (graph[neg_id]["inputs"].get("images") or
                graph[neg_id]["inputs"].get("image_1")) != masked_reference):
        raise AssertionError("Both Swarm vision encoders must see the exact CropAndStitch source crop")
    if any(v["class_type"] == "JoinImageWithAlpha" and
           v["inputs"].get("image") == masked_reference and
           v["inputs"].get("alpha") == [crop_id, 2]
           for v in graph.values()):
        raise AssertionError("Inpainting mask incorrectly reused as source transparency")
    vae_ref = require_single(
        [k for k, v in graph.items() if v["class_type"] == "VAEEncode"
         and v["inputs"].get("pixels") == masked_reference],
        "original crop VAE reference latent",
    )
    positive_ref = require_single(
        [k for k, v in graph.items() if v["class_type"] == "ReferenceLatent"
         and v["inputs"].get("conditioning") == [enc_id, 0]
         and v["inputs"].get("latent") == [vae_ref, 0]],
        "source reference attached to positive conditioning",
    )
    negative_ref = require_single(
        [k for k, v in graph.items() if v["class_type"] == "ReferenceLatent"
         and v["inputs"].get("conditioning") == [neg_id, 0]
         and v["inputs"].get("latent") == [vae_ref, 0]],
        "source reference attached to negative conditioning",
    )

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
    if sampler.get("positive") != [positive_ref, 0] or sampler.get("negative") != [negative_ref, 0]:
        raise AssertionError("Source-attached positive and negative conditioning must reach the actual segment sampler")
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
    if any(v["class_type"] == "LatentMultiply" and
           v["inputs"].get("samples") == [vae_ref, 0]
           for v in graph.values()):
        raise AssertionError("Unmasked Reference Edit latent must not be generated for masked segment sampling")

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
    # SwarmSaveImageWS writes its output straight over WebSocket and returns
    # {}, so /history does NOT expose its images. Observe the exact IMAGE
    # tensor feeding the ORIGINAL save sink via an explicit diagnostic node.
    final_sink_input = graph[final_outputs[0]].get("inputs", {}).get("images")
    if not isinstance(final_sink_input, list) or len(final_sink_input) != 2:
        raise AssertionError("Final output sink has no valid IMAGE tensor input")
    preview("final_input", final_sink_input)
    return graph, {
        "final_output_sink": final_outputs[0],
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

def require_single_final_image(outputs, save_node_id, final_input_probe_id=None):
    # A single SaveImage/SwarmSaveImageWS node may consume a batch. If
    # this backend does not expose output metadata, do not manufacture
    # one-image acceptance from merely having one output node.
    saved = outputs.get(save_node_id, {})
    images = saved.get("images")
    if not isinstance(images, list) and final_input_probe_id:
        # The WebSocket sink intentionally yields no history metadata.
        # PreviewImage observes the exact same input tensor, not a parallel
        # approximation of the compositor.
        images = outputs.get(final_input_probe_id, {}).get("images")
    if not isinstance(images, list):
        raise AssertionError(
            f"Single final IMAGE batch UNVERIFIED: neither sink {save_node_id} "
            "nor its input-tensor probe has an images list")
    if len(images) != 1:
        raise AssertionError(
            f"Expected one final saved image, got {len(images)} in sink {save_node_id}")
    return images[0]


def validate_runtime_node_contracts(base_url, graph):
    # Inspect every actual submitted node, including instrumented previews,
    # splitters and mask converters. A fixed allowlist can miss a real failure.
    schemas = {}
    for class_type in sorted({node["class_type"] for node in graph.values()}):
        registered = fetch_json(base_url, "object_info/" + class_type)
        if class_type not in registered:
            raise AssertionError(
                f"Actual Comfy backend is missing node {class_type} "
                "required by the submitted graph")
        schemas[class_type] = registered[class_type]
    for node_id, node in graph.items():
        schema_inputs = schemas[node["class_type"]].get("input", {})
        required = set(schema_inputs.get("required", {}))
        missing = sorted(required - set(node["inputs"]))
        if missing:
            raise AssertionError(
                f"Native node {node_id} ({node['class_type']}) is missing required "
                f"Comfy inputs: {missing}")
        # A stale SwarmTextEncodeAdvanced declares the same class_type but not
        # the newly added per-reference image_N sockets. Comfy's node schema
        # must advertise every independently wired image before inference.
        if node["class_type"] == "SwarmTextEncodeAdvanced":
            known_sockets = required | set(schema_inputs.get("optional", {}))
            missing_images = sorted(
                name for name in node["inputs"]
                if name.startswith("image_") and name[6:].isdigit()
                and name not in known_sockets)
            if missing_images:
                raise AssertionError(
                    f"Selected Comfy backend has obsolete SwarmTextEncodeAdvanced "
                    f"node: missing image sockets {missing_images} in /object_info. "
                    "Update SwarmText.py on this backend and refresh its schema.")
            for name in node["inputs"]:
                if (name.startswith("image_") and name[6:].isdigit()
                        and name in known_sockets):
                    definition = schema_inputs.get("required", {}).get(
                        name, schema_inputs.get("optional", {}).get(name))
                    if not isinstance(definition, list) or not definition or definition[0] != "IMAGE":
                        raise AssertionError(
                            f"Selected Comfy backend declares {name} as a non-IMAGE socket")
            # Qwen 2.1's native text node uses prevent_empty_text=True for
            # BOTH conditionings. The Swarm adapter exposes this as an explicit
            # optional node input, which older backends silently lack.
            if node["inputs"].get("qwen21_prevent_empty_text") is not None:
                if "qwen21_prevent_empty_text" not in known_sockets:
                    raise AssertionError(
                        "Selected Comfy backend has obsolete SwarmTextEncodeAdvanced: "
                        "missing qwen21_prevent_empty_text socket in /object_info. "
                        "Update SwarmText.py on this backend and refresh its schema.")
                definition = schema_inputs.get("optional", {}).get(
                    "qwen21_prevent_empty_text",
                    schema_inputs.get("required", {}).get("qwen21_prevent_empty_text"))
                if not isinstance(definition, list) or not definition or definition[0] != "BOOLEAN":
                    raise AssertionError(
                        "Selected Comfy backend must declare qwen21_prevent_empty_text as BOOLEAN")
    return schemas


def read_result(base, prompt_id, timeout_seconds, final_output_sink):
    from PIL import Image  # inference-only: offline graph audit needs no Pillow
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
        if status.get("completed") and all(
            outputs.get(NODE_IDS[kind], {}).get("images") for kind in OUTPUT_KINDS
        ):
            require_single_final_image(outputs, final_output_sink, NODE_IDS["final_input"])
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
    import numpy as np  # inference-only
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
            "native_schema_completeness_verified": False,
            "one_final_output_sink_verified": True,
            "exactly_one_final_image_verified": False,
        }, indent=2))
        return 0
    if not args.comfy_url or args.outdir is None:
        parser.error("--comfy-url and --outdir are required for model-backed inference")
    import numpy as np  # model-backed diagnostics, never needed by --preflight-only
    stats = fetch_json(args.comfy_url, "system_stats")
    if not isinstance(stats, dict) or not isinstance(stats.get("system"), dict):
        raise AssertionError("The requested URL is not a real Comfy backend")
    # Compare ALL actual submitted node schemas before expensive inference.
    # The diagnostic PreviewImage/MaskToImage nodes are also load-bearing.
    validate_runtime_node_contracts(args.comfy_url, graph)
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
        images = read_result(args.comfy_url, pid, args.timeout, spec["final_output_sink"])
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
