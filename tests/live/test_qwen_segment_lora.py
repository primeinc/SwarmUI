"""Live ComfyUI Qwen-segment LoRA acceptance. No fake models, tensors, or Comfy nodes.

Use actual SwarmUI-exported Comfy API graphs, installed model files and a
running, inference-capable Comfy backend. This queues actual model-backed image
generations; unlike source/graph/CPU tensor tests it requires the real weights,
though supported CPU inference is not conceptually excluded. An accelerator
is usually practical, not the criterion for claiming an image-space LoRA effect.

SWARM_LORA_ACCEPTANCE_MANIFEST=/path/to/manifest.json \
    python -m pytest tests/live/test_qwen_segment_lora.py -v -s
"""
from __future__ import annotations

import copy
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, ProxyHandler

from PIL import Image, ImageChops, ImageStat
import pytest


pytestmark = pytest.mark.inference
_OUTPUT = "swarm_lora_live_capture"
_LORAS = {"LoraLoader", "LoraLoaderModelOnly"}
_SAMPLERS = {"SwarmKSampler", "KSampler", "KSamplerAdvanced", "LanPaint_KSampler", "LanPaint_KSamplerAdvanced"}
_IMAGE_OUTPUTS = {"SaveImage", "PreviewImage", "SwarmSaveImageWS"}


def acceptance_cases():
    manifest_path = os.environ.get("SWARM_LORA_ACCEPTANCE_MANIFEST")
    if not manifest_path:
        raise RuntimeError(
            "LIVE LORA ACCEPTANCE NOT RUN: SWARM_LORA_ACCEPTANCE_MANIFEST "
            "must identify real SwarmUI-exported global and segment API graphs. "
            "Passing unit tests is NOT a substitute for this gate."
        )
    file = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(file.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("No LoRA acceptance cases. Refusing an empty green run.")
    names = [case.get("name") for case in cases]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Cases must have unique, nonblank names")
    required = manifest.get("required_loras")
    if not isinstance(required, list) or not required or set(required) != set(names):
        raise ValueError("required_loras must enumerate exactly the LoRAs being tested")
    url = manifest.get("comfy_url")
    parsed = urlsplit(url) if isinstance(url, str) else None
    if not parsed or parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("comfy_url must be an actual Comfy backend HTTP(S) URL without credentials")
    for case in cases:
        for field in ("global_graph", "segment_graph", "lora_name", "global_lora_node",
                      "segment_lora_node", "global_sampler_node", "segment_sampler_node"):
            if not isinstance(case.get(field), str) or not case[field]:
                raise ValueError(f"{case['name']}: missing {field}")
        case["_root"] = file.parent
        case["_url"] = url.rstrip("/")
        case["_timeout"] = int(manifest.get("timeout_seconds", 900))
        case["_min_rms"] = float(manifest.get("minimum_changed_pixel_rms", 1.0))
    return cases


# Canonical Comfy API protocol: Comfy-Org/ComfyUI/script_examples/
# basic_api_example.py and websockets_api_example.py.
class ComfyClient:
    def __init__(self, base_url, timeout):
        self.base = base_url
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def request(self, route, data=None):
        body = None if data is None else json.dumps(data).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = Request(self.base + "/" + route.lstrip("/"), data=body, headers=headers)
        with self.opener.open(req, timeout=30) as response:
            return response.read()

    def run_image(self, graph, label):
        queued = json.loads(self.request("prompt", {"prompt": graph}))
        if "prompt_id" not in queued:
            raise AssertionError(f"{label}: Comfy rejected actual graph: {queued}")
        prompt_id = queued["prompt_id"]
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            response = json.loads(self.request("history/" + prompt_id))
            if prompt_id not in response:
                time.sleep(1)
                continue
            history = response[prompt_id]
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise AssertionError(f"{label}: Comfy execution failed: {status}")
            if status.get("status_str") != "success" or status.get("completed") is not True:
                time.sleep(1)
                continue
            images = history.get("outputs", {}).get(_OUTPUT, {}).get("images", [])
            if len(images) != 1:
                raise AssertionError(f"{label}: expected one Comfy saved image; got {images}")
            image = images[0]
            query = urlencode({"filename": image["filename"], "subfolder": image["subfolder"],
                               "type": image["type"]})
            with Image.open(BytesIO(self.request("view?" + query))) as decoded:
                return decoded.convert("RGB").copy()
        raise AssertionError(f"{label}: Comfy prompt {prompt_id} timed out after {self.timeout}s")


def read_graph(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    graph = data.get("prompt", data) if isinstance(data, dict) else None
    if not isinstance(graph, dict) or not graph or any(
        not isinstance(v, dict) or "class_type" not in v or "inputs" not in v
        for v in graph.values()
    ):
        raise ValueError(f"{path}: export the actual Comfy API graph, NOT editor JSON")
    return graph


def model_chain(graph, sampler_id):
    sampler = graph.get(sampler_id)
    if not sampler or sampler["class_type"] not in _SAMPLERS:
        raise AssertionError(f"Missing sampler {sampler_id!r} in real API graph")
    link = sampler["inputs"].get("model")
    chain = []
    while isinstance(link, list) and len(link) == 2 and str(link[0]) in graph:
        node_id = str(link[0])
        if node_id in chain:
            raise AssertionError(f"Model chain cycle at {node_id}")
        chain.append(node_id)
        link = graph[node_id]["inputs"].get("model")
    return chain


def graph_variant(original, case, stage, strength):
    graph = copy.deepcopy(original)
    node_id = case[f"{stage}_lora_node"]
    node = graph.get(node_id)
    if not node or node["class_type"] not in _LORAS:
        raise AssertionError(f"{stage}: {node_id} is not an actual Comfy LoRA loader")
    # The LoRA's full relative path is its identity. Comparing only the
    # basename could silently test a different adapter with the same filename
    # in another folder, falsely passing a rendered ON/OFF comparison.
    def normalize_lora(name):
        return str(name).replace("\\", "/").strip("/").casefold().removesuffix(".safetensors")

    actual = normalize_lora(node["inputs"].get("lora_name", ""))
    expected = normalize_lora(case["lora_name"])
    if actual != expected:
        raise AssertionError(f"{stage}: requested adapter {expected} but graph uses {actual}")
    if node_id not in model_chain(graph, case[f"{stage}_sampler_node"]):
        raise AssertionError(f"{stage}: LoRA {node_id} is not on the REAL SAMPLER'S model path")
    node["inputs"]["strength_model"] = strength
    if "strength_clip" in node["inputs"]:
        node["inputs"]["strength_clip"] = strength
    if stage == "segment":
        if any(n["class_type"] == "SwarmQwenImage21SegmentEncode" for n in graph.values()):
            raise AssertionError("Obsolete homemade Qwen encoder still present in exported graph")
        if not any(n["class_type"] == "SwarmTextEncodeAdvanced" for n in graph.values()):
            raise AssertionError("Segment workflow must use native SwarmTextEncodeAdvanced for Qwen tags")
        if any(n["class_type"] == "TextEncodeQwenImage21" for n in graph.values()):
            raise AssertionError("Segment unexpectedly uses the legacy tag-flattening Qwen encoder")
        crops = [key for key, node in graph.items() if node["class_type"] == "InpaintCropImproved"]
        stitches = [key for key, node in graph.items() if node["class_type"] == "InpaintStitchImproved"]
        if len(crops) != 1 or len(stitches) != 1:
            raise AssertionError("Segment must use one upstream CropAndStitch adapter pair")
        if graph[stitches[0]]["inputs"].get("stitcher") != [crops[0], 0]:
            raise AssertionError("Stitch must receive the metadata from its matching upstream crop")
        # LoRA effect does not prove the edited subject is cleanly replaced.
        # Enforce the proven native mask chain before any OFF/ON/OFF GPU runs:
        # raw detector -> confidence guard -> binary ThresholdMask ->
        # CropAndStitch's single native blend. Fractional raw confidence
        # permits old-subject double exposure even with an effective LoRA.
        crop_inputs = graph[crops[0]]["inputs"]
        cutoff = float(crop_inputs.get("mask_hipass_filter", -1))
        threshold_ref = crop_inputs.get("mask")
        if abs(cutoff - 0.5) > 1e-6 or not isinstance(threshold_ref, list):
            raise AssertionError("Native segment crop lacks its 0.5-cutoff binary subject mask")
        threshold_node = graph.get(str(threshold_ref[0])) if len(threshold_ref) == 2 else None
        if (not threshold_node or threshold_node["class_type"] != "ThresholdMask" or
                abs(float(threshold_node["inputs"].get("value", -1)) - cutoff) > 1e-6):
            raise AssertionError("Native subject mask must be binarized before CropAndStitch")
        guard_ref = threshold_node["inputs"].get("mask")
        guard = graph.get(str(guard_ref[0])) if isinstance(guard_ref, list) and len(guard_ref) == 2 else None
        if (not guard or guard["class_type"] != "SwarmRequireNonEmptyMask" or
                abs(float(guard["inputs"].get("minimum_confidence", -1)) - cutoff) > 1e-6):
            raise AssertionError("Subject mask must be validated at its native crop cutoff")
        detector_ref = guard["inputs"].get("mask")
        detector = graph.get(str(detector_ref[0])) if isinstance(detector_ref, list) and len(detector_ref) == 2 else None
        if not detector or detector["class_type"] != "SwarmClipSeg":
            raise AssertionError("Native crop's binary subject mask is not sourced from the actual detector")
        if any(n["class_type"] == "JoinImageWithAlpha" and
                   n["inputs"].get("image") == [crops[0], 1] and
                   n["inputs"].get("alpha") == [crops[0], 2] for n in graph.values()):
            raise AssertionError("Inpaint target mask was misused as input source alpha")
        sampler_id = case["segment_sampler_node"]
        sampler = graph[sampler_id]
        source_ref = [crops[0], 1]
        reference_encodes = [(key, n) for key, n in graph.items()
                             if n["class_type"] == "VAEEncode"
                             and n["inputs"].get("pixels") == source_ref]
        if len(reference_encodes) != 1:
            raise AssertionError("Missing or duplicate VAEEncode for the Qwen source crop")
        vae_ref = [reference_encodes[0][0], 0]

        def conditioning_source(direction):
            current, visited, refs = sampler["inputs"].get(direction), set(), []
            while isinstance(current, list) and len(current) == 2 and str(current[0]) in graph:
                idx = str(current[0])
                if idx in visited:
                    raise AssertionError("Cyclic Qwen segment conditioning")
                visited.add(idx)
                upstream = graph[idx]
                kind = upstream["class_type"]
                if kind == "ReferenceLatent":
                    refs.append(upstream["inputs"].get("latent"))
                    current = upstream["inputs"].get("conditioning")
                elif kind in ("ConditioningMultiply", "ConditioningZeroOut"):
                    current = upstream["inputs"].get("conditioning")
                elif kind == "SwarmTextEncodeAdvanced":
                    return upstream, refs
                else:
                    raise AssertionError(f"Unexpected {direction} conditioning stage {kind}")
            raise AssertionError(f"Missing SwarmTextEncodeAdvanced on sampler {direction} path")

        def vision_contains_source(ref):
            visited = set()
            while isinstance(ref, list) and len(ref) == 2:
                if ref == source_ref:
                    return True
                idx = str(ref[0])
                if idx in visited or idx not in graph or graph[idx]["class_type"] != "ImageBatch":
                    return False
                visited.add(idx)
                ref = graph[idx]["inputs"].get("image1")
            return False

        for direction in ("positive", "negative"):
            encoder, latent_refs = conditioning_source(direction)
            if vae_ref not in latent_refs:
                raise AssertionError(f"Sampler {direction} lacks its original crop reference latent")
            if not vision_contains_source(encoder["inputs"].get("images")):
                raise AssertionError(f"Sampler {direction} vision encoder does not see source crop")

        # The Qwen editing modes do not share the same latent path.
        if sampler["class_type"] == "KSampler":
            latent_ref = sampler["inputs"].get("latent_image")
            empty = graph.get(str(latent_ref[0])) if isinstance(latent_ref, list) and len(latent_ref) == 2 else None
            if (not empty or empty["class_type"] != "LatentMultiply" or
                    float(empty["inputs"].get("multiplier", -1)) != 0.0 or
                    empty["inputs"].get("samples") != vae_ref):
                raise AssertionError("Reference Edit must sample the zeroed original Qwen crop latent")
            if any(n["class_type"] == "LanPaint_ImageEncode" for n in graph.values()):
                raise AssertionError("Reference Edit cannot use masked LanPaint latent encoding")
        elif sampler["class_type"] == "LanPaint_KSampler":
            encoded = [(key, n) for key, n in graph.items()
                       if n["class_type"] == "LanPaint_ImageEncode"
                       and n["inputs"].get("image") == [crops[0], 1]
                       and n["inputs"].get("mask") == [crops[0], 2]]
            if len(encoded) != 1 or sampler["inputs"].get("latent_image") != [encoded[0][0], 0]:
                raise AssertionError("Masked Img2Img must sample LanPaint's exact masked crop")
        else:
            raise AssertionError(f"Unexpected native Qwen segment sampler: {sampler['class_type']}")
        decoded = [(key, n) for key, n in graph.items()
                   if n["class_type"] == "LanPaint_ImageDecode"
                   and n["inputs"].get("samples") == [sampler_id, 0]
                   and n["inputs"].get("image") == [crops[0], 1]
                   and "mask" not in n["inputs"]]
        if len(decoded) != 1:
            raise AssertionError("Native Qwen decoder must size-match the source crop without blending a second time")
        split = [(key, n) for key, n in graph.items()
                 if n["class_type"] == "SplitImageWithAlpha"
                 and n["inputs"].get("image") == [decoded[0][0], 0]]
        if len(split) != 1 or graph[stitches[0]]["inputs"].get("inpainted_image") != [split[0][0], 0]:
            raise AssertionError("Original native decoder result must reach the exact stitch adapter")
    ref = case.get(f"{stage}_output_image_ref")
    if ref is None:
        outputs = [n["inputs"]["images"] for n in graph.values()
                   if n["class_type"] in _IMAGE_OUTPUTS and "images" in n["inputs"]]
        if len(outputs) != 1:
            raise ValueError(f"{stage}: identify the exact final output_image_ref; found {len(outputs)}")
        ref = outputs[0]
    if not isinstance(ref, list) or len(ref) != 2 or str(ref[0]) not in graph:
        raise ValueError(f"{stage}: invalid real final image connection {ref!r}")
    if _OUTPUT in graph:
        raise ValueError(f"Output id {_OUTPUT} already exists in workflow")
    graph[_OUTPUT] = {"class_type": "SaveImage", "inputs": {
        "images": [str(ref[0]), int(ref[1])], "filename_prefix": "swarm_lora_live_acceptance"
    }}
    return graph


def rms(a, b):
    if a.size != b.size:
        raise AssertionError(f"Output image sizes changed: {a.size} versus {b.size}")
    stats = ImageStat.Stat(ImageChops.difference(a, b))
    return math.sqrt(sum(value * value for value in stats.rms) / len(stats.rms))


@pytest.mark.parametrize("case", acceptance_cases(), ids=lambda c: c["name"])
def test_real_global_and_segment_lora_effect(case):
    """Actual Comfy execution; off-on-off negative control in BOTH workflows."""
    client = ComfyClient(case["_url"], case["_timeout"])
    if not isinstance(json.loads(client.request("system_stats")).get("system"), dict):
        raise AssertionError("Endpoint is not the actual ComfyUI runtime")
    strength = float(case.get("strength", 1))
    if not math.isfinite(strength) or strength == 0:
        raise ValueError("Nonzero finite LoRA strength required")
    metrics = {}
    for stage in ("global", "segment"):
        graph = read_graph(case["_root"] / case[f"{stage}_graph"])
        off = graph_variant(graph, case, stage, 0.0)
        on = graph_variant(graph, case, stage, strength)
        image_off_first = client.run_image(off, f"{case['name']} {stage} OFF-1")
        image_on = client.run_image(on, f"{case['name']} {stage} ON")
        image_off_again = client.run_image(off, f"{case['name']} {stage} OFF-2")
        baseline_noise = rms(image_off_first, image_off_again)
        effect = rms(image_off_first, image_on)
        threshold = max(case["_min_rms"], baseline_noise * 8)
        print(f"{case['name']} {stage}: real-image LoRA RMS {effect:.3f}; "
              f"zero-weight repeat noise {baseline_noise:.3f}; required > {threshold:.3f}")
        assert effect > threshold, (
            f"{case['name']}: {stage.upper()} LORA INEFFECTIVE. Real ComfyUI forward "
            f"and saved image have no distinct adapter effect; RMS {effect:.3f}."
        )
        metrics[stage] = effect
    print(f"{case['name']} accepted: global RMS={metrics['global']:.3f}, "
          f"segment RMS={metrics['segment']:.3f}. Real weights; no fakes.")
