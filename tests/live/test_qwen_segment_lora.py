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
import re
from pathlib import Path
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, ProxyHandler

from PIL import Image, ImageChops, ImageStat
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from qwen_segment_lora_confinement import region_rms, assert_same_detector_mask, normalized_lora_path, same_lora_path


pytestmark = pytest.mark.inference
_OUTPUT = "swarm_lora_live_capture"
_MASK_IMAGE = "swarm_lora_binary_mask_image"
_MASK_OUTPUT = "swarm_lora_binary_mask_capture"
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
    # Case names label experiments. The required_loras manifest field instead
    # enumerates actual installed adapter identities, which may have multiple
    # OFF/ON/OFF cases with different human-readable names and seeds.
    for case in cases:
        if not isinstance(case.get("lora_name"), str) or not case["lora_name"]:
            raise ValueError(f"{case['name']}: missing lora_name")
    required = manifest.get("required_loras")
    if (not isinstance(required, list) or not required
            or any(not isinstance(name, str) or not name for name in required)):
        raise ValueError("required_loras must name at least one real LoRA adapter")
    if any(same_lora_path(required[i], required[j])
           for i in range(len(required)) for j in range(i + 1, len(required))):
        raise ValueError("duplicate adapter identity in required_loras")
    expected = [case["lora_name"] for case in cases]
    if (any(sum(same_lora_path(name, target) for target in required) != 1
            for name in expected)
            or any(not any(same_lora_path(target, name) for target in expected)
                   for name in required)):
        raise ValueError(
            "required_loras must enumerate exactly the distinct case lora_name "
            "adapter identities, not case display names; filename case matters")
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
            outputs = history.get("outputs", {})
            def read_one_saved(node_id, mode):
                images = outputs.get(node_id, {}).get("images", [])
                if len(images) != 1:
                    raise AssertionError(
                        f"{label}: expected one Comfy saved {node_id} image; got {images}")
                media = images[0]
                query = urlencode({
                    "filename": media["filename"], "subfolder": media.get("subfolder", ""),
                    "type": media.get("type", "output"),
                })
                with Image.open(BytesIO(self.request("view?" + query))) as decoded:
                    return decoded.convert(mode).copy()
            final = read_one_saved(_OUTPUT, "RGB")
            mask = read_one_saved(_MASK_OUTPUT, "L") if _MASK_OUTPUT in graph else None
            return final, mask
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
    while True:
        if not isinstance(link, list) or len(link) != 2 or link[0] is None:
            raise AssertionError(f"Malformed sampler MODEL connection: {link!r}")
        node_id = str(link[0])
        if node_id in chain:
            raise AssertionError(f"Model chain cycle at {node_id}")
        node = graph.get(node_id)
        if node is None:
            raise AssertionError(f"Sampler MODEL connection references missing node {node_id}")
        chain.append(node_id)
        inputs = node["inputs"]
        parent = inputs.get("model")
        if parent is None:
            # UNET/Checkpoint source loaders terminate here. A LoRA or other
            # MODEL wrapper without its upstream connection is malformed.
            if node["class_type"] in (_LORAS | {"NunchakuFluxLoraLoader", "DifferentialDiffusion"}) or "model" in inputs:
                raise AssertionError(f"MODEL wrapper {node_id} has no upstream MODEL connection")
            return chain
        link = parent


def assert_segment_final_output_path(graph, reference, stitch_id):
    """Require the compared image to be the one user-visible stitched result.

    A LoRA loader on a sampler chain is not evidence of segment effect if the
    screenshot comes from a different branch or an arbitrary manifest ref.
    """
    outputs = [
        (node_id, node["inputs"].get("images"))
        for node_id, node in graph.items()
        if node["class_type"] in _IMAGE_OUTPUTS
    ]
    if len(outputs) != 1:
        raise AssertionError(
            f"Expected one user-visible final output sink, got {len(outputs)}")
    sink_id, saved_ref = outputs[0]
    if saved_ref != reference:
        raise AssertionError(
            f"Compared image {reference!r} is not the actual saved result "
            f"at {sink_id}: {saved_ref!r}")
    pending = [reference]
    seen = set()
    while pending:
        link = pending.pop()
        if not isinstance(link, list) or len(link) != 2:
            continue
        node_id = str(link[0])
        if node_id == stitch_id and link[1] == 0:
            return
        if node_id in seen:
            continue
        seen.add(node_id)
        node = graph.get(node_id)
        if node is None:
            continue
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                pending.append(value)
    raise AssertionError(
        f"Compared final output {reference!r} does not depend on the actual "
        f"InpaintStitchImproved result {stitch_id}")


def validate_qwen_subject_mask_chain(graph, crop_id):
    """Preflight native detector -> guard -> opaque mask -> CropAndStitch.

    CLIPSeg already applies the author's confidence threshold and normalizes.
    The second native threshold MUST be zero; a legacy 0.5 check falsely
    rejects valid low-threshold edits. Guard coverage rejects broad detections
    before zero-threshold binarization can overwrite the full source image.
    """
    crop_inputs = graph[crop_id]["inputs"]
    hipass = float(crop_inputs.get("mask_hipass_filter", -1))
    threshold_ref = crop_inputs.get("mask")
    if abs(hipass - 0.5) > 1e-6 or not isinstance(threshold_ref, list) or len(threshold_ref) != 2:
        raise AssertionError("Native segment crop must consume a checked binary subject mask at hipass 0.5")
    binary = graph.get(str(threshold_ref[0]))
    if (not binary or binary["class_type"] != "ThresholdMask"
            or int(threshold_ref[1]) != 0
            or abs(float(binary["inputs"].get("value", -1))) > 1e-6):
        raise AssertionError("Native ThresholdMask must binarize all author-selected CLIPSeg support at zero")
    guard_ref = binary["inputs"].get("mask")
    guard = graph.get(str(guard_ref[0])) if isinstance(guard_ref, list) and len(guard_ref) == 2 else None
    if (not guard or guard["class_type"] != "SwarmRequireNonEmptyMask"
            or int(guard_ref[1]) != 0
            or abs(float(guard["inputs"].get("minimum_confidence", -1))) > 1e-6):
        raise AssertionError("Subject mask must preserve the already-applied authored CLIPSeg threshold")
    # Native Qwen supports both regular and inverted <segment:...> tags.
    # The latter intentionally selects the complement of the detector target,
    # which may cover >99% of the canvas. It is safe ONLY when its original
    # detector is guarded and binarized BEFORE InvertMask. Otherwise a soft
    # 0.6-confidence mask inverts to 0.4 and ThresholdMask(0) reselects it.
    def checked_input(ref, kind):
        if (not isinstance(ref, list) or len(ref) != 2 or ref[1] != 0):
            raise AssertionError(f"Expected connected {kind} output 0, got {ref!r}")
        node = graph.get(str(ref[0]))
        if not node or node["class_type"] != kind:
            if kind == "SwarmClipSeg":
                raise AssertionError("Binary subject mask must derive from the actual text detector")
            raise AssertionError(f"Expected {kind} on native Qwen mask chain, got {node!r}")
        return node

    coverage = guard["inputs"].get("maximum_coverage")
    detector_ref = guard["inputs"].get("mask")
    origin = (graph.get(str(detector_ref[0]))
              if isinstance(detector_ref, list) and len(detector_ref) == 2 and detector_ref[1] == 0
              else None)
    if origin and origin["class_type"] == "InvertMask":
        if coverage != 1.0:
            raise AssertionError("Explicit inverted Qwen target must allow 100% complement coverage")
        original_binary = checked_input(origin["inputs"].get("mask"), "ThresholdMask")
        if float(original_binary["inputs"].get("value", -1)) != 0.0:
            raise AssertionError("Original detector must be binarized BEFORE inversion at threshold zero")
        original_guard = checked_input(
            original_binary["inputs"].get("mask"), "SwarmRequireNonEmptyMask")
        if (original_guard["inputs"].get("minimum_confidence") != 0.0
                or original_guard["inputs"].get("maximum_coverage") != 1.0):
            raise AssertionError("Original detector must be nonempty before inversion")
        checked_input(original_guard["inputs"].get("mask"), "SwarmClipSeg")
    else:
        if not isinstance(coverage, (int, float)) or not (0 < coverage <= 0.99):
            raise AssertionError("Normal Qwen detector must reject near-full-canvas support")
        checked_input(detector_ref, "SwarmClipSeg")
    return threshold_ref



def validate_dynamic_hook_on_sampler(graph, sampler_id, hook_node_id):
    """Check the REAL positive/negative conditioning hook path before inference.

    A CreateHookLora is NOT a MODEL-chain LoraLoader. The shipped Swarm text
    encoder selects it by its //hook=N prompt marker and lora_hooks HOOKS
    connection. Checking only the sampler's MODEL path misses this entire
    supported scheduled/alternate LoRA class.
    """
    sampler = graph.get(str(sampler_id))
    if not sampler or sampler["class_type"] not in _SAMPLERS:
        raise AssertionError(f"Dynamic LoRA sampler {sampler_id!r} is missing")
    marker_seen = False
    for direction in ("positive", "negative"):
        link = sampler["inputs"].get(direction)
        visited_cond = set()
        while isinstance(link, list) and len(link) == 2 and str(link[0]) in graph:
            node_id = str(link[0])
            if node_id in visited_cond:
                raise AssertionError(f"{direction} conditioning cycle before dynamic LoRA")
            visited_cond.add(node_id)
            item = graph[node_id]
            kind = item["class_type"]
            if kind == "SwarmTextEncodeAdvanced":
                break
            if kind not in ("ReferenceLatent", "ConditioningMultiply", "ConditioningZeroOut"):
                raise AssertionError(f"Unexpected dynamic {direction} conditioning node {kind}")
            link = item["inputs"].get("conditioning")
        else:
            raise AssertionError(f"{direction} has no active SwarmTextEncodeAdvanced")

        chain, visited_hooks = [], set()
        link = item["inputs"].get("lora_hooks")
        while link is not None:
            if not (isinstance(link, list) and len(link) == 2
                    and link[1] == 0 and str(link[0]) in graph):
                raise AssertionError(f"{direction} has a broken dynamic HOOKS connection")
            current_id = str(link[0])
            if current_id in visited_hooks:
                raise AssertionError("Dynamic LoRA hook chain contains a cycle")
            visited_hooks.add(current_id)
            hook = graph[current_id]
            if hook["class_type"] != "CreateHookLora":
                raise AssertionError(f"Expected CreateHookLora in {direction} HOOKS chain")
            chain.append(current_id)
            link = hook["inputs"].get("prev_hooks")
        chain.reverse()
        if str(hook_node_id) not in chain:
            raise AssertionError(f"{direction} HOOKS chain does not include selected adapter {hook_node_id}")
        index = chain.index(str(hook_node_id))
        markers = [int(n) for n in re.findall(
            r"//hook=(\\d+)", item["inputs"].get("prompt", ""))]
        marker_seen = marker_seen or index in markers
    if not marker_seen:
        raise AssertionError("Selected dynamic LoRA has no matching //hook=N marker on sampler conditioning")


def graph_variant(original, case, stage, strength):
    graph = copy.deepcopy(original)
    node_id = case[f"{stage}_lora_node"]
    node = graph.get(node_id)
    if not node or node["class_type"] not in (_LORAS | {"CreateHookLora"}):
        raise AssertionError(f"{stage}: {node_id} is not a static or dynamic Comfy LoRA loader")
    dynamic = node["class_type"] == "CreateHookLora"
    if dynamic and stage != "segment":
        raise AssertionError("Dynamic hook LoRA case must identify a Qwen segment stage")
    # The LoRA's full relative path is its identity. Comparing only the
    # basename could silently test a different adapter with the same filename
    # in another folder, falsely passing a rendered ON/OFF comparison.
    actual = normalized_lora_path(node["inputs"].get("lora_name", ""))
    expected = normalized_lora_path(case["lora_name"])
    if not same_lora_path(actual, expected):
        raise AssertionError(f"{stage}: requested adapter {expected!r} but graph uses {actual!r}")
    if dynamic:
        validate_dynamic_hook_on_sampler(graph, case[f"{stage}_sampler_node"], node_id)
    elif node_id not in model_chain(graph, case[f"{stage}_sampler_node"]):
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
        threshold_ref = validate_qwen_subject_mask_chain(graph, crops[0])
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
            vision = encoder["inputs"].get("image_1", encoder["inputs"].get("images"))
            if not vision_contains_source(vision):
                raise AssertionError(f"Sampler {direction} vision encoder does not see source crop")
            if encoder["inputs"].get("image_2") is not None and encoder["inputs"].get("images") is not None:
                raise AssertionError("Independent Qwen references must not share a resized ImageBatch")

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
    if stage == "segment":
        assert_segment_final_output_path(graph, ref, stitches[0])
    if _OUTPUT in graph:
        raise ValueError(f"Output id {_OUTPUT} already exists in workflow")
    graph[_OUTPUT] = {"class_type": "SaveImage", "inputs": {
        "images": [str(ref[0]), int(ref[1])], "filename_prefix": "swarm_lora_live_acceptance"
    }}
    if stage == "segment":
        # Diagnostic-only output. This is not an additional production result:
        # it records the real full-canvas detector mask used by the native crop.
        if _MASK_IMAGE in graph or _MASK_OUTPUT in graph:
            raise AssertionError("LoRA mask diagnostic node ID collision")
        graph[_MASK_IMAGE] = {"class_type": "MaskToImage", "inputs": {
            "mask": threshold_ref
        }}
        graph[_MASK_OUTPUT] = {"class_type": "SaveImage", "inputs": {
            "images": [_MASK_IMAGE, 0],
            "filename_prefix": "swarm_lora_segment_mask_acceptance",
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
        image_off_first, mask_off_first = client.run_image(off, f"{case['name']} {stage} OFF-1")
        image_on, mask_on = client.run_image(on, f"{case['name']} {stage} ON")
        image_off_again, mask_off_again = client.run_image(off, f"{case['name']} {stage} OFF-2")
        if stage == "segment":
            detector_mask = assert_same_detector_mask(
                mask_off_first, mask_on, mask_off_again)
            crop_node = next(node for node in off.values()
                             if node["class_type"] == "InpaintCropImproved")
            # The upstream crop expands/feathers the mask. Exclude only the
            # user-configured expand/feather halo plus an 8px interpolation
            # allowance; a hardcoded 48px floor hid real LoRA changes just
            # outside small masks and left tiny images unmeasurable.
            # This is a conservative detector-space bound, NOT an exported
            # exact native stitch mask. Fail rather than claim no leakage if
            # the remaining outside region is empty.
            crop_inputs = crop_node["inputs"]
            blend_margin = int(math.ceil(
                float(crop_inputs.get("mask_expand_pixels", 0))
                + float(crop_inputs.get("mask_blend_pixels", 0)) + 8))
            baseline_inside, baseline_outside = region_rms(
                image_off_first, image_off_again, detector_mask, margin=blend_margin)
            effect_inside, effect_outside = region_rms(
                image_off_first, image_on, detector_mask, margin=blend_margin)
            threshold = max(case["_min_rms"], baseline_inside * 8)
            leakage_limit = max(1.0, baseline_outside * 8)
            print(f"{case['name']} segment: LoRA inside RMS {effect_inside:.3f} "
                  f"(required > {threshold:.3f}); outside RMS {effect_outside:.3f} "
                  f"(allowed <= {leakage_limit:.3f}); mask stable; "
                  f"blend exclusion margin={blend_margin}px")
            assert effect_inside > threshold, (
                f"{case['name']}: SEGMENT LORA INEFFECTIVE INSIDE THE SUBJECT. "
                f"Actual image-space RMS={effect_inside:.3f}.")
            assert effect_outside <= leakage_limit, (
                f"{case['name']}: SEGMENT LORA LEAKS OUTSIDE THE DETECTED SUBJECT. "
                f"Outside/blend-excluded RMS={effect_outside:.3f}; "
                f"limit={leakage_limit:.3f}.")
            metrics[stage] = effect_inside
        else:
            baseline_noise = rms(image_off_first, image_off_again)
            effect = rms(image_off_first, image_on)
            threshold = max(case["_min_rms"], baseline_noise * 8)
            print(f"{case['name']} global: real-image LoRA RMS {effect:.3f}; "
                  f"zero-weight repeat noise {baseline_noise:.3f}; "
                  f"required > {threshold:.3f}")
            assert effect > threshold, (
                f"{case['name']}: GLOBAL LORA INEFFECTIVE. Real ComfyUI forward "
                f"and saved image have no distinct adapter effect; RMS {effect:.3f}.")
            metrics[stage] = effect
    print(f"{case['name']} accepted: global RMS={metrics['global']:.3f}, "
          f"segment RMS={metrics['segment']:.3f}. Real weights; no fakes.")
