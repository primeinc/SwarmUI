"""Fail-closed Qwen segment LoRA acceptance against REAL ComfyUI inference.

Requires a genuine SwarmUI-generated API graph and a reachable Comfy backend
with the actual model and LoRA files installed. Never substitutes graph fixtures,
mocked nodes, or simulated images. This test QUEUES real generations; run only
against an explicitly selected test backend.

Comfy endpoints/response handling follow upstream ComfyUI tests/inference/test_inference.py.
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
from urllib.request import Request, urlopen
import uuid

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "launchtools"))
import qwen_segment_capture as capture


def fetch_json(base: str, route: str, payload: dict | None = None) -> dict:
    url = base.rstrip("/") + "/" + route.lstrip("/")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Accept": "application/json", "Content-Type": "application/json"})
    with urlopen(req, timeout=30) as response:
        parsed = json.loads(response.read())
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected Comfy API response at {route}")
    return parsed


def choose_compositor(graph: dict, preferred: str | None) -> str:
    report = capture.analyze(graph)
    if not report["segment_sampler_count"]:
        raise RuntimeError("Not a SwarmUI segment workflow: no segment sampler feeding a compositor")
    ids = {c["id"] for seg in report["segments"] for c in seg["compositors"]}
    if preferred:
        if preferred not in ids:
            raise RuntimeError(f"Compositor {preferred} does not consume a segmented sampler")
        return preferred
    # Terminal compositor is the one whose image reaches the saved image, not
    # an earlier compositor consumed by another compositor.
    terminal = {
        node_id for node_id in ids
        if not any(other != node_id and node_id in capture.upstream(
            graph, [other]
        ) for other in ids)
    }
    if len(terminal) != 1:
        raise RuntimeError(f"Need --compositor-id; ambiguous terminal compositor: {sorted(terminal)}")
    return terminal.pop()


def find_scoped_loras(graph: dict, compositor: str) -> dict[str, float]:
    report = capture.analyze(graph)
    segments = [
        seg for seg in report["segments"]
        if any(c["id"] == compositor for c in seg["compositors"])
    ]
    scoped_ids = {id_ for seg in segments for id_ in seg["lora_nodes_not_shared_with_base"]}
    if not scoped_ids:
        raise RuntimeError("No confined LoRA found on the segment model path")
    strengths = {}
    for node_id in scoped_ids:
        node = graph[node_id]
        if node["class_type"] not in {"LoraLoaderModelOnly", "LoraLoader"}:
            raise RuntimeError(f"Unsupported confined adapter loader {node['class_type']}: {node_id}")
        strength = node["inputs"].get("strength_model")
        if not isinstance(strength, (int, float)) or strength == 0:
            raise RuntimeError(f"Confined LoRA has zero/invalid strength at node {node_id}")
        strengths[node_id] = float(strength)
    return strengths


def image_from_history(base: str, prompt_id: str, preview_id: str, deadline: float) -> np.ndarray:
    while time.monotonic() < deadline:
        record = fetch_json(base, "history/" + prompt_id).get(prompt_id)
        if record is not None:
            status = record.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"Real Comfy workflow failed: {json.dumps(status)[:4000]}")
            output = record.get("outputs", {}).get(preview_id, {})
            images = output.get("images", [])
            if images:
                if len(images) != 1:
                    raise RuntimeError("Expected exactly one final image per experiment")
                media = images[0]
                uri = "view?" + urlencode({
                    "filename": media["filename"], "subfolder": media.get("subfolder", ""),
                    "type": media.get("type", "temp"),
                })
                with urlopen(base.rstrip("/") + "/" + uri, timeout=30) as response:
                    data = response.read()
                return np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.float32) / 255
            if status.get("completed"):
                raise RuntimeError("Comfy workflow completed without final PreviewImage")
        time.sleep(1)
    raise TimeoutError(f"Comfy did not produce image for {prompt_id} before deadline")


def execute(base: str, graph: dict, preview_id: str, seconds: int) -> np.ndarray:
    queued = fetch_json(base, "prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    if "node_errors" in queued and queued["node_errors"]:
        raise RuntimeError(f"Comfy rejected workflow: {queued['node_errors']}")
    prompt_id = queued.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError(f"Comfy did not queue workflow: {queued}")
    return image_from_history(base, prompt_id, preview_id, time.monotonic() + seconds)


def measure(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise RuntimeError(f"Real images differ in dimensions: {first.shape}, {second.shape}")
    return float(np.abs(first - second).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True, help="Real SwarmUI-exported Comfy API graph (not an editor graph)")
    parser.add_argument("--comfy-url", required=True, help="Explicitly selected real Comfy test backend")
    parser.add_argument("--compositor-id", help="Required if graph has more than one terminal segmented compositor")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--min-mean-difference", type=float, default=0.001)
    parser.add_argument("--report", type=Path, help="Write measured results as JSON")
    args = parser.parse_args()
    graph = capture.graph_or_none(json.loads(args.workflow.read_text(encoding="utf-8")))
    if graph is None:
        raise RuntimeError("Expected a genuine Comfy API graph with class_type and inputs")
    selected = choose_compositor(graph, args.compositor_id)
    loras = find_scoped_loras(graph, selected)
    print(f"REAL INFERENCE: compositor={selected}, scoped_adapters={sorted(loras)}", flush=True)
    # Verify the endpoint is a live Comfy API before submitting.
    capture.get_json(args.comfy_url, "system_stats")
    preview_id = str(max((int(node_id) for node_id in graph if node_id.isdecimal()), default=900000) + 1)
    graph[preview_id] = {"class_type": "PreviewImage", "inputs": {"images": [selected, 0]}}
    result = {"compositor": selected, "tests": [], "verdict": "NOT_CLOSED"}
    off = copy.deepcopy(graph)
    for node_id in loras:
        off[node_id]["inputs"]["strength_model"] = 0
        if "strength_clip" in off[node_id]["inputs"]:
            off[node_id]["inputs"]["strength_clip"] = 0
    baseline1 = execute(args.comfy_url, off, preview_id, args.timeout)
    baseline2 = execute(args.comfy_url, off, preview_id, args.timeout)
    negative_difference = measure(baseline1, baseline2)
    if negative_difference > 1.0 / 255.0:
        raise AssertionError(
            f"FAILED NEGATIVE CONTROL: LoRA-off repeated output differs by {negative_difference:.6f}."
            " Inference is nondeterministic, so A/B cannot be trusted."
        )
    result["negative_control_mean_difference"] = negative_difference
    # Probe each actual confined adapter independently; do not let one successful
    # LoRA hide another adapter whose segment confinement is broken.
    for node_id, strength in sorted(loras.items()):
        one = copy.deepcopy(off)
        one[node_id]["inputs"]["strength_model"] = strength
        if "strength_clip" in one[node_id]["inputs"]:
            one[node_id]["inputs"]["strength_clip"] = graph[node_id]["inputs"]["strength_clip"]
        active = execute(args.comfy_url, one, preview_id, args.timeout)
        difference = measure(baseline1, active)
        outcome = {"node": node_id, "lora_name": graph[node_id]["inputs"]["lora_name"],
                   "strength": strength, "mean_pixel_difference": difference,
                   "pass": difference >= args.min_mean_difference}
        result["tests"].append(outcome)
        print(json.dumps(outcome), flush=True)
    result["verdict"] = "EFFECT_OBSERVED" if all(x["pass"] for x in result["tests"]) else "NOT_CLOSED"
    if args.report:
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["verdict"] != "EFFECT_OBSERVED":
        raise AssertionError("One or more real confined LoRAs produced no measurable image change")
    print("REAL confined-LoRA influence observed. Identity correctness is a separate visual acceptance.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
