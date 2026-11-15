"""Read-only capture of the actual Comfy graph used for a Swarm segment.

Uses only Python's standard library. Does not queue generations, edit settings,
change branches, install packages, or upload the resulting bundle anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler
import zipfile

SAMPLERS = {"SwarmKSampler", "KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced", "LanPaint_KSampler"}
COMPOSITES = {"SwarmImageCompositeMaskedColorCorrecting", "ImageCompositeMasked", "InpaintStitchImproved"}
LORAS = {"LoraLoader", "LoraLoaderModelOnly", "NunchakuFluxLoraLoader"}
ENCODERS = {"SwarmTextEncodeAdvanced", "CLIPTextEncode", "TextEncodeQwenImage21"}
MAX_BYTES = 32 * 1024 * 1024
MAX_NODES = 20000


class CaptureError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CaptureError(f"HTTP redirect {code}; supply an authorized backend URL explicitly")


def get_json(base: str, route: str) -> Any:
    url = base.rstrip("/") + "/" + route.lstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise CaptureError("Use an http(s) URL without embedded credentials")
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        # No proxy discovery or credential forwarding for local backend requests.
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=8) as response:
            raw = response.read(MAX_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise CaptureError(f"GET {route}: {exc}") from exc
    if len(raw) > MAX_BYTES:
        raise CaptureError(f"GET {route} exceeded the 32 MiB safety limit")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CaptureError(f"GET {route} returned something other than JSON") from exc


def graph_or_none(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value or len(value) > MAX_NODES:
        return None
    if not all(isinstance(v, dict) and isinstance(v.get("class_type"), str)
               and isinstance(v.get("inputs", {}), dict) for v in value.values()):
        return None
    return {str(k): v for k, v in value.items()}


def link(value: Any, graph: dict[str, Any]) -> str | None:
    if (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], (str, int)) and not isinstance(value[0], bool)
            and isinstance(value[1], int) and not isinstance(value[1], bool)):
        node = str(value[0])
        if node in graph:
            return node
    return None


def links(value: Any, graph: dict[str, Any]):
    node = link(value, graph)
    if node is not None:
        yield node
    elif isinstance(value, dict):
        for child in value.values():
            yield from links(child, graph)
    elif isinstance(value, list):
        for child in value:
            yield from links(child, graph)


def upstream(graph: dict[str, Any], starts: list[str], stop: set[str] | None = None) -> set[str]:
    todo = list(starts)
    seen: set[str] = set()
    while todo:
        node_id = todo.pop()
        if node_id in seen or node_id not in graph:
            continue
        seen.add(node_id)
        if len(seen) > MAX_NODES:
            raise CaptureError("Graph traversal limit exceeded")
        node = graph[node_id]
        if stop and node["class_type"] in stop:
            continue
        todo.extend(links(node.get("inputs", {}), graph))
    return seen


def roots(value: Any, graph: dict[str, Any]) -> list[str]:
    return list(links(value, graph))


def model_nodes(graph: dict[str, Any], sampler: dict[str, Any]) -> set[str]:
    inputs = sampler.get("inputs", {})
    starts = roots(inputs.get("model", inputs.get("guider")), graph)
    # Deliberately follow only model/guider edges, not conditioning, latent or image edges.
    seen: set[str] = set()
    while starts:
        node_id = starts.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node_inputs = graph[node_id].get("inputs", {})
        for key in ("model", "model1", "model2", "guider", "model_positive", "model_negative"):
            starts.extend(roots(node_inputs.get(key), graph))
    return seen


def conditioning_nodes(graph: dict[str, Any], starts: list[str]) -> set[str]:
    """Follow conditioning edges only; a reference image can depend on the base sampler."""
    pending = list(starts)
    seen: set[str] = set()
    keys = {"conditioning", "conditioning_1", "conditioning_2", "positive", "negative", "cond1", "cond2"}
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        if graph[node_id]["class_type"] in ENCODERS:
            continue
        for key, value in graph[node_id].get("inputs", {}).items():
            if key in keys:
                pending.extend(roots(value, graph))
    return seen


def describe_node(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    node = graph[node_id]
    return {"id": node_id, "class_type": node["class_type"], "inputs": node.get("inputs", {})}


def analyze(graph: dict[str, Any]) -> dict[str, Any]:
    samplers = {k for k, n in graph.items() if n["class_type"] in SAMPLERS}
    by_composite: dict[str, list[str]] = {}
    for node_id, node in graph.items():
        if node["class_type"] in COMPOSITES:
            # Native CropAndStitch composites its sampled result through
            # inpainted_image, while legacy composites use source.
            key = "inpainted_image" if node["class_type"] == "InpaintStitchImproved" else "source"
            candidates = upstream(graph, roots(node.get("inputs", {}).get(key), graph), SAMPLERS)
            by_composite[node_id] = sorted(candidates & samplers)
    segment_ids = {s for values in by_composite.values() for s in values}
    other_model_nodes: set[str] = set()
    for sampler_id in samplers - segment_ids:
        other_model_nodes.update(model_nodes(graph, graph[sampler_id]))
    findings = []
    for sampler_id in sorted(segment_ids):
        node = graph[sampler_id]
        inp = node.get("inputs", {})
        chain = model_nodes(graph, node)
        loaders = sorted(n for n in chain if graph[n]["class_type"] in LORAS)
        conditions = conditioning_nodes(graph, roots([inp.get("positive"), inp.get("negative"), inp.get("guider")], graph))
        encoders = sorted(n for n in conditions if graph[n]["class_type"] in ENCODERS)
        references = sorted(n for n in conditions if graph[n]["class_type"] == "ReferenceLatent")
        latent_nodes = upstream(graph, roots(inp.get("latent_image"), graph), SAMPLERS)
        settings = {k: inp[k] for k in ("steps", "start_at_step", "end_at_step", "denoise", "cfg", "sampler_name", "scheduler", "add_noise", "noise_seed", "seed") if k in inp}
        if all(isinstance(settings.get(k), (int, float)) for k in ("steps", "start_at_step", "end_at_step")):
            settings["effective_step_count"] = max(0, min(settings["steps"], settings["end_at_step"]) - max(0, settings["start_at_step"]))
        warnings = []
        if not loaders:
            warnings.append("NO_LORA_NODE_ON_MODEL_PATH: graph evidence only; merged checkpoint adapters cannot be inferred")
        if references and encoders and all(not any(
                value is not None and (key in ("images", "image", "image1")
                                       or key.startswith("image_"))
                for key, value in graph[e].get("inputs", {}).items())
                for e in encoders):
            warnings.append("REFERENCE_WITHOUT_ENCODER_IMAGE: compare with the installed Qwen 2.1 conditioning contract; not proof of visual failure")
        if settings.get("effective_step_count") == 0:
            warnings.append("ZERO_EFFECTIVE_STEPS")
        if inp.get("add_noise") == "disable":
            warnings.append("SAMPLER_NOISE_DISABLED")
        findings.append({
            "sampler": sampler_id, "class_type": node["class_type"], "settings": settings,
            "model_path": [describe_node(graph, n) for n in sorted(chain)],
            "lora_nodes": [describe_node(graph, n) for n in loaders],
            "lora_nodes_not_shared_with_base": [n for n in loaders if n not in other_model_nodes],
            "conditioning_encoders": [describe_node(graph, n) for n in encoders],
            "reference_latents": [describe_node(graph, n) for n in references],
            "latent_mask_nodes": [describe_node(graph, n) for n in sorted(latent_nodes) if graph[n]["class_type"] in {"SetLatentNoiseMask", "LanPaint_ImageEncode"}],
            "compositors": [describe_node(graph, c) for c, ss in by_composite.items() if sampler_id in ss],
            "warnings": warnings,
        })
    return {"node_count": len(graph), "sampler_count": len(samplers),
            "segment_sampler_count": len(findings), "segments": findings,
            "scope": "Observed graph wiring, not GPU execution or visible LoRA effectiveness"}


def candidates(history: Any, queue: Any) -> list[tuple[float, str, str, dict[str, Any]]]:
    found = []
    if isinstance(history, dict):
        records = [("history", v.get("prompt")) for v in history.values() if isinstance(v, dict)]
    else:
        records = []
    if isinstance(queue, dict):
        for key in ("queue_running", "queue_pending"):
            records.extend((key, item) for item in queue.get(key, []) if isinstance(item, list))
    for source, item in records:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        graph = graph_or_none(item[2])
        if graph is not None and isinstance(item[0], (int, float)):
            found.append((float(item[0]), str(item[1]), source, graph))
    return sorted(found, key=lambda entry: entry[0], reverse=True)


def redact(value: Any, key: str = "") -> Any:
    compact = key.lower().replace("_", "").replace("-", "")
    if any(word in compact for word in ("password", "apikey", "authorization", "accesstoken", "refreshtoken", "secret")):
        return "[REDACTED CREDENTIAL]"
    if isinstance(value, dict):
        return {k: redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    if isinstance(value, str) and ("base64" in compact or compact.endswith("b64") or value.startswith("data:image/") or len(value) > 262144):
        return {"redacted": "embedded data", "characters": len(value), "sha256": hashlib.sha256(value.encode()).hexdigest()}
    return value


def git_info(root: Path) -> dict[str, Any]:
    result = {}
    for key, args in (("head", ["rev-parse", "HEAD"]), ("branch", ["branch", "--show-current"]), ("status", ["status", "--short"])):
        try:
            run = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, timeout=10, check=False)
            result[key] = run.stdout.strip() if run.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            result[key] = "unavailable"
    return result


def capture(base: str, wait: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats = get_json(base, "system_stats")
    if not isinstance(stats, dict) or not isinstance(stats.get("system"), dict):
        raise CaptureError("URL is not a compatible Comfy backend")
    runtime = {"system": {k: stats["system"].get(k) for k in ("os", "comfyui_version", "python_version", "pytorch_version")}, "devices": stats.get("devices", [])}
    deadline = time.monotonic() + wait
    while True:
        history = get_json(base, "history?max_items=8")
        queue = get_json(base, "queue")
        selected, seen = [], set()
        for _, prompt_id, source, graph in candidates(history, queue):
            if prompt_id in seen:
                continue
            seen.add(prompt_id)
            report = analyze(graph)
            if report["segment_sampler_count"]:
                selected.append({"prompt_id": prompt_id, "source": source, "graph": graph, "analysis": report})
            if len(selected) == 2:
                break
        if selected:
            return selected, runtime
        if time.monotonic() >= deadline:
            raise CaptureError("No segment workflow in the last eight history entries or current queue. Run with --wait 90, then repeat the failing generation once. Nothing was queued by this tool.")
        time.sleep(1)


def write_bundle(records: list[dict[str, Any]], runtime: dict[str, Any], root: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    directory = Path(tempfile.mkdtemp(prefix=f"swarm-qwen-{stamp}-"))
    path = directory / "qwen-segment-evidence.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("runtime.json", json.dumps({"swarm_git": git_info(root), "backend": runtime}, indent=2))
        for index, record in enumerate(records, 1):
            archive.writestr(f"case-{index}-graph.json", json.dumps(redact(record["graph"]), indent=2))
            report = {k: v for k, v in record.items() if k != "graph"}
            archive.writestr(f"case-{index}-report.json", json.dumps(redact(report), indent=2))
        archive.writestr("README.txt", "Read-only capture. Prompts and model/LoRA filenames remain in this bundle. Embedded media and common credential keys are redacted. Review before sharing. This tool did not run GPU inference and does not certify a fix.\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("SWARM_COMFY_URL", "http://127.0.0.1:7801/ComfyBackendDirect"))
    parser.add_argument("--workflow", type=Path, help="Analyze an already exported Comfy API graph without network access")
    parser.add_argument("--wait", type=int, default=120, help="Wait up to this many seconds for a user-started segment generation (0-120)")
    args = parser.parse_args(argv)
    if not 0 <= args.wait <= 120:
        parser.error("--wait must be between 0 and 120")
    try:
        if args.workflow:
            if args.workflow.stat().st_size > MAX_BYTES:
                raise CaptureError("Workflow file exceeds the safety limit")
            data = json.loads(args.workflow.read_text(encoding="utf-8-sig"))
            graph = graph_or_none(data) or graph_or_none(data.get("prompt") if isinstance(data, dict) else None)
            if graph is None:
                raise CaptureError("Expected a Comfy API graph, not the editor-layout JSON")
            records, runtime = [{"prompt_id": "offline", "source": "file", "graph": graph, "analysis": analyze(graph)}], {}
        else:
            print("Read-only capture is armed. Repeat the failing segment generation once in SwarmUI.", flush=True)
            print("Swarm normally deletes completed Comfy history; this also watches the active queue.", flush=True)
            records, runtime = capture(args.url, args.wait)
        bundle = write_bundle(records, runtime, Path(__file__).resolve().parents[1])
        for record in records:
            print(f"Case {record['prompt_id']} ({record['source']})")
            for segment in record["analysis"]["segments"]:
                print(f"  Sampler {segment['sampler']}: {segment['settings']}")
                for lora in segment["lora_nodes"]:
                    print(f"  LoRA {lora['id']}: {lora['inputs'].get('lora_name')} strength={lora['inputs'].get('strength_model', lora['inputs'].get('lora_strength'))}")
                for warning in segment["warnings"]:
                    print(f"  {warning}")
        print(f"\nEvidence saved: {bundle}")
        print("This is a diagnostic capture, NOT a passing image-generation test. No generations or settings were changed.")
        return 0
    except (CaptureError, OSError, ValueError) as exc:
        print(f"CAPTURE FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
