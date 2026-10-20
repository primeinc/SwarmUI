"""Real ComfyUI HTTP and graph helpers for live Qwen segment acceptance.

No mocked API responses, workflow fixtures, or synthetic model objects.
This module only prepares a real exported workflow and talks to an explicitly
selected live ComfyUI backend. It does NOT attest to the semantic image result.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def fetch_json(base_url: str, endpoint: str, payload: dict | None = None) -> dict:
    """Call ComfyUI's documented /prompt, /history, or /system_stats HTTP API."""
    address = urlsplit(base_url)
    if (address.scheme not in ("http", "https") or not address.netloc
            or address.username or address.password):
        raise ValueError("A real ComfyUI HTTP(S) URL without embedded credentials is required")
    url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=body,
                      headers={"Accept": "application/json",
                               **({"Content-Type": "application/json"} if body is not None else {})})
    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise AssertionError(
            f"Actual ComfyUI rejected {endpoint} (HTTP {error.code}): {message}"
        ) from error
    if not isinstance(result, dict):
        raise AssertionError(f"Actual ComfyUI {endpoint} returned {type(result).__name__}, expected JSON object")
    return result


def choose_compositor(graph: dict, explicit_id: str | None = None) -> str:
    """Locate the compositor consuming the generated Qwen segment in real API JSON.

    Fail on ambiguous branches instead of inspecting the wrong crop or final image.
    """
    expected = {"SwarmImageCompositeMaskedColorCorrecting", "ImageCompositeMasked"}
    if explicit_id is not None:
        node = graph.get(str(explicit_id))
        if node is None or node.get("class_type") not in expected:
            raise AssertionError(f"Requested compositor {explicit_id!r} is not a real segment compositor")
        if "destination" not in node.get("inputs", {}) or "source" not in node["inputs"]:
            raise AssertionError(f"Compositor {explicit_id!r} lacks source or destination")
        return str(explicit_id)
    candidates = [
        node_id
        for node_id, node in graph.items()
        if node.get("class_type") in expected
        and "destination" in node.get("inputs", {})
        and "source" in node["inputs"]
    ]
    if len(candidates) != 1:
        raise AssertionError(
            f"Expected one actual Qwen segment compositor, got {candidates!r}. "
            "Export a single-segment Comfy API workflow for this reproduction."
        )
    return candidates[0]
