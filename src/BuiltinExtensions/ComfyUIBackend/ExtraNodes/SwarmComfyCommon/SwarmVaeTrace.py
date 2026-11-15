"""Opt-in H3 VAE decode timeline over ComfyUI's own VAE implementation.

Uses ComfyUI's actual VAE.decode / VAE.decode_tiled methods, unchanged,
instead of implementing a second decoder. Enabled only with SWARM_H3_TRACE=1.

Upstream behavior inspected:
Comfy-Org/ComfyUI@c844cb672186f187bf9455e2808821417a10d296
  comfy/sd.py :: VAE.decode, VAE.decode_tiled
  nodes.py :: VAEDecode.decode, VAEDecodeTiled.decode
  comfy_extras/nodes_audio.py :: VAEDecodeAudio.execute
"""
import functools
import json
import logging
import threading
import time


def _is_h3_vae(vae):
    first = getattr(vae, "first_stage_model", None)
    name = type(first).__name__
    if name in ("MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE", "MiniMaxH3ImageVAE", "PH3ImageVAE"):
        return name
    return None


def _stage(model_name):
    if "Audio" in model_name:
        return "audio"
    if "Video" in model_name:
        return "video"
    return "image"


def _current_identity():
    # Take the executing ComfyUI context from this thread before starting a
    # heartbeat thread. Never use process-global 'last_node_id': parallel jobs
    # would silently acquire somebody else's identity.
    try:
        from comfy_execution.utils import get_executing_context
        context = get_executing_context()
        if context is not None:
            return {"prompt_id": context.prompt_id, "node_id": context.node_id}
    except (ImportError, AttributeError):
        pass
    return {"prompt_id": None, "node_id": None}


def _emit(data):
    # The ComfyUI backend captures Python logging on STDERR.
    # All records are single-line and contain no input/output tensor values.
    try:
        logging.info("[H3VAE] %s", json.dumps(data, sort_keys=True))
    except (TypeError, ValueError):
        logging.warning("[H3VAE] telemetry serialization failed")


def _wrap_decoder(original, method, heartbeat_seconds, identity, clock):
    @functools.wraps(original)
    def instrumented(self, samples, *args, **kwargs):
        model_name = _is_h3_vae(self)
        if model_name is None:
            return original(self, samples, *args, **kwargs)

        context = identity()
        started = clock()
        base = {
            "prompt_id": context.get("prompt_id"),
            "node_id": context.get("node_id"),
            "stage": _stage(model_name),
            "model": model_name,
            "method": method,
            "latent_shape": list(getattr(samples, "shape", ())),
        }
        # Record non-content decode configuration without ever logging images.
        tile_names = ("tile_x", "tile_y", "tile_t", "overlap", "overlap_t")
        tiles = {key: kwargs[key] for key in tile_names if key in kwargs}
        if tiles:
            base["tiles"] = tiles
        _emit({**base, "event": "start"})

        finished = threading.Event()
        def heartbeat():
            while not finished.wait(heartbeat_seconds):
                _emit({**base, "event": "working",
                       "elapsed_s": round(clock() - started, 3)})

        worker = threading.Thread(target=heartbeat, name="swarm-h3-vae-progress", daemon=True)
        worker.start()
        status = "success"
        error_type = None
        result = None
        try:
            result = original(self, samples, *args, **kwargs)
            return result
        except BaseException as error:
            status = "error"
            error_type = type(error).__name__
            raise
        finally:
            finished.set()
            worker.join(timeout=0.1)
            record = {**base, "event": "finish", "status": status,
                      "elapsed_s": round(clock() - started, 3)}
            if error_type:
                record["error_type"] = error_type
            if result is not None and hasattr(result, "shape"):
                record["output_shape"] = list(result.shape)
            _emit(record)
    return instrumented


def install_trace(vae_class, *, heartbeat_seconds=15.0, identity=None, clock=None):
    """Wrap the existing ComfyUI VAE API once. No effect on unrelated models.

    Called by SwarmComfyCommon.__init__ only when SWARM_H3_TRACE=1.
    This measures video+audio VAE calls, *not* subsequent video muxing/saving.
    """
    if getattr(vae_class, "_swarm_h3_vae_traced", False):
        return False
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    identity = identity or _current_identity
    clock = clock or time.perf_counter
    methods = ("decode", "decode_tiled")
    for method in methods:
        original = getattr(vae_class, method, None)
        if callable(original):
            setattr(vae_class, method, _wrap_decoder(original, method, heartbeat_seconds, identity, clock))
    vae_class._swarm_h3_vae_traced = True
    return True
