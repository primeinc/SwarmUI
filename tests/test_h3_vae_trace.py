"""CPU-only VAE decode timeline contract using the actual Swarm hook module."""
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

SOURCE = (Path(__file__).resolve().parents[1] / "src" / "BuiltinExtensions" /
          "ComfyUIBackend" / "ExtraNodes" / "SwarmComfyCommon" / "SwarmVaeTrace.py")
INIT = SOURCE.parent / "__init__.py"


def module():
    spec = importlib.util.spec_from_file_location("swarm_vae_trace_test", SOURCE)
    instance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instance)
    return instance


class FakeTensor:
    shape = (1, 16, 97, 60, 78)


class MiniMaxH3VideoVAE:
    pass


class MiniMaxH3AudioVAE:
    pass


class SDXL:
    pass


def make_vae(model_type):
    class VAE:
        def __init__(self):
            self.first_stage_model = model_type()
            self.calls = []

        def decode(self, samples, **kwargs):
            self.calls.append(("decode", samples, kwargs))
            return {"unchanged": samples}

        def decode_tiled(self, samples, **kwargs):
            self.calls.append(("decode_tiled", samples, kwargs))
            return {"unchanged": samples}

    return VAE


class H3VaeTraceTests(unittest.TestCase):
    def setUp(self):
        self.trace = module()
        self.logs = []
        self.trace.logging.info = self.capture

    def capture(self, format_string, payload):
        self.assertEqual(format_string, "[H3VAE] %s")
        self.logs.append(json.loads(payload))

    def test_video_decode_tiled_emits_start_and_finish_without_changing_return(self):
        VAE = make_vae(MiniMaxH3VideoVAE)
        self.trace.install_trace(VAE, heartbeat_seconds=1000, identity=lambda: {"prompt_id": "P", "node_id": "33"})
        vae = VAE()
        result = vae.decode_tiled(FakeTensor(), tile_t=97, tile_x=32)
        self.assertIs(result["unchanged"].__class__, FakeTensor)
        self.assertEqual([x[0] for x in vae.calls], ["decode_tiled"])
        self.assertEqual([x["event"] for x in self.logs], ["start", "finish"])
        self.assertEqual(self.logs[0]["stage"], "video")
        self.assertEqual(self.logs[0]["node_id"], "33")
        self.assertEqual(self.logs[0]["prompt_id"], "P")
        self.assertEqual(self.logs[0]["latent_shape"], [1, 16, 97, 60, 78])
        self.assertEqual(self.logs[1]["status"], "success")
        self.assertGreaterEqual(self.logs[1]["elapsed_s"], 0)

    def test_audio_decode_is_covered(self):
        VAE = make_vae(MiniMaxH3AudioVAE)
        self.trace.install_trace(VAE, heartbeat_seconds=1000, identity=lambda: {"prompt_id": "P", "node_id": "80"})
        instance = VAE()
        instance.decode(FakeTensor())
        self.assertEqual([x["event"] for x in self.logs], ["start", "finish"])
        self.assertEqual(self.logs[0]["stage"], "audio")
        self.assertEqual(self.logs[0]["method"], "decode")

    def test_unrelated_vae_is_not_instrumented(self):
        VAE = make_vae(SDXL)
        self.trace.install_trace(VAE, heartbeat_seconds=1000)
        instance = VAE()
        instance.decode(FakeTensor())
        self.assertEqual(self.logs, [])
        self.assertEqual(len(instance.calls), 1)

    def test_failure_keeps_original_exception_and_reports_it(self):
        VAE = make_vae(MiniMaxH3VideoVAE)
        def broken(self, samples):
            raise RuntimeError("intentional original failure")
        VAE.decode = broken
        self.trace.install_trace(VAE, heartbeat_seconds=1000)
        with self.assertRaisesRegex(RuntimeError, "intentional original failure"):
            VAE().decode(FakeTensor())
        self.assertEqual([x["event"] for x in self.logs], ["start", "finish"])
        self.assertEqual(self.logs[-1]["status"], "error")
        self.assertEqual(self.logs[-1]["error_type"], "RuntimeError")

    def test_install_is_idempotent_no_duplicate_events(self):
        VAE = make_vae(MiniMaxH3VideoVAE)
        self.trace.install_trace(VAE, heartbeat_seconds=1000)
        first = VAE.decode
        self.trace.install_trace(VAE, heartbeat_seconds=1000)
        self.assertIs(first, VAE.decode)
        VAE().decode(FakeTensor())
        self.assertEqual([x["event"] for x in self.logs], ["start", "finish"])

    def test_periodic_working_event_during_slow_decoder(self):
        entered = threading.Event()
        release = threading.Event()
        VAE = make_vae(MiniMaxH3VideoVAE)
        def blocked(self, samples):
            entered.set()
            self.assert_unused = release.wait(1)
            return ("ok",)
        VAE.decode = blocked
        self.trace.install_trace(VAE, heartbeat_seconds=0.01)
        errors = []
        def run():
            try:
                VAE().decode(FakeTensor())
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(0.6))
            import time
            deadline = time.monotonic() + 0.6
            while not any(e["event"] == "working" for e in self.logs) and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(any(e["event"] == "working" for e in self.logs))
        finally:
            release.set()
            thread.join(1)
        self.assertFalse(errors)
        self.assertEqual(self.logs[-1]["event"], "finish")

    def test_real_startup_gate_and_hook_are_explicit(self):
        init = INIT.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("SWARM_H3_TRACE") == "1"', init)
        self.assertIn("SwarmVaeTrace.install_trace(comfy.sd.VAE)", init)


if __name__ == "__main__":
    unittest.main()
