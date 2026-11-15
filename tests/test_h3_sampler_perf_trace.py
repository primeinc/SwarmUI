"""CPU-only contract tests for the opt-in H3 sampler performance trace.

Extract real production functions without importing GPU/Comfy dependencies.
This validates reported shape/conditioning and step timing, not H3 inference.
"""
import ast
import pathlib
import types
import unittest

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmKSampler.py"


class DummyTensor:
    def __init__(self, *shape):
        self.shape = shape
        self.is_nested = False


class DummyNested:
    is_nested = True

    def unbind(self):
        return [DummyTensor(1, 16, 97, 60, 60), DummyTensor(1, 4, 2, 530)]


class MiniMaxH3:
    pass


class H3SamplerTraceTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        names = {"_swarm_h3_trace_shapes", "_swarm_h3_trace_snapshot", "_swarm_h3_trace_timed_callback", "_swarm_h3_trace_residency"}
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in selected}, names)
        env = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), env)
        self.shape = env["_swarm_h3_trace_shapes"]
        self.snapshot = env["_swarm_h3_trace_snapshot"]
        self.timed_callback = env["_swarm_h3_trace_timed_callback"]
        self.residency = env["_swarm_h3_trace_residency"]

    def test_real_reference_and_keyframe_are_reported_separately(self):
        model = types.SimpleNamespace(model=MiniMaxH3(), model_options={"disable_cfg1_optimization": False})
        reference = {"kind": "image", "latent_h": 60, "latent_w": 60, "latent": DummyTensor(1, 16, 2, 60, 60)}
        keyframe = {"resolved_frame_index": 0, "latent": DummyTensor(1, 16, 2, 60, 60)}
        positive = [(DummyTensor(1, 384, 4096), {"minimax_refs": [reference], "minimax_keyframes": [keyframe]})]
        record = self.snapshot(model, positive, DummyNested())
        self.assertEqual(record["model_type"], "MiniMaxH3")
        self.assertEqual(record["latent_shapes"], [[1, 16, 97, 60, 60], [1, 4, 2, 530]])
        self.assertEqual(record["text_tokens"], [384])
        self.assertEqual(record["refs"], [{"kind": "image", "latent_h": 60, "latent_w": 60,
                                         "latent_shape": [1, 16, 2, 60, 60]}])
        self.assertEqual(record["keyframes"], [{"frame": 0, "latent_shape": [1, 16, 2, 60, 60]}])
        self.assertFalse(record["disable_cfg1_optimization"])

    def test_no_reference_has_no_fake_keyframe(self):
        model = types.SimpleNamespace(model=MiniMaxH3(), model_options={})
        record = self.snapshot(model, [(DummyTensor(1, 100, 2048), {})], DummyTensor(1, 16, 97, 60, 60))
        self.assertEqual(record["refs"], [])
        self.assertEqual(record["keyframes"], [])
        self.assertEqual(record["latent_shapes"], [[1, 16, 97, 60, 60]])

    def test_step_times_use_monotonic_clock_and_forward_callback(self):
        times = iter([100.0, 164.5, 166.0, 166.1, 230.6, 231.1, 231.2])
        events = []
        calls = []
        def original(step, x0, x, total):
            calls.append((step, total))
            return "ok"
        wrapped = self.timed_callback(original, events.append, clock=lambda: next(times))
        self.assertEqual(wrapped(0, None, None, 4), "ok")
        self.assertEqual(wrapped(1, None, None, 4), "ok")
        self.assertEqual(calls, [(0, 4), (1, 4)])
        self.assertEqual(events, [
            {"event": "step", "step": 1, "total_steps": 4, "elapsed_s": 64.5, "callback_s": 1.5},
            {"event": "step", "step": 2, "total_steps": 4, "elapsed_s": 64.5, "callback_s": 0.5},
        ])

    def test_residency_uses_patcher_loaded_size_not_cuda_allocation(self):
        model = types.SimpleNamespace(
            loaded_size=lambda: 4 * 1024 * 1024,
            model_size=lambda: 14 * 1024 * 1024,
            is_dynamic=lambda: True,
        )
        self.assertEqual(self.residency(model), {
            "weights_resident_mb": 4.0,
            "weights_offloaded_mb": 10.0,
            "weights_total_mb": 14.0,
            "dynamic_model": True,
        })

    def test_step_sampling_time_excludes_opt_in_trace_query_time(self):
        times = iter([100.0, 164.5, 166.0, 171.0, 235.5, 236.0, 240.0])
        emitted = []
        def emit(data):
            emitted.append(data)
        def probe():
            return {"weights_resident_mb": 8.0}
        cb = self.timed_callback(lambda *args: None, emit, clock=lambda: next(times), residency=probe)
        cb(0, None, None, 4)
        cb(1, None, None, 4)
        self.assertEqual(emitted[0]["elapsed_s"], 64.5)
        self.assertEqual(emitted[0]["callback_s"], 1.5)
        self.assertEqual(emitted[1]["elapsed_s"], 64.5)
        self.assertEqual(emitted[1]["callback_s"], 0.5)
        self.assertEqual(emitted[1]["weights_resident_mb"], 8.0)

    def test_diagnostics_are_opt_in_not_an_inference_default(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("SWARM_H3_TRACE") == "1"', source)
        self.assertIn('if trace_enabled:', source)


if __name__ == "__main__":
    unittest.main()
