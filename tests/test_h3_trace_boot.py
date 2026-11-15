"""Startup visibility contract for H3 diagnostics in the actual launch path.

A green batch child-CMD test is not sufficient: these checks cover the
SwarmUI->ComfyUI ProcessStartInfo boundary and the custom-node import gate.
"""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "src/Backends/NetworkBackendUtils.cs"
PLUGIN = ROOT / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/__init__.py"
SAMPLER = ROOT / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmKSampler.py"


class StartupTraceContract(unittest.TestCase):
    def test_parent_logs_both_enabled_and_disabled_backend_state_before_start(self):
        code = BACKEND.read_text(encoding="utf-8")
        boundary = code.index("runningProcess.Start();")
        before = code[:boundary]
        self.assertIn('Environment.GetEnvironmentVariable("SWARM_H3_TRACE")', before)
        self.assertIn('start.Environment["SWARM_H3_TRACE"]', before)
        self.assertIn('[H3TRACE-BOOT]', before)
        self.assertIn("traceEnabled", before)
        self.assertIn("Logs.Info", before)

    def test_plugin_import_emits_progress_even_without_vae_hook(self):
        code = PLUGIN.read_text(encoding="utf-8")
        parsed = ast.parse(code)
        self.assertTrue(parsed)
        before = code.split("from . import SwarmBlending", 1)[0]
        self.assertIn("[H3TRACE-BOOT]", before)
        self.assertIn("sys.stderr", before)
        self.assertIn("SWARM_H3_TRACE", before)
        self.assertIn("file=", code)
        self.assertIn("vae_hook=", code)
        self.assertIn("SwarmVaeTrace.install_trace(comfy.sd.VAE)", code)

    def test_sampler_route_is_identifiable_even_when_native_used(self):
        generator = (ROOT / "src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs").read_text(encoding="utf-8")
        self.assertIn("[H3TRACE-ROUTE]", generator)
        self.assertIn('"KSamplerAdvanced"', generator)
        self.assertIn('"SwarmKSampler"', generator)

    def test_comfy_stderr_trace_tags_are_forwarded_at_info_level(self):
        code = BACKEND.read_text(encoding="utf-8")
        monitor = code.split("async void MonitorErrLoop()", 1)[1]
        promotion = monitor.split("else if (keepShowing)", 1)[0]
        for marker in ("[H3TRACE-BOOT]", "[H3TRACE-ROUTE]", "[H3PERF]", "[H3VAE]"):
            self.assertIn(marker, promotion)
        self.assertIn('Logs.Info($"[{nameSimple}/STDERR] {line}")', promotion)
        self.assertIn('Logs.Debug($"[{nameSimple}/STDERR] {line}")', monitor)
        self.assertIn('Logs.Warning($"[{nameSimple}/STDERR] {line}")', monitor)

    def test_no_tracing_on_unrelated_model_without_opt_in(self):
        code = SAMPLER.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("SWARM_H3_TRACE") == "1"', code)
        self.assertIn("[H3TRACE-ROUTE] source=SwarmKSampler", code)
        self.assertIn("active={trace_enabled}", code)
        self.assertIn("file=sys.stderr, flush=True", code)


if __name__ == "__main__":
    unittest.main()
