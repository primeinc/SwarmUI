"""Isolate the real latent-preview monkeypatch to its originating sampler.

The previous process-global active_count applied Swarm's async conversion to
unrelated concurrent samplers; the patched function must consult a ContextVar
instead, shared across module reloads but isolated per execution context.
"""
import contextvars
from pathlib import Path
import threading
import types
import unittest

SOURCE = (Path(__file__).resolve().parents[1] /
          "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmKSampler.py")


def load_preview_context():
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index('if not hasattr(latent_preview, "_swarm_preview_context"):')
    end = source.index('if not getattr(latent_preview.preview_to_image, "_swarm_patched", False):', start)
    namespace = {
        "latent_preview": types.SimpleNamespace(),
        "ContextVar": contextvars.ContextVar,
    }
    exec(compile(source[start:end], str(SOURCE), "exec"), namespace)
    return namespace, source


class SamplerPreviewContextTests(unittest.TestCase):
    def test_preview_patch_flag_is_local_to_current_execution_context(self):
        scope, source = load_preview_context()
        active = scope["_swarm_preview_active"]
        begin = scope["_swarm_preview_begin"]
        end = scope["_swarm_preview_end"]
        self.assertFalse(active())
        token = begin()
        self.assertTrue(active())

        result = []
        thread = threading.Thread(target=lambda: result.append(active()))
        thread.start()
        thread.join(1)
        self.assertEqual(result, [False], "Unrelated worker must use normal Comfy preview")
        end(token)
        self.assertFalse(active())

    def test_nested_activation_restores_prior_sampler_context(self):
        scope, source = load_preview_context()
        active = scope["_swarm_preview_active"]
        begin = scope["_swarm_preview_begin"]
        end = scope["_swarm_preview_end"]
        first = begin()
        second = begin()
        end(second)
        self.assertTrue(active(), "Nested sampler must not disable outer preview")
        end(first)
        self.assertFalse(active())

    def test_reloaded_sampler_shares_one_contextvar_with_existing_monkeypatch(self):
        scope, source = load_preview_context()
        preview_module = scope["latent_preview"]
        shared = preview_module._swarm_preview_context
        start = source.index('if not hasattr(latent_preview, "_swarm_preview_context"):')
        stop = source.index('if not getattr(latent_preview.preview_to_image, "_swarm_patched", False):', start)
        next_scope = {"latent_preview": preview_module, "ContextVar": contextvars.ContextVar}
        exec(compile(source[start:stop], str(SOURCE), "exec"), next_scope)
        self.assertIs(next_scope["_preview_context"], shared)
        token = scope["_swarm_preview_begin"]()
        try:
            self.assertTrue(next_scope["_swarm_preview_active"]())
        finally:
            scope["_swarm_preview_end"](token)

    def test_negative_global_state_cannot_return(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn('["active_count"]', source)
        self.assertIn("preview_token = _swarm_preview_begin()", source)
        self.assertIn("_swarm_preview_end(preview_token)", source)


if __name__ == "__main__":
    unittest.main()
