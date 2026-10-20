"""Executes the shipped Swarm advanced encoder against a Qwen tokenizer probe.

The mock CLIP records the exact kwargs that reach tokenize(), not model output.
This tests native Qwen's empty-negative conditioning contract without GPU weights.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / "src/BuiltinExtensions/ComfyUIBackend/"
          "ExtraNodes/SwarmComfyCommon/SwarmText.py")


class EmptyHookGroup:
    def __len__(self):
        return 0


class ClipProbe:
    apply_hooks_to_conds = None

    def __init__(self):
        self.calls = []
        self.patcher = SimpleNamespace(forced_hooks=None)

    def tokenize(self, prompt, **kwargs):
        self.calls.append((prompt, dict(kwargs)))
        return [("fake-token", 1)]

    def encode_from_tokens_scheduled(self, tokens):
        return [[object(), {"pooled_output": object()}]]


class Parsed:
    def __init__(self, prompt):
        self.prompt = prompt

    def has_steps(self):
        return False

    def flatten(self, step, steps):
        return [SimpleNamespace(text=self.prompt, weight=1.0, lora_hook=None,
                                embed=None, is_break=False)]


def real_shipped_encoder(source_override=None):
    source = SOURCE.read_text(encoding="utf-8") if source_override is None else source_override
    tree = ast.parse(source, filename=str(SOURCE))
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "SwarmTextEncodeAdvanced")

    # Execute the actual encoder class, replacing only unrelated Comfy/runtime
    # dependencies. combine_leaves deliberately invokes its real tokenize
    # closure, allowing us to observe the native kwargs passed to clip.tokenize.
    def combine_leaves(clip, leaves, tokenize_fn, per_leaf=False):
        return tokenize_fn("".join(item.text or "" for item in leaves)), False

    scope = {
        "MAX_RESOLUTION": 8192,
        "comfy": SimpleNamespace(hooks=SimpleNamespace(HookGroup=EmptyHookGroup)),
        "token_batches_have_weights": lambda tokens: True,
        "parse_prompt": Parsed,
        "uniform_weight": lambda leaves: 1.0,
        "chunks_from_leaves": lambda leaves: [leaves],
        "combine_leaves": combine_leaves,
    }
    exec(compile(ast.Module(body=[klass], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["SwarmTextEncodeAdvanced"]()


class QwenNativeEmptyTextTokenizationTests(unittest.TestCase):
    def encode(self, authored, use_native):
        clip = ClipProbe()
        real_shipped_encoder().encode(
            clip, steps=4, prompt=authored, width=64, height=64,
            target_width=64, target_height=64,
            qwen21_prevent_empty_text=use_native)
        return clip.calls

    def test_qwen_empty_negative_uses_upstream_prevent_empty_text(self):
        calls = self.encode("", True)
        self.assertTrue(any(text == "" and kw.get("prevent_empty_text") is True
                            for text, kw in calls), calls)

    def test_native_empty_negative_gate_rejects_removed_forwarding(self):
        original = SOURCE.read_text(encoding="utf-8")
        needle = 'tokenize_kwargs["prevent_empty_text"] = True'
        self.assertIn(needle, original)
        mutant = original.replace(needle, 'tokenize_kwargs.pop("prevent_empty_text", None)', 1)
        clip = ClipProbe()
        real_shipped_encoder(mutant).encode(
            clip, steps=4, prompt="", width=64, height=64,
            target_width=64, target_height=64, qwen21_prevent_empty_text=True)
        with self.assertRaises(AssertionError):
            self.assertTrue(any(text == "" and kw.get("prevent_empty_text") is True
                                for text, kw in clip.calls))

    def test_regular_encoder_does_not_change_empty_prompt_tokenization(self):
        calls = self.encode("", False)
        self.assertTrue(any(text == "" and "prevent_empty_text" not in kw
                            for text, kw in calls), calls)

    def test_qwen_mode_preserves_authored_nonempty_instruction(self):
        authored = "Replace the girl with a black boy with blue eyes"
        calls = self.encode(authored, True)
        self.assertTrue(any(text == authored and kw.get("prevent_empty_text") is True
                            for text, kw in calls), calls)


if __name__ == "__main__":
    unittest.main()
