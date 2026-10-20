"""CPU execution of the shipped Swarm text encoder's Qwen tokenizer contract.

This extracts the *actual* parser and encoder source by AST and uses a
Qwen-shaped fake CLIP, not a replacement encoder. It verifies tokenizer
keyword dispatch; semantic Qwen inference and LoRA effect are not covered.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

SOURCE = (Path(__file__).resolve().parents[1] /
          "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmText.py")
REQUIRED = (
    "ParsedText", "quick_simple_tag_filler", "index_of_noncontained",
    "split_tag", "split_smart", "parse_prompt", "chunks_from_leaves",
    "join_text", "token_batches_have_weights", "apply_comfy_token_weights",
    "stamp_token_weight", "multiply_cond_by_token_weights",
    "token_weights_from_tokens", "tokenizer_for_key", "flatten_content",
    "pack_batches", "calc_leaf", "wrap_in_shell", "combine_leaves",
    "uniform_weight", "SwarmTextEncodeAdvanced",
)


def actual_encoder():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    nodes = {node.name: node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    missing = set(REQUIRED) - set(nodes)
    if missing:
        raise AssertionError(f"Shipped Swarm encoder changed: missing {sorted(missing)}")

    class EmptyHookGroup:
        def __len__(self):
            return 0

    scope = {
        "torch": torch,
        "node_helpers": SimpleNamespace(conditioning_set_values=lambda values, _: values),
        "comfy": SimpleNamespace(hooks=SimpleNamespace(HookGroup=EmptyHookGroup)),
        "MAX_RESOLUTION": 16384,
        "PROMPT_TEMPLATE_ENCODE_VIDEO_I2V": "",
        "PROMPT_TEMPLATE_QWEN_IMAGE_EDIT_PLUS": "",
        "KREA2_TEMPLATE": "",
    }
    exec(compile(ast.Module(body=[nodes[name] for name in REQUIRED], type_ignores=[]),
                 str(SOURCE), "exec"), scope)
    return scope["SwarmTextEncodeAdvanced"]()


class FakeQwenClip:
    def __init__(self):
        self.apply_hooks_to_conds = None
        self.patcher = SimpleNamespace(forced_hooks=None)
        self.tokenizer = object()
        self.calls = []

    def tokenize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        # Upstream Qwen tokenizer has a scalar keep_vision entry besides tokens.
        return {"qwen3vl_8b": [[(42, 1.0)]], "keep_vision": False}

    def encode_from_tokens_scheduled(self, _tokens):
        return [[torch.ones(1, 1, 4), {"pooled_output": torch.ones(1, 4)}]]


class NativeEmptyTextContractCPU(unittest.TestCase):
    def run_encoder(self, native_qwen_option):
        clip = FakeQwenClip()
        (conds,) = actual_encoder().encode(
            clip=clip, steps=4, prompt="", width=1024, height=1024,
            target_width=1024, target_height=1024,
            images=torch.zeros(1, 32, 32, 3),
            qwen21_prevent_empty_text=native_qwen_option,
        )
        self.assertEqual(len(conds), 1)
        return clip.calls

    def test_empty_qwen_negative_uses_native_prevent_empty_text(self):
        calls = self.run_encoder(True)
        self.assertTrue(calls)
        self.assertTrue(all(kw.get("prevent_empty_text") is True for _, kw in calls))
        self.assertIn("", [text for text, _ in calls])

    def test_default_legacy_swarm_encoder_does_not_change(self):
        calls = self.run_encoder(False)
        self.assertTrue(calls)
        self.assertFalse(any("prevent_empty_text" in kw for _, kw in calls))


if __name__ == "__main__":
    unittest.main()
