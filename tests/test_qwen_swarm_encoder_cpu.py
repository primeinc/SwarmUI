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
    "uniform_weight", "is_special_token", "embed_token_items",
    "SwarmTextEncodeAdvanced",
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
    encoder = scope["SwarmTextEncodeAdvanced"]()
    encoder._source_cpu_scope = scope
    return encoder


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


class RealWeightedQwenTokenizationCPU(unittest.TestCase):
    """Probe real Swarm leaf combiner, including chat template and image slots."""

    def test_weighted_prompt_has_exactly_one_chat_template_and_one_image(self):
        scope = actual_encoder()._source_cpu_scope
        calls = []
        image = object()

        class FakeQwenSDTokenizer:
            start_token = None
            end_token = None
            pad_token = -1

            def _try_get_embedding(self, _name):
                return None, None, None

            def tokenize_with_weights(self, text, disable_weights=False,
                                      skip_template=False, **kwargs):
                calls.append((text, skip_template, kwargs))
                # Mirror upstream Qwen3VLTokenizer.prevent_empty_text. A
                # nonempty weighted prompt must not acquire this blank token.
                if text == "" and kwargs.get("prevent_empty_text", False):
                    text = " "
                user = [(ord(ch) + 1000, 1.0) for ch in text]
                if skip_template:
                    return [user]
                # Three native chat-turn boundaries and one Qwen image placeholder.
                shell = [(151644, 1.0), (101, 1.0), (151644, 1.0), (102, 1.0)]
                vision = [({"type": "image", "data": image}, 1.0)]
                end = [(151645, 1.0), (151644, 1.0), (103, 1.0)]
                return [shell + vision + user + end]

        class FakeQwenTokenizer:
            def __init__(self):
                self.qwen3vl_8b = FakeQwenSDTokenizer()

        class FakeFullClip:
            def __init__(self):
                self.tokenizer = FakeQwenTokenizer()

            def tokenize(self, text, **kwargs):
                return {
                    "qwen3vl_8b": self.tokenizer.qwen3vl_8b.tokenize_with_weights(text, **kwargs),
                    "keep_vision": False,
                }

        clip = FakeFullClip()
        ParsedText = scope["ParsedText"]
        authored = [
            ParsedText(text="Girl", weight=1.5),
            ParsedText(text=" and ", weight=1),
            ParsedText(text="boy", weight=0.5),
        ]
        combine = scope["combine_leaves"]
        tokens, weighted = combine(
            clip, authored, lambda text: clip.tokenize(text, prevent_empty_text=True),
            per_leaf=True, qwen21_mode=True,
            blank_tokenize_fn=lambda text: clip.tokenize(text, prevent_empty_text=False))
        self.assertTrue(weighted)
        self.assertFalse(tokens["keep_vision"])
        batch = tokens["qwen3vl_8b"][0]
        values = [item[0] for item in batch]
        self.assertEqual(values.count(151644), 3,
                         "One native chat template, not a full template for each weighted leaf")
        self.assertEqual(sum(isinstance(value, dict) for value in values), 1,
                         "Image reference must occur exactly once")
        self.assertEqual(
            "".join(chr(t - 1000) for t in values
                    if isinstance(t, int) and t >= 1000 and t != 151644
                    and t != 151645),
            "Girl and boy",
            "Neither author text nor a synthetic empty-negative space may be inserted")
        authored_weights = [w for t, w in batch if isinstance(t, int)
                            and 1000 <= t < 2000]
        self.assertIn(1.5, authored_weights)
        self.assertIn(0.5, authored_weights)
        self.assertTrue(all(skip for text, skip, kw in calls
                            if text in ("Girl", " and ", "boy")),
                        "Fragments must tokenize without duplicating Qwen's template")
        self.assertTrue(any(text == "" and not kw.get("prevent_empty_text", True)
                            for text, skip, kw in calls),
                        "The nonempty prompt's structural shell cannot include the empty-negative space")


if __name__ == "__main__":
    unittest.main()
