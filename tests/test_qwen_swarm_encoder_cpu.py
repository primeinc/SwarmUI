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
    """Prove the real Swarm leaf combiner preserves the native Qwen template.

    Canonical QwenImage21Tokenizer adds the complete chat shell. Its inner
    Qwen3VLSDTokenizer (used by calc_leaf) already emits raw BPE token items:
    it does NOT add a chat template and needs no skip_template argument.
    The only special handling needed is an empty structural shell without
    prevent_empty_text's synthetic single-space negative instruction.
    """

    @staticmethod
    def fixture():
        scope = actual_encoder()._source_cpu_scope
        calls = {"outer": [], "raw": []}
        vision_image = object()

        class FakeQwenSDTokenizer:
            start_token = None
            end_token = None
            pad_token = -1

            def _try_get_embedding(self, _name):
                return None, None, None

            def tokenize_with_weights(self, text, disable_weights=False, **kwargs):
                # Inner Qwen3VLSDTokenizer inherits SDTokenizer and emits ONLY
                # raw leaf tokens; its accepted **kwargs are not chat-template
                # instructions. Adding Qwen shells here models the WRONG API.
                calls["raw"].append((text, disable_weights, kwargs))
                return [[(ord(ch) + 1000, 1.0) for ch in text]]

        class FakeQwenTokenizer:
            def __init__(self):
                self.qwen3vl_8b = FakeQwenSDTokenizer()

        class FakeFullClip:
            def __init__(self):
                self.tokenizer = FakeQwenTokenizer()

            def tokenize(self, text, **kwargs):
                calls["outer"].append((text, kwargs))
                if not text and kwargs.get("prevent_empty_text", False):
                    text = " "  # Upstream Qwen3VLTokenizer's actual rule.
                # Outer QwenImage21Tokenizer owns the SINGLE complete chat
                # shell, with one unchanged vision object across probes.
                prefix = [
                    (151644, 1.0), (101, 1.0),
                    (151644, 1.0), (102, 1.0),
                    ({"type": "image", "data": vision_image}, 1.0),
                ]
                suffix = [(151645, 1.0), (151644, 1.0), (103, 1.0)]
                user = [(ord(ch) + 1000, 1.0) for ch in text]
                return {"qwen3vl_8b": [prefix + user + suffix], "keep_vision": False}

        return scope, FakeFullClip(), calls

    @staticmethod
    def original_characters(batch):
        return "".join(chr(t - 1000) for t, _ in batch
                       if isinstance(t, int) and 1000 <= t < 2000)

    def test_weighted_prompt_has_one_chat_template_one_image_and_exact_words(self):
        scope, clip, calls = self.fixture()
        Leaf = scope["ParsedText"]
        authored = [
            Leaf(text="Girl", weight=1.5),
            Leaf(text=" and ", weight=1.0),
            Leaf(text="boy", weight=0.5),
        ]
        tokens, weighted = scope["combine_leaves"](
            clip, authored,
            lambda text: clip.tokenize(text, prevent_empty_text=True),
            per_leaf=True,
            blank_tokenize_fn=lambda text: clip.tokenize(text, prevent_empty_text=False))
        self.assertTrue(weighted)
        self.assertFalse(tokens["keep_vision"])
        batch = tokens["qwen3vl_8b"][0]
        values = [item[0] for item in batch]
        self.assertEqual(values.count(151644), 3, "Exactly one native chat shell")
        self.assertEqual(sum(isinstance(value, dict) for value in values), 1,
                         "One original image reference")
        self.assertEqual(self.original_characters(batch), "Girl and boy",
                         "No generated replacement text, duplicate template or extra space")
        # Adversarial control: recreate the previous unsafe shell builder.
        # Native Qwen substitutes one space for the empty prompt used as the
        # template shell, so older combining silently appended that space.
        unsafe, _ = scope["combine_leaves"](
            clip, authored, lambda text: clip.tokenize(text, prevent_empty_text=True),
            per_leaf=True)
        self.assertEqual(self.original_characters(unsafe["qwen3vl_8b"][0]),
                         "Girl and boy ",
                         "The negative control must actually reproduce the trailing-space bug")
        self.assertIn(1.5, [w for t, w in batch if isinstance(t, int) and t < 2000])
        self.assertIn(0.5, [w for t, w in batch if isinstance(t, int) and t < 2000])
        self.assertEqual([text for text, _, _ in calls["raw"]], ["Girl", " and ", "boy"])
        self.assertTrue(any(text == "" and kw.get("prevent_empty_text") is False
                            for text, kw in calls["outer"]),
                        "Nonempty authored text needs a BLANK structural shell")

    def test_true_empty_negative_still_gets_native_space(self):
        scope, clip, calls = self.fixture()
        tokens, weighted = scope["combine_leaves"](
            clip, [],
            lambda text: clip.tokenize(text, prevent_empty_text=True),
            per_leaf=True,
            blank_tokenize_fn=lambda text: clip.tokenize(text, prevent_empty_text=False))
        self.assertFalse(weighted)
        batch = tokens["qwen3vl_8b"][0]
        self.assertEqual(self.original_characters(batch), " ")
        self.assertTrue(any(text == "" and kw.get("prevent_empty_text") is True
                            for text, kw in calls["outer"]),
                        "The actual empty negative must retain upstream substitution")


if __name__ == "__main__":
    unittest.main()
