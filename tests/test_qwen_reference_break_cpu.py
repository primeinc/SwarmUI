"""CPU execution of the shipped Swarm Qwen text encoder's image-slot contract.

This deliberately uses the real Swarm parser/encoder, with a small deterministic
CLIP test double. Qwen model weights are not loaded. The oracle for image_slots
comes from ComfyUI's QwenImage21Tokenizer / build_sequence: a reference latent
must be spliced once, not repeated in every Swarm <break> chunk.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmText.py"


def load_shipped_encoder():
    helpers = types.ModuleType("node_helpers")
    helpers.conditioning_set_values = lambda x, vals, append=False: x
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    hooks = types.ModuleType("comfy.hooks")
    class HookGroup:
        def __init__(self):
            self.hooks = []
        def __len__(self):
            return len(self.hooks)
    hooks.HookGroup = HookGroup
    comfy.hooks = hooks
    nodes = types.ModuleType("nodes")
    nodes.MAX_RESOLUTION = 16384
    package = types.ModuleType("swarm_qwen_break_test")
    package.__path__ = []
    handling = types.ModuleType("swarm_qwen_break_test.SwarmTextHandling")
    for name in ("PROMPT_TEMPLATE_ENCODE_VIDEO_I2V",
                 "PROMPT_TEMPLATE_QWEN_IMAGE_EDIT_PLUS", "KREA2_TEMPLATE"):
        setattr(handling, name, "{}")
    modules = {
        "node_helpers": helpers,
        "comfy": comfy,
        "comfy.hooks": hooks,
        "nodes": nodes,
        package.__name__: package,
        handling.__name__: handling,
    }
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "swarm_qwen_break_test.SwarmText", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.SwarmTextEncodeAdvanced


class FakeQwenTokenizer:
    start_token = None
    end_token = None
    pad_token = -1
    max_length = 16384
    max_word_length = 16384
    min_length = 1
    pad_to_max_length = False

    def _try_get_embedding(self, name):
        return None, None, None

    def tokenize_with_weights(self, text, **kwargs):
        return [[(1000 + ord(char), 1.0) for char in text]]


class RecordingQwenClip:
    """Models Qwen tokenizer's reference embedding and image_slots metadata."""
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(qwen3vl_8b=FakeQwenTokenizer())
        self.apply_hooks_to_conds = None
        self.patcher = types.SimpleNamespace(forced_hooks=None)
        self.image_slots_per_encoding = []

    def tokenize(self, text, images=None, **kwargs):
        image_refs = [({"type": "image", "data": image}, 1.0)
                      for image in (images if images is not None else [])]
        system_template = [(151644, 1.0)] + [
            (20 + index, 1.0) for index in range(6)]
        words = [(1000 + ord(char), 1.0) for char in text]
        return {
            "qwen3vl_8b": [system_template + image_refs + words + [(151645, 1.0)]],
            "keep_vision": False,
        }

    def encode_from_tokens_scheduled(self, tokens):
        content = tokens["qwen3vl_8b"][0]
        positions = [index for index, item in enumerate(content)
                     if isinstance(item[0], dict) and item[0].get("type") == "image"]
        self.image_slots_per_encoding.append(positions)
        # In Qwen, vision embeddings are replaced by reference latents.
        return [[torch.ones((1, len(content) - len(positions), 4)),
                 {"pooled_output": torch.ones((1, 4)), "image_slots": positions}]]


class QwenSwarmBreakReferenceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = load_shipped_encoder()

    def encode(self, prompt, **image_inputs):
        clip = RecordingQwenClip()
        values = self.encoder().encode(
            clip, steps=4, prompt=prompt, width=64, height=64,
            target_width=64, target_height=64,
            qwen21_prevent_empty_text=True,
            **image_inputs)
        return clip, values[0][0][1]

    def test_single_reference_is_not_duplicated_across_break(self):
        source = torch.ones((1, 64, 64, 3))
        clip, result = self.encode("Replace <break> the girl", images=source)
        self.assertEqual(len(clip.image_slots_per_encoding), 2)
        self.assertEqual([len(x) for x in clip.image_slots_per_encoding], [1, 0],
                         "Qwen reference latent must occur once, never again in the second Swarm break chunk")
        self.assertEqual(len(result["image_slots"]), 1)

    def test_two_independent_references_are_not_duplicated_across_break(self):
        refs = {"image_1": torch.ones((1, 64, 96, 3)),
                "image_2": torch.ones((1, 96, 64, 3))}
        clip, result = self.encode("Change <break> the person", **refs)
        self.assertEqual([len(x) for x in clip.image_slots_per_encoding], [2, 0])
        self.assertEqual(len(result["image_slots"]), 2)

    def test_single_chunk_and_plain_text_remain_unchanged(self):
        ref = torch.ones((1, 64, 64, 3))
        clip, result = self.encode("Replace the person", images=ref)
        self.assertEqual([len(x) for x in clip.image_slots_per_encoding], [1])
        self.assertEqual(len(result["image_slots"]), 1)
        plain, result = self.encode("First <break> second")
        self.assertEqual([len(x) for x in plain.image_slots_per_encoding], [0, 0])
        self.assertEqual(len(result["image_slots"]), 0)


if __name__ == "__main__":
    unittest.main()
