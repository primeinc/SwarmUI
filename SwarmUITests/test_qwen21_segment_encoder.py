"""CPU-only contract tests for the real Swarm Qwen 2.1 Comfy encoder.

Uses stand-in tensor/CLIP/VAE objects to execute the shipped node without
requiring a GPU, torch installation, or a running Comfy backend.
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        shape = list(self.shape)
        for axis, item in enumerate(keys):
            if isinstance(item, slice):
                first, last, stride = item.indices(shape[axis])
                shape[axis] = len(range(first, last, stride))
            else:
                raise AssertionError("Only slice indexing is expected in Qwen segment encoding")
        return FakeTensor(shape)

    def movedim(self, source, destination):
        shape = list(self.shape)
        size = shape.pop(source)
        shape.insert(destination, size)
        return FakeTensor(shape)

    def __mul__(self, _other):
        return FakeTensor(self.shape)

    __rmul__ = __mul__

    def __add__(self, _other):
        return FakeTensor(self.shape)

    __radd__ = __add__

    def __rsub__(self, _other):
        return FakeTensor(self.shape)


class FakeVae:
    def __init__(self):
        self.calls = 0
        self.tiled_calls = []

    def encode(self, image):
        assert image.shape == (1, 64, 64, 4)
        self.calls += 1
        return FakeTensor((1, 64, 4, 4))

    def encode_tiled(self, image, **kwargs):
        assert image.shape == (1, 64, 64, 4)
        self.tiled_calls.append(kwargs)
        return FakeTensor((1, 64, 4, 4))


class FakeClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, images, keep_vision, prevent_empty_text):
        self.calls.append({
            "prompt": prompt,
            "images": images,
            "keep_vision": keep_vision,
            "prevent_empty_text": prevent_empty_text,
        })
        return {"prompt": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return dict(tokens)


def load_encoder():
    torch_stub = types.ModuleType("torch")
    torch_stub.zeros = lambda shape, device=None: FakeTensor(shape)

    comfy_stub = types.ModuleType("comfy")
    comfy_stub.__path__ = []
    utils_stub = types.ModuleType("comfy.utils")
    def unexpected_upscale(*args):
        raise AssertionError("Aligned fake image must not be rescaled")
    utils_stub.common_upscale = unexpected_upscale
    model_management_stub = types.ModuleType("comfy.model_management")
    model_management_stub.intermediate_device = lambda: "cpu"
    comfy_stub.utils = utils_stub
    comfy_stub.model_management = model_management_stub

    node_helpers_stub = types.ModuleType("node_helpers")
    def conditioning_set_values(conditioning, values, append=False):
        result = dict(conditioning)
        result.update(values)
        return result
    node_helpers_stub.conditioning_set_values = conditioning_set_values

    stubs = {
        "torch": torch_stub,
        "comfy": comfy_stub,
        "comfy.utils": utils_stub,
        "comfy.model_management": model_management_stub,
        "node_helpers": node_helpers_stub,
    }
    path = Path(__file__).resolve().parents[1] / (
        "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/"
        "SwarmComfyCommon/SwarmQwen.py"
    )
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("fake_swarm_contract.SwarmQwen", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.SwarmQwenImage21SegmentEncode


class Qwen21SegmentEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Encoder = load_encoder()

    def setUp(self):
        self.clip = FakeClip()
        self.vae = FakeVae()
        self.image = FakeTensor((1, 64, 64, 4))

    def test_existing_workflows_default_to_reference_enabled(self):
        encoder = self.Encoder()
        positive, negative, latent = encoder.encode(
            self.clip, self.vae, self.image, "person", "negative"
        )
        self.assertEqual(self.vae.calls, 1)
        self.assertEqual(len(self.clip.calls), 2)
        self.assertTrue(all(len(call["images"]) == 1 for call in self.clip.calls))
        self.assertTrue(all(call["keep_vision"] is False for call in self.clip.calls))
        self.assertEqual(len(positive["reference_latents"]), 1)
        self.assertEqual(len(negative["reference_latents"]), 1)
        self.assertEqual(latent["samples"].shape, (1, 64, 4, 4))

    def test_reference_disabled_removes_both_vision_and_vae_references(self):
        encoder = self.Encoder()
        positive, negative, latent = encoder.encode(
            self.clip, self.vae, self.image, "person", "negative",
            use_original_reference=False,
        )
        self.assertEqual(self.vae.calls, 0)
        self.assertEqual(len(self.clip.calls), 2)
        self.assertTrue(all(call["images"] == [] for call in self.clip.calls))
        self.assertTrue(all(call["keep_vision"] is True for call in self.clip.calls))
        self.assertNotIn("reference_latents", positive)
        self.assertNotIn("reference_latents", negative)
        self.assertEqual(latent["samples"].shape, (1, 64, 4, 4))

    def test_optional_prompt_image_uses_its_own_qwen_reference_slot(self):
        encoder = self.Encoder()
        positive, negative, _latent = encoder.encode(
            self.clip, self.vae, self.image, "person", "negative",
            image_1=FakeTensor((1, 64, 64, 4)),
        )
        self.assertEqual(self.vae.calls, 2)
        self.assertTrue(all(len(call["images"]) == 2 for call in self.clip.calls))
        self.assertEqual(len(positive["reference_latents"]), 2)
        self.assertEqual(len(negative["reference_latents"]), 2)

    def test_tiled_reference_encoder_uses_configured_options_for_all_references(self):
        encoder = self.Encoder()
        positive, negative, _ = encoder.encode(
            self.clip, self.vae, self.image, "person", "negative",
            image_1=FakeTensor((1, 64, 64, 4)),
            reference_tile_size=512,
            reference_tile_overlap=64,
            reference_temporal_tile_size=32,
            reference_temporal_tile_overlap=4,
        )
        self.assertEqual(self.vae.calls, 0)
        self.assertEqual(len(self.vae.tiled_calls), 2)
        for call in self.vae.tiled_calls:
            self.assertEqual(call, {
                "tile_x": 512, "tile_y": 512,
                "overlap": 64, "tile_t": 32, "overlap_t": 4,
            })
        self.assertEqual(len(positive["reference_latents"]), 2)
        self.assertEqual(len(negative["reference_latents"]), 2)

    def test_encoder_refuses_to_alias_a_batched_source_to_first_crop(self):
        encoder = self.Encoder()
        with self.assertRaisesRegex(ValueError, "one source crop"):
            encoder.encode(
                self.clip, self.vae, FakeTensor((2, 64, 64, 4)),
                "person", "negative",
            )

    def test_advanced_prompt_routes_hooks_steps_and_qwen_image_slots(self):
        class FakeAdvanced:
            calls = []

            def encode(self, clip, steps, prompt, width, height, target_width, target_height, **kwargs):
                self.calls.append({
                    "steps": steps, "prompt": prompt, "images": kwargs["images"],
                    "lora_hooks": kwargs["lora_hooks"],
                    "tokenize_kwargs": kwargs["tokenize_kwargs"],
                })
                return ({"scheduled_prompt": prompt},)

        pkg = types.ModuleType("fake_swarm_contract")
        pkg.__path__ = []
        advanced_module = types.ModuleType("fake_swarm_contract.SwarmText")
        advanced_module.SwarmTextEncodeAdvanced = FakeAdvanced
        hooks = object()
        with patch.dict(sys.modules, {
            "fake_swarm_contract": pkg,
            "fake_swarm_contract.SwarmText": advanced_module,
        }):
            pos, neg, _ = self.Encoder().encode(
                self.clip, self.vae, self.image,
                "<fromto[0.5]:first,second> <lora:face//hook=0>", "negative",
                steps=12, lora_hooks=hooks,
            )
        self.assertEqual(len(FakeAdvanced.calls), 2)
        self.assertEqual(FakeAdvanced.calls[0]["steps"], 12)
        self.assertIs(FakeAdvanced.calls[0]["lora_hooks"], hooks)
        self.assertIn("fromto", FakeAdvanced.calls[0]["prompt"])
        self.assertEqual(len(FakeAdvanced.calls[0]["images"]), 1)
        self.assertEqual(FakeAdvanced.calls[0]["tokenize_kwargs"],
                         {"keep_vision": False, "prevent_empty_text": True})
        self.assertIn("reference_latents", pos)
        self.assertIn("reference_latents", neg)

    def test_optional_input_preserves_saved_workflow_compatibility(self):
        inputs = self.Encoder.INPUT_TYPES()
        self.assertIn("use_original_reference", inputs["optional"])
        self.assertTrue(inputs["optional"]["use_original_reference"][1]["default"])


if __name__ == "__main__":
    unittest.main()
