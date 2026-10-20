"""CPU controls of the registered Swarm Qwen Image 2.1 empty-latent node.

The native ComfyUI TextEncodeQwenImage21 produces 64 channels at a 16x
spatial ratio. Generic EmptyLatentImage produces only 4 channels.
These tests run the actual custom node, not a separate tensor reimplementation.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

LATENTS = (Path(__file__).resolve().parents[1] /
           "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmLatents.py")


def load_native_node():
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.SimpleNamespace(
        intermediate_device=lambda: torch.device("cpu"))
    spec = importlib.util.spec_from_file_location("swarm_qwen21_latents", LATENTS)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"comfy": comfy}):
        spec.loader.exec_module(module)
    return module.NODE_CLASS_MAPPINGS["SwarmEmptyQwenImage21Latent"]()


class EmptyQwenImage21LatentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = load_native_node()

    def test_reference_shape_is_64_channels_one_sixteenth(self):
        result, = self.node.generate(width=1024, height=1024, batch_size=1)
        self.assertEqual(tuple(result["samples"].shape), (1, 64, 64, 64))
        self.assertEqual(torch.count_nonzero(result["samples"]).item(), 0)
        self.assertEqual(result["samples"].device.type, "cpu")

    def test_rectangle_and_batch_are_not_forced_square(self):
        result, = self.node.generate(width=1024, height=768, batch_size=2)
        self.assertEqual(tuple(result["samples"].shape), (2, 64, 48, 64))
        self.assertEqual(torch.count_nonzero(result["samples"]).item(), 0)

    def test_schema_is_real_comfy_required_socket_contract(self):
        required = self.node.INPUT_TYPES()["required"]
        self.assertEqual(set(required), {"width", "height", "batch_size"})
        self.assertEqual(self.node.RETURN_TYPES, ("LATENT",))
        self.assertEqual(self.node.FUNCTION, "generate")


if __name__ == "__main__":
    unittest.main()
