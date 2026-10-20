"""Real CPU-tensor checks of the Swarm mask guard, including a negative control."""
from pathlib import Path
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

import torch

MASKS = (Path(__file__).resolve().parents[1] /
         "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmMasks.py")


def get_real_node():
    module = types.ModuleType("comfy")
    module.model_management = types.SimpleNamespace(
        intermediate_device=lambda: torch.device("cpu"),
        get_torch_device=lambda: torch.device("cpu"),
    )
    spec = importlib.util.spec_from_file_location("actual_swarm_masks", MASKS)
    loaded = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"comfy": module}):
        spec.loader.exec_module(loaded)
    return loaded.SwarmRequireNonEmptyMask()


class MaskGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = get_real_node()

    def test_positive_real_mask_is_passed_through_unchanged(self):
        mask = torch.zeros((1, 64, 80), dtype=torch.float32)
        mask[0, 12:33, 16:48] = 0.6
        output, = self.node.validate(mask)
        self.assertIs(output, mask)
        self.assertTrue(torch.equal(output, mask))

    def test_empty_mask_causes_real_failing_control(self):
        with self.assertRaisesRegex(ValueError, "no subject detected"):
            self.node.validate(torch.zeros((1, 64, 80), dtype=torch.float32))

    def test_mixed_batch_rejects_even_if_another_image_detects_subject(self):
        mask = torch.zeros((2, 32, 32), dtype=torch.float32)
        mask[0, 8:16, 8:16] = 0.8
        with self.assertRaisesRegex(ValueError, "one or more masks"):
            self.node.validate(mask)

    def test_rejects_all_zero_batch(self):
        with self.assertRaises(ValueError):
            self.node.validate(torch.zeros((2, 32, 32), dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
