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

    def test_weak_detection_rejected_before_upstream_hipass_removes_it(self):
        weak_mask = torch.full((1, 64, 80), 0.2, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "native crop cutoff"):
            self.node.validate(weak_mask, minimum_confidence=0.5)

    def test_exact_cutoff_is_rejected_because_native_threshold_discards_it(self):
        mask = torch.zeros((1, 64, 80), dtype=torch.float32)
        mask[0, 20, 30] = 0.5
        with self.assertRaises(ValueError):
            self.node.validate(mask, minimum_confidence=0.5)

    def test_above_cutoff_survives(self):
        mask = torch.zeros((1, 64, 80), dtype=torch.float32)
        mask[0, 20, 30] = 0.5001
        self.assertIs(self.node.validate(mask, minimum_confidence=0.5)[0], mask)

    def test_batch_with_only_weak_second_detection_fails(self):
        mask = torch.full((2, 64, 80), 0.2, dtype=torch.float32)
        mask[0, 20, 30] = 0.8
        with self.assertRaises(ValueError):
            self.node.validate(mask, minimum_confidence=0.5)

    def test_authored_low_clipseg_threshold_preserves_all_selected_pixels(self):
        # SwarmClipSeg already thresholds confidence at the user's 0.1.
        # Its subsequent normalization leaves valid selected pixels at 0.25
        # and 0.5; a second hard 0.5 cutoff silently deletes both.
        scores = torch.tensor([0.0, 0.2, 0.4, 0.8], dtype=torch.float32).reshape(1, 1, 4)
        detected = torch.nn.functional.threshold(scores, 0.1, 0.0)
        detected -= detected.min()
        detected /= detected.max()
        output, = self.node.validate(detected, minimum_confidence=0.0)
        self.assertIs(output, detected)
        binary = (output > 0.0).float()  # production ThresholdMask(value=0)
        self.assertEqual(binary.flatten().tolist(), [0.0, 1.0, 1.0, 1.0])
        self.assertEqual((binary > 0.5).float().flatten().tolist(),
                         [0.0, 1.0, 1.0, 1.0])  # upstream crop hipass=0.5 is unchanged
        self.assertEqual((output > 0.5).float().flatten().tolist(),
                         [0.0, 0.0, 0.0, 1.0])  # negative control: old graph lost pixels

    def test_rejects_all_zero_batch(self):
        with self.assertRaises(ValueError):
            self.node.validate(torch.zeros((2, 32, 32), dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
