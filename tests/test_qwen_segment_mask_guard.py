"""Real CPU-tensor checks of the Swarm mask guard, including a negative control."""
from pathlib import Path
import ast
import numpy as np
from PIL import Image
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


def actual_clipseg_with_controlled_logits():
    """Execute the shipped SwarmClipSeg.seg, replacing only model logits.

    This is a negative detector-mask contract test, NOT model inference.
    A controlled weak background response exposes the normalization that can
    produce almost whole-canvas positive support after authored thresholding.
    """
    source_path = (Path(__file__).resolve().parents[1] /
                   "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmClipSeg.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    klass = next(n for n in tree.body
                 if isinstance(n, ast.ClassDef) and n.name == "SwarmClipSeg")

    class ProbeProcessor:
        @classmethod
        def from_pretrained(cls, _path):
            return cls()

        def __call__(self, **_kwargs):
            return {}

    class ControlledLogits:
        @classmethod
        def from_pretrained(cls, _path):
            return cls()

        def __call__(self, **_kwargs):
            probabilities = torch.linspace(0.2, 0.4, 64).reshape(8, 8)
            probabilities[2:6, 2:6] = 0.95
            return [torch.logit(probabilities)]

    bindings = {
        "np": np, "torch": torch, "Image": Image,
        "get_path": lambda: "/no-network",
        "download_model": lambda *_args: None,
        "CLIPSegProcessor": ProbeProcessor,
        "CLIPSegForImageSegmentation": ControlledLogits,
    }
    exec(compile(ast.Module(body=[klass], type_ignores=[]),
                 str(source_path), "exec"), bindings)
    return bindings["SwarmClipSeg"]()


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

    def test_nearly_full_canvas_detection_fails_instead_of_replacing_background(self):
        # Adversarial scores: authored CLIPSeg threshold 0.1 retains weak
        # background everywhere except a tiny corner. Native ThresholdMask(0)
        # would otherwise turn 99.6% of the original image opaque.
        mask = torch.full((1, 32, 32), 0.2, dtype=torch.float32)
        mask[:, :2, :2] = 0.0
        with self.assertRaisesRegex(ValueError, "near.*entire source canvas"):
            self.node.validate(mask, minimum_confidence=0.0, maximum_coverage=0.99)

    def test_centered_subject_at_low_authored_threshold_remains_valid(self):
        mask = torch.zeros((1, 64, 80), dtype=torch.float32)
        mask[:, 10:36, 15:45] = 0.2
        output, = self.node.validate(
            mask, minimum_confidence=0.0, maximum_coverage=0.99)
        self.assertIs(output, mask)
        self.assertAlmostEqual(float(output.max()), 0.2, places=6)

    def test_batch_rejects_only_broad_member_without_hiding_behind_valid_mask(self):
        mask = torch.zeros((2, 32, 32), dtype=torch.float32)
        mask[0, 10:20, 10:20] = 0.9
        mask[1] = 0.1
        mask[1, :2, :2] = 0
        with self.assertRaisesRegex(ValueError, "nearly the entire source canvas"):
            self.node.validate(mask, minimum_confidence=0.0, maximum_coverage=0.99)

    def test_exact_maximum_coverage_allowed_but_larger_rejected(self):
        mask = torch.zeros((1, 10, 10), dtype=torch.float32)
        mask[:, :, :9] = 0.2
        self.node.validate(mask, minimum_confidence=0.0, maximum_coverage=0.90)
        with self.assertRaises(ValueError):
            self.node.validate(mask, minimum_confidence=0.0, maximum_coverage=0.89)

    def test_invalid_maximum_coverage_rejected(self):
        mask = torch.zeros((1, 10, 10), dtype=torch.float32)
        mask[:, 2:8, 2:8] = 0.6
        for invalid in (0, -0.01, 1.1):
            with self.assertRaises(ValueError):
                self.node.validate(mask, minimum_confidence=0.0,
                                   maximum_coverage=invalid)

    def test_real_clipseg_postprocess_near_full_support_is_rejected(self):
        # This executes shipped SwarmClipSeg.seg normalization against known
        # detector logits. It does NOT pretend to detect a real subject.
        # At authored threshold 0.1, weak background scores survive and the
        # native ThresholdMask(value=0) would make them opaque.
        image = torch.zeros((1, 128, 128, 3), dtype=torch.float32)
        (detected,) = actual_clipseg_with_controlled_logits().seg(image, "girl", 0.1)
        coverage = float((detected > 0).float().mean())
        self.assertGreater(coverage, 0.99,
                           "The positive-control postprocess no longer exposes broad support")
        with self.assertRaisesRegex(ValueError, "nearly the entire source canvas"):
            self.node.validate(detected, minimum_confidence=0.0, maximum_coverage=0.99)

    def test_inverse_segment_can_intentionally_edit_more_than_99_percent(self):
        original = torch.zeros((1, 100, 100), dtype=torch.float32)
        original[0, 0, 0] = 0.9  # explicit small subject selected by author
        self.assertIs(self.node.validate(
            original, minimum_confidence=0.0, maximum_coverage=1.0)[0], original)
        complement = 1.0 - original
        self.assertGreater(float((complement > 0).float().mean()), 0.99)
        # Old hardcoded positive-only cap wrongly rejected this valid inverse.
        with self.assertRaises(ValueError):
            self.node.validate(
                complement, minimum_confidence=0.0, maximum_coverage=0.99)
        self.assertIs(self.node.validate(
            complement, minimum_confidence=0.0, maximum_coverage=1.0)[0], complement)

    def test_inverse_segment_rejects_empty_detector_before_inverting(self):
        absent = torch.zeros((1, 64, 64), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "no subject detected"):
            self.node.validate(absent, minimum_confidence=0.0, maximum_coverage=1.0)
        # A naive complement would be 100% opaque and wrongly edit the frame.
        inverted_empty = 1.0 - absent
        self.assertEqual(float(inverted_empty.min()), 1.0)
        # A detector that selected literally every pixel instead has an EMPTY
        # inverse target; its post-invert guard must reject rather than crop all.
        fully_detected = torch.ones((1, 64, 64), dtype=torch.float32)
        self.node.validate(fully_detected, minimum_confidence=0.0, maximum_coverage=1.0)
        with self.assertRaisesRegex(ValueError, "no subject detected"):
            self.node.validate(1.0 - fully_detected,
                               minimum_confidence=0.0, maximum_coverage=1.0)

    def test_rejects_all_zero_batch(self):
        with self.assertRaises(ValueError):
            self.node.validate(torch.zeros((2, 32, 32), dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
