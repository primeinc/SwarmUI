"""Adversarial CPU pixel controls for segmented LoRA effect versus leakage.

No inference. Tests evaluate the same independent image-space acceptance
metric used when real Qwen LoRA OFF/ON/OFF images are supplied.
"""
import unittest
from PIL import Image, ImageDraw
from integration.qwen_segment_lora_confinement import (
    region_rms, assert_same_detector_mask,
)


class QwenSegmentLoraConfinementCPUTests(unittest.TestCase):
    @staticmethod
    def sample():
        base = Image.new("RGB", (256, 256), (20, 30, 40))
        mask = Image.new("L", base.size, 0)
        ImageDraw.Draw(mask).rectangle((104, 104, 151, 151), fill=255)
        return base, mask

    def test_change_on_detected_subject_is_measured_inside_only(self):
        base, mask = self.sample()
        variant = base.copy()
        ImageDraw.Draw(variant).rectangle((112, 112, 142, 142), fill=(220, 30, 40))
        inside, outside = region_rms(base, variant, mask)
        self.assertGreater(inside, 30.0)
        self.assertEqual(outside, 0.0)

    def test_change_only_outside_mask_cannot_fake_segment_lora_effect(self):
        base, mask = self.sample()
        variant = base.copy()
        ImageDraw.Draw(variant).rectangle((0, 0, 31, 31), fill=(220, 30, 40))
        inside, outside = region_rms(base, variant, mask)
        self.assertEqual(inside, 0.0)
        self.assertGreater(outside, 5.0)

    def test_soft_blend_near_mask_edge_is_excluded_from_leakage_measure(self):
        base, mask = self.sample()
        variant = base.copy()
        ImageDraw.Draw(variant).rectangle((151, 120, 160, 130), fill=(220, 30, 40))
        _, outside = region_rms(base, variant, mask)
        self.assertEqual(outside, 0.0)

    def test_empty_and_whole_frame_detection_fail_clearly(self):
        base, _ = self.sample()
        for mask, expected in [(Image.new("L", base.size, 0), "empty"),
                               (Image.new("L", base.size, 255), "covers")]:
            with self.assertRaisesRegex(AssertionError, expected):
                region_rms(base, base.copy(), mask)

    def test_cannot_compare_images_or_masks_with_different_geometry(self):
        base, mask = self.sample()
        with self.assertRaisesRegex(AssertionError, "dimensions"):
            region_rms(base, Image.new("RGB", (100, 100)), mask)
        with self.assertRaisesRegex(AssertionError, "dimensions"):
            region_rms(base, base, Image.new("L", (100, 100)))

    def test_off_on_off_must_share_actual_detector_mask(self):
        _, mask = self.sample()
        self.assertEqual(assert_same_detector_mask(mask, mask.copy(), mask).size, mask.size)
        changed = mask.copy()
        changed.putpixel((0, 0), 255)
        with self.assertRaisesRegex(AssertionError, "different detector masks"):
            assert_same_detector_mask(mask, changed, mask)
        with self.assertRaisesRegex(AssertionError, "Missing actual"):
            assert_same_detector_mask(mask, None, mask)


if __name__ == "__main__":
    unittest.main()
