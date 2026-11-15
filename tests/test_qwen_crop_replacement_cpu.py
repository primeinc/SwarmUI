"""Deterministic real-tensor CropAndStitch replacement test.

This checks pixel compositing and original-canvas preservation using the
unchanged upstream adapter. It does NOT test Qwen or LoRA inference.
"""
import unittest
import torch

from inpaint_cropandstitch import InpaintCropImproved, InpaintStitchImproved
from test_qwen_segment_mask_guard import get_real_node


class CropReplacementTests(unittest.TestCase):
    def test_old_subject_pixels_are_replaced_when_model_supplies_new_pixels(self):
        source = torch.ones((1, 96, 96, 3), dtype=torch.float32)
        source[:, 30:66, 30:66] = 0.0  # original dark rectangular subject
        mask = torch.zeros((1, 96, 96), dtype=torch.float32)
        mask[:, 30:66, 30:66] = 1.0
        # Exercise the ACTUAL Swarm guard before the native threshold and the
        # unchanged upstream crop/stitch. No Qwen weights or GPU are needed to
        # execute and falsify this pixel-compositor contract.
        detector_confidence = mask * 0.6
        guard = get_real_node()
        # CLIPSeg already applied the author's detection threshold. Native
        # Qwen's CURRENT graph guards nonempty support at zero and binarizes
        # at zero; the previous 0.5/0.5 fixture silently tested an obsolete
        # graph and discarded detector-selected soft pixels.
        validated, = guard.validate(detector_confidence, minimum_confidence=0.0)
        self.assertIs(validated, detector_confidence)
        weak_selected = mask * 0.2
        self.assertIs(guard.validate(weak_selected, minimum_confidence=0.0)[0],
                      weak_selected)
        with self.assertRaisesRegex(ValueError, "no subject detected"):
            guard.validate(torch.zeros_like(mask), minimum_confidence=0.0)
        # Comfy-Org/ComfyUI ThresholdMask.execute uses (mask > value).float().
        solid_mask = (validated > 0.0).float()
        self.assertEqual(torch.count_nonzero(solid_mask).item(), 36 * 36)

        crop = InpaintCropImproved()
        stitch = InpaintStitchImproved()
        args = dict(
            image=source, mask=solid_mask,
            downscale_algorithm="bilinear", upscale_algorithm="bicubic",
            preresize=False, preresize_mode="ensure minimum resolution",
            preresize_min_width=1024, preresize_min_height=1024,
            preresize_max_width=16384, preresize_max_height=16384,
            mask_fill_holes=True, mask_expand_pixels=8,
            mask_invert=False, mask_blend_pixels=4, mask_hipass_filter=0.5,
            extend_for_outpainting=False, extend_up_factor=1.0,
            extend_down_factor=1.0, extend_left_factor=1.0,
            extend_right_factor=1.0, context_from_mask_extend_factor=1.2,
            output_resize_to_target_size=True, output_target_width=128,
            output_target_height=128, output_padding="32",
            device_mode="cpu (compatible)"
        )
        stitcher, crop_image, crop_mask = crop.inpaint_crop(**args)[:3]
        self.assertGreater(torch.count_nonzero(crop_mask).item(), 0)
        replacement = torch.ones_like(crop_image) * 0.8
        result = stitch.inpaint_stitch(stitcher, replacement)[0]
        self.assertEqual(result.shape, source.shape)
        # Source subject must be gone if the sampler emits a truly different
        # crop and the adapter's grown mask fully covers the old subject.
        old_region = result[0, 30:66, 30:66, :]
        # The replacement crop is 0.8; accepting >0.05 would let an
        # 80-percent-visible dark old subject pass as "no ghosting".
        # Grow exceeds the blend radius, so the entire original subject
        # should receive mostly replacement pixels.
        self.assertGreater(old_region.min().item(), 0.65)
        # A real CLIPSeg detector produces fractional confidence values.
        # The native stitcher algebra blends old and new pixels if that
        # fractional confidence is fed directly into its mask input.
        # Contrast the native adapter with the unthresholded 0.6 mask to
        # establish that thresholding is necessary, not cosmetic.
        soft_mask = detector_confidence
        soft_stitcher, soft_image, _ = crop.inpaint_crop(
            **{**args, "mask": soft_mask})[:3]
        soft_result = stitch.inpaint_stitch(
            soft_stitcher, torch.ones_like(soft_image) * 0.8)[0]
        self.assertLess(soft_result[0, 45, 45, 0].item(), 0.55)
        self.assertGreater(result[0, 45, 45, 0].item(), 0.7)
        # Preserve untouched context, not a newly generated full frame.
        self.assertTrue(torch.allclose(result[0, :8, :8], source[0, :8, :8], atol=0.0001))


    def test_native_context_filter_rejects_weak_background_speckles(self):
        """Native CropAndStitch filters optional context too: do not invent a bug."""
        image = torch.ones((1, 256, 256, 3), dtype=torch.float32)
        subject = torch.zeros((1, 256, 256), dtype=torch.float32)
        subject[:, 112:145, 112:145] = 1.0
        weak_context = subject.clone()
        weak_context[:, :8, :8] = 0.1
        weak_context[:, -8:, -8:] = 0.2
        strong_context = weak_context.clone()
        strong_context[:, :8, :8] = 0.8

        args = dict(
            image=image, mask=subject,
            downscale_algorithm="bilinear", upscale_algorithm="bicubic",
            preresize=False, preresize_mode="ensure minimum resolution",
            preresize_min_width=1024, preresize_min_height=1024,
            preresize_max_width=16384, preresize_max_height=16384,
            mask_fill_holes=True, mask_expand_pixels=0,
            mask_invert=False, mask_blend_pixels=0, mask_hipass_filter=0.5,
            extend_for_outpainting=False, extend_up_factor=1.0,
            extend_down_factor=1.0, extend_left_factor=1.0,
            extend_right_factor=1.0, context_from_mask_extend_factor=1.2,
            output_resize_to_target_size=True, output_target_width=128,
            output_target_height=128, output_padding="32",
            device_mode="cpu (compatible)",
        )
        crop = InpaintCropImproved()
        def bounds(context):
            stitcher, _, _ = crop.inpaint_crop(
                **{**args, "optional_context_mask": context})[:3]
            return tuple(stitcher[k][0] for k in (
                "cropped_to_canvas_x", "cropped_to_canvas_y",
                "cropped_to_canvas_w", "cropped_to_canvas_h"))
        baseline = bounds(subject)
        self.assertEqual(bounds(weak_context), baseline,
                         "Upstream filters confidence below the hipass cutoff")
        # A false positive ABOVE the cutoff can legitimately expand context.
        # That is a detector-quality issue, not evidence that subthreshold
        # mask confidence evades the native crop filter.
        noisy = bounds(strong_context)
        self.assertNotEqual(noisy, baseline,
                            "Strong remote context must change native bounds")


    def test_native_crop_inverse_mask_must_keep_enclosed_excluded_subject(self):
        # This is the exact inverse-segment adversarial case: an author's
        # subject is the BLACK hole inside a WHITE complement selection.
        # Native fillholes_iterative_hipass_fill_m would change the black
        # hole to white, silently sampling over the excluded person.
        image = torch.full((1, 256, 256, 3), 0.15, dtype=torch.float32)
        image[:, 96:160, 96:160, :] = 0.9
        inverse = torch.ones((1, 256, 256), dtype=torch.float32)
        inverse[:, 96:160, 96:160] = 0.0
        args = dict(
            image=image, mask=inverse,
            downscale_algorithm="bilinear", upscale_algorithm="bicubic",
            preresize=False, preresize_mode="ensure minimum resolution",
            preresize_min_width=1024, preresize_min_height=1024,
            preresize_max_width=16384, preresize_max_height=16384,
            mask_expand_pixels=0, mask_invert=False,
            mask_blend_pixels=0, mask_hipass_filter=0.5,
            extend_for_outpainting=False, extend_up_factor=1.0,
            extend_down_factor=1.0, extend_left_factor=1.0,
            extend_right_factor=1.0, context_from_mask_extend_factor=1.2,
            output_resize_to_target_size=True, output_target_width=256,
            output_target_height=256, output_padding="32",
            device_mode="cpu (compatible)"
        )
        crop = InpaintCropImproved()
        stitch = InpaintStitchImproved()
        def composited(fill_holes):
            metadata, pixels, _ = crop.inpaint_crop(
                **{**args, "mask_fill_holes": fill_holes})[:3]
            return stitch.inpaint_stitch(
                metadata, torch.full_like(pixels, 0.4))[0]
        preserved = composited(False)
        wrongly_filled = composited(True)
        self.assertGreater(float(preserved[0, 128, 128, 0]), 0.85,
                           "An intentionally EXCLUDED subject must remain the source pixel")
        self.assertLess(float(wrongly_filled[0, 128, 128, 0]), 0.55,
                        "Negative control: native hole-fill must demonstrate why it is unsafe")
        self.assertGreater(float(preserved[0, 10, 10, 0]), 0.3,
                           "The INVERTED surrounding area still receives replacement pixels")

if __name__ == "__main__":
    unittest.main()
