"""Deterministic real-tensor CropAndStitch replacement test.

This checks pixel compositing and original-canvas preservation using the
unchanged upstream adapter. It does NOT test Qwen or LoRA inference.
"""
import unittest
import torch

from inpaint_cropandstitch import InpaintCropImproved, InpaintStitchImproved


class CropReplacementTests(unittest.TestCase):
    def test_old_subject_pixels_are_replaced_when_model_supplies_new_pixels(self):
        source = torch.ones((1, 96, 96, 3), dtype=torch.float32)
        source[:, 30:66, 30:66] = 0.0  # original dark rectangular subject
        mask = torch.zeros((1, 96, 96), dtype=torch.float32)
        mask[:, 30:66, 30:66] = 1.0

        crop = InpaintCropImproved()
        stitch = InpaintStitchImproved()
        args = dict(
            image=source, mask=mask,
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
        soft_mask = mask * 0.6
        soft_stitcher, soft_image, _ = crop.inpaint_crop(
            **{**args, "mask": soft_mask})[:3]
        soft_result = stitch.inpaint_stitch(
            soft_stitcher, torch.ones_like(soft_image) * 0.8)[0]
        self.assertLess(soft_result[0, 45, 45, 0].item(), 0.55)
        self.assertGreater(result[0, 45, 45, 0].item(), 0.7)
        # Preserve untouched context, not a newly generated full frame.
        self.assertTrue(torch.allclose(result[0, :8, :8], source[0, :8, :8], atol=0.0001))


if __name__ == "__main__":
    unittest.main()
