"""Adversarial CPU pixel controls for segmented LoRA effect versus leakage.

No inference. Tests evaluate the same independent image-space acceptance
metric used when real Qwen LoRA OFF/ON/OFF images are supplied.
"""
import unittest
from PIL import Image, ImageDraw
from integration.qwen_segment_lora_confinement import (
    region_rms, assert_same_detector_mask, normalized_lora_path, same_lora_path,
)


def load_actual_live_model_chain():
    """Execute the exact shipped live model_chain AST without queuing inference."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent / "live/test_qwen_segment_lora.py"
    syntax = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [node for node in syntax.body
                 if isinstance(node, ast.FunctionDef) and node.name == "model_chain"]
    if len(functions) != 1:
        raise AssertionError("Live LoRA acceptance model_chain missing or duplicated")
    scope = {"_SAMPLERS": {"KSampler", "LanPaint_KSampler"},
             "_LORAS": {"LoraLoader", "LoraLoaderModelOnly"}}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    return scope["model_chain"]


class LiveLoRAModelChainContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.check = staticmethod(load_actual_live_model_chain())

    @staticmethod
    def graph():
        return {
            "sampler": {"class_type": "LanPaint_KSampler", "inputs": {"model": ["patched", 0]}},
            "patched": {"class_type": "LoraLoaderModelOnly",
                        "inputs": {"model": ["raw", 0], "lora_name": "subjects/portrait.safetensors",
                                   "strength_model": 1.0}},
            "raw": {"class_type": "UNETLoader", "inputs": {"unet_name": "test.safetensors"}},
        }

    def test_valid_lora_to_real_source_loader(self):
        self.assertEqual(self.check(self.graph(), "sampler"), ["patched", "raw"])

    def test_missing_model_upstream_rejected(self):
        graph = self.graph()
        graph["patched"]["inputs"]["model"] = ["lost_source", 0]
        with self.assertRaisesRegex(AssertionError, "missing node"):
            self.check(graph, "sampler")

    def test_selected_lora_self_cycle_rejected(self):
        graph = self.graph()
        graph["patched"]["inputs"]["model"] = ["patched", 0]
        with self.assertRaisesRegex(AssertionError, "cycle"):
            self.check(graph, "sampler")

    def test_lora_wrapper_without_parent_rejected(self):
        graph = self.graph()
        del graph["patched"]["inputs"]["model"]
        with self.assertRaisesRegex(AssertionError, "no upstream MODEL"):
            self.check(graph, "sampler")

    def test_malformed_link_rejected(self):
        graph = self.graph()
        graph["patched"]["inputs"]["model"] = ["raw"]
        with self.assertRaisesRegex(AssertionError, "Malformed"):
            self.check(graph, "sampler")


def load_actual_live_output_gate():
    """Exercise the real live acceptance gate without importing GPU pytest."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent / "live/test_qwen_segment_lora.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    funcs = [node for node in tree.body
             if isinstance(node, ast.FunctionDef)
             and node.name == "assert_segment_final_output_path"]
    if len(funcs) != 1:
        raise AssertionError("Missing actual live final output gate")
    scope = {"_IMAGE_OUTPUTS": {"SwarmSaveImageWS", "SaveImage", "PreviewImage"}}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), "exec"), scope)
    return scope["assert_segment_final_output_path"]


class LiveLoRAFinalOutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.check = staticmethod(load_actual_live_output_gate())

    @staticmethod
    def graph():
        return {
            "original": {"class_type": "LoadImage", "inputs": {"image": "original.png"}},
            "metadata": {"class_type": "InpaintCropImproved", "inputs": {"image": ["original", 0]}},
            "crop": {"class_type": "LanPaint_ImageDecode", "inputs": {}},
            "stitch": {"class_type": "InpaintStitchImproved",
                       "inputs": {"stitcher": ["metadata", 0], "inpainted_image": ["crop", 0]}},
            "saved": {"class_type": "SwarmSaveImageWS", "inputs": {"images": ["stitch", 0]}},
        }

    def test_comparison_uses_single_real_stitched_output(self):
        self.assertIsNone(self.check(self.graph(), ["stitch", 0], "stitch"))

    def test_comparison_must_not_use_unrelated_image_even_when_stitch_exists(self):
        g = self.graph()
        g["saved"]["inputs"]["images"] = ["original", 0]
        with self.assertRaisesRegex(AssertionError, "not the actual saved result"):
            self.check(g, ["stitch", 0], "stitch")

    def test_saved_base_frame_not_accepted_as_stitched_segment(self):
        g = self.graph()
        g["saved"]["inputs"]["images"] = ["original", 0]
        with self.assertRaisesRegex(AssertionError, "does not depend"):
            self.check(g, ["original", 0], "stitch")

    def test_gallery_of_multiple_save_sinks_is_not_one_result(self):
        g = self.graph()
        g["debug"] = {"class_type": "PreviewImage", "inputs": {"images": ["crop", 0]}}
        with self.assertRaisesRegex(AssertionError, "one user-visible final output"):
            self.check(g, ["stitch", 0], "stitch")

    def test_legitimate_final_image_scale_still_reaches_stitch(self):
        g = self.graph()
        g["scaled"] = {"class_type": "ImageScale", "inputs": {"image": ["stitch", 0]}}
        g["saved"]["inputs"]["images"] = ["scaled", 0]
        self.assertIsNone(self.check(g, ["scaled", 0], "stitch"))


def load_shipped_qwen_mask_preflight():
    """Execute actual live acceptance mask-contract function without a model."""
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parent / "live/test_qwen_segment_lora.py"
    syntax = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    funcs = [n for n in syntax.body
             if isinstance(n, ast.FunctionDef) and n.name == "validate_qwen_subject_mask_chain"]
    if len(funcs) != 1:
        raise AssertionError("Missing or duplicate live Qwen mask contract")
    scope = {}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"), scope)
    return scope["validate_qwen_subject_mask_chain"]


class LiveLoRAMaskPreflightContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validate_mask = staticmethod(load_shipped_qwen_mask_preflight())

    @staticmethod
    def graph():
        return {
            "detector": {"class_type": "SwarmClipSeg", "inputs": {"threshold": 0.1}},
            "guard": {"class_type": "SwarmRequireNonEmptyMask",
                      "inputs": {"mask": ["detector", 0],
                                 "minimum_confidence": 0.0, "maximum_coverage": 0.99}},
            "binary": {"class_type": "ThresholdMask",
                       "inputs": {"mask": ["guard", 0], "value": 0.0}},
            "crop": {"class_type": "InpaintCropImproved",
                     "inputs": {"mask": ["binary", 0], "mask_hipass_filter": 0.5}},
        }

    def test_authored_low_threshold_and_safe_coverage_passes(self):
        self.assertEqual(self.validate_mask(self.graph(), "crop"), ["binary", 0])

    def test_stale_second_half_threshold_rejected(self):
        graph = self.graph()
        graph["binary"]["inputs"]["value"] = 0.5
        with self.assertRaisesRegex(AssertionError, "at zero"):
            self.validate_mask(graph, "crop")

    def test_stale_guard_half_threshold_rejected(self):
        graph = self.graph()
        graph["guard"]["inputs"]["minimum_confidence"] = 0.5
        with self.assertRaisesRegex(AssertionError, "authored CLIPSeg"):
            self.validate_mask(graph, "crop")

    def test_missing_near_full_frame_guard_rejected(self):
        graph = self.graph()
        del graph["guard"]["inputs"]["maximum_coverage"]
        with self.assertRaisesRegex(AssertionError, "near-full-canvas"):
            self.validate_mask(graph, "crop")

    def test_disabled_near_full_frame_guard_rejected(self):
        graph = self.graph()
        graph["guard"]["inputs"]["maximum_coverage"] = 1.0
        with self.assertRaisesRegex(AssertionError, "near-full-canvas"):
            self.validate_mask(graph, "crop")

    def test_non_detector_mask_cannot_be_accepted(self):
        graph = self.graph()
        graph["detector"]["class_type"] = "SolidMask"
        with self.assertRaisesRegex(AssertionError, "actual text detector"):
            self.validate_mask(graph, "crop")


class QwenSegmentLoraConfinementCPUTests(unittest.TestCase):
    @staticmethod
    def sample():
        base = Image.new("RGB", (256, 256), (20, 30, 40))
        mask = Image.new("L", base.size, 0)
        ImageDraw.Draw(mask).rectangle((104, 104, 151, 151), fill=255)
        return base, mask

    def test_lora_paths_match_separators_but_not_filename_case(self):
        self.assertEqual(normalized_lora_path("subjects\\portrait.safetensors"),
                         "subjects/portrait.safetensors")
        self.assertEqual(normalized_lora_path("subjects/portrait.SAFETENSORS"),
                         "subjects/portrait.SAFETENSORS")
        self.assertNotEqual(normalized_lora_path("subjects/Portrait.safetensors"),
                            normalized_lora_path("subjects/portrait.safetensors"))
        self.assertNotEqual(normalized_lora_path("subjects/portrait.SAFETENSORS"),
                            normalized_lora_path("subjects/portrait.safetensors"),
                            "Two full Linux LoRA filenames differing only in suffix case are distinct")
        self.assertTrue(same_lora_path("subjects/portrait",
                                       "subjects\\portrait.safetensors"))
        self.assertTrue(same_lora_path("subjects/portrait.SAFETENSORS",
                                       "subjects/portrait.SAFETENSORS"))
        self.assertFalse(same_lora_path("subjects/portrait.SAFETENSORS",
                                        "subjects/portrait.safetensors"))
        self.assertFalse(same_lora_path(" subjects/portrait.safetensors",
                                        "subjects/portrait.safetensors"))
        self.assertFalse(same_lora_path("subjects/portrait .safetensors",
                                        "subjects/portrait.safetensors"))

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
        # Only the real configured blend halo may be excluded. The former
        # implicit 48px exclusion also hid leaks well outside that halo.
        _, outside = region_rms(base, variant, mask, margin=10)
        self.assertEqual(outside, 0.0)

    def test_near_boundary_leak_outside_actual_margin_is_not_hidden(self):
        base, mask = self.sample()
        variant = base.copy()
        # Mask ends at x=151; configured grow/feather guard is only 8px.
        # A 48px hard floor gave a false-zero outside RMS for this change.
        ImageDraw.Draw(variant).rectangle((168, 120, 178, 131), fill=(220, 30, 40))
        _, outside = region_rms(base, variant, mask, margin=8)
        self.assertGreater(outside, 2.0)
        _, old_exclusion = region_rms(base, variant, mask, margin=48)
        self.assertEqual(old_exclusion, 0.0,
                         "Negative control: the old 48px floor must conceal this defect")

    def test_small_canvas_has_meaningful_outside_support_at_configured_margin(self):
        base = Image.new("RGB", (128, 128), (20, 30, 40))
        mask = Image.new("L", base.size, 0)
        ImageDraw.Draw(mask).rectangle((48, 48, 80, 80), fill=255)
        variant = base.copy()
        ImageDraw.Draw(variant).rectangle((0, 0, 8, 8), fill=(220, 30, 40))
        _, outside = region_rms(base, variant, mask, margin=34)
        self.assertGreater(outside, 0.0)
        with self.assertRaisesRegex(AssertionError, "covers"):
            region_rms(base, variant, mask, margin=48)

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
