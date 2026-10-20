"""Adversarial CPU pixel controls for segmented LoRA effect versus leakage.

No inference. Tests evaluate the same independent image-space acceptance
metric used when real Qwen LoRA OFF/ON/OFF images are supplied.
"""
import unittest
from PIL import Image, ImageDraw
from integration.qwen_segment_lora_confinement import (
    region_rms, assert_same_detector_mask,
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
