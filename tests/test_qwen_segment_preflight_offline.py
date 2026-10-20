"""CPU/no-model negative controls for the actual exported-graph preflight code.

These programmatically assembled graphs are UNIT TEST FIXTURES for validating
the preflight verifier, never evidence that Swarm generated a working workflow.
Real generated-graph preflight requires exporting the graph from SwarmUI, and
semantic image acceptance requires model-backed inference.
"""
from contextlib import redirect_stdout
import copy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from integration import qwen_segment_text_replacement as probe


def native_contract_fixture():
    """Minimal Comfy API-shaped graph to exercise offline validation branches."""
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    return {
        "source": node("LoadImage", image="fixture.png"),
        "detector": node("SwarmClipSeg", images=["source", 0],
                         match_text="girl", threshold=0.1),
        "guard": node("SwarmRequireNonEmptyMask", mask=["detector", 0],
                      minimum_confidence=0.5),
        "binary": node("ThresholdMask", mask=["guard", 0], value=0.5),
        "crop": node("InpaintCropImproved", image=["source", 0],
                     mask=["binary", 0], mask_hipass_filter=0.5,
                     output_resize_to_target_size=True),
        "condition": node("TextEncodeQwenImage21", clip=["clip", 0],
                          prompt=probe.PROMPT, negative_prompt="",
                          **{"images.image_1": ["crop", 1]}),
        "masked": node("LanPaint_ImageEncode", image=["crop", 1],
                       mask=["crop", 2], vae=["vae", 0]),
        "raw_model": node("UNETLoader", unet_name="fixture.safetensors"),
        "sampler": node("LanPaint_KSampler", model=["raw_model", 0],
                        seed=3, steps=4, cfg=1.0, denoise=0.8,
                        sampler_name="euler", scheduler="simple",
                        positive=["condition", 0], negative=["condition", 1],
                        latent_image=["masked", 0], LanPaint_NumSteps=5,
                        LanPaint_PromptMode="Image First",
                        LanPaint_Info="LanPaint KSampler.",
                        Inpainting_mode="🖼️ Image Inpainting"),
        "decoded": node("LanPaint_ImageDecode", samples=["sampler", 0],
                        image=["crop", 1], vae=["vae", 0]),
        "split": node("SplitImageWithAlpha", image=["decoded", 0]),
        "stitch": node("InpaintStitchImproved",
                       stitcher=["crop", 0], inpainted_image=["split", 0]),
        "final": node("SwarmSaveImageWS", images=["stitch", 0]),
    }


class OfflineGraphPreflightUnitTests(unittest.TestCase):

    def test_offline_command_never_contacts_a_backend_or_infers_a_model(self):
        graph = native_contract_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            export = Path(tmp) / "contract-fixture.json"
            export.write_text(json.dumps(graph), encoding="utf-8")
            out = StringIO()
            with patch.object(probe, "fetch_json", side_effect=AssertionError(
                "Offline graph check unexpectedly contacted a backend"
            )), redirect_stdout(out):
                code = probe.main(["--workflow", str(export), "--preflight-only"])
        self.assertEqual(code, 0)
        summary = json.loads(out.getvalue())
        self.assertEqual(summary["mode"], "offline")
        self.assertEqual(summary["model_inference_runs"], 0)
        self.assertFalse(summary["semantic_subject_replacement_verified"])
        self.assertFalse(summary["image_space_lora_effect_verified"])
        self.assertFalse(summary["runtime_node_registration_verified"])
        self.assertFalse(summary["native_schema_completeness_verified"])

    def test_detector_mask_must_reach_native_crop_through_guard_and_threshold(self):
        graph = native_contract_fixture()
        graph["crop"]["inputs"]["mask"] = ["detector", 0]
        with self.assertRaises(AssertionError):
            probe.inspect_and_instrument(graph)

    def test_presegment_sampler_on_crop_source_fails(self):
        graph = native_contract_fixture()
        graph["source"]["class_type"] = "KSampler"
        with self.assertRaises(AssertionError):
            probe.inspect_and_instrument(graph)

    def test_unmasked_latent_cannot_claim_to_be_lanpaint(self):
        graph = native_contract_fixture()
        graph["sampler"]["inputs"]["latent_image"] = ["condition", 2]
        with self.assertRaises(AssertionError):
            probe.inspect_and_instrument(graph)

    def test_double_outputs_are_not_accepted_as_one_generation(self):
        graph = native_contract_fixture()
        graph["extra"] = {"class_type": "PreviewImage",
                          "inputs": {"images": ["crop", 1]}}
        with self.assertRaises(AssertionError):
            probe.inspect_and_instrument(graph)

    def test_two_final_saved_images_fail_even_with_one_save_sink(self):
        with self.assertRaisesRegex(AssertionError, "got 2"):
            probe.require_single_final_image(
                {"final": {"images": [{"filename": "a.png"}, {"filename": "b.png"}]}},
                "final")
        self.assertEqual(
            probe.require_single_final_image(
                {"final": {"images": [{"filename": "single.png"}]}}, "final"
            )["filename"],
            "single.png")
        with self.assertRaisesRegex(AssertionError, "UNVERIFIED"):
            probe.require_single_final_image({}, "final")

    def test_runtime_preflight_rejects_missing_splitter_and_diagnostic_nodes(self):
        graph, _ = probe.inspect_and_instrument(native_contract_fixture())
        available = {node["class_type"] for node in graph.values()}
        def fake_fetch(base, route):
            name = route.removeprefix("object_info/")
            return {name: {"input": {"required": {}}}} if name in available else {}
        with patch.object(probe, "fetch_json", side_effect=fake_fetch):
            self.assertTrue(probe.validate_runtime_node_contracts("http://example.invalid", graph))
        for missing in ("SplitImageWithAlpha", "MaskToImage", "PreviewImage"):
            available.remove(missing)
            with patch.object(probe, "fetch_json", side_effect=fake_fetch):
                with self.assertRaisesRegex(AssertionError, missing):
                    probe.validate_runtime_node_contracts("http://example.invalid", graph)
            available.add(missing)

    def test_wrong_subject_or_target_instruction_is_rejected(self):
        graph = native_contract_fixture()
        graph["detector"]["inputs"]["match_text"] = "bird"
        with self.assertRaises(AssertionError):
            probe.inspect_and_instrument(graph)
        graph = native_contract_fixture()
        graph["condition"]["inputs"]["prompt"] = "portrait"
        with self.assertRaises(AssertionError):
            probe.inspect_and_instrument(graph)


if __name__ == "__main__":
    unittest.main()
