"""CPU-only regression of the shipped read-only Qwen graph capture classifier.

The native Qwen paths use LanPaint_KSampler or KSampler under the unmodified
InpaintStitchImproved compositor. A capture returning zero segmented samplers
for this graph is a false negative, not evidence that Qwen did not run.
"""
import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "launchtools/qwen_segment_capture.py"
SPEC = importlib.util.spec_from_file_location("qwen_segment_capture_under_test", MODULE_PATH)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def qwen_graph(sampler_type="LanPaint_KSampler"):
    return {
        "source": {"class_type": "LoadImage", "inputs": {}},
        "crop": {"class_type": "InpaintCropImproved",
                 "inputs": {"image": ["source", 0]}},
        "model": {"class_type": "UNETLoader", "inputs": {}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["model", 0], "lora_name": "subject.safetensors",
                            "strength_model": 1.0}},
        "vision": {"class_type": "SwarmTextEncodeAdvanced",
                   "inputs": {"prompt": "A person", "image_1": ["crop", 1],
                              "qwen21_prevent_empty_text": True}},
        "vae": {"class_type": "VAEEncode",
                "inputs": {"pixels": ["crop", 1]}},
        "ref": {"class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["vision", 0], "latent": ["vae", 0]}},
        "masked": {"class_type": "LanPaint_ImageEncode",
                   "inputs": {"image": ["crop", 1], "mask": ["crop", 2]}},
        "sampler": {"class_type": sampler_type,
                    "inputs": {"model": ["lora", 0], "positive": ["ref", 0],
                               "negative": ["vision", 0], "latent_image": ["masked", 0],
                               "steps": 20, "seed": 5}},
        "decode": {"class_type": "LanPaint_ImageDecode",
                   "inputs": {"samples": ["sampler", 0]}},
        "stitch": {"class_type": "InpaintStitchImproved",
                   "inputs": {"stitcher": ["crop", 0],
                              "inpainted_image": ["decode", 0]}},
        "final": {"class_type": "SwarmSaveImageWS",
                  "inputs": {"images": ["stitch", 0]}},
    }


class QwenCaptureCPU(unittest.TestCase):
    def test_masked_lanpaint_segment_identifies_actual_sampler_and_lora(self):
        report = capture.analyze(qwen_graph())
        self.assertEqual(report["segment_sampler_count"], 1)
        segment, = report["segments"]
        self.assertEqual(segment["sampler"], "sampler")
        self.assertEqual(segment["class_type"], "LanPaint_KSampler")
        self.assertEqual([n["id"] for n in segment["lora_nodes"]], ["lora"])
        self.assertEqual([n["id"] for n in segment["conditioning_encoders"]], ["vision"])
        self.assertEqual([n["id"] for n in segment["reference_latents"]], ["ref"])
        self.assertEqual([n["id"] for n in segment["compositors"]], ["stitch"])
        self.assertIn("masked", [n["id"] for n in segment["latent_mask_nodes"]])
        self.assertFalse(any("REFERENCE_WITHOUT_ENCODER_IMAGE" in x
                             for x in segment["warnings"]),
                         "Native Qwen independent image_1 socket must count as vision")

    def test_reference_edit_ordinary_ksampler_through_stitch_is_identified(self):
        graph = qwen_graph("KSampler")
        graph["sampler"]["inputs"]["latent_image"] = ["vae", 0]
        report = capture.analyze(graph)
        self.assertEqual(report["segment_sampler_count"], 1)

    def test_unrelated_sampler_without_stitch_is_not_a_segment(self):
        graph = qwen_graph()
        del graph["stitch"]
        report = capture.analyze(graph)
        self.assertEqual(report["segment_sampler_count"], 0)

    def test_obsolete_composite_remains_supported(self):
        graph = qwen_graph()
        graph["legacy"] = {"class_type": "ImageCompositeMasked",
                           "inputs": {"source": ["decode", 0]}}
        report = capture.analyze(graph)
        self.assertEqual(report["segment_sampler_count"], 1)
        self.assertEqual({n["id"] for n in report["segments"][0]["compositors"]},
                         {"legacy", "stitch"})


if __name__ == "__main__":
    unittest.main()
