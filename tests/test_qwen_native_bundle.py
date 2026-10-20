"""Read-only regression for Qwen node shipping and native workflow plumbing.

This proves the repository source and unchanged upstream node bytes, NOT
live ComfyUI loading, real GPU inference, or semantic subject replacement.
"""
from pathlib import Path
import hashlib
import json
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "src/BuiltinExtensions/ComfyUIBackend"
BUNDLE = BACKEND / "ExtraNodes/ComfyQwenNodes_Upstream.py"
PINNED_UPSTREAM_GIT_BLOB = "4e7a150a23f9235e48c8fb1d8f18bf759eb33916"
CROP_PLUGIN = BACKEND / "ExtraNodes/ComfyInpaintCropAndStitch"
PINNED_CROP_GIT_BLOB = "89d4c43593d3d5ec1ee131bd6dbdbce666fe425f"
PINNED_CROP_INIT_BLOB = "d24dd44d381507953fc59f7a593c36f94179bf85"
LANPAINT_PLUGIN = BACKEND / "ExtraNodes/LanPaintUpstream"
PINNED_LANPAINT_NODES_BLOB = "6c2f8e4a7ebf238da66d2e3c885e82685880f78d"
PINNED_LANPAINT_INIT_BLOB = "057359402c29c697197d643fd499825209684592"
UPSTREAM_WORKFLOW = ROOT / "tests/references/LanPaint_Qwen_Image_2.1_Edit_Masked_Inpaint.json"
PINNED_WORKFLOW_BLOB = "52ffc874c1a6d4cb380a4a7374485d660dec7725"


class NativeQwenBundleTest(unittest.TestCase):
    def test_unmodified_upstream_source_and_license_are_shipped(self):
        data = BUNDLE.read_bytes()
        git_blob = b"blob " + str(len(data)).encode("ascii") + b"\x00" + data
        self.assertEqual(hashlib.sha1(git_blob).hexdigest(), PINNED_UPSTREAM_GIT_BLOB)
        self.assertTrue((BACKEND / "ExtraNodes/ComfyQwenNodes_Upstream.LICENSE").is_file())

    def test_shipped_custom_nodes_directory_is_loaded(self):
        loader = (BACKEND / "ComfyUISelfStartBackend.cs").read_text()
        self.assertIn('ComfyUIBackendExtension.Folder + "/ExtraNodes"', loader)
        self.assertIn('custom_nodes:', loader)
        native = BUNDLE.read_text()
        self.assertIn('node_id="TextEncodeQwenImage21"', native)
        self.assertIn("async def comfy_entrypoint", native)

    def test_verbatim_upstream_crop_and_stitch_are_bundled_separately(self):
        for relative, blob in (
            ("inpaint_cropandstitch.py", PINNED_CROP_GIT_BLOB),
            ("__init__.py", PINNED_CROP_INIT_BLOB),
        ):
            contents = (CROP_PLUGIN / relative).read_bytes()
            git_blob = b"blob " + str(len(contents)).encode("ascii") + b"\x00" + contents
            self.assertEqual(hashlib.sha1(git_blob).hexdigest(), blob, relative)
        self.assertTrue((CROP_PLUGIN / "LICENSE").is_file())
        self.assertTrue((CROP_PLUGIN / "js/showcontrol.js").is_file())
        registrations = (CROP_PLUGIN / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('"InpaintCropImproved"', registrations)
        self.assertIn('"InpaintStitchImproved"', registrations)

    def test_verbatim_upstream_masked_sampler_adapter_is_shipped(self):
        for relative, blob in (
            ("src/LanPaint/nodes.py", PINNED_LANPAINT_NODES_BLOB),
            ("__init__.py", PINNED_LANPAINT_INIT_BLOB),
        ):
            contents = (LANPAINT_PLUGIN / relative).read_bytes()
            git_blob = b"blob " + str(len(contents)).encode("ascii") + b"\x00" + contents
            self.assertEqual(hashlib.sha1(git_blob).hexdigest(), blob, relative)
        self.assertTrue((LANPAINT_PLUGIN / "LICENSE").is_file())

    def test_real_upstream_qwen_21_workflow_is_unmodified_and_matches_the_connected_chain(self):
        data = UPSTREAM_WORKFLOW.read_bytes()
        git_blob = b"blob " + str(len(data)).encode("ascii") + b"\x00" + data
        self.assertEqual(hashlib.sha1(git_blob).hexdigest(), PINNED_WORKFLOW_BLOB)
        workflow = json.loads(data)
        nodes = {node["type"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}

        def edge_to(target_type, name):
            incoming = next(x for x in nodes[target_type]["inputs"] if x["name"] == name)
            self.assertIsInstance(incoming["link"], int, (target_type, name))
            return links[incoming["link"]]

        def assert_edge(source_type, source_slot, target_type, input_name):
            link = edge_to(target_type, input_name)
            self.assertEqual((link[1], link[2], link[3]),
                             (nodes[source_type]["id"], source_slot, nodes[target_type]["id"]))

        # These are the real links in the author's Qwen 2.1 example, not
        # values invented by our Swarm fixture. One IMAGE goes to both Qwen's
        # image1 conditioning and LanPaint's masked encode/decode. One MASK
        # goes to both mask inputs. The downstream source is the decoder.
        assert_edge("JoinImageWithAlpha", 0, "TextEncodeQwenImage21", "images.image_1")
        assert_edge("JoinImageWithAlpha", 0, "LanPaint_ImageEncode", "image")
        assert_edge("JoinImageWithAlpha", 0, "LanPaint_ImageDecode", "image")
        assert_edge("LoadImageMask", 0, "LanPaint_ImageEncode", "mask")
        assert_edge("LoadImageMask", 0, "LanPaint_ImageDecode", "mask")
        assert_edge("TextEncodeQwenImage21", 0, "LanPaint_KSampler", "positive")
        assert_edge("TextEncodeQwenImage21", 1, "LanPaint_KSampler", "negative")
        assert_edge("LanPaint_ImageEncode", 0, "LanPaint_KSampler", "latent_image")
        assert_edge("LanPaint_KSampler", 0, "LanPaint_ImageDecode", "samples")
        assert_edge("LanPaint_ImageDecode", 0, "SaveImage", "images")

        # Proof attack: if someone swaps the sampler input to the wrong
        # native Qwen empty latent output, the reference edge assertion
        # must reject it. No silent acceptance of the old faulty path.
        self.assertNotEqual(
            edge_to("LanPaint_KSampler", "latent_image")[1],
            nodes["TextEncodeQwenImage21"]["id"]
        )

    def test_crop_and_stitch_native_required_inputs_are_supplied(self):
        upstream = (CROP_PLUGIN / "inpaint_cropandstitch.py").read_text()
        steps = (BACKEND / "WorkflowGeneratorSteps.cs").read_text()
        crop_schema = upstream[upstream.index("class InpaintCropImproved:"):
                               upstream.index("class InpaintStitchImproved:")]
        required = crop_schema.split('"required": {', 1)[1].split('"optional": {', 1)[0]
        required_names = re.findall(r'^\s+"([a-z_]+)": \(', required, re.MULTILINE)
        self.assertGreater(len(required_names), 20)
        crop_inputs = steps.split('JObject cropInputs = new()', 1)[1].split('};', 1)[0]
        for field in required_names:
            self.assertIn('["' + field + '"] = ', crop_inputs, field)
        # Stitch's two mandatory inputs must reach the native class unchanged.
        self.assertIn('["stitcher"] = cropStitcher', steps)
        self.assertIn('["inpainted_image"] = stitchInput', steps)

    def test_nonempty_mask_guard_uses_native_crop_hipass_cutoff(self):
        steps = (BACKEND / "WorkflowGeneratorSteps.cs").read_text()
        masks = (BACKEND / "ExtraNodes/SwarmComfyCommon/SwarmMasks.py").read_text()
        self.assertIn('["minimum_confidence"] = cropInputs["mask_hipass_filter"]', steps)
        self.assertIn('("FLOAT", {"default": 0.5', masks)
        self.assertIn('string solidSubjectMask = g.CreateNode("ThresholdMask"', steps)
        self.assertIn('["mask"] = NodePath(requiredMask, 0)', steps)
        self.assertIn('["value"] = cropInputs["mask_hipass_filter"]', steps)
        self.assertIn('cropInputs["mask"] = NodePath(solidSubjectMask, 0)', steps)

    def test_masked_sampler_required_native_inputs_are_supplied(self):
        steps = (BACKEND / "WorkflowGeneratorSteps.cs").read_text()
        nodes = (LANPAINT_PLUGIN / "src/LanPaint/nodes.py").read_text()
        required = ("LanPaint_NumSteps", "LanPaint_PromptMode",
                    "LanPaint_Info", "Inpainting_mode")
        # These are declared under the upstream REQUIRED schema, not inferred
        # from Python sample() default values.
        sample_required = nodes[nodes.index("class LanPaint_KSampler():"):
                                nodes.index("class LanPaint_KSamplerAdvanced:")]
        for field in required:
            self.assertIn('"' + field + '": (', sample_required)
            self.assertIn('samplerInputs["' + field + '"] = ', steps)

    def test_generation_uses_the_original_adapters_end_to_end(self):
        helper = (BACKEND / "WorkflowGeneratorQwenSegment.cs").read_text()
        steps = (BACKEND / "WorkflowGeneratorSteps.cs").read_text()
        init = (BACKEND / "ExtraNodes/SwarmComfyCommon/__init__.py").read_text()
        self.assertIn('CreateNode("TextEncodeQwenImage21"', helper)
        self.assertIn('CreateNode("LanPaint_KSampler"', steps)
        self.assertIn('CreateNode("LanPaint_ImageEncode"', helper)
        self.assertIn('CreateNode("LanPaint_ImageDecode"', steps)
        self.assertIn('CreateNode("InpaintCropImproved"', steps)
        self.assertIn('CreateNode("InpaintStitchImproved"', steps)
        self.assertIn('["stitcher"] = cropStitcher', steps)
        self.assertIn('["inpainted_image"] = stitchInput', steps)
        self.assertIn('["images.image_1"] = croppedImage', helper)
        for obsolete in (
            "SwarmQwenImage21SegmentEncode",
            "SwarmQwen21SegmentDiagnostics",
            "SwarmQwen21CompositeDiagnostics",
        ):
            self.assertNotIn('CreateNode("' + obsolete + '"', helper + steps)
            self.assertNotIn(obsolete, init)
        self.assertFalse((BACKEND / "ExtraNodes/SwarmComfyCommon/SwarmQwen.py").exists())


if __name__ == "__main__":
    unittest.main()
