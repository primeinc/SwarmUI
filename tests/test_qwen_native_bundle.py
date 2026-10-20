"""Read-only regression for Qwen node shipping and native workflow plumbing.

This proves the repository source and unchanged upstream node bytes, NOT
live ComfyUI loading, real GPU inference, or semantic subject replacement.
"""
from pathlib import Path
import hashlib
import unittest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "src/BuiltinExtensions/ComfyUIBackend"
BUNDLE = BACKEND / "ExtraNodes/ComfyQwenNodes_Upstream.py"
PINNED_UPSTREAM_GIT_BLOB = "4e7a150a23f9235e48c8fb1d8f18bf759eb33916"


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

    def test_generation_uses_native_qwen_encoder_and_sampler(self):
        helper = (BACKEND / "WorkflowGeneratorQwenSegment.cs").read_text()
        steps = (BACKEND / "WorkflowGeneratorSteps.cs").read_text()
        composite = (BACKEND / "WorkflowGenerator.cs").read_text()
        init = (BACKEND / "ExtraNodes/SwarmComfyCommon/__init__.py").read_text()
        self.assertIn('CreateNode("TextEncodeQwenImage21"', helper)
        self.assertIn('CreateNode("KSampler"', steps)
        self.assertIn('"ImageCompositeMasked"', composite)
        self.assertIn('"ImageScaleToTotalPixels"', composite)
        self.assertIn('"ImageCrop"', composite)
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
