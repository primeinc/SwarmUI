"""Isolated tests for SwarmKSampler's preview conversion, without importing ComfyUI."""
import ast
from io import BytesIO
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmKSampler.py"


def load_helper():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_swarm_preview_pil_image"), None)
    if function is None:
        raise AssertionError("SwarmKSampler preview image normalizer is missing")
    scope = {"Image": Image}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope[function.name], tree


class TestSwarmKSamplerPreview(unittest.TestCase):
    def test_pil_is_already_an_image(self):
        convert, _ = load_helper()
        image = Image.new("RGB", (4, 4), (50, 100, 150))
        self.assertIs(convert(image), image)
        encoded = BytesIO()
        convert(image).save(encoded, format="JPEG")
        self.assertTrue(encoded.getvalue().startswith(b"\\xff\\xd8"))

    def test_tensor_npy_preview_is_converted(self):
        convert, _ = load_helper()
        class FakeCpuTensor:
            def numpy(self):
                return np.full((4, 4, 3), 127, dtype=np.uint8)
        result = convert(FakeCpuTensor())
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.getpixel((0, 0)), (127, 127, 127))

    def test_both_preview_emission_paths_use_normalizer(self):
        _, tree = load_helper()
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_swarm_preview_pil_image"]
        self.assertEqual(len(calls), 2, "Animated and still-image preview paths must both normalize frame type")


if __name__ == "__main__":
    unittest.main()
