"""H3 prompt-image resize policy contract and CPU-only reference-node tests.

This exercises the real Swarm H3 Python reference collector with stubbed tensor
and VAE dependencies, plus checks the C# workflow-node binding. It does not
execute a C# workflow or GPU inference.
"""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs"
NODE = ROOT / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmModels.py"


class FakeImage:
    def __init__(self, height, width):
        self.shape = (1, height, width, 3)

    def __getitem__(self, item):
        return self


def load_h3_collector():
    source = ast.parse(NODE.read_text(encoding="utf-8"))
    h3_class = next(n for n in source.body if isinstance(n, ast.ClassDef)
                    and n.name == "SwarmMiniMaxH3CollectReferences")
    execute = copy.deepcopy(next(n for n in h3_class.body if isinstance(n, ast.FunctionDef)
                                 and n.name == "execute"))
    execute.decorator_list = []
    resized = []
    env = {
        "math": math,
        "H3_CANVAS_MULTIPLE": 32,
        "H3_REF_IMAGE_SHORT_EDGE": 2048,
        "temporal_shape": lambda length: (length, None, None),
        "_resize": lambda image, width, height, crop: (
            resized.append((width, height)) or FakeImage(height, width)),
        "io": SimpleNamespace(NodeOutput=lambda value: value),
    }
    exec(compile(ast.Module(body=[execute], type_ignores=[]), str(NODE), "exec"), env)
    return env["execute"], resized


class H3PromptImageResizeTests(unittest.TestCase):
    def test_workflow_passes_existing_smart_resize_toggle(self):
        source = GEN.read_text(encoding="utf-8")
        start = source.index("else if (isPositive && IsMiniMaxH3())")
        finish = source.index("bool hasAny = false;", start)
        h3_inputs = source[start:finish]
        # The H3 node's input is the actual implementation seam.
        self.assertIn('["ref_image_size"] = UserInput.Get(T2IParamTypes.SmartImagePromptResizing, true) ? "match" : "max",', h3_inputs)

    def test_h3_collector_match_vs_max_changes_real_reference_size(self):
        execute, sizes = load_h3_collector()
        vae = SimpleNamespace(encode=lambda image: object())
        input_image = FakeImage(3000, 4000)
        def run(mode):
            output = execute(None, vae, None, 1024, 576, 124,
                             ref_image_size=mode,
                             ref_images={"ref_image_0": input_image})
            self.assertEqual(len(output["ref_blocks"]), 1)
            width, height = sizes[-1]
            self.assertEqual(output["ref_blocks"][0]["latent_w"], width // 16)
            self.assertEqual(output["ref_blocks"][0]["latent_h"], height // 16)
            return width, height

        fit_width, fit_height = run("match")
        max_width, max_height = run("max")
        self.assertLess(fit_width * fit_height, max_width * max_height / 2)
        self.assertLess(fit_width * fit_height, 1024 * 576 * 1.1)
        self.assertEqual(max_height, 2048)

    def test_default_collector_policy_remains_match(self):
        execute, sizes = load_h3_collector()
        vae = SimpleNamespace(encode=lambda image: object())
        execute(None, vae, None, 1024, 576, 124,
                ref_images={"ref_image_0": FakeImage(3000, 4000)})
        default_size = sizes[-1]
        execute(None, vae, None, 1024, 576, 124, ref_image_size="match",
                ref_images={"ref_image_0": FakeImage(3000, 4000)})
        self.assertEqual(default_size, sizes[-1])


if __name__ == "__main__":
    unittest.main()
