"""Execute the shipped live Qwen dynamic-hook model-path validator on CPU.

This tests Comfy's CreateHookLora -> SwarmTextEncodeAdvanced.lora_hooks path,
not an imaginary static LoraLoader on the sampler MODEL path. It cannot
establish model-backed image-space LoRA effect.
"""
import ast
from pathlib import Path
import re
import unittest

SOURCE = (Path(__file__).resolve().parents[1] /
          "tests/live/test_qwen_segment_lora.py")


def real_validator():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "validate_dynamic_hook_on_sampler")
    scope = {"_SAMPLERS": {"LanPaint_KSampler", "KSampler"}, "re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope[node.name]


def native_hook_graph():
    return {
        "model": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "sampler": {"class_type": "LanPaint_KSampler", "inputs": {
            "model": ["model", 0], "positive": ["posref", 0],
            "negative": ["negref", 0]}},
        "posref": {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": ["pos", 0]}},
        "negref": {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": ["neg", 0]}},
        "pos": {"class_type": "SwarmTextEncodeAdvanced", "inputs": {
            "lora_hooks": ["hook", 0],
            "prompt": "Girl <alternate:boy,girl> <lora:Audit/HannahLynn//hook=0>"}},
        "neg": {"class_type": "SwarmTextEncodeAdvanced", "inputs": {
            "lora_hooks": ["hook", 0], "prompt": ""}},
        "hook": {"class_type": "CreateHookLora", "inputs": {
            "lora_name": "Audit/HannahLynn.safetensors",
            "strength_model": 0.8, "strength_clip": 0.0}},
    }


class ShippedDynamicLoRAHookContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validate = staticmethod(real_validator())

    def test_valid_dynamic_hook_connected_through_both_conditioners(self):
        self.assertIsNone(self.validate(native_hook_graph(), "sampler", "hook"))

    def test_missing_authored_hook_marker_must_fail(self):
        graph = native_hook_graph()
        graph["pos"]["inputs"]["prompt"] = "Girl without hook"
        with self.assertRaisesRegex(AssertionError, "matching //hook"):
            self.validate(graph, "sampler", "hook")

    def test_missing_negative_hook_connection_must_fail(self):
        graph = native_hook_graph()
        graph["neg"]["inputs"]["lora_hooks"] = None
        with self.assertRaisesRegex(AssertionError, "negative"):
            self.validate(graph, "sampler", "hook")

    def test_missing_positive_hook_connection_must_fail(self):
        graph = native_hook_graph()
        graph["pos"]["inputs"]["lora_hooks"] = None
        with self.assertRaisesRegex(AssertionError, "positive"):
            self.validate(graph, "sampler", "hook")

    def test_wrong_marker_index_is_not_the_selected_adapter(self):
        graph = native_hook_graph()
        graph["pos"]["inputs"]["prompt"] = "<lora:Audit/HannahLynn//hook=1>"
        with self.assertRaisesRegex(AssertionError, "matching //hook"):
            self.validate(graph, "sampler", "hook")

    def test_cyclic_hooks_chain_is_rejected(self):
        graph = native_hook_graph()
        graph["hook"]["inputs"]["prev_hooks"] = ["hook", 0]
        with self.assertRaisesRegex(AssertionError, "cycle"):
            self.validate(graph, "sampler", "hook")

    def test_a_model_loader_is_not_a_native_hook(self):
        graph = native_hook_graph()
        graph["hook"]["class_type"] = "LoraLoader"
        with self.assertRaisesRegex(AssertionError, "CreateHookLora"):
            self.validate(graph, "sampler", "hook")

    def test_previous_hook_changes_selected_numeric_index(self):
        graph = native_hook_graph()
        graph["hook"]["inputs"]["prev_hooks"] = ["previous", 0]
        graph["previous"] = {"class_type": "CreateHookLora", "inputs": {
            "lora_name": "Other.safetensors", "strength_model": 1.0}}
        with self.assertRaisesRegex(AssertionError, "matching //hook"):
            self.validate(graph, "sampler", "hook")
        graph["pos"]["inputs"]["prompt"] = "<lora:Audit/HannahLynn//hook=1>"
        self.assertIsNone(self.validate(graph, "sampler", "hook"))


if __name__ == "__main__":
    unittest.main()
