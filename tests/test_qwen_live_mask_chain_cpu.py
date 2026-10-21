"""CPU contract for the shipped *live acceptance* Qwen mask validator.

Executes the actual validator via AST without invoking pytest collection, which
correctly requires a real model-backed acceptance manifest. No fake inference.
The negative fixture deliberately removes the upstream inverse-mask binary
threshold; pre-change code must fail the valid inverted fixture.
"""
import ast
import copy
from pathlib import Path
import unittest

SOURCE = (Path(__file__).resolve().parents[1] /
          "tests/live/test_qwen_segment_lora.py")


def shipped_mask_validator():
    module = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == "validate_qwen_subject_mask_chain")
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["validate_qwen_subject_mask_chain"]


def normal_graph():
    return {
        "detector": {"class_type": "SwarmClipSeg", "inputs": {}},
        "guard": {"class_type": "SwarmRequireNonEmptyMask", "inputs": {
            "mask": ["detector", 0], "minimum_confidence": 0.0,
            "maximum_coverage": 0.99}},
        "binary": {"class_type": "ThresholdMask", "inputs": {
            "mask": ["guard", 0], "value": 0.0}},
        "crop": {"class_type": "InpaintCropImproved", "inputs": {
            "mask": ["binary", 0], "mask_hipass_filter": 0.5}},
    }


def inverted_graph():
    graph = normal_graph()
    graph["pre_guard"] = {"class_type": "SwarmRequireNonEmptyMask", "inputs": {
        "mask": ["detector", 0], "minimum_confidence": 0.0,
        "maximum_coverage": 1.0}}
    graph["pre_binary"] = {"class_type": "ThresholdMask", "inputs": {
        "mask": ["pre_guard", 0], "value": 0.0}}
    graph["invert"] = {"class_type": "InvertMask", "inputs": {
        "mask": ["pre_binary", 0]}}
    graph["guard"]["inputs"]["mask"] = ["invert", 0]
    graph["guard"]["inputs"]["maximum_coverage"] = 1.0
    return graph


class LiveQwenMaskChainContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validate = staticmethod(shipped_mask_validator())

    def test_normal_detector_support_remains_valid(self):
        self.assertEqual(self.validate(normal_graph(), "crop"), ["binary", 0])

    def test_genuine_inverse_detects_and_binarizes_before_complement(self):
        self.assertEqual(self.validate(inverted_graph(), "crop"), ["binary", 0])

    def test_inverse_without_binary_original_is_rejected(self):
        graph = inverted_graph()
        graph["invert"]["inputs"]["mask"] = ["pre_guard", 0]
        with self.assertRaises(AssertionError):
            self.validate(graph, "crop")

    def test_inverse_without_original_nonempty_guard_is_rejected(self):
        graph = inverted_graph()
        graph["pre_binary"]["inputs"]["mask"] = ["detector", 0]
        with self.assertRaises(AssertionError):
            self.validate(graph, "crop")

    def test_inverse_requires_allowed_full_complement_coverage(self):
        graph = inverted_graph()
        graph["guard"]["inputs"]["maximum_coverage"] = 0.99
        with self.assertRaises(AssertionError):
            self.validate(graph, "crop")

    def test_normal_must_still_reject_full_canvas_coverage(self):
        graph = normal_graph()
        graph["guard"]["inputs"]["maximum_coverage"] = 1.0
        with self.assertRaises(AssertionError):
            self.validate(graph, "crop")

    def test_second_confidence_threshold_still_invalid(self):
        graph = inverted_graph()
        graph["binary"]["inputs"]["value"] = 0.5
        with self.assertRaises(AssertionError):
            self.validate(graph, "crop")


if __name__ == "__main__":
    unittest.main()
