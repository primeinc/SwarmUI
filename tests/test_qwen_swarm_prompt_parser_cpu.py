"""Exercise the shipped Swarm prompt parser on CPU, without Comfy or Qwen weights.

The code under test is extracted by AST from the real SwarmText.py file, not
reimplemented here or substituted by a fake encoder. No claim is made about
visual model output or LoRA effectiveness: this proves only parsing semantics.
"""
import ast
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[1] /
          "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmText.py")
REQUIRED = {
    "ParsedText", "quick_simple_tag_filler", "index_of_noncontained",
    "split_tag", "split_smart", "parse_prompt",
}


def load_actual_parser():
    syntax = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    picked = [
        node for node in syntax.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in REQUIRED
    ]
    found = {node.name for node in picked}
    if found != REQUIRED:
        raise AssertionError(f"Shipped parser changed: missing {sorted(REQUIRED - found)}")
    scope = {}
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["parse_prompt"]


class RealSwarmQwenSegmentPromptParserTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parse_prompt = staticmethod(load_actual_parser())

    def leaves(self, prompt, step=0, steps=4):
        return self.parse_prompt(prompt).flatten(step, steps)

    def test_user_writes_edit_instruction_verbatim(self):
        authored = "Edit the image, replace the girl with a black boy with blue eyes"
        self.assertEqual("".join(x.text or "" for x in self.leaves(authored)), authored)
        self.assertEqual("".join(x.text or "" for x in self.leaves("删除女孩")), "删除女孩")

    def test_nested_weights_retain_words_and_multiply_weights(self):
        leaves = self.leaves("<weight[1.2]:portrait <weight[1.5]:HannahLynn>>")
        self.assertEqual("".join(x.text or "" for x in leaves), "portrait HannahLynn")
        self.assertAlmostEqual(leaves[0].weight, 1.2)
        self.assertAlmostEqual(leaves[1].weight, 1.8)

    def test_alternates_and_fromto_are_evaluated_per_step(self):
        for prompt in ("<alternate:girl,boy>", "<fromto[0.5]:girl,boy>"):
            self.assertEqual("".join(x.text or "" for x in self.leaves(prompt, 0)), "girl")
            self.assertEqual("".join(x.text or "" for x in self.leaves(prompt, 3)), "boy")

    def test_segment_lora_hook_is_not_dropped(self):
        leaves = self.leaves("HannahLynn <lora:Model//hook=0>")
        self.assertEqual([x.lora_hook for x in leaves if x.lora_hook is not None], [0])
        self.assertIn("HannahLynn", "".join(x.text or "" for x in leaves))

    def test_break_and_embeddings_keep_structural_meaning(self):
        leaves = self.leaves("head<break><embed:face>")
        self.assertEqual(sum(bool(x.is_break) for x in leaves), 1)
        self.assertEqual([x.embed for x in leaves if x.embed is not None], ["face"])

    def test_unknown_image_reference_and_bad_weight_remain_literal(self):
        self.assertEqual(
            "".join(x.text or "" for x in self.leaves("Replace <image1>")),
            "Replace <image1>")
        malformed = "<weight[abc]:Remove the hat>"
        self.assertEqual("".join(x.text or "" for x in self.leaves(malformed)), malformed)


# Reuse the already scheduled CPU prompt-parser CI job for Qwen image-slot
# execution regressions, without changing the workflow definition.
from test_qwen_reference_break_cpu import QwenSwarmBreakReferenceContract, QwenNativeWeightedShellContract


if __name__ == "__main__":
    unittest.main()
