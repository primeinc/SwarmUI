"""CPU-only checks of the real Swarm advanced parser used by Qwen segments."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_swarm_text():
    parent = types.ModuleType("fake_swarm_text_package")
    parent.__path__ = []
    handling = types.ModuleType("fake_swarm_text_package.SwarmTextHandling")
    handling.PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = ""
    handling.PROMPT_TEMPLATE_QWEN_IMAGE_EDIT_PLUS = ""
    handling.KREA2_TEMPLATE = ""
    torch = types.ModuleType("torch")
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    hooks = types.ModuleType("comfy.hooks")
    comfy.hooks = hooks
    nodes = types.ModuleType("nodes")
    nodes.MAX_RESOLUTION = 16384
    stubs = {
        "fake_swarm_text_package": parent,
        "fake_swarm_text_package.SwarmTextHandling": handling,
        "torch": torch,
        "comfy": comfy,
        "comfy.hooks": hooks,
        "nodes": nodes,
        "node_helpers": types.ModuleType("node_helpers"),
    }
    source = Path(__file__).resolve().parents[1] / (
        "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/"
        "SwarmComfyCommon/SwarmText.py"
    )
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "fake_swarm_text_package.SwarmText", source
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class SwarmQwenAdvancedTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = load_swarm_text()

    def test_qwen_keep_vision_flag_not_misparsed_as_weighted_tokens(self):
        self.assertFalse(self.parser.token_batches_have_weights({
            "qwen3vl_8b": [[(10, 1.0)]],
            "keep_vision": True,
        }))
        self.assertTrue(self.parser.token_batches_have_weights({
            "qwen3vl_8b": [[(10, 1.5)]],
            "keep_vision": False,
        }))

    def test_real_parser_preserves_alternate_fromto_and_dynamic_hooks(self):
        parsed = self.parser.parse_prompt(
            "<fromto[0.5]:first,second> <alternate:left,right>"
            " <lora:face//hook=0>"
        )
        first = parsed.flatten(0, 10)
        second = parsed.flatten(6, 10)
        self.assertIn("first", " ".join(x.text or "" for x in first))
        self.assertIn("second", " ".join(x.text or "" for x in second))
        self.assertEqual([x.lora_hook for x in first if x.lora_hook is not None], [0])
        self.assertEqual([x.lora_hook for x in second if x.lora_hook is not None], [0])


if __name__ == "__main__":
    unittest.main()
