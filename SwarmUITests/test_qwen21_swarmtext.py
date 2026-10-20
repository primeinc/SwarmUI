"""CPU-only checks of the real Swarm advanced parser used by Qwen segments."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeHook:
    def clone(self):
        return FakeHook()


class FakeHookGroup:
    def __init__(self, hooks=None):
        self.hooks = list(hooks or [])

    def __len__(self):
        return len(self.hooks)

    def add(self, hook):
        self.hooks.append(hook)

    def clone(self):
        return FakeHookGroup(self.hooks)


class FakeTokenLeaf:
    start_token = None
    end_token = None
    pad_token = -1
    min_length = None
    max_length = 1000
    max_word_length = 1000
    pad_to_max_length = False

    def _try_get_embedding(self, name):
        return (None, None, None)

    def tokenize_with_weights(self, text, **kwargs):
        return [[(ord(character), 1.0) for character in text]]


class FakeClip:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(qwen3vl_8b=FakeTokenLeaf())
        self.patcher = types.SimpleNamespace(
            forced_hooks=None, register_all_hook_patches=self.register_hooks
        )
        self.apply_hooks_to_conds = None
        self.tokens_seen = []
        self.encoded_texts = []
        self.registered_hooks = None

    def register_hooks(self, hooks, targets):
        self.registered_hooks = hooks

    def clone(self, disable_dynamic=False):
        return self

    def tokenize(self, text, **kwargs):
        self.tokens_seen.append((text, kwargs))
        return {
            "qwen3vl_8b": [[(ord(character), 1.0) for character in text]],
            "keep_vision": kwargs.get("keep_vision", False),
        }

    def encode_from_tokens_scheduled(self, tokens):
        chars = tokens["qwen3vl_8b"][0]
        self.encoded_texts.append("".join(chr(code) for code, *_ in chars))
        return [[None, {"pooled_output": 1.0}]]


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
    hooks.HookGroup = FakeHookGroup
    hooks.create_target_dict = lambda target: {}
    hooks.EnumWeightTarget = types.SimpleNamespace(Clip="clip")
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

    def test_advanced_encoder_executes_real_schedule_with_qwen_images_and_hooks(self):
        clip = FakeClip()
        hooks = FakeHookGroup([FakeHook()])
        prompt = "<fromto[0.5]:first,second> <lora:face//hook=0>"
        conditionings = self.parser.SwarmTextEncodeAdvanced().encode(
            clip, 10, prompt, 64, 64, 64, 64,
            images=[object(), object()], lora_hooks=hooks,
            tokenize_kwargs={"keep_vision": False, "prevent_empty_text": True},
        )[0]
        self.assertEqual(len(conditionings), 2)
        self.assertIn("first", clip.encoded_texts[0])
        self.assertIn("second", clip.encoded_texts[1])
        self.assertLessEqual(conditionings[0][1]["start_percent"], 0)
        self.assertGreaterEqual(conditionings[1][1]["start_percent"], 0.499)
        self.assertIs(clip.registered_hooks, hooks)
        self.assertTrue(all(token_kwargs.get("keep_vision") is False
                            for _, token_kwargs in clip.tokens_seen))
        self.assertTrue(all(len(token_kwargs.get("images", [])) == 2
                            for _, token_kwargs in clip.tokens_seen
                            if "images" in token_kwargs))

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
