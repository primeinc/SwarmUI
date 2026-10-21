"""ComfyUI source contract: Qwen conditioning LoRA hooks reach MODEL patches.

Executes pinned first-party CLIP -> conditioning -> sampler -> ModelPatcher
functions with a fake LoRA weight. This is CPU control-plane proof, NOT image
inference or proof that a real adapter changes generated pixels.
"""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS
from urllib.request import urlopen
import unittest

UPSTREAM_SHA = "e3bee7ffcb7d6547ed0bcd15221792f3b8646b17"
UPSTREAM_BLOBS = {
    "comfy/sampler_helpers.py": "bdce2f2d85a5ffaae59f1f6fc715bd44e47d5853",
    "comfy/sd.py": "e0b9d276b38d80e14f485b9c77e302bd1307c041",
    "comfy/model_patcher.py": "8472b4a4533f23f17a27ebeb4c83be1d99a1d54c",
}


def read_pinned_upstream(path):
    local = os.environ.get("COMFY_SOURCE_ROOT")
    if local:
        data = (Path(local) / path).read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/Comfy-Org/ComfyUI/{UPSTREAM_SHA}/{path}"
        with urlopen(url, timeout=30) as response:
            data = response.read()
    git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if git_blob != UPSTREAM_BLOBS[path]:
        raise AssertionError(f"Canonical Comfy source {path} does not match pinned blob: {git_blob}")
    return ast.parse(data.decode("utf-8"), filename=path)


def extract_method(tree, clsname, method):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == clsname)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)


class HookGroup:
    def __init__(self, hooks=()):
        self.hooks = list(hooks)

    def __len__(self):
        return len(self.hooks)

    def add(self, hook):
        if hook not in self.hooks:
            self.hooks.append(hook)

    def get_type(self, kind):
        return [hook for hook in self.hooks if hook.kind == kind]

    def clone(self):
        return HookGroup(self.hooks)

    @staticmethod
    def combine_all_hooks(groups):
        return HookGroup(h for group in groups for h in group.hooks) if groups else None


class TestWeightHook:
    kind = "Weight"

    def __init__(self):
        self.hook_ref = "qwen-model-lora-marker"
        self.received_targets = []

    def clone(self):
        return self

    def add_hook_patches(self, model, options, targets, registered):
        self.received_targets.append(targets)
        model.hook_patches[self.hook_ref] = {"sample_weight": [0.5]}
        registered.add(self)


class FakeModel:
    wrappers = {}
    callbacks = {}

    def __init__(self):
        self.hook_patches = {}
        self.hook_patches_backup = {}

    def restore_hook_patches(self):
        pass

    def get_all_callbacks(self, _kind):
        return []


class PinnedComfyModelHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers = read_pinned_upstream("comfy/sampler_helpers.py")
        sd = read_pinned_upstream("comfy/sd.py")
        patcher = read_pinned_upstream("comfy/model_patcher.py")
        functions = [n for n in helpers.body
                     if isinstance(n, ast.FunctionDef)
                     and n.name in ("get_hooks_from_cond", "prepare_model_patcher")]
        assert len(functions) == 2
        comfy = NS(
            hooks=NS(HookGroup=HookGroup,
                     create_target_dict=lambda target: {"target": target},
                     EnumWeightTarget=NS(Model="MODEL"),
                     EnumHookType=NS(TransformerOptions="TransformerOptions",
                                     AdditionalModels="AdditionalModels",
                                     Weight="Weight")),
            patcher_extension=NS(merge_nested_dicts=lambda target, source, copy_dict1=False:
                                 target.update(source)),
        )
        scope = {
            "comfy": comfy, "ModelPatcher": FakeModel,
            "ControlBase": type("ControlBase", (), {}),
            "create_hook_patches_clone": lambda patches: dict(patches),
            "CallbacksMP": NS(ON_REGISTER_ALL_HOOK_PATCHES="registration-callbacks"),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     "comfy/sampler_helpers.py", "exec"), scope)
        exec(compile(ast.Module(body=[
            extract_method(patcher, "ModelPatcher", "register_all_hook_patches"),
            extract_method(sd, "CLIP", "add_hooks_to_dict"),
        ], type_ignores=[]), "pinned_comfy_methods.py", "exec"), scope)
        FakeModel.register_all_hook_patches = scope["register_all_hook_patches"]
        cls._prepare_model_patcher = staticmethod(scope["prepare_model_patcher"])
        cls._add_hooks_to_dict = staticmethod(scope["add_hooks_to_dict"])

    def run_path(self, include_condition_hook):
        hook = TestWeightHook()
        clip = NS(apply_hooks_to_conds=HookGroup([hook]) if include_condition_hook else None)
        metadata = self._add_hooks_to_dict(clip, {"pooled_output": "test"})
        model = FakeModel()
        options = {"transformer_options": {}}
        self._prepare_model_patcher(model, {"positive": [metadata]}, options)
        return hook, metadata, model, options

    def test_conditioning_hook_reaches_real_comfy_model_patcher_registration(self):
        hook, metadata, model, options = self.run_path(True)
        self.assertIs(metadata["hooks"].hooks[0], hook)
        self.assertEqual(hook.received_targets, [{"target": "MODEL"}])
        self.assertEqual(model.hook_patches,
                         {"qwen-model-lora-marker": {"sample_weight": [0.5]}})
        self.assertEqual(options["registered_hooks"].hooks, [hook])

    def test_negative_control_dropping_cond_hook_prevents_model_registration(self):
        hook, metadata, model, options = self.run_path(False)
        self.assertNotIn("hooks", metadata)
        self.assertEqual(hook.received_targets, [])
        self.assertEqual(model.hook_patches, {})
        self.assertNotIn("registered_hooks", options)

    def run_shipped_swarm_encoder_to_real_comfy_model(self, drop_cond_hooks=False):
        # Unlike the standalone Comfy contract, this exercises the real
        # shipped SwarmTextEncodeAdvanced implementation and actual Swarm
        # <lora:...//hook=0> parser before entering pinned Comfy methods.
        import torch
        from test_qwen_swarm_encoder_cpu import actual_encoder, FakeQwenClip, SOURCE

        encoder = actual_encoder()
        scope = encoder._source_cpu_scope
        scope["comfy"].hooks = NS(
            HookGroup=HookGroup,
            create_target_dict=lambda target: {"target": target},
            EnumWeightTarget=NS(Clip="CLIP", Model="MODEL"),
        )
        if drop_cond_hooks:
            shipped = SOURCE.read_text(encoding="utf-8")
            target = "            clip.apply_hooks_to_conds = active_hooks\\n"
            if shipped.count(target) != 1:
                raise AssertionError("Shipped Swarm encoder hook assignment changed")
            altered = shipped.replace(target, "            # mutation: remove conditioning hook assignment\\n", 1)
            klass = next(node for node in ast.parse(altered, filename=str(SOURCE)).body
                         if isinstance(node, ast.ClassDef)
                         and node.name == "SwarmTextEncodeAdvanced")
            exec(compile(ast.Module(body=[klass], type_ignores=[]), str(SOURCE), "exec"), scope)
            encoder = scope["SwarmTextEncodeAdvanced"]()

        hook = TestWeightHook()
        add_hooks = self._add_hooks_to_dict
        class HookAwareClip(FakeQwenClip):
            def __init__(self):
                super().__init__()
                self.clip_registered = []
                self.patcher = NS(
                    forced_hooks=None,
                    register_all_hook_patches=lambda hooks, target:
                        self.clip_registered.append((hooks, target)),
                )

            def clone(self, disable_dynamic=False):
                if not disable_dynamic:
                    raise AssertionError("Expected native Swarm dynamic hook clone")
                return self

            def encode_from_tokens_scheduled(self, tokens):
                metadata = add_hooks(self, {"pooled_output": torch.ones(1, 4)})
                return [[torch.ones(1, 1, 4), metadata]]

        clip = HookAwareClip()
        (conds,) = encoder.encode(
            clip=clip, steps=4,
            prompt="HannahLynn <lora:Audit/HannahLynn//hook=0>",
            width=64, height=64, target_width=64, target_height=64,
            lora_hooks=HookGroup([hook]),
            qwen21_prevent_empty_text=True,
        )
        metadata = conds[0][1]
        model, options = FakeModel(), {"transformer_options": {}}
        self._prepare_model_patcher(model, {"positive": [metadata]}, options)
        return hook, clip, metadata, model, options

    def test_actual_swarm_qwen_tag_hooks_reach_real_comfy_sampler_model(self):
        hook, clip, metadata, model, options = self.run_shipped_swarm_encoder_to_real_comfy_model()
        self.assertTrue(clip.clip_registered, "Dynamic LoRA must register on CLIP before encoding")
        self.assertIs(metadata["hooks"].hooks[0], hook,
                      "Actual SwarmText conditioning must carry its tagged LoRA hook")
        self.assertEqual(hook.received_targets, [{"target": "MODEL"}])
        self.assertEqual(model.hook_patches,
                         {"qwen-model-lora-marker": {"sample_weight": [0.5]}})
        self.assertIn("registered_hooks", options)

    def test_mutating_real_swarm_qwen_hook_forwarding_breaks_model_patches(self):
        hook, clip, metadata, model, options = (
            self.run_shipped_swarm_encoder_to_real_comfy_model(drop_cond_hooks=True))
        self.assertNotIn("hooks", metadata,
                         "Without actual clip-to-condition forwarding, the model must not see the LoRA")
        self.assertFalse(model.hook_patches)
        self.assertFalse(hook.received_targets)
        self.assertNotIn("registered_hooks", options)


if __name__ == "__main__":
    unittest.main()
