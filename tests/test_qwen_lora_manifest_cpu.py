"""Execute the actual live-LoRA manifest validator without a backend or model.

Manifest required_loras is an inventory of adapter files, not case display
labels. Distinct case names can legitimately test the same adapter twice.
"""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "tests/live/test_qwen_segment_lora.py"
INTEGRATION = ROOT / "tests/integration/qwen_segment_lora_confinement.py"


def actual_acceptance_cases():
    integration = ast.parse(INTEGRATION.read_text(encoding="utf-8"))
    related = [n for n in integration.body if isinstance(n, ast.FunctionDef)
               and n.name in {"normalized_lora_path", "same_lora_path"}]
    scope = {"os": os, "Path": Path, "json": json, "urlsplit": urlsplit}
    exec(compile(ast.Module(body=related, type_ignores=[]),
                 str(INTEGRATION), "exec"), scope)
    tree = ast.parse(LIVE.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "acceptance_cases")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(LIVE), "exec"), scope)
    return scope["acceptance_cases"]


def make_case(label, lora):
    return {
        "name": label, "lora_name": lora,
        "global_graph": "global.json", "segment_graph": "segment.json",
        "global_lora_node": "lora_global",
        "segment_lora_node": "lora_segment",
        "global_sampler_node": "sampler_global",
        "segment_sampler_node": "sampler_segment",
    }


class LiveLoraManifestCPU(unittest.TestCase):
    def invoke(self, cases, required):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "acceptance.json"
            path.write_text(json.dumps({
                "comfy_url": "http://localhost:8188",
                "cases": cases, "required_loras": required,
            }), encoding="utf-8")
            with patch.dict(os.environ, {"SWARM_LORA_ACCEPTANCE_MANIFEST": str(path)}):
                return actual_acceptance_cases()()

    def test_case_display_name_is_not_adapter_identity(self):
        result = self.invoke([make_case("girl swap / run one",
                                        "subjects/portrait.safetensors")],
                             ["subjects/portrait.safetensors"])
        self.assertEqual(result[0]["name"], "girl swap / run one")

    def test_multiple_test_cases_can_share_one_required_adapter(self):
        cases = [make_case("portrait / seed 1", "subjects/portrait.safetensors"),
                 make_case("portrait / seed 2", "subjects/portrait.safetensors")]
        self.assertEqual(len(self.invoke(cases, ["subjects/portrait.safetensors"])), 2)

    def test_required_adapter_can_have_optional_safetensors_suffix_and_slashes(self):
        result = self.invoke([make_case("portrait A", "subjects/portrait")],
                             ["subjects\\portrait.safetensors"])
        self.assertEqual(len(result), 1)

    def test_missing_and_extra_adapters_are_rejected_by_file_not_case_name(self):
        cases = [make_case("friendly label", "subjects/portrait.safetensors")]
        with self.assertRaisesRegex(ValueError, "required_loras"):
            self.invoke(cases, ["friendly label"])
        with self.assertRaisesRegex(ValueError, "required_loras"):
            self.invoke(cases, ["subjects/portrait.safetensors", "subjects/other.safetensors"])

    def test_case_sensitive_adapter_identity_not_papered_over(self):
        with self.assertRaisesRegex(ValueError, "required_loras"):
            self.invoke([make_case("portrait", "subjects/Portrait.safetensors")],
                        ["subjects/portrait.safetensors"])

    def test_duplicate_required_names_for_same_adapter_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.invoke([make_case("portrait", "subjects/portrait.safetensors")],
                        ["subjects/portrait", "subjects/portrait.safetensors"])

    def test_missing_lora_identity_is_rejected_even_if_case_name_matches(self):
        broken = make_case("portrait", "")
        with self.assertRaisesRegex(ValueError, "lora_name"):
            self.invoke([broken], ["portrait"])


if __name__ == "__main__":
    unittest.main()
