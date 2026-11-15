"""Ensure H3 sampler logging derives from the actual graph node decision.

Previously the trace route recomputed a separate Features predicate before
CreateNode, which could diverge from the execution path after future edits.
"""
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs"


def verify_route_matches_created_node(content):
    body = content.split("public string CreateKSampler(", 1)[1].split(
        "/// <summary>Returns a reference to the first prompt image", 1)[0]
    required = (
        'bool useSwarmSampler = Features.Contains("variation_seed") && !RestrictCustomNodes;',
        'if (useSwarmSampler)',
        'created = CreateNode("SwarmKSampler", inputs, firstId);',
        'created = CreateNode("KSamplerAdvanced", inputs, firstId);',
        'sampler={samplerNodeClass} node={created}',
        'per_step_trace={useSwarmSampler}',
        'if (!useSwarmSampler && isH3TraceRoute)',
    )
    for item in required:
        if item not in body:
            raise AssertionError("Missing shared sampler route: " + item)
    if body.count('Features.Contains("variation_seed") && !RestrictCustomNodes') != 1:
        raise AssertionError("Sampler selection must have exactly one predicate")
    if body.index("if (useSwarmSampler)") < body.index('bool useSwarmSampler ='):
        raise AssertionError("Route must be decided before graph generation")
    if body.index("node={created}") < body.index('created = CreateNode("KSamplerAdvanced"'):
        raise AssertionError("Log must report the actual generated node, not a guess")


class H3SamplerDispatchContractTests(unittest.TestCase):
    def test_actual_graph_route_and_log_share_one_predicate(self):
        verify_route_matches_created_node(SOURCE.read_text(encoding="utf-8"))

    def test_negative_duplicate_predicate_is_rejected(self):
        source = SOURCE.read_text(encoding="utf-8")
        mutant = source.replace('if (useSwarmSampler)',
                                'if (Features.Contains("variation_seed") && !RestrictCustomNodes)', 1)
        with self.assertRaises(AssertionError):
            verify_route_matches_created_node(mutant)

    def test_negative_log_must_report_real_node(self):
        source = SOURCE.read_text(encoding="utf-8")
        mutant = source.replace('node={created}', 'node={firstId ?? "auto"}', 1)
        with self.assertRaises(AssertionError):
            verify_route_matches_created_node(mutant)


if __name__ == "__main__":
    unittest.main()
