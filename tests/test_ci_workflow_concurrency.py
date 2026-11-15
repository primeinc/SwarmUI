"""Prevent CI backlog regressions from dropping workflow-level cancellation.

These checks enforce one running/pending run per workflow and ref when newer
commits arrive. Push and PR refs remain distinct, and the two workflows do
not cancel each other.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/unit-tests.yml",
    ROOT / ".github/workflows/build-and-check.yml",
)
EXPECTED_GROUP = 'group: ${{ github.workflow }}-${{ github.ref }}'


def validate_workflow(source):
    if source.count("concurrency:") != 1:
        raise AssertionError("Workflow must have exactly one top-level concurrency rule")
    header = source.split("\njobs:\n", 1)[0]
    if source.count("\njobs:\n") != 1:
        raise AssertionError("Workflow must retain its jobs")
    if "on:\n    push:\n    pull_request:" not in header:
        raise AssertionError("Push and pull-request validation must remain enabled")
    if "\nconcurrency:\n" not in header:
        raise AssertionError("Concurrency must be at workflow level, not job level")
    if EXPECTED_GROUP not in header:
        raise AssertionError("Group must isolate same workflow and same Git ref")
    if "cancel-in-progress: true" not in header:
        raise AssertionError("New commits must cancel superseded active checks")


class WorkflowConcurrencyTests(unittest.TestCase):
    def test_both_workflows_cancel_superseded_checks(self):
        for path in WORKFLOWS:
            with self.subTest(path=path.name):
                validate_workflow(path.read_text(encoding="utf-8"))

    def test_negative_control_cancellation_disabled(self):
        content = WORKFLOWS[0].read_text(encoding="utf-8")
        with self.assertRaises(AssertionError):
            validate_workflow(content.replace("cancel-in-progress: true",
                                              "cancel-in-progress: false"))

    def test_negative_control_cross_workflow_cancellation(self):
        content = WORKFLOWS[1].read_text(encoding="utf-8")
        with self.assertRaises(AssertionError):
            validate_workflow(content.replace(EXPECTED_GROUP,
                                              'group: ${{ github.ref }}'))


if __name__ == "__main__":
    unittest.main()
