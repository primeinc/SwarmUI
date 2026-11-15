"""The step-2 stale-preview defect is at the actual SwarmUI websocket seam.

The real .NET helper is executed by tests/h3-preview-guard; these assertions
ensure the parser is not 'fixed' in isolation while the live path bypasses it.
"""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONT = ROOT / "src/BuiltinExtensions/ComfyUIBackend/ComfyUIAPIAbstractBackend.cs"
SENDER = ROOT / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmKSampler.py"
SHIM = ROOT / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmInternalUtil.py"


def require_guard_before_dispatch(source):
    method = source.split("public async Task AwaitJobLive(", 1)[1].split("public static (MediaType", 1)[0]
    binary_path = method.split("ComfyRawWebsocketOutputToFormatLabel(output);", 1)[0]
    guard = "if (!isMe || !H3PreviewPromptGuard.Matches(output, promptId))"
    if guard not in binary_path:
        raise AssertionError("Binary preview path must reject unowned or stale frames")
    if binary_path.index(guard) > binary_path.index("else\n                    {", binary_path.index("byte[] output")) + 1000:
        raise AssertionError("Prompt guard must be immediate in the binary path")
    if "continue;" not in binary_path[binary_path.index(guard):]:
        raise AssertionError("Rejected preview must not be sent to the UI")
    if "takeOutput(new JObject()" not in method:
        raise AssertionError("Real preview output consumer disappeared")


class H3PreviewDeliveryTests(unittest.TestCase):
    def test_live_websocket_drops_prior_job_before_preview_output(self):
        require_guard_before_dispatch(FRONT.read_text(encoding="utf-8"))

    def test_a_mutation_that_removes_the_live_guard_fails(self):
        source = FRONT.read_text(encoding="utf-8")
        mutant = source.replace("if (!isMe || !H3PreviewPromptGuard.Matches(output, promptId))",
                                "if (false)", 1)
        with self.assertRaises(AssertionError):
            require_guard_before_dispatch(mutant)

    def test_custom_sender_carries_prompt_id_in_event_four_metadata(self):
        sender = SENDER.read_text(encoding="utf-8")
        shim = SHIM.read_text(encoding="utf-8")
        self.assertIn('"prompt_id": prompt_id', sender)
        self.assertIn("metadata_json = json.dumps(metadata)", sender)
        self.assertIn("server.send_sync(9999123, combined_data", sender)
        self.assertIn("if event == 9999123:", shim)
        self.assertIn("event = 4", shim)


if __name__ == "__main__":
    unittest.main()
