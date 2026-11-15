"""Refactor guard: one ComfyUI websocket lease per submitted prompt.

The original socket pool was a proven cross-generation seam: late previews
could be consumed by the next job when it reused the same socket. Inspect the
actual production AwaitJobLive/ClearSockets code rather than a test-only parser.
"""
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src/BuiltinExtensions/ComfyUIBackend/ComfyUIAPIAbstractBackend.cs"


def verify_session_isolation(source):
    session = source.split("public async Task AwaitJobLive(", 1)[1].split(
        "public static (MediaType, int, int, int) ComfyRawWebsocketOutputToFormatLabel", 1)[0]
    if "ReusableSockets" in source or "ReusableSocket " in source:
        raise AssertionError("Old cross-request websocket pool survives")
    required = (
        "Guid.NewGuid().ToString()",
        "NetworkBackendUtils.ConnectWebsocket(APIAddress",
        'client_id',
        "ActiveJobSockets.TryAdd(socket",
        "ActiveJobSockets.TryRemove(socket",
        "socket.Dispose()",
        "H3PreviewPromptGuard.Matches(output, promptId)",
    )
    for token in required:
        if token not in session:
            raise AssertionError(f"Missing per-request lifecycle: {token}")
    if "ReusableSockets.Enqueue" in session or "ReusableSockets.TryDequeue" in session:
        raise AssertionError("Prompt socket is still recycled")
    if session.index("Guid.NewGuid().ToString()") > session.index("NetworkBackendUtils.ConnectWebsocket"):
        raise AssertionError("Request-scoped client identity must exist before socket connect")
    if session.index("ActiveJobSockets.TryAdd(socket") > session.index("socket.SendJson"):
        raise AssertionError("Register connected socket before feature negotiation")
    if not session.count("socket.Dispose()") >= 2:
        raise AssertionError("Both startup failure and normal finalizer must dispose socket")


class H3RequestSessionTests(unittest.TestCase):
    def test_production_has_no_cross_request_socket_reuse(self):
        verify_session_isolation(SOURCE.read_text(encoding="utf-8"))

    def test_negative_control_restored_reuse_fails(self):
        content = SOURCE.read_text(encoding="utf-8")
        mutant = content.replace("ActiveJobSockets.TryRemove(socket, out _);",
                                 "ReusableSockets.Enqueue(new() { Socket = socket });", 1)
        with self.assertRaises(AssertionError):
            verify_session_isolation(mutant)

    def test_negative_control_missing_failure_dispose_fails(self):
        content = SOURCE.read_text(encoding="utf-8")
        mutant = content.replace("socket.Dispose();", "/* lost dispose */", 1)
        with self.assertRaises(AssertionError):
            verify_session_isolation(mutant)

    def test_shutdown_closes_active_jobs_without_pooling(self):
        content = SOURCE.read_text(encoding="utf-8")
        clear = content.split("public async Task ClearSockets()", 1)[1].split(
            "public override async Task Shutdown()", 1)[0]
        self.assertIn("ActiveJobSockets.Keys", clear)
        self.assertIn("socket.Dispose()", clear)
        self.assertIn("await ClearSockets();", content)


if __name__ == "__main__":
    unittest.main()
