"""Regression checks for SwarmKSampler's real asynchronous preview callback.

Execute the selected production functions via AST so no ComfyUI/GPU installation
is needed. This tests dispatch, not GPU inference or browser rendering.
"""
import ast
import contextvars
import pathlib
import threading
import types
import unittest

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmKSampler.py"


class QueuedThread:
    jobs = []

    def __init__(self, target, daemon=False):
        self.target = target

    def start(self):
        self.jobs.append(self.target)


class FakeLatent:
    ndim = 4
    shape = (1, 3, 8, 8)
    device = types.SimpleNamespace(type="cpu")

    def __init__(self, marker):
        self.marker = marker

    def __getitem__(self, index):
        return self


class FakePreviewer:
    def decode_latent_to_preview_image(self, fmt, latent):
        return (fmt, latent.marker, 128)


class PreviewDispatchTests(unittest.TestCase):
    def setUp(self):
        QueuedThread.jobs = []
        self.sent = []
        self.targeted_clients = []
        self.current_context = contextvars.ContextVar("context", default=None)
        self.server = types.SimpleNamespace(last_prompt_id=None, last_node_id=None, client_id=None)
        self.model = types.SimpleNamespace(model=types.SimpleNamespace(latent_format=object()))
        module = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        names = {"get_preview_metadata", "make_swarm_preview_state", "make_swarm_sampler_callback"}
        selected = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in selected}, names)
        progress = lambda steps: types.SimpleNamespace(update_absolute=lambda *args: None)
        def capture(idx, frame, metadata, *, client_id=None):
            self.sent.append((idx, frame, metadata.copy()))
            self.targeted_clients.append(client_id)
        env = {
            "threading": types.SimpleNamespace(Lock=threading.Lock, Thread=QueuedThread),
            "get_executing_context": self.current_context.get,
            "PromptServer": types.SimpleNamespace(instance=self.server),
            "latent_preview": types.SimpleNamespace(get_previewer=lambda device, latent_format: FakePreviewer()),
            "comfy": types.SimpleNamespace(utils=types.SimpleNamespace(ProgressBar=progress)),
            "swarm_send_extra_preview": capture,
            "swarm_send_animated_preview": capture,
            "_swarm_preview_frame_to_image": lambda frame: frame,
        }
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), env)
        self.state = env["make_swarm_preview_state"]
        self.callback = env["make_swarm_sampler_callback"]

    def _prepare(self, job):
        self.server.last_prompt_id = job
        self.server.last_node_id = job + "-sampler"
        self.server.client_id = job + "-client"
        self.current_context.set(types.SimpleNamespace(prompt_id=job, node_id=job + "-sampler"))
        state = self.state()
        cb = self.callback(4, object(), self.model, "one", state)
        return state, cb

    def test_old_job_preview_cannot_appear_in_new_job(self):
        old_state, old_cb = self._prepare("A")
        old_cb(1, FakeLatent("old-pixels"), None, 4)
        old_state["active"] = False
        new_state, new_cb = self._prepare("B")
        new_cb(1, FakeLatent("new-pixels"), None, 4)
        self.current_context.set(None)  # Background threads do not inherit ContextVar.
        self.assertEqual(len(QueuedThread.jobs), 2)
        for job in QueuedThread.jobs:
            job()
        self.assertEqual(self.sent, [(0, "new-pixels", {
            "prompt_id": "B", "node_id": "B-sampler",
            "display_node_id": "B-sampler",
            "parent_node_id": "B-sampler",
            "real_node_id": "B-sampler",
        })])
        new_state["active"] = False

    def test_out_of_order_steps_cannot_overwrite_latest_preview(self):
        state, cb = self._prepare("A")
        cb(0, FakeLatent("step1"), None, 4)
        cb(1, FakeLatent("step2"), None, 4)
        QueuedThread.jobs[1]()  # Delayed step 2 wins.
        QueuedThread.jobs[0]()  # Step 1 must not overwrite it.
        self.assertEqual([item[1] for item in self.sent], ["step2"])
        state["active"] = False

    def test_preview_targets_originating_client_when_global_client_changes(self):
        # A queued worker runs after Comfy's process-global client_id changes.
        # It must never deliver A's pixels to B's websocket client.
        state, cb = self._prepare("A")
        cb(1, FakeLatent("A-current-pixels"), None, 4)
        self.server.client_id = "B-client"
        self.current_context.set(None)
        self.assertEqual(len(QueuedThread.jobs), 1)
        QueuedThread.jobs[0]()
        self.assertEqual(self.targeted_clients, ["A-client"])
        self.assertEqual(self.sent[0][2]["prompt_id"], "A")

    def test_real_preview_serializer_targets_captured_client_not_global_client(self):
        # Exercise the actual sender implementation, not the callback mock.
        import io
        import json
        import struct
        class FakeImage:
            def save(self, stream, **kwargs):
                stream.write(b"PREVIEW-PIXELS")
        sent = []
        server = types.SimpleNamespace(
            client_id="B-client",
            send_sync=lambda event, data, sid=None: sent.append((event, data, sid)),
        )
        module = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        names = {"swarm_send_extra_preview", "swarm_send_animated_preview"}
        selected = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in selected}, names)
        env = {
            "PromptServer": types.SimpleNamespace(instance=server),
            "BytesIO": io.BytesIO, "json": json, "struct": struct,
        }
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), env)
        image = FakeImage()
        env["swarm_send_extra_preview"](0, image, {"prompt_id": "A"}, client_id="A-client")
        env["swarm_send_animated_preview"](0, [image], {"prompt_id": "A"}, client_id="A-client")
        self.assertEqual([sid for event, data, sid in sent], ["A-client", "A-client"])
        self.assertTrue(all(event == 9999123 for event, data, sid in sent))
        for event, data, sid in sent:
            size = struct.unpack(">I", data[:4])[0]
            meta = json.loads(data[4:4 + size])
            self.assertEqual(meta["prompt_id"], "A")
            self.assertIn(b"PREVIEW-PIXELS", data)

    def test_cpu_preview_copy_is_not_nonblocking(self):
        module = ast.parse(SOURCE.read_text(encoding="utf-8"))
        candidates = [n for n in ast.walk(module)
                      if isinstance(n, ast.FunctionDef) and n.name == "_swarm_preview_to_image"]
        self.assertEqual(len(candidates), 1)
        transfers = [n for n in ast.walk(candidates[0]) if isinstance(n, ast.Call)
                     and any(k.arg == "device" and isinstance(k.value, ast.Constant)
                             and k.value.value == "cpu" for k in n.keywords)]
        self.assertTrue(transfers)
        self.assertTrue(all(k.arg != "non_blocking" for n in transfers for k in n.keywords))


if __name__ == "__main__":
    unittest.main()
