"""CPU regression of the shipped CLIPSeg model download/cache function.

These run real filesystem reads/writes and real safetensors header validation,
using an in-memory HTTP response stub. No network, CLIPSeg weights or GPU.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save

SOURCE = (Path(__file__).resolve().parents[1] /
          "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmClipSeg.py")


def load_real_module():
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.CLIPSegProcessor = object
    fake_transformers.CLIPSegForImageSegmentation = object
    fake_folder = types.ModuleType("folder_paths")
    fake_folder.folder_names_and_paths = {}
    spec = importlib.util.spec_from_file_location("real_swarm_clipseg", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"transformers": fake_transformers, "folder_paths": fake_folder}):
        spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, body, status=200, declared_length=None, interrupt=False):
        self.body = body
        self.status = status
        self.interrupt = interrupt
        self.headers = {"Content-Length": str(len(body) if declared_length is None else declared_length)}
        self.closed = False

    def raise_for_status(self):
        if self.status >= 400:
            raise IOError(f"HTTP {self.status}")

    def iter_content(self, chunk_size=1024 * 1024):
        half = max(1, len(self.body) // 2)
        yield self.body[:half]
        if self.interrupt:
            raise IOError("connection lost mid-file")
        yield self.body[half:]

    def close(self):
        self.closed = True


class HTTPStub:
    def __init__(self, contents):
        self.contents = contents
        self.calls = []
        self.overrides = {}

    def get(self, url, **kwargs):
        name = url.rsplit("/", 1)[-1]
        self.calls.append((name, kwargs))
        response_config = self.overrides.get(name, {})
        return FakeResponse(self.contents[name], **response_config)


class ClipSegCacheCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_real_module()
        cls.bodies = {
            "model.safetensors": save({"weight": torch.ones(2)}),
            "merges.txt": b"#version: 0.2\n",
        }
        for name in cls.module._CLIPSEG_FILES:
            if name not in cls.bodies:
                cls.bodies[name] = json.dumps({"fixture": name}).encode("utf-8")

    def fake_download(self, directory, http):
        with patch.object(self.module.requests, "get", http.get):
            self.module.download_model(str(directory), "https://example.invalid/models/")

    def test_existing_empty_cache_directory_downloads_all_required_files(self):
        with tempfile.TemporaryDirectory() as root:
            location = Path(root) / "model"
            location.mkdir()
            http = HTTPStub(self.bodies)
            self.fake_download(location, http)
            self.assertEqual({call[0] for call in http.calls},
                             set(self.module._CLIPSEG_FILES))
            for name in self.module._CLIPSEG_FILES:
                self.assertEqual((location / name).read_bytes(), self.bodies[name])
            http.calls.clear()
            self.fake_download(location, http)
            self.assertEqual(http.calls, [], "Complete cache must never redownload")

    def test_corrupt_json_and_safetensors_cache_files_are_repaired(self):
        with tempfile.TemporaryDirectory() as root:
            location = Path(root)
            self.fake_download(location, HTTPStub(self.bodies))
            (location / "config.json").write_bytes(b"{invalid")
            (location / "model.safetensors").write_bytes(b"partial")
            http = HTTPStub(self.bodies)
            self.fake_download(location, http)
            self.assertEqual({call[0] for call in http.calls},
                             {"config.json", "model.safetensors"})
            self.assertEqual((location / "model.safetensors").read_bytes(),
                             self.bodies["model.safetensors"])

    def test_truncated_response_must_not_publish_partial_file(self):
        with tempfile.TemporaryDirectory() as root:
            location = Path(root)
            http = HTTPStub(self.bodies)
            http.overrides["config.json"] = {
                "declared_length": len(self.bodies["config.json"]) + 12
            }
            with self.assertRaisesRegex(IOError, "truncated"):
                self.fake_download(location, http)
            self.assertFalse((location / "config.json").exists())
            self.assertEqual(list(location.glob("*.partial")), [])
            self.assertEqual(list(location.glob(".*.partial")), [])
            http.overrides.clear()
            self.fake_download(location, http)
            self.assertTrue((location / "model.safetensors").exists())

    def test_http_error_and_interrupted_transfer_leave_no_poisoned_cache(self):
        for response_override in ({"status": 404}, {"interrupt": True}):
            with self.subTest(override=response_override), tempfile.TemporaryDirectory() as root:
                location = Path(root)
                http = HTTPStub(self.bodies)
                http.overrides["config.json"] = response_override
                with self.assertRaises((OSError, IOError)):
                    self.fake_download(location, http)
                self.assertFalse((location / "config.json").exists())
                self.assertFalse(any(location.glob("*.partial")))
                self.assertFalse(any(location.glob(".*.partial")))
                http.overrides.clear()
                self.fake_download(location, http)
                self.assertEqual((location / "config.json").read_bytes(),
                                 self.bodies["config.json"])


if __name__ == "__main__":
    unittest.main()
