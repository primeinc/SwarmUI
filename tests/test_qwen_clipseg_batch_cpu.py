"""Execute the shipped SwarmClipSeg.seg on CPU with controlled logits, not weights.

Positive case must retain a mask for every supplied image in order; a detector
that silently processes only images[0] fails this test without a model or GPU.
"""
import ast
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch


SOURCE = (Path(__file__).resolve().parents[1] /
          "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmClipSeg.py")


def native_detector():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    native = next(n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == "SwarmClipSeg")

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, _path):
            return cls()

        def __call__(self, *, text, images, return_tensors, padding):
            assert text == "girl" and return_tensors == "pt" and padding
            return {"brightness": int(np.asarray(images)[0, 0, 0])}

    class FakeModel:
        load_count = 0

        @classmethod
        def from_pretrained(cls, _path):
            cls.load_count += 1
            return cls()

        def __call__(self, *, brightness):
            probs = torch.full((8, 8), 0.2)
            if brightness > 128:
                probs[:, 4:] = 0.95
            else:
                probs[:, :4] = 0.95
            return [torch.logit(probs)]

    scope = {
        "np": np,
        "torch": torch,
        "Image": Image,
        "get_path": lambda: "/no-network",
        "download_model": lambda *_args: None,
        "CLIPSegProcessor": FakeProcessor,
        "CLIPSegForImageSegmentation": FakeModel,
        "_load_cached_clipseg": lambda path: (
            FakeProcessor.from_pretrained(path), FakeModel.from_pretrained(path)),
    }
    exec(compile(ast.Module(body=[native], type_ignores=[]),
                 str(SOURCE), "exec"), scope)
    return scope["SwarmClipSeg"](), FakeModel


class SwarmClipSegBatchCPU(unittest.TestCase):
    def test_two_input_images_get_distinct_ordered_mask_frames(self):
        detector, fake_model = native_detector()
        images = torch.stack([
            torch.zeros((32, 48, 3), dtype=torch.float32),
            torch.ones((32, 48, 3), dtype=torch.float32),
        ])
        mask, = detector.seg(images, "girl", 0.5)
        self.assertEqual(tuple(mask.shape), (2, 32, 48))
        self.assertNotEqual(float(mask[0, 12, 5]), float(mask[1, 12, 5]))
        self.assertGreater(float(mask[0, 12, 5]), 0.95)
        self.assertEqual(float(mask[1, 12, 5]), 0.0)
        self.assertEqual(float(mask[0, 12, 42]), 0.0)
        self.assertGreater(float(mask[1, 12, 42]), 0.95)
        self.assertEqual(fake_model.load_count, 1,
                         "A batch must reuse one loaded CLIPSeg model per call")

    def test_single_image_legacy_shape_preserved(self):
        detector, _ = native_detector()
        mask, = detector.seg(torch.zeros((1, 48, 32, 3)), "girl", 0.5)
        self.assertEqual(tuple(mask.shape), (1, 48, 32))

    def test_empty_or_unbatched_images_fail_instead_of_silent_drop(self):
        detector, _ = native_detector()
        for invalid in [torch.empty(0, 8, 8, 3), torch.zeros(8, 8, 3)]:
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaisesRegex(ValueError, "nonempty BHWC"):
                    detector.seg(invalid, "girl", 0.5)


if __name__ == "__main__":
    unittest.main()
