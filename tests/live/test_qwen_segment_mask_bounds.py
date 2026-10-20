"""CPU checks for the production SwarmMaskBounds node used by Qwen segments.

These execute the actual node method with real torch tensors, but do not
exercise CLIPSeg, Qwen inference, or the compositor. Live image acceptance is
independently required by qwen_segment_text_replacement.py.

Run: python -m pytest tests/live/test_qwen_segment_mask_bounds.py -q
"""
import ast
from pathlib import Path

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmMasks.py"
)
source_ast = ast.parse(SOURCE.read_text(encoding="utf-8"))
node_ast = next(n for n in source_ast.body if isinstance(n, ast.ClassDef) and n.name == "SwarmMaskBounds")
namespace = {"torch": torch}
exec(compile(ast.Module(body=[node_ast], type_ignores=[]), str(SOURCE), "exec"), namespace)
bounds = namespace["SwarmMaskBounds"]().get_bounds


def qwen_bounds_mask(soft, cutoff=0.5):
    # The actual ThresholdMask(value=0.5) operation from ComfyUI:
    # comfy_extras/nodes_mask.py#ThresholdMask.execute, which is wired in
    # WorkflowGenerator.CreateImageMaskCrop(boundsThreshold=0.5).
    return (soft > cutoff).float()


def test_low_confidence_background_does_not_expand_qwen_crop():
    mask = torch.full((1, 128, 192), 0.02)
    mask[:, 40:90, 60:110] = 1
    assert bounds(qwen_bounds_mask(mask), 8) == (52, 32, 66, 66)


def test_empty_detected_mask_fails_instead_of_full_frame_crop():
    with pytest.raises(ValueError, match="empty"):
        bounds(torch.zeros((1, 128, 192)), 8)


def test_empty_after_qwen_threshold_fails_closed():
    mask = torch.full((1, 128, 192), 0.1)
    with pytest.raises(ValueError, match="empty"):
        bounds(qwen_bounds_mask(mask), 8)


def test_preserves_binary_mask_subject_geometry():
    mask = torch.zeros((1, 128, 192))
    mask[:, 40:90, 60:110] = 1
    assert bounds(mask, 8) == (52, 32, 66, 66)


def test_four_dimensional_comfy_mask_shape():
    mask = torch.zeros((1, 1, 128, 192))
    mask[:, :, 40:90, 60:110] = 1
    assert bounds(mask, 8) == (52, 32, 66, 66)


def test_unscaled_non_qwen_soft_mask_keeps_legacy_behavior():
    mask = torch.full((1, 128, 192), 0.02)
    mask[:, 40:90, 60:110] = 1
    assert bounds(mask, 8) == (0, 0, 192, 128)


def test_non_finite_mask_fails_closed():
    mask = torch.zeros((1, 128, 192))
    mask[0, 60, 80] = torch.nan
    with pytest.raises(ValueError, match="invalid"):
        bounds(mask, 8)
