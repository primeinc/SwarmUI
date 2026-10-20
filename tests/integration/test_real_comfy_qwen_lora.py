"""Real ComfyUI/Qwen 2.1 loader-to-forward contract, not a graph or mocked-node test.

Uses the pinned ComfyUI checkout and its actual:
  - QwenImage21 transformer / BaseModel
  - LoraLoaderModelOnly and LoRA safetensors reader
  - LoRA key conversion / ModelPatcher weight application
  - Qwen transformer forward

A reduced-size real Qwen 2.1 architecture keeps CI CPU-compatible.
It is NOT a substitute for the full-size user's LoRA identity acceptance.
"""
import os
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

comfy_root = os.environ.get("COMFYUI_REPO_PATH")
if not comfy_root or not Path(comfy_root, "comfy/model_patcher.py").is_file():
    raise RuntimeError(
        "Real ComfyUI checkout required: set COMFYUI_REPO_PATH to pinned upstream."
        " No stubs or skipped fallback are permitted."
    )
sys.path.insert(0, str(Path(comfy_root).resolve()))

from comfy.cli_args import args
args.cpu = True

import comfy.model_management
comfy.model_management.in_training = True

import comfy.ops
import comfy.model_base
import comfy.model_patcher
import comfy.supported_models
import folder_paths
from nodes import LoraLoaderModelOnly


@pytest.fixture()
def qwen_model(tmp_path):
    """Instantiate the upstream Qwen Image 2.1 implementation, reduced only in size."""
    torch.manual_seed(35017)
    config = comfy.supported_models.QwenImage21({
        "image_model": "qwen_image21",
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 1,
        "context_in_dim": 16,
        "mlp_ratio": 1,
        "axes_dims_rope": (4, 6, 6),
        "fused_mlp": False,
        "dtype": torch.float32,
    })
    config.custom_operations = comfy.ops.disable_weight_init
    model = comfy.model_base.QwenImage21(config, device=torch.device("cpu"))
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(mean=0.0, std=0.15)
    patcher = comfy.model_patcher.ModelPatcher(
        model, torch.device("cpu"), torch.device("cpu")
    )

    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    key = "diffusion_model.proj_out"
    original_weight = model.diffusion_model.proj_out.weight.detach().clone()
    rank = 2
    torch.manual_seed(35018)
    up = torch.randn(original_weight.shape[0], rank) * 0.6
    down = torch.randn(rank, original_weight.shape[1]) * 0.6
    save_file({
        f"{key}.lora_up.weight": up.contiguous(),
        f"{key}.lora_down.weight": down.contiguous(),
    }, str(lora_dir / "actual-comfy-lora.safetensors"))
    folder_paths.add_model_folder_path("loras", str(lora_dir), is_default=True)

    x = torch.randn(1, 4, 2, 2)
    t = torch.tensor([0.3], dtype=torch.float32)
    context = torch.randn(1, 3, 16)

    def real_forward():
        with torch.no_grad():
            return model.diffusion_model(
                x.clone(), t.clone(), context.clone(),
                transformer_options={},
            ).detach().clone()

    yield patcher, model, real_forward, "diffusion_model.proj_out.weight"
    patcher.unpatch_model()


def test_real_qwen_lora_is_applied_to_actual_forward_and_can_be_removed(qwen_model):
    base, model, forward, target = qwen_model
    original = forward()
    loader = LoraLoaderModelOnly()

    off = loader.load_lora_model_only(base, "actual-comfy-lora.safetensors", 0)[0]
    assert off is base
    torch.testing.assert_close(forward(), original, atol=0, rtol=0)

    scoped = loader.load_lora_model_only(base, "actual-comfy-lora.safetensors", 1.0)[0]
    assert scoped is not base
    assert target in scoped.patches, "Real Comfy loader did not accept any model weights"
    assert scoped.patches_uuid != base.patches_uuid

    scoped.patch_weight_to_device(target, device_to=torch.device("cpu"))
    modified = forward()
    assert not torch.allclose(original, modified, atol=1e-6, rtol=1e-6), (
        "Real Qwen transformer output did not change after patch application."
    )

    scoped.unpatch_model(device_to=torch.device("cpu"))
    torch.testing.assert_close(forward(), original, atol=0, rtol=0)


def test_real_qwen_confined_variant_overlays_global_and_stays_independent(qwen_model):
    base, model, forward, target = qwen_model
    original = forward()
    loader = LoraLoaderModelOnly()
    global_model = loader.load_lora_model_only(base, "actual-comfy-lora.safetensors", 0.25)[0]
    confined = loader.load_lora_model_only(global_model, "actual-comfy-lora.safetensors", 0.75)[0]
    assert len(confined.patches[target]) == 2, (
        "Real segment variant lost its inherited global adapter."
    )
    assert len(global_model.patches[target]) == 1
    assert target not in base.patches

    global_model.patch_weight_to_device(target, device_to=torch.device("cpu"))
    global_result = forward()
    global_model.unpatch_model(device_to=torch.device("cpu"))

    confined.patch_weight_to_device(target, device_to=torch.device("cpu"))
    confined_result = forward()
    confined.unpatch_model(device_to=torch.device("cpu"))

    assert not torch.allclose(original, global_result, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(global_result, confined_result, atol=1e-6, rtol=1e-6), (
        "Segment-specific adapter has no effect beyond the global adapter."
    )
    torch.testing.assert_close(forward(), original, atol=0, rtol=0)
