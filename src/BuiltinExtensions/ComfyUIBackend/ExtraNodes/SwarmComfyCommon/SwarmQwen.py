import torch
import comfy.utils
import comfy.model_management
import node_helpers


class SwarmQwenImage21SegmentEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    CATEGORY = "SwarmUI/qwen"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    DESCRIPTION = "Qwen Image 2.1 segment conditioning compatible with ComfyUI TextEncodeQwenImage21."

    def encode(self, clip, vae, image, prompt, negative_prompt):
        # Match Qwen 2.1's native edit contract without depending on the newer upstream
        # TextEncodeQwenImage21 node being registered in the selected Comfy backend.
        source = image[:1]
        samples = source.movedim(-1, 1)
        width = max(32, round(samples.shape[3] / 32) * 32)
        height = max(32, round(samples.shape[2] / 32) * 32)
        if (width, height) == (samples.shape[3], samples.shape[2]):
            resized = source
        else:
            resized = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)

        rgb = resized[:, :, :, :3]
        if resized.shape[-1] > 3:
            rgb = rgb * resized[:, :, :, 3:] + (1.0 - resized[:, :, :, 3:])

        reference = vae.encode(resized)
        images_vl = [rgb]
        positive = clip.encode_from_tokens_scheduled(
            clip.tokenize(prompt or "", images=images_vl, keep_vision=False, prevent_empty_text=True)
        )
        negative = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative_prompt or "", images=images_vl, keep_vision=False, prevent_empty_text=True)
        )
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [reference]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [reference]}, append=True)

        latent = {
            "samples": torch.zeros(
                [1, 64, height // 16, width // 16],
                device=comfy.model_management.intermediate_device(),
            )
        }

        print(
            f"[QWEN21-SEG] encoder crop={tuple(resized.shape)} "
            f"latent={tuple(reference.shape)} prompt={repr((prompt or '')[:160])} "
            f"negative={repr((negative_prompt or '')[:120])}",
            flush=True,
        )
        return (positive, negative, latent)


class SwarmQwen21SegmentDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "generated": ("IMAGE",),
                "mask": ("MASK",),
                "section_id": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "prompt": ("STRING", {"multiline": True}),
                "mode": ("STRING", {"default": "Reference Edit"}),
            }
        }

    CATEGORY = "SwarmUI/qwen"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "diagnose"
    DESCRIPTION = "Pass-through runtime diagnostics for Qwen 2.1 segment edits."

    def diagnose(self, source, generated, mask, section_id, prompt, mode):
        src = source[:1, :, :, :3].detach().float()
        out = generated[:1, :, :, :3].detach().float()

        if src.shape[1:3] != out.shape[1:3]:
            src_cmp = comfy.utils.common_upscale(
                src.movedim(-1, 1), out.shape[2], out.shape[1], "bilinear", "disabled"
            ).movedim(1, -1)
        else:
            src_cmp = src

        diff = (out - src_cmp).abs().mean(dim=-1)
        m = mask[:1].detach().float().to(diff.device)
        if m.dim() == 4:
            m = m[:, 0]
        if m.shape[-2:] != diff.shape[-2:]:
            m = torch.nn.functional.interpolate(
                m.unsqueeze(1), size=diff.shape[-2:], mode="bilinear", align_corners=False
            )[:, 0]
        m = m.clamp(0, 1)

        mask_sum = m.sum().item()
        masked_mae = ((diff * m).sum() / max(mask_sum, 1e-8)).item()
        total_mae = diff.mean().item()
        max_delta = diff.max().item()
        changed = (diff > (1.0 / 255.0)).float()
        changed_masked = ((changed * (m > 0.01)).sum() / max((m > 0.01).sum().item(), 1.0)).item()

        level = "WARNING" if masked_mae < 0.002 else "INFO"
        print(
            f"[QWEN21-SEG] {level} section={section_id} mode={mode!r} "
            f"source={tuple(source.shape)} generated={tuple(generated.shape)} mask={tuple(mask.shape)} "
            f"mae={total_mae:.6f} masked_mae={masked_mae:.6f} max_delta={max_delta:.6f} "
            f"masked_changed_fraction={changed_masked:.4f} prompt={repr((prompt or '')[:160])}",
            flush=True,
        )
        return (generated,)


NODE_CLASS_MAPPINGS = {
    "SwarmQwenImage21SegmentEncode": SwarmQwenImage21SegmentEncode,
    "SwarmQwen21SegmentDiagnostics": SwarmQwen21SegmentDiagnostics,
}
