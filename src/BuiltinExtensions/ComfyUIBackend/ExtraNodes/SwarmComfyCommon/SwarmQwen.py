import torch


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


class SwarmQwen21CompositeDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before": ("IMAGE",),
                "after": ("IMAGE",),
                "mask": ("MASK",),
                "section_id": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    CATEGORY = "SwarmUI/qwen"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "diagnose"
    DESCRIPTION = "Pass through the final composite while reporting actual before/after changes."

    def diagnose(self, before, after, mask, section_id):
        src = before[:1, :, :, :3].detach().float()
        final = after[:1, :, :, :3].detach().float()
        if src.shape != final.shape:
            raise ValueError(
                f"[QWEN21-SEG] Final compositor changed spatial shape: "
                f"source={tuple(src.shape)}, final={tuple(final.shape)}"
            )
        diff = (final - src).abs().mean(dim=-1)
        m = mask[:1].detach().float().to(diff.device)
        if m.dim() == 4:
            m = m[:, 0]
        if m.shape[-2:] != diff.shape[-2:]:
            m = torch.nn.functional.interpolate(
                m.unsqueeze(1), size=diff.shape[-2:], mode="bilinear", align_corners=False
            )[:, 0]
        m = m.clamp(0, 1)
        mask_weight = m.sum().item()
        mask_coverage = (m > 0.05).float().mean().item()
        masked_mae = ((diff * m).sum() / max(mask_weight, 1e-8)).item()
        full_mae = diff.mean().item()
        masked_changed = (
            ((diff > (1.0 / 255)).float() * (m > 0.05)).sum()
            / max((m > 0.05).sum().item(), 1.0)
        ).item()
        level = "WARNING" if masked_mae < 0.002 else "INFO"
        print(
            f"[QWEN21-SEG] COMPOSITE {level} section={section_id} "
            f"before={tuple(before.shape)} after={tuple(after.shape)} "
            f"mask_mean={m.mean().item():.4f} mask_coverage={mask_coverage:.4f} "
            f"full_mae={full_mae:.6f} masked_mae={masked_mae:.6f} "
            f"masked_changed_fraction={masked_changed:.4f}",
            flush=True,
        )
        return (after,)


NODE_CLASS_MAPPINGS = {
    "SwarmQwen21SegmentDiagnostics": SwarmQwen21SegmentDiagnostics,
    "SwarmQwen21CompositeDiagnostics": SwarmQwen21CompositeDiagnostics,
}
