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
            },
            "optional": {
                "use_original_reference": ("BOOLEAN", {"default": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "lora_hooks": ("HOOKS",),
                "reference_tile_size": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "reference_tile_overlap": ("INT", {"default": 64, "min": 0, "max": 4096}),
                "reference_temporal_tile_size": ("INT", {"default": 32, "min": 8, "max": 4096}),
                "reference_temporal_tile_overlap": ("INT", {"default": 4, "min": 4, "max": 4096}),
                **{f"image_{i}": ("IMAGE",) for i in range(1, 17)},
            },
        }

    CATEGORY = "SwarmUI/qwen"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    DESCRIPTION = "Qwen Image 2.1 segment conditioning compatible with ComfyUI TextEncodeQwenImage21."

    def encode(
        self, clip, vae, image, prompt, negative_prompt,
        use_original_reference=True, reference_tile_size=0, reference_tile_overlap=64,
        reference_temporal_tile_size=32, reference_temporal_tile_overlap=4,
        steps=20, lora_hooks=None, **reference_images,
    ):
        # TextEncodeQwenImage21 supports one image per native vision slot. The C# workflow
        # isolates batch items before this node, preventing one source image from
        # silently providing references for the rest of a batch.
        if image.shape[0] != 1:
            raise ValueError("Qwen segment encoder requires one source crop per invocation")
        source = image[:1]

        def resize_reference(ref_image):
            samples = ref_image[:1].movedim(-1, 1)
            width = max(32, round(samples.shape[3] / 32) * 32)
            height = max(32, round(samples.shape[2] / 32) * 32)
            if (width, height) == (samples.shape[3], samples.shape[2]):
                return ref_image[:1]
            return comfy.utils.common_upscale(
                samples, width, height, "lanczos", "disabled"
            ).movedim(1, -1)

        resized = resize_reference(source)
        height, width = resized.shape[1:3]
        references = [resized] if use_original_reference else []
        for key in sorted(reference_images, key=lambda key: int(key.rsplit("_", 1)[-1])):
            ref_image = reference_images[key]
            if ref_image is not None:
                if len(references) >= 16:
                    raise ValueError("Qwen Image 2.1 supports at most sixteen reference images")
                references.append(resize_reference(ref_image))

        images_vl = []
        ref_latents = []
        for ref_image in references:
            rgb = ref_image[:, :, :, :3]
            if ref_image.shape[-1] > 3:
                rgb = rgb * ref_image[:, :, :, 3:] + (1.0 - ref_image[:, :, :, 3:])
            images_vl.append(rgb)
            if reference_tile_size > 0:
                ref_latents.append(vae.encode_tiled(
                    ref_image, tile_x=reference_tile_size, tile_y=reference_tile_size,
                    overlap=reference_tile_overlap, tile_t=reference_temporal_tile_size,
                    overlap_t=reference_temporal_tile_overlap,
                ))
            else:
                ref_latents.append(vae.encode(ref_image))

        keep_vision = not ref_latents
        # Route Swarm-specific syntax and step-scheduled LoRA hooks through the
        # same parser as ordinary Swarm conditioning, preserving Qwen's image
        # slots and keep_vision behavior in each scheduled token sequence.
        advanced_markers = ("<weight", "<embed", "<embedding", "<break", "<fromto", "<alternate",
                            "<alt:", "<alt[", "<lora:", "(")
        use_advanced = lora_hooks is not None or any(
            marker in text for text in (prompt or "", negative_prompt or "")
            for marker in advanced_markers
        )
        if use_advanced:
            from .SwarmText import SwarmTextEncodeAdvanced
            tokenizer_flags = {"keep_vision": keep_vision, "prevent_empty_text": True}
            advanced = SwarmTextEncodeAdvanced()
            def encode_prompt(text):
                return advanced.encode(
                    clip, steps, text or "", width, height, width, height,
                    images=images_vl, lora_hooks=lora_hooks,
                    tokenize_kwargs=tokenizer_flags,
                )[0]
            positive = encode_prompt(prompt)
            negative = encode_prompt(negative_prompt)
        else:
            positive = clip.encode_from_tokens_scheduled(
                clip.tokenize(prompt or "", images=images_vl, keep_vision=keep_vision, prevent_empty_text=True)
            )
            negative = clip.encode_from_tokens_scheduled(
                clip.tokenize(negative_prompt or "", images=images_vl, keep_vision=keep_vision, prevent_empty_text=True)
            )

        if ref_latents:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        # Keep the native-compatible third output for workflow/debug parity. Segment Reference Edit
        # intentionally does not sample from it; it uses a VAE encoding of the crop as its edit anchor.
        latent = {
            "samples": torch.zeros(
                [1, 64, height // 16, width // 16],
                device=comfy.model_management.intermediate_device(),
            )
        }

        print(
            f"[QWEN21-SEG] encoder crop={tuple(resized.shape)} "
            f"reference_latents={len(ref_latents)} "
            f"original_reference={use_original_reference} prompt={repr((prompt or '')[:160])} "
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
