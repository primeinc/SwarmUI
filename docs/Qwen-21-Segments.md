# Qwen Image 2.1 segments

Qwen Image 2.1 `<segment:...>` passes use Swarm's bundled `SwarmTextEncodeAdvanced` multimodal encoder with the actual scaled segment crop supplied as the first vision image. Positive and negative segment text are encoded separately, and the same crop is attached to both conditionings through `ReferenceLatent` using the selected segment VAE.

This deliberately does **not** require upstream ComfyUI's newer `TextEncodeQwenImage21` node. A Swarm backend that can already run Qwen Image 2.1 can therefore run the segment path without a separate Comfy update solely for that node.

The existing section-confined LoRA loader still supplies the segment sampler's model input. There is no second LoRA injector. Krea and other model families retain their prior paths. The preview implementation is unchanged.

## Sampling modes

`Segment Param Overrides` > `Segment Qwen 2.1 Sampling`:

- **Reference Edit** (default): the segment uses the masked crop latent, Qwen vision/reference conditioning from that crop, and the full configured segment step schedule. Creativity zero skips the edit; other creativity values do not shorten sampling in this mode.
- **Masked Img2Img**: uses the same crop-aware conditioning while preserving the existing partial-denoise Segment Creativity calculation.

Optional global Prompt Images are appended after the segment crop for vision conditioning and are also attached as reference latents. The crop itself is always present even when no Prompt Images are configured.

Because the segment path uses `SwarmTextEncodeAdvanced`, it retains Swarm prompt parsing rather than downgrading segment text to a plain-string-only Comfy node.

## Verification scope

`QwenSegmentPipelineTests` invokes the registered segment generation step for base-stage and post-refiner placement. It checks that Qwen segmentation works without the `qwen21_native_conditioning` feature, uses the bundled Swarm encoder, sends the crop into both positive and negative multimodal conditioning, attaches reference latents, preserves scoped LoRA model wiring, supports text-only segment instructions, handles multiple independent segments, preserves zero-creativity behavior and partial-denoise mode, and leaves Krea on its existing path.

Graph construction and compilation do not certify GPU output. User-visible acceptance remains: with the same source crop, seed, and settings, a text-only segment edit visibly follows its instruction; LoRA-on versus LoRA-off produces the adapter's expected effect; and compositing keeps the final change within the actual segment mask.

## Inspected source contracts

- `primeinc/SwarmUI@32c888b6ded07d4afa1e2fd2403e4cace258a9ac`, `WorkflowGenerator.cs#CreateConditioning`: normal Qwen Image 2.1 generation already uses `SwarmTextEncodeAdvanced` for multimodal prompt encoding. Adapted for the segment crop.
- `primeinc/SwarmUI@32c888b6ded07d4afa1e2fd2403e4cace258a9ac`, `WorkflowGenerator.cs#BuildInputImageHandling`: Qwen Image 2.1 reference images are VAE-encoded and applied to both positive and negative conditioning through `ReferenceLatent`. Adapted for the segment crop and optional prompt images.
- `Comfy-Org/ComfyUI@e3bee7ffcb7d6547ed0bcd15221792f3b8646b17`, `comfy_extras/nodes_qwen.py#TextEncodeQwenImage21`: compared for the intended Qwen 2.1 invariant that the same references affect both vision-token text encoding and reference-latent conditioning. The implementation does not depend on this node being installed.
