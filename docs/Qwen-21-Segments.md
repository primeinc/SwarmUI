# Native Qwen Image 2.1 segments

Qwen Image 2.1 `<segment:...>` passes now use `TextEncodeQwenImage21`, with the actual scaled crop on `images.image_1`, the selected segment VAE, and the segment's positive and negative prompts. Both conditioning outputs feed the segment sampler. Automatic reference injection is disabled for this call so references are not duplicated.

The existing section-confined LoRA loader still supplies the sampler's model input. There is no new LoRA injector. Krea and other model families retain their prior conditioning and sampling paths. The preview implementation is unchanged.

## Sampling modes

`Segment Param Overrides` > `Segment Qwen 2.1 Sampling`:

- **Reference Edit** (default): use the native encoder's correctly sized empty latent and the full configured segment step schedule. The existing pixel-space mask restricts where the regenerated crop is pasted. Creativity zero skips the edit; other creativity values do not shorten sampling in this mode.
- **Masked Img2Img**: use the same native paired conditioning, but retain the encoded crop, its latent noise mask, and the existing partial-denoise creativity calculation.

The earlier experimental `Segment Qwen 2.1 Original Reference` switch does not remove the crop from the native encoder. It only affects the legacy automatic-reference helper, which this segment path bypasses.

The crop is reference image 1. Optional global Prompt Images follow as references 2 through 16 and remain separate inputs. The crop is already scaled to multiples of 64; `resolution=0` preserves that size through the native encoder. Negative conditioning receives the same references. The native encoder's actual serialized input name is `images.image_1`, not `image_1`.

This path requires a Comfy backend that advertises `TextEncodeQwenImage21`. Missing support produces an explicit error instead of silently falling back to text-only conditioning. Plain segment instructions and static `<lora:...>` tags are supported. Advanced weighted/step-scheduled Swarm prompt tags produce an explicit unsupported-feature error, not an ineffective literal prompt.

## Verification scope

`QwenSegmentPipelineTests` invokes the registered segment generation step for base-stage and post-refiner placement. It checks the crop/encoder/sampler/compositor connections, scoped LoRA weight, text-only segment prompts, multiple independent segments, an explicitly selected same-model override, zero creativity, masked mode, missing backend support, and the unchanged Krea path.

Graph construction and compilation do not certify GPU output. User-visible acceptance remains: identical source crop, seed and settings; a text-only edit visibly follows its instruction; LoRA-on vs LoRA-off produces the adapter's expected effect; compositing keeps the final change within the actual segment mask.

## Inspected source contracts

- `Comfy-Org/ComfyUI@e3bee7ffcb7d6547ed0bcd15221792f3b8646b17`, `comfy_extras/nodes_qwen.py#TextEncodeQwenImage21`: paired conditioning, reference processing, resolution and empty latent outputs. Adapted in `WorkflowGeneratorQwenSegment.cs`.
- `jtreminio/SwarmUI-Base2Edit@77a8c2d3192b0dafeb1d5314870f1aa83e291e09`, `src/StageRunner.cs#CreateQwenImage21Conditioning,#ExecuteSampler` and `Tests/QwenImage21Tests.cs`: compared for native paired conditioning, grouped input serialization and avoiding duplicate references. This is not a claim that its GPU output was tested.
- `primeinc/SwarmUI@86b0189e3215b2567715d924802c1aa6f46f8d65`, `WorkflowGeneratorSteps.cs#RunSegmentationProcessing`, `WorkflowGenerator.cs#CreateKSampler,#LoadLorasForConfinement`: existing loop and sampler connections, adapted without changing the preview branch's source.
