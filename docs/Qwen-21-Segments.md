# Qwen Image 2.1 segmentation

Swarm's Qwen Image 2.1 `<segment:...>` path uses **the same
`SwarmTextEncodeAdvanced` CLIP encoder and `ReferenceLatent` inputs as
ordinary Qwen edit sampling**. There is no separate
`SwarmQwenImage21SegmentEncode` node.

The image to edit at this stage is the detected region of the base/refined
image. Swarm crops and resizes that region, supplies the **crop** to both the
vision-token conditioner and VAE reference-latent conditioner, samples the edit,
and composites the result back over the selected soft segmentation mask.

## Source image and Prompt Images

With **Segment Qwen 2.1 Original Reference** enabled (default), the segment
crop replaces the first Prompt Image as the edit source. The **first full-frame
Prompt Image is NOT added a second time**; adding it alongside the crop can
overconstrain subject replacement and create conflicting image references.
Additional Prompt Images (indices 1 onward) are still provided unless
Text Encoded Image is `none`. If Original Reference is disabled, explicitly
supplied Prompt Images, including index 0, retain their usual meanings.

Swarm's regular Qwen conditioner also supports `<weight>`, `<fromto>`,
`<alternate>`, `<embed>`, and scheduled `<lora:...//hook=...>` syntax.
Static segment-confined LoRAs are applied on the model path during the same
model-loader stage as global adapters. ComfyUI's existing `ReferenceLatent`
nodes attach VAE references to positive and negative conditioning.

The regular Qwen conditioner batches multiple image references just as the
normal Qwen path does, which can rescale extra images to match the first
reference. Independent-aspect-ratio reference inputs are NOT yet certified for
this path. The overall reference count is limited to 16.

## Segment sampling and effective steps

The segment's **Steps** value means executed denoising passes, rather than the
total schedule length before skipping. ComfyUI `KSamplerAdvanced` computes
`steps - start_at_step` passes. For a Qwen segment with `Steps=4` and
creativity `0.8`, Swarm requests a five-step schedule and starts at step 1:
**four model evaluations, not three**. This mirrors the relationship between
the total sigma schedule and denoise ratio in the upstream `KSampler` path.

- **Reference Edit** (default): encodes the cropped source to a plain VAE
  latent; samples with partial-denoise schedule; blends the result back over
  the detected segment mask
- **Masked Img2Img**: uses the existing cropped noise-masked latent and applies
  DifferentialDiffusion when configured
- **Zero segment creativity**: skip Qwen segment sampling, but preserve a
  requested Save Segment Mask output

Qwen segment compositing retains the antialiased/blurred segmentation mask
instead of applying the legacy `ThresholdMask(value=0.001)` to it.
Other model families, including Krea, keep their previous composite behavior.

Batch size greater than one is handled per input image, with a distinct
detected crop, sampler, and composite before the output is rebatched.

A segment after a refiner uses the refiner-stage model chain when rebuilding
the confined-LoRA variant. A segment before the refiner uses the base-stage
chain. Qwen overrides do not replace the original global LoRAs.

## Verification, scope, and explicit limits

`SwarmUITests/QwenSegmentPipelineTests.cs` exercises the actual C# workflow
builder with synthetic model paths. It checks prompt section parsing, scoped
LoRA node attachment, positive/negative reference lineage, crop-substituted
vision input, 4 effective steps from the user's failing prompt, batch isolation,
refiner stage, and no hard-thresholded Qwen composite mask.

These are **graph and C# checks, not rendered acceptance**.
A green build or node-connection test does not prove that the model has
replaced the subject in the generated pixels.

The user-reported control and failure are:

- Normal Qwen edit: `edit image 1 replace the girl with a black boy with blue eyes`
  visibly succeeds
- Segmented Qwen edit: `<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes`
  visibly fails in 10/10 attempts on the previous code
- The old segmented log records `steps=4, start=1` and **three effective
  sampler iterations**, not four

**Runtime proof remains required**: use the same actual model and LoRA with a
matched no-segment, segmented, no-LoRA, and wrong-section negative control.
Check both the generated crop and the final composite. The changes here do
not claim a GPU pass or semantic replacement success without that result.

## References

- Swarm source: `src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs`
  (`CreateConditioningDirect`, `BuildInputImageHandling`, `RecompositeCropped`)
  and `WorkflowGeneratorSteps.cs#RunSegmentationProcessing`
- ComfyUI `e3bee7ffcb7d6547ed0bcd15221792f3b8646b17`:
  `nodes.py#KSamplerAdvanced.sample`,
  `nodes.py#KSampler.sample`, and
  `comfy_extras/nodes_qwen.py#TextEncodeQwenImage21.execute`
