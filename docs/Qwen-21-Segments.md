# Qwen Image 2.1 segments: upstream node contract

Qwen segment editing in SwarmUI now depends on the **native ComfyUI
`TextEncodeQwenImage21` and `KSampler` nodes**, not an independent Python
implementation of Qwen encoding, latents, or denoise scheduling.

SwarmUI ships the **unaltered** `Comfy-Org/ComfyUI` source file
`comfy_extras/nodes_qwen.py` at revision
`e3bee7ffcb7d6547ed0bcd15221792f3b8646b17`, as
`src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/ComfyQwenNodes_Upstream.py`.
Its original GPL-3.0 license is included as `ComfyQwenNodes_Upstream.LICENSE`.
The bundled Comfy custom-node path is supplied in
`ComfyUISelfStartBackend.cs`. ComfyUI does not override an already-loaded
built-in node with the same node ID.

**Important:** This is source/deployment wiring verification, not a claim
that a particular running Comfy process has loaded that file. The current
runtime must list `TextEncodeQwenImage21` in `/object_info`.

## Existing node composition

The Qwen-only path consists of:

1. The existing Swarm `SwarmClipSeg` detector selects the named region. This
   is retained because the segment syntax depends on text-selected masks,
   and replacing it with SAM3 would add an unrequested detector dependency.
2. Existing `SwarmMaskBounds` determines tight crop bounds on the **raw
   detection before blur/grow**, using native `ThresholdMask(0.5)`. Bounds
   are not determined from barely-nonzero blurred-mask pixels.
3. Native `ImageCrop` extracts the image at those bounds and native
   `ImageScaleToTotalPixels` scales it to the configured megapixel count,
   preserving its aspect ratio. Only existing Swarm `SwarmMaskBounds`
   remains to calculate dynamic coordinates for both the image and mask.
4. Native `TextEncodeQwenImage21` receives the selected crop as
   `images.image_1`; it performs all Qwen vision-token preprocessing, image
   reference VAE encoding, positive/negative conditioning, and outputs its
   native empty edit latent.
5. Native `KSampler` receives the segment-scoped, LoRA-patched model and
   the native encoder's latent; the sampler itself owns denoising.
   **Steps=4 means 4 sampling evaluations**; creativity 0.8 becomes
   `denoise=0.8`, not `start_at_step=1`.
6. Native `ImageScale`, `ThresholdMask`, and `ImageCompositeMasked`
   resize and apply the result at the original crop coordinates.
   Where requested, mask blur uses native `MaskToImage` + `ImageBlur` +
   `ImageToMask` instead of a Qwen-specific Python mask transform.
   Native `PreviewImage` nodes added by the live integration test inspect
   the crop before and after, the mask fed to the compositor, and the final
   full image. The regular `SaveOptionalIntermediate` path saves the
   actual segment result before compositing.

Swarm-specific C# **glue** remains for reading the `<segment:...>` region,
choosing masks/bounds and the LoRA-scoped model stage, and retaining the
pre-existing `CreateImageMaskCrop` / `RecompositeCropped` operations.
No standalone Qwen encoder, scheduler, or Qwen diagnostic Python nodes
remain registered.

## Reference images and supported options

For the user's one-image InitEdit, the detected crop **replaces** the whole
source photo, `Prompt Images[0]`. Do not include both the crop and original
full-frame photo as image references. Additional `Prompt Images[1..]` remain
separate native reference slots, limited to 16 inputs in total.

`Segment Qwen 2.1 Original Reference` disables or enables automatic
inclusion of the selected crop. In Reference Edit mode the sampler uses
ComfyUI's native empty latent; Masked Img2Img continues to use the masked
crop latent and the existing DifferentialDiffusion option.

The native Qwen node does **not** implement Swarm's dynamic prompt LoRA
hooks or tiled reference-image VAE encoding. These options currently produce
an explicit error rather than pretending to work. Static global/segment
LoRAs are patched on the model-loader path, independently from conditioning.
This is a documented capability gap, not certified equivalent behavior.

For non-Qwen models, the original sampler and compositing paths are unchanged.

## Verification and acceptance

The exact user-reported acceptance case is:

```
<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes
```

Ordinary editing worked immediately; old segmented editing failed 10/10
times and returned ghosted/incorrect results. The live graph inspection in
`tests/integration/qwen_segment_text_replacement.py` now rejects an
obsolete `SwarmQwenImage21SegmentEncode`, requires the native encoder,
native four-step sampler, crop input and final native compositor, and
captures real before/crop/mask/after image outputs.

`tests/live/test_qwen_segment_lora.py` performs real on/off/on-adjacent
LoRA negative controls against a running Comfy backend, using the actual
model and user-provided exported workflows. It is not executed by ordinary
CPU CI. Source checks and build checks do not prove subject replacement;
real GPU output from the current branch must show the requested boy, the
correct source crop/preview and no unwanted background replacement.

## Inspected references

- ComfyUI@e3bee7ffcb7d6547ed0bcd15221792f3b8646b17:
  `comfy_extras/nodes_qwen.py#TextEncodeQwenImage21.execute`
  (three native outputs, same resized reference for CLIP and VAE);
  `nodes.py#KSampler.sample` (denoise/sigma-step handling);
  `comfy_extras/nodes_images.py#ImageCrop`;
  `comfy_extras/nodes_mask.py#ImageCompositeMasked,ThresholdMask`.
- SwarmUI current branch:
  `WorkflowGeneratorQwenSegment.cs#CreateQwen21SegmentConditioning`;
  `WorkflowGeneratorSteps.cs#RunSegmentationProcessing`;
  `WorkflowGenerator.cs#CreateImageMaskCrop,RecompositeCropped`.
