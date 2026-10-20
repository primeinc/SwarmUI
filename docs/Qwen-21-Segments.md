# Qwen Image 2.1 segments: upstream node composition

## The runnable node stack

SwarmUI's Qwen 2.1 `<segment:...>` Reference Edit path uses existing ComfyUI
nodes and the **unchanged Inpaint Crop & Stitch plugin**:

```
SwarmClipSeg(text, source frame) -> detected MASK
   |
   +--> InpaintCropImproved(image=source frame, mask=detected MASK)
           0: STITCHER ----------------------------> InpaintStitchImproved.stitcher
           1: IMAGE --> TextEncodeQwenImage21.images.image_1
                  |           0: positive -----> LanPaint_KSampler.positive
                  |           1: negative -----> LanPaint_KSampler.negative
                  |
                  +------> LanPaint_ImageEncode.image
           2: MASK ------> LanPaint_ImageEncode.mask
                                      | 0: masked LATENT
                                      +------------> LanPaint_KSampler.latent_image
                                                      (steps=4, denoise=0.8,
                                                       LanPaint_NumSteps=5)
                                                            | 0: sampled LATENT
                                      +---------------------+
                                      v
                       LanPaint_ImageDecode(samples, VAE,
                                              image=crop output 1,
                                              mask=crop output 2)
                                      | 0: image-space crop
                                      v
                              SplitImageWithAlpha
                                      |
                                      +--> InpaintStitchImproved.inpainted_image
                                                        |
                                                    final IMAGE
```

The *same* crop adapter produces the Qwen reference, cropped mask and
opaque stitch metadata. Nothing recomputes crop coordinates for the final
composite. The native Qwen text encoder owns vision processing and reference
conditioning, while LanPaint's unmodified image encoder, masked sampler and
image decoder handle inpainting at the exact crop resolution.

The user-requested outer schedule is `steps=4`, `denoise=0.8`, routed through
`LanPaint_KSampler` and its call to ComfyUI's `nodes.common_ksampler`.
`LanPaint_NumSteps=5` is a **separate inner Langevin update count**, not the
four outer sampler steps and not evidence that just four GPU forwards occur.
Actual inference time and results require the live backend.

### Copied, licensed, and pinned upstream packages

- ComfyUI `Comfy-Org/ComfyUI@e3bee7ffcb7d6547ed0bcd15221792f3b8646b17`,
  `comfy_extras/nodes_qwen.py`, copied **unchanged** into
  `ExtraNodes/ComfyQwenNodes_Upstream.py`. Original GPL-3.0 license alongside.
- Luis Quesada's `lquesada/ComfyUI-Inpaint-CropAndStitch@348a8bbaad1a9000523e42dd3eaf6ab34a8fe62f`,
  version **3.0.18**, copied **unchanged** into
  `ExtraNodes/ComfyInpaintCropAndStitch/`, including Python node
  implementation, registration, JS, license, metadata and original README.
  The crop/stitch plugin is a **separately GPL-3.0-licensed component**;
  SwarmUI's own MIT license is not a relicense of that upstream code.
- `scraed/LanPaint@2d7912f9a5efe5ece8de334c7ca18317b8288c39`,
  `src/LanPaint/nodes.py` and supporting files, copied **unchanged** into
  `ExtraNodes/LanPaintUpstream/`. Original GPL-3.0 license included.

Swarm's managed Comfy backend adds its shipped `ExtraNodes` directory to
ComfyUI's custom-node search path. A real backend must expose
`TextEncodeQwenImage21`, `InpaintCropImproved`,
`InpaintStitchImproved`, `LanPaint_ImageEncode`, `LanPaint_KSampler` and
`LanPaint_ImageDecode` from `/object_info` before inference. The live test
checks all six. No additional ComfyUI-Manager installation is required for a
Swarm-managed backend. An external backend must have equivalent registered
nodes; shipping a file does not alter that external process.

### Swarm-specific glue that remains

Swarm still interprets `<segment:...>`, uses the already-established
`SwarmClipSeg` or selected YOLO detector and routes static LoRAs to the
segment's model. This is the application-specific orchestration. It does NOT
implement its own Qwen tokenizer, patch sampler steps, invent new reference
latents, compute Qwen crop bounds, or paste Qwen pixels back by hand.

The intact crop plugin handles:
- mask high-pass filtering (0.5), hole filling, optional expansion and blending;
- crop context and target-resolution resizing;
- stitch geometry, resampling and blending into the original frame.

Swarm maps `Segment Mask Grow` to the upstream crop node's
`mask_expand_pixels`, `Segment Mask Blur` to `mask_blend_pixels`, and
`Segment Mask Oversize` to an optional ComfyUI `GrowMask` context mask.
The upstream `context_from_mask_extend_factor` stays at 1.2.
`Segment Target Resolution` supplies target crop width/height, otherwise the
model's preferred dimensions are used. The adapter uses its upstream
`cpu (compatible)` mode by default.

For an InitEdit with one Prompt Image, the detected crop replaces the original
whole frame in Qwen's `images.image_1` slot. The original full frame is
**not duplicated** as image 2. Additional explicit Prompt Images are preserved.

Both Qwen sampling modes use the upstream `LanPaint_ImageEncode` result,
including its masked latent. The `Masked Img2Img` option additionally routes
through the established `DifferentialDiffusion` model transform where enabled;
the native Qwen empty latent output is **not** used as the sampler input.
LanPaint's `LanPaint_ImageDecode` blends the decoded image with the original
crop according to the mask before the crop/stitch plugin handles the final
canvas composite. The non-Qwen segmentation pipeline is unchanged.

### Always-on Qwen segment diagnostics

Until runtime acceptance passes, every Qwen segment produces intermediate
captures through SwarmUI's existing output path, without new Comfy Python nodes:

- `00_source_full_frame`
- `01_raw_detector_mask`
- `02_actual_qwen_reference_crop`
- `03_adapter_crop_mask`
- `04_sampler_decoded_crop`
- `05_exact_stitch_input`
- `06_stitched_full_frame`

The backend logs `[QWEN21-SEG] DIAG` with a stage label, output ID and
precise Comfy node ID/output index. The crop plugin's output 2 is the
**cropped mask**, not necessarily the internal final softened blend mask
inside its opaque `STITCHER` state. Do not mislabel it as the final applied
alpha. Captures are real graph outputs, not post-hoc simulated images.

### Evidence and acceptance

The reproduced failure was:

```
<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes
```

Normal non-segmented Qwen editing had succeeded while segmented edits had
failed ten times and generated subject ghosting and birds instead of the
requested replacement. A correct mask or a graph that executes is NOT visual
acceptance.

`tests/test_qwen_native_bundle.py` checks immutable upstream Git blob
digests, registrations and source routing. The build and Python compilation
gates are source-level. `tests/integration/qwen_segment_text_replacement.py`
requires the actual backend's six registered nodes, validates the direct
Crop -> Qwen conditioning + masked LanPaint encode -> LanPaint sampler ->
LanPaint decode -> Stitch references in an exported real workflow,
then queues and captures actual GPU output. `tests/live/test_qwen_segment_lora.py`
exercises real off/on/off LoRA effects through ordinary and segmented samplers.

**The semantic replacement remains UNVERIFIED until real GPU output visibly
replaces the girl with the boy and the background remains intact.** CI passing
does not certify pixel or semantic correctness.

## Exact sources

- `lquesada/ComfyUI-Inpaint-CropAndStitch@348a8bba...`,
  `inpaint_cropandstitch.py::InpaintCropImproved.inpaint_crop` and
  `InpaintStitchImproved.inpaint_stitch`, plus upstream test suite.
- `Comfy-Org/ComfyUI@e3bee7ff...`,
  `comfy_extras/nodes_qwen.py::TextEncodeQwenImage21.execute`,
  `nodes.py::common_ksampler`.
- `scraed/LanPaint@2d7912f9...`,
  `src/LanPaint/nodes.py::LanPaint_ImageEncode.encode`,
  `LanPaint_KSampler.sample`, `LanPaint_ImageDecode.decode`.
- SwarmUI `WorkflowGeneratorSteps.cs::RunSegmentationProcessing`,
  `WorkflowGeneratorQwenSegment.cs::CreateQwen21SegmentConditioning`.
