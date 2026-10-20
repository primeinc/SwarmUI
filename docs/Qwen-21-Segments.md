# Qwen Image 2.1 segmented editing

## Actual native paths

SwarmUI retains its existing `<segment:...>` prompt parser and CLIPSeg/YOLO
detection. It uses **unmodified** upstream ComfyUI Qwen 2.1 conditioning and
Luis Quesada's Inpaint Crop & Stitch, with optional upstream LanPaint nodes.
No custom Qwen tokenizer, VAE encoder, sampler, crop math, or image stitcher.

For Qwen edits with an explicitly supplied Init Image or Prompt Image,
the crop and detector must receive that original photograph, regardless of
whether the prompt also contains global scene text. Using a generated base
frame here silently destroys the subject before segmentation. An instruction-only
segment without a source image is an error. Ordinary text-to-image prompts with
global scene content and no supplied image may still segment their generated
frame.

The two selectable segment sampling modes are intentionally different:

```
Original source -> SwarmClipSeg(text) -> normalized detector mask
        |                                      |
        |                 SwarmRequireNonEmptyMask (>0.5 support)
        |                                      |
        |                      native ThresholdMask (value=0.5)
        |                                      |
        +--------> InpaintCropImproved(image, binary subject mask)
                               |0 stitcher   |1 cropped image   |2 cropped mask
                                                |
                         TextEncodeQwenImage21.images.image_1
                              positive / negative / output-2 empty edit latent
                                              |
Reference Edit [advanced option]: KSampler(model, Qwen output-2 latent, denoise=1)
                                              |
                             LanPaint_ImageDecode(image=crop, no mask)
                                              |
                           SplitImageWithAlpha -> InpaintStitchImproved(stitcher)
                                              |
                                        final original-size image

Masked Img2Img [default]: same crop -> LanPaint_ImageEncode(image=crop, mask=crop mask)
                                    -> LanPaint_KSampler(denoise=requested strength)
                                    -> LanPaint_ImageDecode -> SplitImageWithAlpha
                                    -> InpaintStitchImproved(stitcher)
```

Previously, Swarm defaulted to Reference Edit, which fed a zero-filled
latent to an unmasked KSampler and used the segment mask only after generation.
That is **not masked inpainting** and can fail to replace the selected subject
even if the selected LoRA model is correctly attached. Qwen segments now
default to the existing upstream masked path, which sends the actual crop
mask into LanPaint_ImageEncode and its KSampler. Reference Edit remains an
explicit advanced option for users who specifically want unmasked crop edits.

The segment's text is authored by the user and is **never rewritten**.
The existing Swarm prompt parser removes the `<segment:...>` directive and
resolves Swarm tags. The Qwen adapter flattens residual Swarm weight markup
only and passes the resulting text directly to native
`TextEncodeQwenImage21`. It must not turn LoRA triggers into replacement
sentences, classify edit verbs, add a target description, or insert
`<image1>` unless the user wrote that token. Image references are supplied
through the encoder's image inputs, not inferred from text. A bare LoRA
trigger remains a bare LoRA trigger; the user determines any edit instruction.
Values at or below 0.5 in SwarmClipSeg's **normalized output mask** fail
before upstream CropAndStitch can fall back to the entire image. Accepted
fractional confidence values are converted to a solid subject mask using
ComfyUI's native ThresholdMask before the unchanged crop adapter. Its own
feathering still handles the mask boundary; the subject interior no longer
blends old and replacement identities.

The Qwen native edit encoder returns a **zero-filled** output-2 latent, not
the original image encoded as an img2img latent. Therefore Reference Edit
uses `denoise=1`, a full native noise schedule. The segment tag's creativity
value remains the `denoise` control of **Masked Img2Img**, not of the
empty-latent Reference Edit sampler. InpaintStitchImproved owns the
only pixel-space blend into the source image. Masked Img2Img additionally
feeds the crop mask to LanPaint_ImageEncode. LanPaint_ImageDecode
only decodes and size-matches; giving it the mask as well causes a
second blend and ghosts the original pixels around the edited subject.
Switching modes is explicit.

Comfy-Org's Qwen 2.1 documentation specifies `cfg=1`, `euler`, `simple`
and 25 steps for the base edit workflow. CFG 5 is explicitly described as
degrading output badly. Swarm's generic CFG default is 7, so segmented Qwen
now defaults to **1.0**, independently of generic/SD model defaults.
Explicit section CFG or Segment CFG overrides remain effective.
The reproduction uses four sampler steps to honor its requested workflow
and is **not** a blanket recommendation for all subject replacements.
If the chosen model is a distilled Turbo/Lightning variant, follow that
model's scheduler and step requirements instead of extrapolating them.

## Exact bundled upstream provenance

- `Comfy-Org/ComfyUI@e3bee7ffcb7d6547ed0bcd15221792f3b8646b17`:
  `comfy_extras/nodes_qwen.py::TextEncodeQwenImage21.execute` and native
  `KSampler`. Qwen module is bundled unchanged in
  `ExtraNodes/ComfyQwenNodes_Upstream.py`.
- `lquesada/ComfyUI-Inpaint-CropAndStitch@348a8bbaad1a9000523e42dd3eaf6ab34a8fe62f`:
  `inpaint_cropandstitch.py::InpaintCropImproved.inpaint_crop` and
  `InpaintStitchImproved.inpaint_stitch`, bundled separately unchanged.
- `scraed/LanPaint@2d7912f9a5efe5ece8de334c7ca18317b8288c39`:
  `src/LanPaint/nodes.py::LanPaint_ImageEncode.encode`,
  `LanPaint_KSampler.sample`, `LanPaint_ImageDecode.decode`, bundled
  separately unchanged. The upstream GPL-3.0 license remains included
  with each component and is not superseded by Swarm's license.
- `Comfy-Org/docs@2eff886d14c92c9d8d8dc586d142821d22cd0d6b`:
  `tutorials/image/qwen/qwen-image-2-1.mdx`, "Workflow settings / Sampler
  settings": `cfg=1`, `euler`, `simple`; higher CFG harms image quality.

A Swarm-managed Comfy backend must actually register
`TextEncodeQwenImage21`, `InpaintCropImproved`,
`InpaintStitchImproved`, and `LanPaint_ImageDecode` from `/object_info`.
Masked Img2Img additionally requires `LanPaint_ImageEncode` and
`LanPaint_KSampler`; Reference Edit uses stock `KSampler`.
An external Comfy instance may require the same upstream nodes installed.

## Diagnostics and acceptance

Log lines beginning `[QWEN21-SEG]` identify the actual source, detected
mask, crop, conditioning and model nodes, sampler mode, requested creativity,
actual sampler denoise, CFG, and stitch graph. Optional intermediate images
are saved **only when the user enables Output Intermediate Images**.
The crop output-2 mask is not equivalent to the opaque stitcher's internal
post-processed blending mask.

Reported reproduction:

```text
<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes
```

`tests/integration/qwen_segment_text_replacement.py` rejects real exported
graphs whose crop source contains an upstream base sampler, validates native
Qwen Reference Edit sampler links and parameters, queues a real Comfy backend,
and captures before/crop/mask/after images for semantic review.
`tests/live/test_qwen_segment_lora.py` checks an actual LoRA node is on the
sampler model path and compares OFF/ON/OFF model-backed output from real global
and segment workflows.

**Testability does not require a GPU as a blanket rule.** C# graph generation,
prompt/LoRA-path invariants, native node contracts, exported-graph preflight,
the actual PyTorch CPU mask guard, and unchanged upstream CPU crop/stitch
operations are executable without Qwen weights or GPU hardware. The pixel tests
include failing controls for old-subject ghosting and empty masks. These do
not establish what Qwen *generates*: semantic replacement of the target,
actual detector recognition, and image-space LoRA effectiveness require running
the corresponding model weights and inspecting inference results. A capable CPU
backend is theoretically acceptable; a GPU is a performance/resource choice,
not a logical prerequisite. Ordinary CI has not run model-backed Qwen inference.

**Status: semantic subject replacement and LoRA effect remain UNVERIFIED**
until the real model-backed workflows succeed with the user's source image and
LoRAs. A valid graph and passing CI do not prove image correctness.
