# Qwen Image 2.1 segmented editing

## Actual native paths

SwarmUI retains its existing `<segment:...>` prompt parser and CLIPSeg/YOLO
detection. It uses **unmodified** upstream ComfyUI Qwen 2.1 conditioning and
Luis Quesada's Inpaint Crop & Stitch, with optional upstream LanPaint nodes.
No custom Qwen tokenizer, VAE encoder, sampler, crop math, or image stitcher.

For a **segment-only** request (no global/base/refiner prompt), the crop and
detector must receive the unmodified Init Image or the first Prompt Image,
**not** the blank-prompt base sampler's generated image. A missing source
image is an error. Ordinary prompts with non-empty global content retain
generated-frame segment processing.

The two selectable segment sampling modes are intentionally different:

```
Original source -> SwarmClipSeg(text) -> detected mask
        |                                 |
        +--------> InpaintCropImproved(image, detected mask)
                               |0 stitcher   |1 cropped image   |2 cropped mask
                                                |
                         TextEncodeQwenImage21.images.image_1
                              positive / negative / output-2 empty edit latent
                                              |
Reference Edit [default]: KSampler(model, Qwen output-2 latent, denoise=1)
                                              |
                             LanPaint_ImageDecode(image=crop, mask=cropped mask)
                                              |
                           SplitImageWithAlpha -> InpaintStitchImproved(stitcher)
                                              |
                                        final original-size image

Masked Img2Img: same crop -> LanPaint_ImageEncode(image=crop, mask=crop mask)
                                    -> LanPaint_KSampler(denoise=requested strength)
                                    -> LanPaint_ImageDecode -> SplitImageWithAlpha
                                    -> InpaintStitchImproved(stitcher)
```

The Qwen native edit encoder returns a **zero-filled** output-2 latent, not
the original image encoded as an img2img latent. Therefore Reference Edit
uses `denoise=1`, a full native noise schedule. The segment tag's creativity
value remains the `denoise` control of **Masked Img2Img**, not of the
empty-latent Reference Edit sampler. Both use the native crop's mask for
blending and stitching. Switching modes is explicit.

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
sampler model path and compares OFF/ON/OFF GPU output from real global and
segment workflows.

**Status: semantic subject replacement and LoRA effect remain UNVERIFIED**
until these real GPU workflows succeed with the user's model/image/LoRAs.
A valid graph and passing CI do not prove image correctness.
