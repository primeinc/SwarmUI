# Qwen Image 2.1 segmented editing

## Actual native paths

SwarmUI retains its existing `<segment:...>` prompt parser and CLIPSeg/YOLO
detection. For Qwen 2.1 text conditioning it routes the author's Swarm tags
through `SwarmTextEncodeAdvanced` and ComfyUI's unchanged `ReferenceLatent`
node, preserving the upstream Qwen tokenizer and using Luis Quesada's Inpaint
Crop & Stitch with optional LanPaint nodes. No separate Qwen tokenizer,
inpaint sampler, crop math, or stitcher is reimplemented.

For Qwen edits with an explicitly supplied Init Image or Prompt Image,
the crop and detector must receive that original photograph, regardless of
whether the prompt also contains global scene text. Using a generated base
frame here silently destroys the subject before segmentation. An instruction-only
segment without a source image is an error. Ordinary text-to-image prompts with
global scene content and no supplied image may still segment their generated
frame.

For remote Comfy backends without the Swarm base64 loader, ordinary `Init Image`
file uploads are rescaled to the generation canvas by the legacy fallback.
That is incorrect when native Qwen segmentation edits the original photo.
The generated native crop graph now disables only that Init Image rescaling,
so its file upload retains the supplied source dimensions and pixels.
Ordinary img2img and raw custom workflow upload semantics are unchanged.
Independent Prompt Images were already non-resizing inputs.

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
                         SwarmTextEncodeAdvanced(images=crop)
                           | positive and negative text conditioning
                         VAEEncode(crop) -> ReferenceLatent (both paths)
                                              |
Reference Edit [advanced option]: LatentMultiply(VAE crop, 0)
                                    -> KSampler(denoise=1)
                                    -> LanPaint_ImageDecode(image=crop, no mask)
                                    -> SplitImageWithAlpha
                                    -> InpaintStitchImproved(stitcher)
                                    -> final original-size image

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

The segment's text is authored by the user and is **not rewritten**.
The Swarm request parser removes the `<segment:...>` directive; the segment's
remaining authored text is then supplied to `SwarmTextEncodeAdvanced`
with its Swarm weight, scheduling, embedding, and LoRA-hook markup intact.
That existing Swarm node interprets the tags when it builds the positive
and negative conditioning. It must not invent a replacement sentence,
classify verbs, add a target description, or insert an image token.
The crop is provided through the encoder's `images` input, with its
matching `VAEEncode` reference latent attached via the upstream
`ReferenceLatent` nodes on both conditioning paths.

ComfyUI's Qwen 2.1 tokenizer inserts image references into its template
when image tensors are supplied. Its inherited Qwen3VL tokenizer disables
ordinary textual token weighting internally. Swarm therefore processes
its own attention weights and scheduled steps *before* calling that tokenizer
rather than passing literal `<weight...>` strings to the old native Qwen
`TextEncodeQwenImage21` node. Passing through `SwarmTextEncodeAdvanced`
is the branch's graph-level implementation of full Swarm segment-tag support;
this graph construction is not proof of image-space LoRA effectiveness.
Values at or below 0.5 in SwarmClipSeg's **normalized output mask** fail
before upstream CropAndStitch can fall back to the entire image. Accepted
fractional confidence values are converted to a solid subject mask using
ComfyUI's native ThresholdMask before the unchanged crop adapter. Its own
feathering still handles the mask boundary; the subject interior no longer
blends old and replacement identities.

The Reference Edit branch now constructs its zero-filled native-format latent
from `VAEEncode(cropped source) -> LatentMultiply(multiplier=0)`.
The standard `KSampler` uses `denoise=1`, a full noise schedule. The segment tag's creativity
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
  `comfy/text_encoders/qwen_image21.py::QwenImage21Tokenizer.tokenize_with_weights`,
  `comfy_extras/nodes_edit_model.py::ReferenceLatent.execute`, and native
  `KSampler`. The original `TextEncodeQwenImage21.execute` is an
  upstream *comparison* contract; this Swarm segment graph no longer creates
  that node. The upstream Qwen module remains separately bundled at
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

The selected Comfy backend must register every node used in its generated
graph, including `SwarmTextEncodeAdvanced`, `VAEEncode`,
`ReferenceLatent`, `GetImageSize`, `InpaintCropImproved`,
`InpaintStitchImproved`, and `LanPaint_ImageDecode`.
Masked Img2Img also needs `LanPaint_ImageEncode` and
`LanPaint_KSampler`; Reference Edit uses `LatentMultiply` and
`KSampler`. Registration is checked against that backend's actual
`/object_info`, not inferred from packages installed on another backend.

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
graphs whose crop source contains an upstream base sampler, validates the
current Swarm-conditioning/ReferenceLatent and masked-sampler paths, queues a real Comfy backend,
and captures before/crop/mask/after images for semantic review.
`tests/live/test_qwen_segment_lora.py` checks an actual LoRA node is on the
sampler model path and compares OFF/ON/OFF model-backed output from real global
and segment workflows.
The NUnit production-generator checks exercise original Init Image, Prompt Image-only
source, independent reference image slots, exact native base64 bytes, separate
batch detector/crop/sampler paths with different seeds, OFF/ON LoRA loader
placement, and the remote Init Image upload-resizing decision. These are
source and graph contracts, not Qwen inference. Upstream CropAndStitch also
high-pass-filters its optional context mask: weak confidence below the cutoff
cannot enlarge the crop, but strong false detections can, so real mask inspection
remains part of acceptance.

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
