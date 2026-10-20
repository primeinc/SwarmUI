# Qwen Image 2.1 segments

Qwen Image 2.1 `<segment:...>` passes use ComfyUI's native
`TextEncodeQwenImage21` node, not a Swarm-specific replacement. The cropped
original (when enabled) is passed as `image_1`, and optional Prompt Images
occupy subsequent native image slots. `resolution=0` preserves the crop's
reference size, rounded by ComfyUI to a multiple of 32. The native encoder
owns Qwen vision tokens, VAE references, and the empty latent.

Static section-confined LoRAs continue to load through Swarm's existing
model/CLIP loader pipeline. Native Qwen's encoder has no dynamic hook or tiled
reference-VAE input: those unsupported settings fail explicitly rather than
being silently ignored. Advanced prompt syntax requires separate upstream
support, not a locally invented substitute.

Crop bounds use ComfyUI's native `ThresholdMask` at 0.5, while the original
mask remains available to the compositor. This prevents weak CLIPSeg background
confidence from expanding the crop without forcing that same mask onto the
final edit boundary.

## Sampling modes and controls

`Segment Param Overrides` > `Segment Qwen 2.1 Sampling`:

- **Reference Edit** (default): samples the **native Qwen empty latent** returned
  by `TextEncodeQwenImage21` (output 2), with the cropped original supplied
  separately as vision/reference conditioning. This matches the working
  non-segmented Qwen image-reference editing contract: the input girl's pixels
  are **not** re-encoded as the starting latent to be reconstructed. The configured
  `Steps` count is the number of actual denoising iterations. For `0 < Segment Creativity < 1`, it follows ComfyUI's
  `KSampler.set_steps(steps, denoise)` convention: build
  `scheduled_steps = int(steps / creativity)` and begin at
  `start_at_step = scheduled_steps - steps`. At four steps and 0.8 creativity,
  ComfyUI executes **four** iterations from a five-step schedule, not three from
  a four-step schedule. The final pixel composite confines the result with an opaque subject
  mask and feathered boundary, avoiding both faint-source double exposure
  and opaque rectangular paste.
- **Masked Img2Img**: samples the masked crop latent and retains the older
  `start_at_step = round(steps * (1 - Segment Creativity))` partial-iteration
  behavior, with DifferentialDiffusion when configured.

Zero creativity skips Qwen segment sampling. If `Save Segment Mask` is enabled,
the mask is still generated and saved.

`Segment Qwen 2.1 Original Reference` controls whether the cropped source
image is also supplied as vision/reference conditioning. Disabling it does **not**
change Reference Edit's fresh output latent or Masked Img2Img's masked crop
latent, and does not disable other optional Prompt Images. The default is enabled.

Optional `Prompt Images` are retained as individual Qwen references when Text
Encoded Image is not `none`; each reference retains its aspect ratio and its
own vision slot. Qwen supports at most 16 reference images in this workflow:
15 additional images when the original crop is a reference, or 16 when it is
not. Native `TextEncodeQwenImage21` does not expose tiled reference encoding.

Batch size greater than one is handled by splitting the source into individual
images **before** segment detection and conditioning. Each image gets an
independent crop, sampler, and composite, and the images are rebatched afterward.

When a segment is placed after an actual refiner, a confined-LoRA model variant
is rebuilt using the stage that produced the source model. This preserves
refiner-stage LoRA selection and stage-specific model wrappers. Segments before
a refiner retain the base-stage model path.

Other model families, including Krea, retain their existing segment workflow.

## Verification and remaining acceptance

`QwenReferenceEditStepScheduleTests` verifies production step-plan arithmetic,
not image fidelity. The former reduced-size synthetic Qwen/LoRA test file and
its CI workflow were removed; they must **not** be reported as proof.
`tests/integration/qwen_segment_text_replacement.py` instead instruments
a real Swarm-exported Comfy workflow for the exact failed text-only prompt:
it saves the actual CLIPSeg mask, source crop, generated crop, and composite.
Only a live backend and semantic inspection can determine whether the subject
was actually replaced. Its normal syntax check in CI is not a runtime pass.

`tests/live/test_qwen_segment_lora.py` requires real exported workflows,
installed LoRA files, and a live ComfyUI backend. Its on/off controls compare
actual saved images from full inference. It is deliberately not a green CPU-CI
gate without that environment and the specified manifest.

The read-only `qwen_segment_capture.py` tool can report encoder inputs,
model-loader lineage, reference nodes, sampler settings, and compositor.
Pixel-delta diagnostics distinguish no pixel change from a semantic edit failure
but are not a semantic-identity assertion. Never promote a green build, graph
check, or pixel-delta count to full LoRA acceptance.

## Source contracts reviewed

- `primeinc/SwarmUI`: `WorkflowGeneratorSteps.cs#RunSegmentationProcessing`,
  `WorkflowGeneratorQwenSegment.cs#CreateQwen21SegmentConditioning`,
  `SwarmQwen.py#SwarmQwen21SegmentDiagnostics`, and
  `SwarmText.py#SwarmTextEncodeAdvanced`.
- `Comfy-Org/ComfyUI@e3bee7ffcb7d6547ed0bcd15221792f3b8646b17`:
  `comfy_extras/nodes_qwen.py#TextEncodeQwenImage21`,
  `comfy/text_encoders/qwen_image21.py#QwenImage21Tokenizer`, and
  `nodes.py#VAEEncodeTiled`.
