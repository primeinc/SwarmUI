# Capture a real Qwen segmented-generation failure

The read-only `capture-qwen-segment.bat` tool records actual ComfyUI graphs
and node connections, not a success certificate or a synthetic GPU test.

On Windows with SwarmUI and its backend running, invoke
`capture-qwen-segment.bat` and repeat the observed generation once.
The default source is `http://127.0.0.1:7801/ComfyBackendDirect`;
an explicit direct backend URL is also accepted. The tool reads queue
and history without starting another generation, changing configuration,
installing software, switching Git branches, or contacting an agent.

The archive contains the actual sampler inputs, adapter model chain,
conditioning nodes, mask/composite lineage, and version/queue evidence,
with credentials and embedded images redacted. Review the bundle before
sharing because prompts and model filenames may remain.

## Current source and runtime distinction

The current Qwen segment workflow uses `SwarmTextEncodeAdvanced` for
the complete Swarm prompt-tag language, upstream `VAEEncode` and
`ReferenceLatent` for both conditioning references, and unchanged upstream
`InpaintCropImproved` / `InpaintStitchImproved` adapters. The default **Masked Img2Img** path uses upstream
`LanPaint_ImageEncode`, `LanPaint_KSampler`, and `LanPaint_ImageDecode`.
The separate **Reference Edit** mode uses ComfyUI's `KSampler` and
an empty native-format crop latent produced by `VAEEncode` and
`LatentMultiply(0)`. Both modes use upstream CropAndStitch for the final
image; neither generates the old homemade Qwen diagnostics gallery.
An effective Qwen segment-model override controls original-photo source
selection even when the base checkpoint is not Qwen. Replicated source-image
batches offset the segment seed per image so edits are not identical clones.
For API-by-URL Comfy backends without the Swarm base64 image loader,
Qwen's original Init Image also skips the legacy upload-to-canvas resize.
Other img2img and custom workflows retain their existing resize behavior.

The user's segment instruction is passed to `SwarmTextEncodeAdvanced`
without any invented replacement wording or synthetic `<image1>` token.
Swarm's real prompt processor handles residual weight, schedule, embedding
and LoRA-hook markup, rather than flattening it through a token-count
estimator. Native Qwen tokenization then receives the resulting text and
image tensors. This must be verified with the actual generated graph. The selected Comfy backend's own `/object_info` node inventory
is checked before submission; bundled nodes on another backend do not make
external API-by-URL backends compatible.

Swarm bundles unchanged upstream Qwen, LanPaint, and CropAndStitch sources
under separate node packages, retaining their respective licenses. Empty
CLIPSeg masks fail before the crop adapter rather than becoming whole-frame
fallback crops. Neither this wiring nor CPU tests prove semantic replacement.

The live, opt-in `tests/integration/qwen_segment_text_replacement.py`
preflights **real exported Comfy API graphs** and all node classes in the
running backend's `/object_info` before allowing model-backed inference.
It rejects a disconnected detector, wrong full-frame reference, obsolete
custom encoder, non-native sampler, missing crop, blurred-mask-driven
crop bounds, empty-mask fallthrough, and non-native compositor.
It saves actual full-frame before/after, crop before/after and mask results.

The exact previously failing prompt remains:

```
<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes
```

Source-level tests and successful CI do **not** establish semantic identity
replacement. A pixel delta, a LoRA loader on the model path, and four
denoising callbacks are necessary but insufficient evidence. Closure requires actual generated results that replace the child as requested
with correct crop and final composite, plus positive/negative LoRA controls.

No GPU or model weights are required for the source-level C# tests, real CPU
PyTorch mask-guard controls, unchanged upstream crop/stitch tensor tests, or
offline inspection of an exported Comfy API graph. These tests can disprove
incorrect sampler links, LoRA loader identity, output multiplicity, mask
thresholds, and pixel blending. They cannot demonstrate whether Qwen turns
a particular girl into the requested boy or whether a particular LoRA changes
model output. Those two acceptance claims require the relevant weights and a
runnable inference backend, not necessarily GPU hardware. Do not call source
and tensor contracts untestable just because no local GPU is connected.

## References

- SwarmUI: `WorkflowGeneratorQwenSegment.cs`,
  `WorkflowGeneratorSteps.cs#RunSegmentationProcessing`,
  `ExtraNodes/ComfyInpaintCropAndStitch/inpaint_cropandstitch.py`,
  `ExtraNodes/LanPaintUpstream/src/LanPaint/nodes.py`.
- ComfyUI@e3bee7ffcb7d6547ed0bcd15221792f3b8646b17:
  `comfy/text_encoders/qwen_image21.py#QwenImage21Tokenizer.tokenize_with_weights`,
  `comfy_extras/nodes_edit_model.py#ReferenceLatent.execute`,
  `nodes.py#KSampler.sample`. Original `TextEncodeQwenImage21.execute`
  is an upstream comparison, **not** the current segment generator node.
- SwarmUI `launchtools/qwen_segment_capture.py`: backend GET API capture,
  diagnostic connection analysis, no image-generation execution.
