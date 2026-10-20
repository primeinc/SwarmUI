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

The current Qwen segment workflow generates native `TextEncodeQwenImage21`,
`KSampler`, and `ImageCompositeMasked` nodes, and ships the unchanged
upstream `nodes_qwen.py` module as `ComfyQwenNodes_Upstream.py`
(GPL-3.0 license included). The prior
`SwarmQwenImage21SegmentEncode` and Qwen Python diagnostics are removed.

The live, opt-in `tests/integration/qwen_segment_text_replacement.py`
preflights **real exported Comfy API graphs** and the running backend's
`object_info/TextEncodeQwenImage21` before allowing GPU inference.
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
denoising callbacks are necessary but insufficient evidence. Closure
requires actual generated results that replace the child as requested
with correct crop and final composite, plus positive/negative LoRA controls.

## References

- SwarmUI: `WorkflowGeneratorQwenSegment.cs`,
  `WorkflowGeneratorSteps.cs#RunSegmentationProcessing`,
  `WorkflowGenerator.cs#CreateImageMaskCrop,RecompositeCropped`.
- ComfyUI@e3bee7ffcb7d6547ed0bcd15221792f3b8646b17:
  `comfy_extras/nodes_qwen.py#TextEncodeQwenImage21.execute`,
  `nodes.py#KSampler.sample`,
  `comfy_extras/nodes_mask.py#ImageCompositeMasked`.
- SwarmUI `launchtools/qwen_segment_capture.py`: backend GET API capture,
  diagnostic connection analysis, no image-generation execution.
