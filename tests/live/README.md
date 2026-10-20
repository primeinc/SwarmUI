# Qwen Image 2.1 segment acceptance, not graph assertions

Known reproduced failure reported on October 20, 2026, **10 of 10 attempts**:

    <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes

The tag means: use CLIPSeg to find \`girl\`, sample the cropped edit with 0.8 segment creativity, and use **0.1 CLIPSeg detection threshold**. This is a text-only segmentation test and does **not** require a LoRA to fail. Do not misdiagnose it as a defective adapter or GPU.

## Test the actual failing stage

\`tests/integration/qwen_segment_text_replacement.py\` accepts a **real SwarmUI-generated Comfy API graph** from the failed configuration and a **real ComfyUI backend**, not an editor-layout graph, synthesized node fixture, stub model, or fake image.

    python tests/integration/qwen_segment_text_replacement.py \
      --workflow /path/to/actual-exported-comfy-api-workflow.json \
      --comfy-url http://127.0.0.1:8188 \
      --outdir /path/to/local-qwen-probe \
      --attempts 1

The script does not generate or install model assets. It only operates against an explicitly selected backend, and running it **queues real inference**. It requires the exported workflow to contain a CLIPSeg detector matching \`girl\` at \`0.1\`, the exact replacement instruction in native \`TextEncodeQwenImage21\`, a non-empty detector-mask guard, and the upstream CropAndStitch and LanPaint nodes. Otherwise it fails before inference.

It adds real Comfy \`PreviewImage\` output nodes to save: the original full-size image, CLIPSeg mask, source crop, edited crop, and final composite. It measures whether the mask is empty/broad and whether sampler output and final composite differ from their actual inputs. These measurements distinguish detector, sampler, and compositor failures. **Pixel changes do not prove that the intended boy with blue eyes replaced the girl.** The script always reports \`NOT_CLOSED\` until the saved images have been semantically reviewed. For the exact reported ten-attempt failure, use \`--attempts 10\` after an initial one-attempt diagnosis.

The raw graph can be collected read-only through \`launchtools/qwen_segment_capture.py\`; this does not queue inference and may need to observe a running workflow because Swarm often clears completed Comfy history.

## Confined LoRA acceptance

\`tests/live/test_qwen_segment_lora.py\` compares the **same installed LoRA** at strength zero and a nonzero strength through two actual SwarmUI Comfy API exports: working global (positive control) and segmented (reported failure). It queues real OFF-ON-OFF image generations and fails when either model path doesn't use the requested adapter, or its rendered image shows no measurable effect.

    SWARM_LORA_ACCEPTANCE_MANIFEST=/path/to/actual-test-manifest.json \
      python -m pytest tests/live/test_qwen_segment_lora.py -v -s

Use the actual installed model files, adapter names, sampler, input image, seed, and SwarmUI-exported graph. The manifest must enumerate all adapters under test; an empty population fails. No adapter checks should be counted as passing before **text-only segmentation** itself is working.

\`tests/live/test_segment_lora_browser.py\` uses a live Chromium page against the actual SwarmUI LoRA selector and generation parameter. Unlike previous tests, it does not substitute a JavaScript fake \`<select>\` object. This only proves the browser control, not image generation.

## What ordinary CI proves

Normal CI compiles C# and runs actual Qwen segment prompt/source/LoRA-chain graph tests, Python native-source checks, and PyTorch **CPU** guard/crop/stitch tests against the shipped code and unchanged upstream adapter. It can reject wrong source selection, missing/extra LoRA loaders, mask fallthrough, double exposures from fractional masks, and incorrect graph wiring without a GPU or Qwen model weights. These tests are real, falsifiable engineering checks, **not** successful Qwen semantic image generations. No mock visual success should be counted as acceptance.

Model-backed image acceptance requires the installed models, genuine input image, and a Comfy backend capable of running them. A GPU is often the practical accelerator but is not a logical prerequisite for the source and tensor tests, nor inherently for inference on a supported CPU backend. Ordinary GitHub-hosted CI **has not run the full Qwen image-generation acceptance**. Until the actual text edit succeeds and all intended confined LoRAs visibly affect their final images, the correct project verdict remains **NOT_CLOSED**.
