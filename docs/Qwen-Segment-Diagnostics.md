# Capture a Qwen segment failure without changing the generation

The Qwen segment-only LoRA issue is NOT FIXED. This tool captures evidence; it does not claim to repair image generation.

From the SwarmUI directory on Windows, run `capture-qwen-segment.bat`, then repeat the failing generation once. Keep SwarmUI and its Comfy backend running. The tool reads the active queue because Swarm normally deletes completed Comfy history. It prints the path of a ZIP in the Windows temporary directory. It does not queue a generation, alter settings, switch branches, install packages, or send the ZIP anywhere.

The default endpoint is `http://127.0.0.1:7801/ComfyBackendDirect`. For a non-default server use `capture-qwen-segment.bat --url http://127.0.0.1:YOUR_PORT/ComfyBackendDirect`. An explicit direct Comfy backend URL also works. Authentication failures are reported, not bypassed. The tool uses the backend reached by the supplied URL; it does not discover or scan other backends.

The capture reports actual sampler inputs, model-path LoRA loaders and strengths, loader nodes not shared with base sampling, latent masks, reference-latent connections, and compositors. A disconnected LoRA node elsewhere in the graph does not count. The original graph is included with embedded media and common credential fields redacted. Prompts and model/LoRA filenames remain: review the bundle before sharing.

## Verification scope

Run `python -m unittest discover -s tests -p test_qwen_segment_capture.py -v`.

18 tests passed locally on Python 3.13.5/Linux: graph traversal, missing/disconnected LoRA negative controls, conditioning isolation, zero effective steps, empty/deleted history with active queue, GET-only HTTP integration, redaction, and CLI ZIP output. The Windows batch launcher has not been executed in this environment. No GPU inference or user-visible Qwen fix is certified by these tests.

## Evidence and unresolved questions

- Source: `primeinc/SwarmUI@57c12b9a2455c0b777291edf67e27559f838f13d`, `src/BuiltinExtensions/ComfyUIBackend/ComfyUIAPIAbstractBackend.cs#AwaitJobLive`: completed prompt history is deleted unless NoInternalSpecialHandling is enabled. Capture therefore includes `/queue`, not just `/history`.
- Source: `Comfy-Org/ComfyUI@e3bee7ffcb7d6547ed0bcd15221792f3b8646b17`, `server.py#get_history,#get_queue,#system_stats`: GET API routes and response shapes used by the capture. Runtime version is captured separately; equivalence to the inspected revision is not assumed.
- Source: the same ComfyUI revision, `comfy_extras/nodes_qwen.py#TextEncodeQwenImage21.execute` and `comfy/text_encoders/qwen_image21.py#QwenImage21TEModel.encode_token_weights`: reference pixels are encoded for text/vision and VAE, with image-slot metadata. A reference with no image supplied to its conditioning encoder is flagged for investigation, not declared the root cause.
- A progress counter of four steps does not prove that the configured total was four: the sampler may start partway through a larger schedule. The capture reports total, start, end and effective steps separately.

End-to-end acceptance still requires the user's exact Qwen model, LoRA, prompt, graph, and before/after image result. A graph connection alone cannot establish that LoRA weights were applied effectively inside the model.
