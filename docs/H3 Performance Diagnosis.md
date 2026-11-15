# Diagnose MiniMax H3 performance differences

This procedure distinguishes **main-model H3 Ref2VA with one prompt-attached picture**
from **H3 FL2VA Image-to-Video follow-up**. Two videos taking less time than one
is a performance symptom, not proof that a particular setting is wrong.

## Known controlled comparison (reported November 15, 2026)

- Same written prompt and one prompt-attached, model-native-resolution picture
- Four sampler steps, CFG 1, 324 requested output frames (H3 aligns to 328)
- Two-stage I2V: roughly **65 seconds per visible sampler step**, two saved videos
- Main-model video: roughly **123 seconds per visible sampler step**, one saved video
- Output metadata only records the requested parameter; it does **not** prove
  effective sampler model, latent shape, guidance payload, or sampler count

## Capture a discriminating trace

This fork's `SwarmKSampler` has an **opt-in** shape-and-timing trace. Run the
**same two prompts in the same warm ComfyUI process** so a restart does not
confound the comparison.

1. Stop SwarmUI and its managed ComfyUI backend. From PowerShell in the
   updated `dev` checkout, start SwarmUI with the environment variable set:

   ```powershell
   $env:SWARM_H3_TRACE = '1'
   .\launch-windows.bat *>&1 | Tee-Object -FilePath .\h3-session.log
   ```

   For an independently started ComfyUI, set this variable in **that process's**
   environment instead. Restart the ComfyUI backend to reload the patched
   `SwarmKSampler.py`. The variable is disabled by default.

2. Run the **two-video follow-up** case first, then the **one-video main-model**
   case. Keep model files, prompt, image, seed, steps, CFG, frame count,
   resolution, preview mode and other advanced settings the same wherever
   both pathways allow. Do not clear the cache between runs.
3. Stop the trace, then compare both prompts in the single captured log:

   ```powershell
   python .\tests\h3_perf_compare.py .\h3-session.log
   ```

   Alternatively, save each run to a separate log and pass both paths:

   ```powershell
   python .\tests\h3_perf_compare.py .\two-video.log .\one-video.log
   ```

The comparison selects the **largest video latent** in the two-video run so
its short first video cannot accidentally be compared against the final
13.5-second main-model video.

## What the trace proves

Each `[H3PERF]` JSON line is tied to the ComfyUI `prompt_id` and `node_id`.
The sampler `start` event records actual video/audio latent shapes,
number/shape of reference blocks, keyframe blocks, conditioning text token
lengths, CFG, sampler, scheduler, preview type, patch count, and allocated/
reserved CUDA memory. `step` events record sampler-side wall time (`elapsed_s`) separately from
preview/progress callback overhead (`callback_s`), without forcing GPU
synchronization. `finish` reports sampler duration.

Compare `latent_shapes` first. If equal, compare `refs` versus `keyframes`
and `text_tokens`; H3's Ref2VA `ref_img` and FL2VA `cond` tokens are
**different execution modes**, not equivalent inputs even for identical images.
If these agree, compare loaded checkpoint filenames by capturing SwarmUI
**Verbose** logs containing `Will use workflow` and inspecting the nodes
feeding each sampler's `model` input. The trace does **not** identify a
checkpoint filename itself, prove GPU kernel utilization, or fix slow inference.

To stop tracing, unset the environment variable and restart the backend:

```powershell
Remove-Item Env:SWARM_H3_TRACE
```

## Code boundaries

- `WorkflowGeneratorModelSupport.cs::CreateEmptyImage` constructs main H3 latent
- `WorkflowGenerator.cs::CreateConditioningDirect` collects Ref2VA references
- `WorkflowGenerator.cs::ImageToVideoGenInfo.PrepFullCond` builds FL2VA keyframes
- `SwarmKSampler.py::SwarmKSampler.sample` reports actual sampling input/timing
- Upstream ComfyUI `comfy/ldm/minimax/model.py::PackedLayout` distinguishes
  `ref_img` and `cond` segments (inspected revision
  `c844cb672186f187bf9455e2808821417a10d296`)

No performance fix should be chosen from the output metadata alone.
