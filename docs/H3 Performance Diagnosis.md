# Diagnose MiniMax H3 performance differences

This procedure distinguishes **main-model H3 with a prompt-attached reference picture**
from **H3 Image-to-Video follow-up with explicit first-frame conditioning**. Two videos taking less time than one
is a performance symptom, not proof that a particular setting is wrong.

## Known controlled comparison (reported November 15, 2026)

- Same written prompt and one prompt-attached, model-native-resolution picture
- Four sampler steps, CFG 1, 324 requested output frames (H3 aligns to 328)
- Two-stage I2V: roughly **65 seconds per visible sampler step**, two saved videos
- Main-model video: roughly **123 seconds per visible sampler step**, one saved video
- Output metadata only records the requested parameter; it does **not** prove
  effective sampler model, latent shape, guidance payload, or sampler count

## Third controlled run: one-frame H3 base + 324-frame I2V

| User observation | Main 324 frames | Two-stage, base default | Two-stage, base **1 frame** |
| --- | ---: | ---: | ---: |
| Four-step long sampler | 491 s | 261 s | 261 s |
| Total Comfy prompt time | 634 s | 535.90 s | **384.81 s** |
| Seconds per long step | ~122.9 | ~65.3 | ~65.3 |
| Outputs reported | 1 | 2 | 2 |

The initial two-stage base frame count is **not present in the supplied log excerpt**.
SwarmUI defaults the base H3 count to 124 when unset; confirm the graph before
treating the first column as a verified 124-frame run. Both follow-up runs requested
324 frames, aligned to 328 by H3. Both users' requests used one prompt image,
4 steps and CFG 1.

The third run isolates two costs. Shortening the first stage reduces prompt time by
**151.09 s**, while the visible long-video sampler stays **261 s**. Moving from the
one-frame follow-up run to main 324-frame H3 costs **249.19 s total**, of which
approximately **230 s** is in the long sampler (491 vs 261). Those costs are not
explained by extra saved output files, CFG, or the number of prompt images.

**A single frame is a genuine H3 special case, not five frames.** SwarmUI's
`MiniMaxH3AlignFrames(1)` returns 1; the real
`SwarmModels.py::temporal_shape(1)` returns (1 output frame, 1 video latent step,
2 audio latent tokens). By contrast, 124 frames use **37** video latent steps,
and requested 324 frames become 328 actual output frames with **97** video
latent steps. This explains why the first-stage work is far smaller in the
third run. The separate long sampler remains a 328-frame job.

**The two long samplers are different model tasks, not the same input shape
under different parameter labels.** The main H3 reference path builds
`SwarmMiniMaxH3CollectReferences`, then injects `minimax_refs` into the
conditioning. The I2V path creates a dedicated first-frame constraint using
`SwarmMiniMaxH3AddKeyframes`. Upstream H3 packs reference pictures into
`ref_img` attention rows and explicit first frames into `cond` rows.
MiniMax publishes **Ref2VA** and **FL2VA** as separate task-specific checkpoints
(official source: https://github.com/MiniMax-AI/MiniMax-H3/blob/main/README.md).
Checkpoint identity, exact latent resolution, and attention/VRAM behavior must
be read from the generated workflow/trace before attributing the observed
230-second difference to a specific kernel or model weight file.

**Generation-time display:** 6.41 minutes is about 384.6 seconds, matching the
entire 384.81-second Comfy prompt execution. The separately displayed 27.33-second
one-frame generation is part of that total, not another 27.33 seconds to add on.

**Minimal workflow for a single final video from a known opening image:** Use
a regular image model as the UI's required base model, upload the image under
**Init Image**, set **Init Image Creativity = 0** to bypass its still-image
sampler, and select H3 **FL2VA** in **Image-to-Video Follow-up**, with
**Video Frames = 324**. This lets the follow-up use the uploaded image directly
as a first-frame keyframe without first generating a video to harvest frame 0.
`Save Image Before Video` already suppresses the initial output for this
zero-creativity case unless intermediate images are explicitly requested.
This is **FL2VA first-frame conditioning**; it is not identical in semantics
to **Ref2VA image-reference conditioning**, so output fidelity should be judged
against the user's desired opening frame.

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
