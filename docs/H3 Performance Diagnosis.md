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

## Windows log analysis menu

After updating the `primeinc/SwarmUI` `dev` checkout, double-click
[`H3_Log_Analyzer.bat`](../H3_Log_Analyzer.bat) in the repository root.
It requires Python 3 and uses Windows' native file chooser (falling back to
a typed/dragged file path). It never edits source logs.

- **Quick Scan** reads older logs without `[H3PERF]` to extract prompt times,
  model weight staging, compiler graph-break counts and output counts. It
  explicitly labels the absence of detailed sampler measurements.
- **Two Runs / One Log** expects the two-stage I2V request first, then main H3.
  **Three Runs / One Log** expects I2V, main H3, then the one-frame-first-stage
  I2V request. Runs must have distinct ComfyUI prompt IDs and appear in order.
- **Compare Two/Three Files** accepts separate log files in the same order.
  It reuses `tests/h3_perf_compare.py` and selects the longest video sampler
  in each request; short preliminary samplers are not mistaken for final videos.
- **Export H3 Trace** creates sanitized structured `.jsonl` containing only
  `[H3PERF]` records. **Open Reports** opens `H3-Reports`.
- Every successful operation writes timestamped `.txt` and `.json` files
  under `H3-Reports`, which is excluded from Git along with
  `h3-session*.log`. Source log paths appear in the local JSON receipt.

To scan one file directly, drag its `.log` or `.txt` onto the batch file,
or run `H3_Log_Analyzer.bat --scan "C:\\path with spaces\\h3-session.log"`.
For a non-interactive Windows smoke test, run
`H3_Log_Analyzer.bat --self-test`.

For persistent logs, launch the updated Windows updater from a terminal
with stdout/stderr redirected to a file, or use the PowerShell capture
example below. The `SWARM_H3_TRACE=1` variable must be present in the
ComfyUI backend's environment, and the backend must restart to reload the
patched sampler. The analyzer does not change runtime settings or start
model inference.

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

   To include the one-frame-first-stage third run, pass three separate logs:

   ```powershell
   python .\tests\h3_perf_compare.py .\two-video.log .\one-video.log --one-frame-followup-log .\one-frame-followup.log
   ```

   Or, if all three requests ran sequentially in one backend process in that
   order (two-video, main video, one-frame follow-up):

   ```powershell
   python .\tests\h3_perf_compare.py .\h3-session.log --three-in-one-log
   ```

The comparison selects the **largest video latent** in the two-video run so
its short first video cannot accidentally be compared against the final
13.5-second main-model video.

## Preventing stale step-2 previews from previous jobs

**The preview boundary fix** is independent of the sampler's GPU preview
implementation. The custom sampler sends the originating `prompt_id` in a
metadata-bearing event-4 binary preview. Previously
`ComfyUIAPIAbstractBackend.AwaitJobLive` parsed the image MIME type and index
but ignored the `prompt_id`. With reused sockets, a delayed preview from
the preceding job could be forwarded as a preview for the current request.

The production `H3PreviewPromptGuard.Matches` now rejects an event-4
frame whose explicit `prompt_id` differs from the active ComfyUI prompt.
The receiver also drops **all binary frames before the current prompt's
`execution_start`**. A native preview that has no `prompt_id` remains
accepted to avoid changing unrelated ComfyUI model behavior.

The async `SwarmKSampler` preview sender also snapshots
`PromptServer.instance.client_id` when its sampler begins. Queued workers
send to that captured socket client, not whichever client was most recently
assigned when they finish rendering.

**Verification boundaries:**
`tests/h3-preview-guard` compiles and invokes the actual production
C# identity helper against real event-4 binary envelope fixtures and
negative controls; `tests/test_h3_preview_delivery_contract.py` checks
the production call site; `tests/test_swarm_sampler_preview_dispatch.py`
tests the real Python serializers and queued workers under simulated
client switches. These checks do not run your H3 model on your GPU.
The corresponding live acceptance case remains: on two successive H3
video generations, the second sampling-step preview must belong to the
**current** job, never the previous image or a black placeholder.

The live [H3TRACE-ROUTE] marker must also confirm `SwarmKSampler` when
testing the custom preview callback. If ComfyUI did not register the custom
node, SwarmUI may fall back to native `KSamplerAdvanced`, which is outside
the custom callback's scope.

## When [H3PERF] and [H3VAE] do not appear in SwarmUI logs

A launcher echo and a child-CMD smoke test are **not** proof that the
ComfyUI worker loaded the tracing code. This fork emits a startup handshake
from both sides of the real process boundary. After updating `dev`,
**fully restart SwarmUI and its managed ComfyUI backend**, then inspect the
SwarmUI console for (trace records are forwarded at **Info**):

```text
[H3TRACE-BOOT] host=SwarmUI backend=ComfyUI-0 traceEnabled=True childEnv=1
[ComfyUI-0/STDERR] [H3TRACE-BOOT] source=SwarmComfyCommon import_start env=1 file=...
[ComfyUI-0/STDERR] [H3TRACE-BOOT] source=SwarmComfyCommon status=enabled vae_hook=True custom_sampler=True ...
[H3TRACE-ROUTE] stage=video-followup sampler=SwarmKSampler ... per_step_trace=True
[ComfyUI-0/STDERR] [H3TRACE-ROUTE] source=SwarmKSampler model=MiniMaxH3 active=True ...
[ComfyUI-0/STDERR] [H3PERF] {"event": "start", ...}
[ComfyUI-0/STDERR] [H3VAE] {"event": "start", ...}
```

The lines are examples of the expected structure, **not recorded runtime
output**. The parent trace state is reported even when disabled, and the
Python package reports its loaded file path even when tracing is disabled.

Interpret missing stages *in order*:

1. **No SwarmUI host marker:** the running binary is stale, the wrong
   installation is running, or this is not a self-start ComfyUI backend.
2. **Host marker says traceEnabled=False:** the actual SwarmUI process was
   not started with `SWARM_H3_TRACE=1`. Launch using the updated
   `Windows_Update_SwarmUI_primeinc_test.bat`, not only the ordinary
   `launch-windows.bat`.
3. **Host enabled, but no Python package marker:** ComfyUI did not load
   the patched `SwarmComfyCommon` from the expected installation, or
   its import failed before reaching the hook. The Python marker names
   the actual file path for comparison.
4. **Package enabled, but C# route reports native KSamplerAdvanced:**
   the earlier `[H3PERF]` trace was limited to `SwarmKSampler`.
   This diagnosis is **not** proof that the native sampler is equivalent.
5. **Route reports SwarmKSampler, but Python entry is absent:** the
   worker is executing a different node implementation than the source
   that generated the graph.
6. **Python sampler entry reports active=False:** the effective model
   class name did not pass the H3 guard. The entry records that name.
7. **Sampler/VAE events missing despite activated routes:** inspect
   plugin import warnings and actual ComfyUI process stderr. Both trace
   event emitters now write directly to stderr with `flush=True`,
   avoiding Python INFO log-level filtering. SwarmUI's
   `NetworkBackendUtils.ReportLogsFromProcess` now promotes only H3 diagnostic
   markers from the usual Debug subprocess severity to Info.

**H3 Performance Studio Quick Scan** also reports the observable host,
plugin, sampler and VAE checkpoints, and identifies the earliest missing
one. Old logs that omit application startup are labeled inconclusive rather
than proving the environment variable was absent.

**Proof scope:** CPU-only CI verifies startup source gates, JSON formatting,
and stderr emission; it cannot prove the live GPU worker, external
ComfyUI version, or loaded custom-node path on your machine.

## VAE decoder progress and timing

**H3 VAE decoding no longer has to disappear into a silent interval.**
With `SWARM_H3_TRACE=1` on ComfyUI **before backend startup**, the
`SwarmComfyCommon` plugin wraps the native ComfyUI
`comfy.sd.VAE.decode` and `decode_tiled` methods **only for H3 VAEs**.
It logs single-line `[H3VAE]` INFO events on ComfyUI's existing STDERR log:

- `event=start`: prompt/node IDs, video/audio/image VAE model, method,
  input latent shape and requested tiling options
- `event=working`: elapsed seconds every 15 seconds while VAE decoding
  is still in progress; a heartbeat, **not tile-by-tile percent complete**
- `event=finish`: elapsed seconds, status, output shape and exception type
  if the original decoder fails; exceptions are re-raised unchanged

The wrapper preserves the existing VAE decode code path and changes no
sampling, VAE precision, tile sizes, or model checkpoints. Normal SwarmUI
launches with the variable unset are unaffected. Video encoding/muxing and
SwarmUI output-file persistence **are not included in the VAE duration**,
so the remaining gap between `[H3VAE] finish` and `Prompt executed in`
must not automatically be classified as VAE time.

`H3_Log_Analyzer.bat` **Quick Scan** presents video and audio decoder
durations separately, while comparison views show matching per-prompt
VAE totals. **Export H3 Trace** includes both `[H3PERF]` and
`[H3VAE]` entries in JSONL without logging image/audio values.

Upstream provenance: `Comfy-Org/ComfyUI` commit
`c844cb672186f187bf9455e2808821417a10d296`,
`comfy/sd.py::VAE.decode` (around line 1362) and
`::VAE.decode_tiled` (around line 1483);
`nodes.py::VAEDecodeTiled.decode` (around line 362);
`comfy_extras/nodes_audio.py::VAEDecodeAudio.execute` (around line 132).
Implementation:
`src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmVaeTrace.py`,
registered by the package `__init__.py`.

## What the trace proves

Each `[H3PERF]` JSON line is tied to the ComfyUI `prompt_id` and `node_id`.
The sampler `start` event records actual video/audio latent shapes,
number/shape of reference blocks, keyframe blocks, conditioning text token
lengths, CFG, sampler, scheduler, preview type, patch count, and allocated/
reserved CUDA memory. `step` events record sampler-side wall time (`elapsed_s`) separately from
preview/progress callback overhead (`callback_s`), without forcing GPU
synchronization. `finish` reports sampler duration.

The trace also records **actual model-weight residency** from ComfyUI
`ModelPatcher.loaded_size()` and `model_size()` at sampler start, every step,
and finish. `weights_resident_mb` includes dynamic VBAR-resident GPU weights;
`weights_offloaded_mb` is the remaining model size, **not a count of bytes
transferred per step**. This is deliberately different from
`torch.cuda.memory_allocated`: that counter misses dynamically managed
weights. `0 models unloaded` only reports full model-registry unloading
decisions, and `N MB Staged` is dynamically staged model weight storage,
not `N MB` of permanently resident GPU weights.

**Implementation provenance (source consulted):**
`Comfy-Org/ComfyUI@c844cb672186f187bf9455e2808821417a10d296`,
`comfy/model_patcher.py::DynamicVramModelPatcher.loaded_size` (around line
1901), `comfy/model_patcher.py::DynamicVramModelPatcher.load` (around 2128),
and `comfy/model_management.py::load_models_gpu` (around 1000–1033).
`SwarmKSampler.py::_swarm_h3_trace_residency` adapts these APIs. This
proves reported **residency snapshots**, not how much PCIe traffic or
GPU compute time each step consumed; a GPU profiler is needed for that.

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
