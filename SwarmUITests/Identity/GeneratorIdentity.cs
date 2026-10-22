using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime;
using System.Threading;
using System.Threading.Tasks;
using FreneticUtilities.FreneticExtensions;
using FreneticUtilities.FreneticToolkit;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SixLabors.ImageSharp.Memory;
using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Builtin_DynamicThresholding;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests.Identity;

/// <summary>Builds every main model class x option variant with only APIs the upstream commit and the trace branch share, and writes one line per case:
/// label, sha256 of the generated workflow's text and of the exact text <see cref="ComfyUIAPIAbstractBackend.CreateWorkflow"/> sends (only the random SaveImage prefix replaced), and, for the input CreateWorkflow sent, sha256 of the image parameter metadata after unused parameters are removed together with the unused-parameter list (swarm_version left out),
/// its ExtraMeta keys other than debug_trace, and sha256 of the metadata text an image is saved with (<see cref="T2IParamInput.MetadataToString"/>) without swarm_version and the extra data's date and debug_trace; or the error.
/// With IDENTITY_COST_OUT set, it then measures every <see cref="CostSampleStride"/>th case that built, one at a time on this thread, and writes its allocation and time (<see cref="MeasureCosts"/>).
/// Fails if any error is not a generator rejection or fewer than 80% of the cases build. IDENTITY_OUT names the output file; IDENTITY_DEBUG=1 turns AddDebugData on. Skipped without IDENTITY_OUT; SwarmUITests/Identity/justfile runs it in this tree and in the upstream tree and compares the dumps.</summary>
[Category("Identity")]
public class GeneratorIdentity : SwarmUITest
{
    /// <summary>Counts the bytes the shared char and byte array pools allocate on one thread when a rent finds no pooled array.
    /// Whether a rent finds one depends on the pool's trimming at each gen2 GC, which follows system memory pressure, not on the code measured (.NET 8 SharedArrayPool.Rent and Trim):
    /// a Qwen Image 2.1 build's interpolated CreateNode keys (DefaultInterpolatedStringHandler rents from ArrayPool&lt;char&gt;.Shared) moved its count by 2 to 6 MiB from run to run.
    /// ArrayPoolEventSource raises BufferAllocated(bufferId, bufferSize, poolId, bucketId, reason) on the renting thread, and poolId is the pool's GetHashCode.
    /// Defined here because this file also builds alone in the upstream tree; WorkflowTraceBudgetTests uses it too.</summary>
    public sealed class PoolRefills : EventListener
    {
        /// <summary>Pool IDs ArrayPoolEventSource reports for the shared char and byte pools.</summary>
        public static readonly int CharPoolId = ArrayPool<char>.Shared.GetHashCode(), BytePoolId = ArrayPool<byte>.Shared.GetHashCode();

        /// <summary>Managed thread whose refills are counted: the one that creates the listener.</summary>
        public readonly int ThreadId = Environment.CurrentManagedThreadId;

        /// <summary>Bytes of arrays the shared char and byte pools allocated on <see cref="ThreadId"/>.</summary>
        public long Bytes;

        /// <summary>Informational pool events (an array allocated or dropped by any pool) raised on <see cref="ThreadId"/>; each one also allocates its event arguments there.</summary>
        public long Events;

        /// <inheritdoc/>
        protected override void OnEventSourceCreated(EventSource source)
        {
            if (source.Name == "System.Buffers.ArrayPoolEventSource")
            {
                EnableEvents(source, EventLevel.Informational);
            }
        }

        /// <inheritdoc/>
        protected override void OnEventWritten(EventWrittenEventArgs e)
        {
            if (Environment.CurrentManagedThreadId != ThreadId)
            {
                return;
            }
            Events++;
            if (e.EventId != 2)
            {
                return;
            }
            int length = (int)e.Payload[1], pool = (int)e.Payload[2];
            Bytes += pool == CharPoolId ? length * 2L : pool == BytePoolId ? length : 0;
        }
    }

    /// <summary>Every this many cases that built, one has its cost measured (<see cref="MeasureCosts"/>).</summary>
    public const int CostSampleStride = 32;

    /// <summary>Measured runs per no-GC region (<see cref="MeasureCosts"/>).</summary>
    public const int CostRepeats = 3;

    /// <summary>Most no-GC regions <see cref="MeasureCosts"/> opens for one case before it gives up finding a run with full-length prefixes (<see cref="FullLengthPrefixes"/>).</summary>
    public const int CostRegions = 4;

    /// <summary>Bytes a no-GC region of <see cref="MeasureCosts"/> holds beyond its runs, for other threads' allocations.</summary>
    public const long CostRegionSlack = 16L << 20;

    /// <summary>Length of a SaveImage filename prefix ("SwarmUI_{Random.Shared.Next():X4}_", WGNodeData) whose random number has eight hex digits, as Random.Shared.Next() gives 15 times in 16.</summary>
    public const int FullPrefixLength = 17;

    /// <summary>Whether every random SaveImage filename prefix in a workflow is <see cref="FullPrefixLength"/> long: a shorter one shortens every text that holds it, and so the run's allocation.</summary>
    public static bool FullLengthPrefixes(JObject workflow)
    {
        foreach (JProperty property in workflow.Properties())
        {
            if (property.Value["inputs"]?["filename_prefix"] is JValue prefix && $"{prefix}" is string text && text.StartsWith("SwarmUI_") && text.Length != FullPrefixLength)
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>Runs each sampled case once, then, inside a no-GC region (<see cref="GC.TryStartNoGCRegion(long)"/>), once to warm the shared array pools and <see cref="CostRepeats"/> times measured, alone on this thread with ImageSharp allocating every buffer (<see cref="SimpleGcMemoryAllocator"/>).
    /// A measured run counts when every SaveImage prefix it wrote is full length (<see cref="FullLengthPrefixes"/>); a region with none is followed by another, up to <see cref="CostRegions"/>.
    /// Returns one line per case: label, the bytes a counted run allocated on this thread, and the least milliseconds a measured run took.
    /// No GC runs in the region, so no gen2 GC trims the pools (.NET 8 SharedArrayPool.Trim runs after each gen2 GC and, under high memory pressure, drops every thread's arrays) and every rent in a measured run finds an array.
    /// Throws if a GC ran in a region, a pool allocated or dropped an array on this thread in a measured run (<see cref="PoolRefills.Events"/>), or counted runs allocated different byte counts: each is a measurement that does not hold for the code alone.
    /// With AddDebugData on, counted runs may differ by one run: a traced run adds its new texts to the trace's process-wide hash caches (ConditionalWeakTable, keyed by reference), whose dead entries no GC clears inside the region,
    /// and the run that fills one allocates its rebuilt container (a 256-entry container is 5,224 bytes, the extra one of three traced runs allocated); the count at least two runs agree on is reported, and the runs are printed.
    /// Counts repeat only with tiered compilation off (DOTNET_TieredCompilation=0, as the justfile runs it): a method tiering up between runs changes what they allocate.
    /// A run is the dump's work for the case: build the input, generate, build the input again, create the workflow text to send, and write the saved metadata text.</summary>
    public static List<string> MeasureCosts(List<(string Label, Func<T2IParamInput> Make)> cases, IEnumerable<int> sample, string[] features)
    {
        MemoryAllocator priorAllocator = SixLabors.ImageSharp.Configuration.Default.MemoryAllocator;
        SixLabors.ImageSharp.Configuration.Default.MemoryAllocator = new SimpleGcMemoryAllocator();
        List<string> result = [];
        try
        {
            using PoolRefills refills = new();
            foreach (int i in sample)
            {
                (JObject Workflow, string Sent) run()
                {
                    T2IParamInput input = cases[i].Make();
                    JObject workflow = new WorkflowGenerator() { UserInput = input, Features = [.. features] }.Generate();
                    T2IParamInput sendInput = cases[i].Make();
                    string sent = ComfyUIAPIAbstractBackend.CreateWorkflow(sendInput, w => w, null, [.. features]);
                    T2IParamInput.MetadataToString(sendInput.GenFullMetadataObject());
                    return (workflow, sent);
                }
                long first = GC.GetAllocatedBytesForCurrentThread();
                run();
                first = GC.GetAllocatedBytesForCurrentThread() - first;
                long regionBytes = (CostRepeats + 1) * first * 5 / 4 + CostRegionSlack;
                List<long> bytes = [];
                long ticks = long.MaxValue;
                for (int region = 0; bytes.Count == 0; region++)
                {
                    if (region == CostRegions)
                    {
                        throw new InvalidOperationException($"No measured run of {cases[i].Label} in {CostRegions} no-GC regions wrote only full-length SaveImage prefixes.");
                    }
                    if (!GC.TryStartNoGCRegion(regionBytes))
                    {
                        throw new InvalidOperationException($"The runtime refused a no-GC region of {regionBytes} bytes to measure {cases[i].Label}.");
                    }
                    long events;
                    int collections;
                    try
                    {
                        collections = GC.CollectionCount(0);
                        GC.WaitForPendingFinalizers();
                        run();
                        events = refills.Events;
                        for (int repeat = 0; repeat < CostRepeats; repeat++)
                        {
                            long allocated = GC.GetAllocatedBytesForCurrentThread();
                            long start = Stopwatch.GetTimestamp();
                            (JObject workflow, string sent) = run();
                            ticks = Math.Min(ticks, Stopwatch.GetTimestamp() - start);
                            allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
                            if (FullLengthPrefixes(workflow) && FullLengthPrefixes(JObject.Parse(sent)))
                            {
                                bytes.Add(allocated);
                            }
                        }
                        events = refills.Events - events;
                        collections = GC.CollectionCount(0) - collections;
                    }
                    finally
                    {
                        if (GCSettings.LatencyMode == GCLatencyMode.NoGCRegion)
                        {
                            GC.EndNoGCRegion();
                        }
                    }
                    if (collections != 0 || events != 0)
                    {
                        throw new InvalidOperationException($"Cost of {cases[i].Label} is not repeatable: {collections} GCs in a {regionBytes}-byte no-GC region, {events} pool allocations or drops in its measured runs.");
                    }
                }
                if (bytes.Distinct().Count() != 1)
                {
                    long[] agreed = [.. bytes.GroupBy(b => b).Where(g => g.Count() >= 2).Select(g => g.Key)];
                    if (!Program.ServerSettings.AddDebugData || agreed.Length != 1)
                    {
                        throw new InvalidOperationException($"Cost of {cases[i].Label} is not repeatable: runs with full-length prefixes allocated {bytes.JoinString(", ")} bytes.");
                    }
                    TestContext.Out.WriteLine($"COST {cases[i].Label}: traced runs allocated {bytes.JoinString(", ")} bytes; reporting {agreed[0]}, the count at least two runs agree on");
                    bytes = [agreed[0]];
                }
                result.Add($"{cases[i].Label}\t{bytes[0]}\t{ticks * 1000.0 / Stopwatch.Frequency:0.000}");
            }
        }
        finally
        {
            SixLabors.ImageSharp.Configuration.Default.MemoryAllocator = priorAllocator;
        }
        return result;
    }
    /// <summary>Folder the fake models are registered under.</summary>
    public static string Root;

    /// <summary>Registers a fake model of a class in a model set.</summary>
    public static T2IModel Add(string type, string name, T2IModelClass modelClass, bool diffusionModels = false)
    {
        T2IModelHandler handler = Program.T2IModelSets[type];
        string folder = diffusionModels ? $"{Root}/diffusion_models" : $"{Root}/{type}";
        T2IModel model = new(handler, folder, $"{folder}/{name}", name) { ModelClass = modelClass, StandardWidth = modelClass?.StandardWidth ?? 0, StandardHeight = modelClass?.StandardHeight ?? 0, Metadata = new() { ModelName = name, ModelClassType = modelClass?.ID } };
        handler.Models[name] = model;
        return model;
    }

    /// <summary>Returns the fake main model of a class in checkpoint or diffusion_models format, registering it once.</summary>
    public static T2IModel ModelFor(T2IModelClass modelClass, bool diffusionModels, string suffix = "")
    {
        string name = $"test/{(diffusionModels ? "dm" : "ckpt")}/{modelClass.ID.Replace('/', '_')}{suffix}.safetensors";
        return Program.T2IModelSets["Stable-Diffusion"].Models.TryGetValue(name, out T2IModel existing) ? existing : Add("Stable-Diffusion", name, modelClass, diffusionModels);
    }

    /// <summary>Returns a fake ControlNet for a main model class: its compat class's ControlNet class if one exists, otherwise an unclassed one.</summary>
    public static T2IModel ControlNetFor(T2IModelClass modelClass)
    {
        T2IModelClass controlClass = T2IModelClassSorter.ModelClasses.Values.Where(c => c.ID.EndsWith("/controlnet") && c.CompatClass?.ID == modelClass.CompatClass?.ID).OrderBy(c => c.ID).FirstOrDefault();
        string name = $"test/cn/{(controlClass?.ID ?? "generic").Replace('/', '_')}.safetensors";
        return Program.T2IModelSets["ControlNet"].Models.TryGetValue(name, out T2IModel existing) ? existing : Add("ControlNet", name, controlClass);
    }

    /// <summary>Returns a solid-color PNG image.</summary>
    public static Image TestImage(int width, int height)
    {
        using SixLabors.ImageSharp.Image<Rgba32> img = new(width, height, new Rgba32(120, 140, 160, 255));
        return new Image(img);
    }

    /// <summary>Marks every model file name literal in the generator's IL as already present, so the generator never tries to download one.</summary>
    public static void MarkModelFileLiteralsPresent()
    {
        Module module = typeof(WorkflowGenerator).Module;
        BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;
        foreach (MethodBase method in module.GetTypes().SelectMany(t => t.GetMethods(flags).Cast<MethodBase>().Concat(t.GetConstructors(flags))))
        {
            byte[] il;
            try
            {
                il = method.GetMethodBody()?.GetILAsByteArray();
            }
            catch (Exception)
            {
                continue;
            }
            for (int i = 0; il is not null && i + 4 < il.Length; i++)
            {
                if (il[i] != 0x72 || il[i + 4] != 0x70)
                {
                    continue;
                }
                try
                {
                    string str = module.ResolveString(BitConverter.ToInt32(il, i + 1));
                    if (!str.Contains("://") && (str.EndsWith(".safetensors") || str.EndsWith(".pth") || str.EndsWith(".pt") || str.EndsWith(".bin") || str.EndsWith(".gguf")))
                    {
                        WorkflowGenerator.ClipModelsValid[str] = str;
                        WorkflowGenerator.VisionModelsValid[str] = str;
                        WorkflowGenerator.IPAdapterModelsValid[str] = str;
                        WorkflowGenerator.IPAdapterModelsValid[$"LORA-{str}"] = str;
                    }
                }
                catch (Exception)
                {
                }
            }
        }
    }

    /// <summary>Returns workflow JSON text with each random SaveImage filename prefix (read from the parsed workflow) replaced by a fixed one, every other byte as it was.</summary>
    public static string MaskedText(string text)
    {
        foreach (JProperty property in JObject.Parse(text).Properties())
        {
            if (property.Value["inputs"]?["filename_prefix"] is JValue prefix && $"{prefix}".StartsWith("SwarmUI_"))
            {
                text = text.Replace($"\"{prefix}\"", "\"SwarmUI_(random)_\"");
            }
        }
        return text;
    }

    /// <summary>Writes the identity dump to IDENTITY_OUT.</summary>
    [Test]
    public void DumpWorkflowIdentity()
    {
        string outPath = Environment.GetEnvironmentVariable("IDENTITY_OUT");
        if (string.IsNullOrWhiteSpace(outPath))
        {
            Assert.Ignore("Set IDENTITY_OUT, or run 'just run' in SwarmUITests/Identity.");
        }
        if (ComfyUIBackendExtension.SamplerParam is not null)
        {
            Assert.Ignore("Another fixture already ran ComfyUIBackendExtension.OnInit, and running it again would register its parameters twice; run 'just run' in SwarmUITests/Identity, which runs this test alone.");
        }
        Setup();
        Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash = false;
        Program.ServerSettings.AddDebugData = Environment.GetEnvironmentVariable("IDENTITY_DEBUG") == "1";
        Program.GlobalProgramCancel = new CancellationToken(true);
        Root = Path.Combine(Path.GetTempPath(), "swarmui-identity").Replace('\\', '/');
        T2IModelClassSorter.Init();
        foreach (string type in new[] { "Stable-Diffusion", "VAE", "LoRA", "Embedding", "ControlNet", "Clip", "ClipVision" })
        {
            Program.T2IModelSets[type] = new() { ModelType = type, DownloadFolderPath = $"{Root}/{type}" };
        }
        CommonModels.RegisterCoreSet();
        T2IParamTypes.RegisterDefaults();
        Program.Backends = new();
        new ComfyUIBackendExtension().OnInit();
        new DynamicThresholdingExtension().OnInit();
        foreach (CommonModels.ModelInfo info in CommonModels.Known.Values)
        {
            Add(info.FolderType, info.FileName, null);
        }
        MarkModelFileLiteralsPresent();
        Add("LoRA", "test_alien_lora.safetensors", null);
        Image init = TestImage(512, 384), promptImage = TestImage(256, 256), mask = TestImage(512, 384);
        T2IModel video = ModelFor(T2IModelClassSorter.ModelClasses["wan-2_1-image2video-14b"], true), seedvr = ModelFor(T2IModelClassSorter.ModelClasses["seedvr2"], true);
        string basePrompt = "zqxbase man on a road", segmentPrompt = "zqxbase man on a road <segment:man> zqxseg green alien <lora:test_alien_lora>";
        static void refiner(T2IParamInput i)
        {
            i.Set(T2IParamTypes.RefinerMethod, "PostApply");
            i.Set(T2IParamTypes.RefinerControl, 0.4);
        }
        void videoSet(T2IParamInput i)
        {
            i.Set(T2IParamTypes.VideoModel, video);
            i.Set(T2IParamTypes.VideoFrames, 17);
        }
        List<(string Name, Func<T2IModel, (string, List<Action<T2IParamInput>>)> Make, bool CheckpointOnly)> variants =
        [
            ("plain", m => (basePrompt, []), false),
            ("segment-lora", m => (segmentPrompt, []), false),
            ("refiner+segment", m => (segmentPrompt, [refiner]), false),
            ("refinermodel+segment", m => (segmentPrompt, [refiner, i => i.Set(T2IParamTypes.RefinerModel, ModelFor(m.ModelClass, m.IsDiffusionModelsFormat, "-alt"))]), false),
            ("segmentmodel", m => (segmentPrompt, [i => i.Set(T2IParamTypes.SegmentModel, ModelFor(m.ModelClass, m.IsDiffusionModelsFormat, "-alt"))]), false),
            ("applyafterbase+refiner", m => (segmentPrompt, [refiner, i => i.Set(T2IParamTypes.SegmentApplyAfter, "Base")]), false),
            ("initimage+promptimage+segment", m => (segmentPrompt, [i => i.Set(T2IParamTypes.InitImage, init), i => i.Set(T2IParamTypes.InitImageCreativity, 0.6), i => i.Set(T2IParamTypes.PromptImages, [promptImage])]), false),
            ("regions+objects", m => ($"{basePrompt} <region:0,0,0.5,1> zqxregion left side <object:0.5,0.25,0.5,0.5> zqxobject cat", []), true),
            ("initimage+mask", m => (basePrompt, [i => i.Set(T2IParamTypes.InitImage, init), i => i.Set(T2IParamTypes.MaskImage, mask), i => i.Set(T2IParamTypes.InitImageCreativity, 0.6)]), true),
            ("controlnet", m => (basePrompt, [i => i.Set(T2IParamTypes.Controlnets[0].Strength, 0.8), i => i.Set(T2IParamTypes.Controlnets[0].Image, init), i => i.Set(T2IParamTypes.Controlnets[0].Model, ControlNetFor(m.ModelClass))]), true),
            ("refiner-upscale", m => (basePrompt, [refiner, i => i.Set(T2IParamTypes.RefinerUpscale, 1.5)]), true),
            ("clear+variationseed", m => ($"{basePrompt} <clear:background>", [i => i.Set(T2IParamTypes.VariationSeed, 5L), i => i.Set(T2IParamTypes.VariationSeedStrength, 0.3)]), true),
            ("image2video", m => (basePrompt, [videoSet]), true),
            ("image2video+extend", m => ($"{basePrompt} <extend:17> zqxextend walking on", [videoSet, i => i.Set(T2IParamTypes.VideoExtendModel, video)]), true),
            ("seedvr", m => (basePrompt, [i => i.Set(ComfyUIBackendExtension.SeedVRModel, seedvr)]), true)
        ];
        string[] features = ["comfyui", "refiners", "controlnet", "endstepsearly", "seamless", "video", "variation_seed", "freeu", "yolov8", "comfy_loadimage_b64", "comfy_saveimage_ws", "comfy_saveaudio_ws", "comfy_just_load_model", "comfy_latent_blend_masked", "set_clip_device", "gguf", "nunchaku", "bnb_nf4", "teacache", "dynamic_thresholding"];
        string[] markers = ["/lora", "/vae", "/audio-vae", "/controlnet", "/control-", "/embedding", "/textual-inversion"];
        List<(string Label, Func<T2IParamInput> Make)> cases = [];
        foreach (T2IModelClass modelClass in T2IModelClassSorter.ModelClasses.Values.Where(c => !markers.Any(c.ID.Contains)).OrderBy(c => c.ID))
        {
            ControlNetFor(modelClass);
            foreach (bool diffusionModels in new[] { false, true })
            {
                T2IModel model = ModelFor(modelClass, diffusionModels);
                ModelFor(modelClass, diffusionModels, "-alt");
                foreach ((string Name, Func<T2IModel, (string, List<Action<T2IParamInput>>)> Make, bool CheckpointOnly) variant in variants.Where(v => !diffusionModels || !v.CheckpointOnly))
                {
                    cases.Add(($"{modelClass.ID} [{(diffusionModels ? "diffusion_models" : "checkpoint")}] {variant.Name}", () =>
                    {
                        (string prompt, List<Action<T2IParamInput>> setters) = variant.Make(model);
                        T2IParamInput input = new(null);
                        input.Set(T2IParamTypes.Model, model);
                        input.Set(T2IParamTypes.Prompt, prompt);
                        input.Set(T2IParamTypes.NegativePrompt, "");
                        input.Set(T2IParamTypes.Width, 1024);
                        input.Set(T2IParamTypes.Height, 1024);
                        input.Set(T2IParamTypes.Steps, 20);
                        input.Set(T2IParamTypes.CFGScale, 4.0);
                        input.Set(T2IParamTypes.Seed, 1L);
                        input.Set(T2IParamTypes.Images, 1);
                        foreach (Action<T2IParamInput> setter in setters)
                        {
                            setter(input);
                        }
                        input.ApplySpecialLogic();
                        input.ApplyLateSpecialLogic();
                        return input;
                    }
                    ));
                }
            }
        }
        string[] lines = new string[cases.Count];
        List<string> crashes = [];
        int traced = 0;
        Parallel.For(0, cases.Count, i =>
        {
            T2IParamInput input = null;
            try
            {
                input = cases[i].Make();
                WorkflowGenerator generator = new() { UserInput = input, Features = [.. features] };
                JObject workflow = generator.Generate();
                T2IParamInput sendInput = cases[i].Make();
                string sent = ComfyUIAPIAbstractBackend.CreateWorkflow(sendInput, w => w, null, [.. features]);
                if (sendInput.ExtraMeta.ContainsKey("debug_trace"))
                {
                    Interlocked.Increment(ref traced);
                }
                JObject metadata = sendInput.GenFullMetadataObject();
                ((JObject)metadata["sui_image_params"]).Remove("swarm_version");
                string meta = Utilities.HashSHA256($"{metadata["sui_image_params"]}\n{metadata["sui_extra_data"]?["unused_parameters"]}".EncodeUTF8());
                string extra = sendInput.ExtraMeta.Keys.Where(k => k != "debug_trace").Order().JoinString(",");
                if (metadata["sui_extra_data"] is JObject extraData)
                {
                    extraData.Remove("date");
                    extraData.Remove("debug_trace");
                }
                string savedText = Utilities.HashSHA256(T2IParamInput.MetadataToString(metadata).EncodeUTF8());
                lines[i] = $"{cases[i].Label}\t{Utilities.HashSHA256(MaskedText(workflow.ToString()).EncodeUTF8())}\t{Utilities.HashSHA256(MaskedText(sent).EncodeUTF8())}\t{meta}\t{extra}\t{savedText}";
            }
            catch (Exception ex)
            {
                lines[i] = $"{cases[i].Label}\tERROR {ex.GetType().Name}: {ex.Message.Replace('\n', ' ')}";
                if (ex is not SwarmReadableErrorException)
                {
                    lock (crashes)
                    {
                        crashes.Add($"{cases[i].Label}: {ex}");
                    }
                }
            }
        });
        File.WriteAllLines(outPath, lines);
        List<string[]> built = [.. lines.Where(l => !l.Contains("\tERROR ")).Select(l => l.Split('\t'))];
        TestContext.Out.WriteLine($"IDENTITY {cases.Count} cases, {cases.Count - built.Count} errors, debug={Program.ServerSettings.AddDebugData}, {traced} builds carry debug_trace, {built.Select(l => l[1]).Distinct().Count()} distinct workflow hashes, {built.Select(l => l[2]).Distinct().Count()} distinct sent hashes, {built.Select(l => l[3]).Distinct().Count()} distinct metadata hashes, {built.Select(l => l[5]).Distinct().Count()} distinct saved metadata texts, written to {outPath}");
        string costPath = Environment.GetEnvironmentVariable("IDENTITY_COST_OUT");
        if (!string.IsNullOrWhiteSpace(costPath))
        {
            List<int> builtIndexes = [.. Enumerable.Range(0, cases.Count).Where(i => !lines[i].Contains("\tERROR "))];
            List<string> costs = MeasureCosts(cases, builtIndexes.Where((_, n) => n % CostSampleStride == 0), features);
            File.WriteAllLines(costPath, costs);
            TestContext.Out.WriteLine($"COST {costs.Count} cases measured, written to {costPath}");
        }
        Assert.That(crashes, Is.Empty, $"every error is a generator rejection (SwarmReadableErrorException); first: {crashes.FirstOrDefault()}");
        Assert.That(built.Count, Is.GreaterThanOrEqualTo(cases.Count * 8 / 10), "at least 80% of the cases build, so identical dumps compare real workflows");
    }
}
