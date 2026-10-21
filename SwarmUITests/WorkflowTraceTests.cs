using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FreneticUtilities.FreneticExtensions;
using FreneticUtilities.FreneticToolkit;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SixLabors.ImageSharp.Metadata.Profiles.Exif;
using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Backends;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using static SwarmUITests.WorkflowGeneratorHarness;

namespace SwarmUITests;

/// <summary>Tests for <see cref="WorkflowTrace"/>: it must never change the generated workflow, and must account for every node, model load, and sampler chain.</summary>
public class WorkflowTraceTests : SwarmUITest
{
    /// <summary>Prepares the basics.</summary>
    [OneTimeSetUp]
    public static void PreInit()
    {
        Setup();
        Init();
    }

    /// <summary>Text unique to the test prompts, which must never appear raw in a trace.</summary>
    public const string PromptMarkerBase = "zqxbase man on a road", PromptMarkerSegment = "zqxseg green alien";

    /// <summary>Text unique to the other prompt-view parameters the fitness variants set (<see cref="WithPromptParameterMarkers"/>), which must never appear raw in a trace either.</summary>
    public const string PromptMarkerNegative = "zqxneg blurry hands", PromptMarkerStyle = "zqxstyle synthwave";

    /// <summary>Gives a request a marker in the prompt-view parameters besides the prompt that every generation can read: the negative prompt and the Text2Audio style. The unsampler prompt, which adds an unsampler pass, is marked in <see cref="TestUnsamplerHasCallRecord"/>.</summary>
    public static Request WithPromptParameterMarkers(Request request)
    {
        if (string.IsNullOrEmpty(request.NegativePrompt))
        {
            request.NegativePrompt = PromptMarkerNegative;
        }
        request.Setters.Add(i => i.Set(T2IParamTypes.Text2AudioStyle, PromptMarkerStyle));
        return request;
    }

    /// <summary>A named request variant.</summary>
    /// <param name="Name">Variant name.</param>
    /// <param name="Make">Builds the request for a given main model.</param>
    /// <param name="Feature">If true, the variant exists to reach one feature's generator step, which <see cref="TestFeatureVariantsReachTheirSteps"/> checks.</param>
    public record class Variant(string Name, Func<T2IModel, Request> Make, bool Feature = false);

    /// <summary>The option variants every model class is built with, each with a marker in every prompt-view parameter (<see cref="WithPromptParameterMarkers"/>).</summary>
    public static List<Variant> Variants()
    {
        Lora("test_alien_lora");
        string segmentPrompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment} <lora:test_alien_lora>";
        static void refiner(T2IParamInput input)
        {
            input.Set(T2IParamTypes.RefinerMethod, "PostApply");
            input.Set(T2IParamTypes.RefinerControl, 0.4);
        }
        List<Variant> variants =
        [
            new("plain", m => new() { Model = m, Prompt = PromptMarkerBase }),
            new("segment-lora", m => new() { Model = m, Prompt = segmentPrompt }),
            new("refiner+segment", m => new() { Model = m, Prompt = segmentPrompt, Setters = { refiner } }),
            new("refinermodel+segment", m => new() { Model = m, Prompt = segmentPrompt, Setters = { refiner, i => i.Set(T2IParamTypes.RefinerModel, ModelFor(m.ModelClass, m.IsDiffusionModelsFormat, "-alt")) } }),
            new("segmentmodel", m => new() { Model = m, Prompt = segmentPrompt, Setters = { i => i.Set(T2IParamTypes.SegmentModel, ModelFor(m.ModelClass, m.IsDiffusionModelsFormat, "-alt")) } }),
            new("applyafterbase+refiner", m => new() { Model = m, Prompt = segmentPrompt, Setters = { refiner, i => i.Set(T2IParamTypes.SegmentApplyAfter, "Base") } }),
            new("initimage+promptimage+segment", m => new() { Model = m, Prompt = segmentPrompt, Setters = { i => i.Set(T2IParamTypes.InitImage, InitTestImage), i => i.Set(T2IParamTypes.InitImageCreativity, 0.6), i => i.Set(T2IParamTypes.PromptImages, [PromptTestImage]) } }),
            new("regions+objects", m => new() { Model = m, Prompt = $"{PromptMarkerBase} <region:0,0,0.5,1> zqxregion left side <object:0.5,0.25,0.5,0.5> zqxobject cat" }, true),
            new("initimage+mask", m => new() { Model = m, Prompt = PromptMarkerBase, Setters = { i => i.Set(T2IParamTypes.InitImage, InitTestImage), i => i.Set(T2IParamTypes.MaskImage, MaskTestImage), i => i.Set(T2IParamTypes.InitImageCreativity, 0.6) } }, true),
            new("controlnet", m => new() { Model = m, Prompt = PromptMarkerBase, Setters = { i => i.Set(T2IParamTypes.Controlnets[0].Strength, 0.8), i => i.Set(T2IParamTypes.Controlnets[0].Image, InitTestImage), i => i.Set(T2IParamTypes.Controlnets[0].Model, ControlNetFor(m.ModelClass)) } }, true),
            new("refiner-upscale", m => new() { Model = m, Prompt = PromptMarkerBase, Setters = { refiner, i => i.Set(T2IParamTypes.RefinerUpscale, 1.5) } }, true),
            new("clear+variationseed", m => new() { Model = m, Prompt = $"{PromptMarkerBase} <clear:background>", Setters = { i => i.Set(T2IParamTypes.VariationSeed, 5L), i => i.Set(T2IParamTypes.VariationSeedStrength, 0.3) } }, true),
            new("image2video", m => new() { Model = m, Prompt = PromptMarkerBase, Setters = { video } }, true),
            new("image2video+extend", m => new() { Model = m, Prompt = $"{PromptMarkerBase} <extend:17> zqxextend walking on", Setters = { video, i => i.Set(T2IParamTypes.VideoExtendModel, VideoTestModel) } }, true),
            new("image2video+swap", m => new() { Model = m, Prompt = PromptMarkerBase, Setters = { video, i => i.Set(T2IParamTypes.VideoSwapModel, VideoSwapTestModel) } }, true),
            new("seedvr", m => new() { Model = m, Prompt = PromptMarkerBase, Setters = { i => i.Set(ComfyUIBackendExtension.SeedVRModel, SeedVRTestModel) } }, true)
        ];
        return [.. variants.Select(v => v with { Make = m => WithPromptParameterMarkers(v.Make(m)) })];
        static void video(T2IParamInput input)
        {
            input.Set(T2IParamTypes.VideoModel, VideoTestModel);
            input.Set(T2IParamTypes.VideoFrames, 17);
        }
    }

    /// <summary>Shared test images, built once.</summary>
    public static readonly Image InitTestImage = TestImage(512, 384), PromptTestImage = TestImage(256, 256), MaskTestImage = TestImage(512, 384);

    /// <summary>Image-to-video model used by the video variants.</summary>
    public static T2IModel VideoTestModel => ModelFor(T2IModelClassSorter.ModelClasses["wan-2_1-image2video-14b"], true);

    /// <summary>Video swap model used by the video swap variant: a different compat class from <see cref="VideoTestModel"/>, so the swap stage's records show whose class they carry.</summary>
    public static T2IModel VideoSwapTestModel => ModelFor(T2IModelClassSorter.ModelClasses["wan-2_1-image2video-1_3b"], true);

    /// <summary>SeedVR2 model used by the SeedVR variant.</summary>
    public static T2IModel SeedVRTestModel => ModelFor(T2IModelClassSorter.ModelClasses["seedvr2"], true);

    /// <summary>Returns a fake ControlNet for a main model class: its compat class's ControlNet class if one exists, otherwise an unclassed ControlNet.</summary>
    public static T2IModel ControlNetFor(T2IModelClass modelClass)
    {
        T2IModelClass controlClass = T2IModelClassSorter.ModelClasses.Values.Where(c => c.ID.EndsWith("/controlnet") && c.CompatClass?.ID == modelClass.CompatClass?.ID).OrderBy(c => c.ID).FirstOrDefault();
        string name = $"test/cn/{(controlClass?.ID ?? "generic").Replace('/', '_')}.safetensors";
        if (Program.T2IModelSets["ControlNet"].Models.TryGetValue(name, out T2IModel existing))
        {
            return existing;
        }
        return AddModel("ControlNet", name, controlClass);
    }

    /// <summary>Largest EXIF profile that fits one JPEG APP1 segment (ImageSharp splits anything larger across several, which most readers do not join).</summary>
    public const int MaxSingleApp1ExifBytes = 65527;

    /// <summary>Returns the size of the EXIF profile Swarm writes for this metadata text in JPEG and WEBP images and metadata thumbnails.</summary>
    public static int ExifBytes(string metadata)
    {
        ExifProfile profile = new();
        profile.SetValue(ExifTag.UserComment, metadata);
        return profile.ToByteArray().Length;
    }

    /// <summary>Runs what a ComfyUI generation writes into an image's metadata after the trace finishes, in order: the backend posts the workflow (<see cref="WorkflowTrace.RecordPosted"/>) and adds its debug data with the prompt when the output arrives (ComfyUIAPIAbstractBackend.AwaitJobLive),
    /// then T2IEngine copies the input, adds the generation times and fires <see cref="T2IEngine.PostGenerateEvent"/> with the output file, where <see cref="WorkflowTrace.FitRecordToSave"/> refits a tiered record. Returns the copy the saved file's metadata is generated from.</summary>
    /// <param name="input">The input the workflow was created for.</param>
    /// <param name="workflow">The workflow text the backend posts.</param>
    /// <param name="output">The output file, as the backend returns it; null for <see cref="InitTestImage"/>.</param>
    public static T2IParamInput SaveInput(T2IParamInput input, string workflow, MediaFile output = null)
    {
        WorkflowTrace.RecordPosted(input, workflow, nameof(ComfyUISelfStartBackend));
        input.ExtraMeta["debug_backend"] = new JObject()
        {
            ["backend_type"] = nameof(ComfyUISelfStartBackend),
            ["backend_id"] = 12,
            ["debug_internal_prompt"] = input.Get(T2IParamTypes.Prompt),
            ["backend_usages"] = 1234,
            ["comfy_output_node"] = "9",
            ["comfy_is_real"] = true,
            ["comfy_img_type"] = $"{MediaType.ImagePng}",
            ["comfy_event_id"] = 1,
            ["comfy_index"] = 0
        };
        T2IParamInput copy = input.Clone();
        copy.ExtraMeta["prep_time"] = "1.23 min";
        copy.ExtraMeta["generation_time"] = "45.67 sec";
        T2IEngine.PostGenerateEvent?.Invoke(new(output ?? InitTestImage, copy, () => { }, "9"));
        return copy;
    }

    /// <summary>Checks the trace invariants for one traced build, appending violations to the list: every node has provenance, every removal is attributed, every sampler has a call record, the image record holds every sampler, a JPEG save's metadata (<see cref="SaveInput"/>) fits one EXIF segment, and no raw prompt text, untracked node or dangling link appears.
    /// Returns the stored image record's metadata length, and the record tier a JPEG save stores (0 is the most complete).</summary>
    public static (int RecordChars, int JpegTier) CheckTrace(Build build, string label, List<string> failures)
    {
        WorkflowTrace trace = build.Trace;
        foreach (JProperty property in build.Workflow.Properties())
        {
            string classType = $"{property.Value["class_type"]}";
            if (!trace.Nodes.TryGetValue(property.Name, out WorkflowTrace.NodeRecord record))
            {
                failures.Add($"{label}: node {property.Name}:{classType} has no provenance record");
                continue;
            }
            if (record.ClassType != classType)
            {
                failures.Add($"{label}: node {property.Name} recorded as {record.ClassType} but is {classType}");
            }
            if (record.Step < 0 || record.Step >= trace.StepRecords.Count || string.IsNullOrEmpty(trace.StepRecords[record.Step].Name) || string.IsNullOrEmpty(record.Origin) || !record.Origin.Contains('@') || string.IsNullOrEmpty(record.Stage))
            {
                failures.Add($"{label}: node {property.Name}:{classType} provenance incomplete (step={record.Step}, origin={record.Origin}, stage={record.Stage})");
            }
            if (record.RemovedBy is not null)
            {
                failures.Add($"{label}: node {property.Name}:{classType} is present but recorded as removed by {record.RemovedBy}");
            }
        }
        foreach (WorkflowTrace.NodeRecord record in trace.Nodes.Values)
        {
            if (!build.Workflow.ContainsKey(record.Id) && record.RemovedBy is null)
            {
                failures.Add($"{label}: node {record.Id}:{record.ClassType} is gone from the workflow but no step is recorded as removing it");
            }
        }
        foreach (WorkflowTrace.ModelLoadRecord load in trace.Loads)
        {
            if (!load.Finished)
            {
                failures.Add($"{label}: model load #{load.Index} ({load.Type} {load.Model}) never finished");
            }
            if (load.ModelOut is null)
            {
                failures.Add($"{label}: model load #{load.Index} ({load.Type} {load.Model}) has no model output recorded");
            }
        }
        foreach (WorkflowTrace.EditRecord edit in trace.Edits.Where(e => e.Step < 0))
        {
            failures.Add($"{label}: {trace.EditJson(edit).ToString(Formatting.None)}: an edit made outside any generator step");
        }
        if (trace.StepStack.Count != 0 || trace.LoadStack.Count != 0 || trace.Stages.Count != 0)
        {
            failures.Add($"{label}: unbalanced trace scopes (steps={trace.StepStack.Count}, loads={trace.LoadStack.Count}, stages={trace.Stages.Count})");
        }
        if (build.Input.ExtraMeta.GetValueOrDefault(WorkflowTrace.MetadataKey) is not JObject stored)
        {
            failures.Add($"{label}: no {WorkflowTrace.MetadataKey} in ExtraMeta");
            return (0, -1);
        }
        string format = WorkflowTrace.ImageFormatOf(build.Input);
        if (format != "PNG")
        {
            failures.Add($"{label}: the harness saves as {format}, the checks below expect the PNG default");
        }
        JObject summary = trace.Build();
        if (summary["error"] is not null || stored["trace_error"] is not null)
        {
            failures.Add($"{label}: the trace recorded an error: {summary["error"]}{stored["trace_error"]}");
        }
        if ($"{summary["workflow"]["sha256"]}" != Utilities.HashSHA256(build.WorkflowText.EncodeUTF8()))
        {
            failures.Add($"{label}: the summary's workflow sha256 is not the hash of the workflow text");
        }
        List<JObject> tiers = trace.RecordTiers(summary);
        if (!JToken.DeepEquals(stored, tiers[0]))
        {
            failures.Add($"{label}: the record stored for a PNG image is not the first record tier");
        }
        int storedLength = WorkflowTrace.RecordMetadataChars(stored);
        int samplerNodes = build.Workflow.Properties().Count(p => WorkflowTrace.SamplerClasses.Contains($"{p.Value["class_type"]}"));
        T2IParamInput jpegInput = build.Input.Clone();
        jpegInput.Set(T2IParamTypes.ImageFormat, "JPG");
        jpegInput.ExtraMeta[WorkflowTrace.MetadataKey] = WorkflowTrace.PickTier(tiers, jpegInput, "JPG");
        T2IParamInput saved = SaveInput(jpegInput, build.WorkflowText);
        JObject jpeg = (JObject)saved.ExtraMeta[WorkflowTrace.MetadataKey];
        int jpegTier = tiers.FindIndex(t => ReferenceEquals(t, jpeg));
        int exifBytes = ExifBytes(saved.GenRawMetadata());
        if (exifBytes > MaxSingleApp1ExifBytes)
        {
            failures.Add($"{label}: JPEG image metadata EXIF is {exifBytes} bytes, over one APP1 segment ({MaxSingleApp1ExifBytes})");
        }
        foreach ((string name, JObject record) in new[] { ("PNG", stored), ("JPG", jpeg) })
        {
            int samplerEntries = (record["samplers"] as JArray)?.Count(s => s["removed_by"] is null) ?? 0;
            if (samplerEntries < samplerNodes)
            {
                failures.Add($"{label}: {samplerNodes} sampler nodes but the {name} image record has {samplerEntries} sampler records");
            }
            if (record.ToString(Formatting.None).Contains("zqx"))
            {
                failures.Add($"{label}: raw prompt text appears in the {name} image record");
            }
        }
        foreach (JObject sampler in (stored["samplers"] as JArray ?? []).OfType<JObject>())
        {
            foreach (string id in (sampler["inputs"] as JObject ?? []).Properties().SelectMany(p => p.Value.Select(v => $"{v}")))
            {
                if (stored["nodes"]?[id] is null)
                {
                    failures.Add($"{label}: sampler {sampler["node"]} chain node {id} has no entry in the image record's node table");
                }
            }
        }
        foreach (JProperty property in build.Workflow.Properties())
        {
            JObject entry = stored["nodes"]?[property.Name] as JObject;
            if (entry is null || $"{entry["class"]}" != $"{property.Value["class_type"]}" || string.IsNullOrEmpty($"{entry["stage"]}") || string.IsNullOrEmpty($"{entry["origin"]}") || string.IsNullOrEmpty($"{entry["step"]}"))
            {
                failures.Add($"{label}: node {property.Name}:{property.Value["class_type"]} has no entry with its class, stage and provenance in the image record: {entry?.ToString(Formatting.None) ?? "none"}");
            }
        }
        if ((stored["steps"] as JArray)?.Count != trace.StepRecords.Count)
        {
            failures.Add($"{label}: {trace.StepRecords.Count} steps ran but the image record lists {(stored["steps"] as JArray)?.Count ?? 0}");
        }
        if ((stored["params_read"] as JObject)?.Count != trace.ParamReads.Count)
        {
            failures.Add($"{label}: {trace.ParamReads.Count} parameters were read but the image record lists {(stored["params_read"] as JObject)?.Count ?? 0}");
        }
        if (summary.ToString(Formatting.None).Contains("zqx"))
        {
            failures.Add($"{label}: raw prompt text appears in the full trace");
        }
        if (trace.BuildDetail().ToString(Formatting.None).Contains("zqx"))
        {
            failures.Add($"{label}: raw prompt text appears in the log detail");
        }
        if (summary["untracked_nodes"] is not null)
        {
            failures.Add($"{label}: summary lists untracked nodes {summary["untracked_nodes"].ToString(Formatting.None)}");
        }
        if (summary["dangling_links"] is not null)
        {
            failures.Add($"{label}: summary lists dangling links {summary["dangling_links"].ToString(Formatting.None)}");
        }
        int listed = (summary["nodes"] as JObject)?.Count ?? 0;
        if (listed != build.Workflow.Count)
        {
            failures.Add($"{label}: summary lists {listed} nodes but the workflow has {build.Workflow.Count}");
        }
        foreach (JProperty property in build.Workflow.Properties().Where(p => WorkflowTrace.SamplerClasses.Contains($"{p.Value["class_type"]}")))
        {
            if (!trace.SamplerCalls.ContainsKey(property.Name))
            {
                failures.Add($"{label}: sampler {property.Name}:{property.Value["class_type"]} has no call record");
            }
        }
        return (storedLength, jpegTier);
    }

    /// <summary>Outcome of one fitness case.</summary>
    /// <param name="Label">Case label.</param>
    /// <param name="Rejected">The generator's readable rejection of the untraced build, if any.</param>
    /// <param name="Crash">A non-readable exception from the untraced build, if any.</param>
    /// <param name="Failures">Trace invariant violations.</param>
    /// <param name="PlainTicks">Untraced build time.</param>
    /// <param name="TracedTicks">Traced build time.</param>
    /// <param name="RecordChars">Stored image record metadata length.</param>
    /// <param name="JpegTier">Record tier a JPEG save stores, 0 being the most complete.</param>
    /// <param name="EditSteps">Name of the step behind each in-place edit the trace recorded.</param>
    public record class CaseOutcome(string Label, string Rejected, string Crash, List<string> Failures, long PlainTicks, long TracedTicks, int RecordChars = 0, int JpegTier = 0, List<string> EditSteps = null);

    /// <summary>For a request the untraced build throws on, checks that tracing changed nothing about the failure: same exception, same partial workflow, same input state, no trace stored in metadata, tracker detached, error recorded, and the partial trace still builds.</summary>
    public static void CheckFailureIsNoOp(Attempt plain, Attempt traced, string label, List<string> failures)
    {
        if (traced.Error is null)
        {
            failures.Add($"{label}: untraced build threw {plain.Error.GetType().Name} but traced build succeeded");
            return;
        }
        if (traced.Error.GetType() != plain.Error.GetType() || traced.Error.Message != plain.Error.Message)
        {
            failures.Add($"{label}: untraced build threw {plain.Error.GetType().Name}: {plain.Error.Message}, traced build threw {traced.Error.GetType().Name}: {traced.Error.Message}");
        }
        if (plain.Generator is null || traced.Generator is null)
        {
            if (plain.Generator is not null || traced.Generator is not null)
            {
                failures.Add($"{label}: only one build failed before generation started");
            }
            return;
        }
        if (ComparableText(traced.Generator.Workflow) != ComparableText(plain.Generator.Workflow))
        {
            failures.Add($"{label}: tracing changed the partial workflow at the point of failure");
        }
        string plainMeta = plain.Input.ExtraMeta.Keys.Order().JoinString(","), tracedMeta = traced.Input.ExtraMeta.Keys.Order().JoinString(",");
        if (plainMeta != tracedMeta)
        {
            failures.Add($"{label}: failed build ExtraMeta keys differ: untraced [{plainMeta}], traced [{tracedMeta}]");
        }
        if (!traced.Input.ParamsQueried.SetEquals(plain.Input.ParamsQueried))
        {
            failures.Add($"{label}: failed build ParamsQueried differ: traced-only [{traced.Input.ParamsQueried.Except(plain.Input.ParamsQueried).JoinString(",")}], untraced-only [{plain.Input.ParamsQueried.Except(traced.Input.ParamsQueried).JoinString(",")}]");
        }
        if (traced.Input.ParamReadTracker is not null)
        {
            failures.Add($"{label}: parameter read tracker still attached after a failed build");
        }
        WorkflowTrace trace = traced.Generator.Trace;
        if (trace.Error is null || !trace.Error.StartsWith($"{traced.Error.GetType().Name}: "))
        {
            failures.Add($"{label}: trace did not record the failure (Error = {trace.Error ?? "null"})");
        }
        try
        {
            trace.Build();
            trace.BuildDetail();
        }
        catch (Exception ex)
        {
            failures.Add($"{label}: partial trace of a failed build cannot be built: {ex}");
        }
    }

    /// <summary>Builds one request untraced and traced, and checks the traced build, or that tracing is a no-op on the failure path when the generator rejects the request.</summary>
    public static CaseOutcome RunCase(string label, Func<Request> make)
    {
        List<string> failures = [];
        long start = Stopwatch.GetTimestamp();
        Attempt plainAttempt = TryGenerate(make(), false);
        long plainTicks = Stopwatch.GetTimestamp() - start;
        start = Stopwatch.GetTimestamp();
        Attempt tracedAttempt = TryGenerate(make(), true);
        long tracedTicks = Stopwatch.GetTimestamp() - start;
        if (plainAttempt.Error is not null)
        {
            CheckFailureIsNoOp(plainAttempt, tracedAttempt, label, failures);
            bool readable = plainAttempt.Error is SwarmReadableErrorException;
            return new(label, readable ? plainAttempt.Error.Message : null, readable ? null : $"{plainAttempt.Error.GetType().Name}: {plainAttempt.Error.Message}", failures, plainTicks, tracedTicks);
        }
        if (tracedAttempt.Error is not null)
        {
            failures.Add($"{label}: untraced build succeeded but traced build threw {tracedAttempt.Error}");
            return new(label, null, null, failures, plainTicks, tracedTicks);
        }
        Build plain = new(plainAttempt.Input, plainAttempt.Generator.Workflow, plainAttempt.Generator.Workflow.ToString(), null);
        Build traced = new(tracedAttempt.Input, tracedAttempt.Generator.Workflow, tracedAttempt.Generator.Workflow.ToString(), tracedAttempt.Generator.Trace);
        if (!SameWorkflow(traced, plain))
        {
            failures.Add($"{label}: tracing changed the workflow");
        }
        (int recordChars, int jpegTier) = CheckTrace(traced, label, failures);
        return new(label, null, null, failures, plainTicks, tracedTicks, recordChars, jpegTier, [.. traced.Trace.Edits.Select(e => traced.Trace.StepName(e.Step))]);
    }

    /// <summary>The fitness function: every main model class, in both checkpoint and diffusion_models formats, across every option variant.
    /// Tracing must leave the workflow byte-identical, and the trace must account for every node and model load.
    /// Cases run in parallel, as concurrent requests do in production; models are registered before the parallel section.</summary>
    [Test]
    public void TestEveryModelClassAndVariantIsTraced()
    {
        Stopwatch timer = Stopwatch.StartNew();
        List<Variant> variants = Variants();
        List<(string, Func<Request>)> cases = [];
        _ = VideoTestModel;
        _ = SeedVRTestModel;
        foreach (T2IModelClass modelClass in MainModelClasses())
        {
            ControlNetFor(modelClass);
            foreach (bool diffusionModels in new[] { false, true })
            {
                T2IModel model = ModelFor(modelClass, diffusionModels);
                ModelFor(modelClass, diffusionModels, "-alt");
                foreach (Variant variant in variants)
                {
                    cases.Add(($"{modelClass.ID} [{(diffusionModels ? "diffusion_models" : "checkpoint")}] {variant.Name}", () => variant.Make(model)));
                }
            }
        }
        CaseOutcome[] outcomes = new CaseOutcome[cases.Count];
        Parallel.For(0, cases.Count, i => outcomes[i] = RunCase(cases[i].Item1, cases[i].Item2));
        timer.Stop();
        List<string> failures = [.. outcomes.SelectMany(o => o.Failures)];
        List<CaseOutcome> rejected = [.. outcomes.Where(o => o.Rejected is not null)];
        List<CaseOutcome> upstreamCrashes = [.. outcomes.Where(o => o.Crash is not null)];
        int built = outcomes.Count(o => o.Rejected is null && o.Crash is null);
        static long ms(long ticks) => ticks * 1000 / Stopwatch.Frequency;
        TestContext.Out.WriteLine($"{cases.Count} cases in {timer.ElapsedMilliseconds} ms wall: {built} traced and checked, {rejected.Count} rejected by the generator ({rejected.Count(o => o.Failures.Count == 0)} confirmed identical with tracing on: same exception, partial workflow, input state), {upstreamCrashes.Count} upstream crashes ({upstreamCrashes.Count(o => o.Failures.Count == 0)} confirmed identical). Summed build time: untraced {ms(outcomes.Sum(o => o.PlainTicks))} ms, traced {ms(outcomes.Sum(o => o.TracedTicks))} ms.");
        List<CaseOutcome> checkedOutcomes = [.. outcomes.Where(o => o.Rejected is null && o.Crash is null).OrderBy(o => o.RecordChars)];
        if (checkedOutcomes.Count > 0)
        {
            CaseOutcome largest = checkedOutcomes[^1];
            TestContext.Out.WriteLine($"Image record size (PNG, tier 0): median {checkedOutcomes[checkedOutcomes.Count / 2].RecordChars} chars, max {largest.RecordChars} chars ({largest.Label}). JPEG record tier (limit {MetadataHelper.MaxJpegExifChars} chars of metadata): {checkedOutcomes.GroupBy(o => o.JpegTier).OrderBy(g => g.Key).Select(g => $"tier {g.Key} x{g.Count()}").JoinString(", ")}.");
            List<string> editSteps = [.. checkedOutcomes.SelectMany(o => o.EditSteps)];
            TestContext.Out.WriteLine($"In-place edits recorded: {editSteps.Count} in {checkedOutcomes.Count(o => o.EditSteps.Count > 0)} builds; by step: {editSteps.GroupBy(s => s).OrderByDescending(g => g.Count()).Select(g => $"{g.Key} x{g.Count()}").JoinString(", ")}");
        }
        foreach (CaseOutcome crash in upstreamCrashes)
        {
            TestContext.Out.WriteLine($"UPSTREAM CRASH (untraced build): {crash.Label}: {crash.Crash}");
        }
        foreach (IGrouping<string, CaseOutcome> reason in rejected.GroupBy(o => o.Rejected.Replace(o.Label.Before(' ').Replace('/', '_'), "{class}").Replace(o.Label.Before(' '), "{class}")))
        {
            TestContext.Out.WriteLine($"REJECTED x{reason.Count()}: {reason.Key}\n    {reason.Select(o => o.Label).JoinString("\n    ")}");
        }
        List<string> neverTraced = [.. MainModelClasses().Select(c => c.ID).Where(id => !outcomes.Any(o => o.Rejected is null && o.Crash is null && o.Label.StartsWith($"{id} [")))];
        Assert.That(built, Is.GreaterThan(0));
        Assert.That(neverTraced, Is.Empty, "every registered main model class builds and is traced in at least one case");
        Assert.That(failures, Is.Empty, string.Join("\n", failures.Take(50)));
    }

    /// <summary>The bare-ComfyUI path (no Swarm nodes) is traced too.</summary>
    [Test]
    public void TestBareComfyFeaturesAreTraced()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        List<string> failures = [];
        foreach (Variant variant in Variants())
        {
            Request plainRequest = variant.Make(model);
            plainRequest.Features = BareFeatures;
            Request tracedRequest = variant.Make(model);
            tracedRequest.Features = BareFeatures;
            Build plain = Generate(plainRequest, false);
            Build traced = Generate(tracedRequest, true);
            Assert.That(ComparableText(traced.Workflow), Is.EqualTo(ComparableText(plain.Workflow)), variant.Name);
            CheckTrace(traced, $"bare {variant.Name}", failures);
            if (traced.Trace.Build()["media"] is JObject media && media.Properties().Any(p => p.Value["sha8"] is null && (p.Value["raw_sha8"]?.Type ?? JTokenType.Null) == JTokenType.Null))
            {
                failures.Add($"bare {variant.Name}: a media record has no hash: {media.ToString(Formatting.None)}");
            }
        }
        Assert.That(failures, Is.Empty, string.Join("\n", failures));
    }

    /// <summary>Returns the node a link points at.</summary>
    public static (string, JObject) Source(JObject workflow, JToken link)
    {
        string id = $"{link[0]}";
        return (id, workflow[id] as JObject);
    }

    /// <summary>Walks a node's 'model' inputs back to the loader, returning (id, node) pairs nearest first.</summary>
    public static List<(string, JObject)> ModelChain(JObject workflow, JToken link)
    {
        List<(string, JObject)> chain = [];
        while (link is JArray)
        {
            (string id, JObject node) = Source(workflow, link);
            chain.Add((id, node));
            link = node["inputs"]?["model"];
        }
        return chain;
    }

    /// <summary>Returns the node IDs of an image record sampler's input chain.</summary>
    public static List<string> ChainIds(JObject sampler, string input) => [.. (sampler["inputs"]?[input] as JArray ?? []).Select(id => $"{id}")];

    /// <summary>Qwen Image 2.1 with a segment carrying a confined LoRA: the trace must report what the graph actually contains, checked against the graph by an independent walk.</summary>
    [Test]
    public void TestQwen21SegmentTraceMatchesGraph()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);
        Lora("test_alien_lora");
        Request request = new() { Model = model, Prompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment} <lora:test_alien_lora>" };
        Build traced = Generate(request, true);
        JObject workflow = traced.Workflow;
        WorkflowTrace trace = traced.Trace;
        PromptRegion.Part part = new PromptRegion(traced.Input.Get(T2IParamTypes.Prompt)).Parts.Single(p => p.Type == PromptRegion.PartType.Segment);
        Assert.That(part.ContextID, Is.EqualTo(11));
        string stage = $"seg{part.ContextID}";
        // Independent graph facts.
        (string samplerId, JObject sampler) = workflow.Properties().Where(p => WorkflowTrace.SamplerClasses.Contains($"{p.Value["class_type"]}") && trace.Nodes[p.Name].Stage == stage).Select(p => (p.Name, (JObject)p.Value)).Single();
        JObject inputs = (JObject)sampler["inputs"];
        List<(string, JObject)> modelChain = ModelChain(workflow, inputs["model"]);
        (string loraId, JObject loraNode) = modelChain.Single(n => $"{n.Item2["class_type"]}".StartsWith("LoraLoader"));
        Assert.That($"{loraNode["class_type"]}", Is.EqualTo("LoraLoaderModelOnly"), "Qwen 2.1 LoRAs do not target the text encoder");
        Assert.That($"{loraNode["inputs"]["lora_name"]}", Is.EqualTo("test_alien_lora.safetensors"));
        Assert.That($"{modelChain[^1].Item2["class_type"]}", Is.EqualTo("UNETLoader"));
        (string posId, JObject posNode) = Source(workflow, inputs["positive"]);
        Assert.That($"{posNode["class_type"]}", Is.EqualTo("ReferenceLatent"));
        (string encId, JObject encNode) = Source(workflow, posNode["inputs"]["conditioning"]);
        Assert.That($"{encNode["class_type"]}", Is.EqualTo("SwarmTextEncodeAdvanced"));
        Assert.That($"{encNode["inputs"]["prompt"]}", Is.EqualTo(part.Prompt.Trim()), "the segment prompt reaches the encoder unchanged apart from Swarm tag processing");
        Assert.That(encNode["inputs"]["images"]?.Type ?? JTokenType.Null, Is.EqualTo(JTokenType.Null), "no Prompt Images were given, so the encoder gets no images");
        (string refEncodeId, JObject refEncode) = Source(workflow, posNode["inputs"]["latent"]);
        (string refPixelsId, JObject refPixels) = Source(workflow, refEncode["inputs"]["pixels"]);
        Assert.That($"{refPixels["class_type"]}", Is.EqualTo("SwarmImageScaleForMP"), "the segment's reference latent is the scaled crop");
        int steps = inputs["steps"].Value<int>();
        Assert.That(inputs["start_at_step"].Value<int>(), Is.EqualTo((int)Math.Round(steps * (1 - part.Strength2))));
        // The trace must report the same facts.
        Assert.That(trace.Nodes[loraId].Stage, Is.EqualTo(stage));
        Assert.That(trace.Loras.Any(l => l.Lora == "test_alien_lora" && l.Section == part.ContextID && l.Node == $"{loraId}:LoraLoaderModelOnly" && l.Stage == stage), Is.True, string.Join("\n", trace.Loras));
        JObject call = trace.SamplerCalls[samplerId];
        Assert.That(call["start_step"].Value<int>(), Is.EqualTo(inputs["start_at_step"].Value<int>()));
        Assert.That(call["section"].Value<int>(), Is.EqualTo(part.ContextID));
        Assert.That(call["stage"].Value<string>(), Is.EqualTo(stage));
        JObject summary = (JObject)traced.Input.ExtraMeta[WorkflowTrace.MetadataKey];
        JObject nodes = (JObject)summary["nodes"];
        JObject samplerEntry = (JObject)((JArray)summary["samplers"]).Single(s => $"{s["node"]}" == $"{samplerId}:{sampler["class_type"]}");
        Assert.That(ChainIds(samplerEntry, "model"), Does.Contain(loraId));
        Assert.That(($"{nodes[loraId]["class"]}", $"{nodes[loraId]["inputs"]["lora_name"]}", $"{nodes[loraId]["stage"]}"), Is.EqualTo(("LoraLoaderModelOnly", "test_alien_lora.safetensors", stage)));
        List<string> positiveChain = ChainIds(samplerEntry, "positive");
        Assert.That(positiveChain, Does.Contain(encId).And.Contain(refPixelsId));
        Assert.That(($"{nodes[encId]["class"]}", $"{nodes[encId]["inputs"]["prompt"]}"), Is.EqualTo(("SwarmTextEncodeAdvanced", $"@{stage}.pos#{MetadataHelper.Sha8(part.Prompt.Trim())}")));
        Assert.That($"{nodes[refPixelsId]["class"]}", Is.EqualTo("SwarmImageScaleForMP"));
        JObject segmentEntry = (JObject)((JArray)summary["segments"]).Single();
        Assert.That($"{segmentEntry["prompt"]}", Is.EqualTo($"@{stage}.pos#{MetadataHelper.Sha8($"{encNode["inputs"]["prompt"]}")}"), "the segment record names the text its encoder node gets");
        Assert.That(summary["prompts"][$"{segmentEntry["prompt"]}"]?.Value<int>(), Is.EqualTo($"{encNode["inputs"]["prompt"]}".Length), "the segment prompt is listed with its length");
        Assert.That(summary.ToString(Formatting.None), Does.Not.Contain("zqx"));
    }

    /// <summary>Returns the nodes upstream of a link, nearest first, recursing only through nodes of the given stage: the nodes a sampler input chain in the trace must list.</summary>
    public static List<(string, JObject)> StageWalk(JObject workflow, JToken link, WorkflowTrace trace, string stage, bool modelOnly)
    {
        List<(string, JObject)> found = [];
        HashSet<string> seen = [];
        Queue<JToken> queue = new([link]);
        while (queue.Count > 0)
        {
            (string id, JObject node) = Source(workflow, queue.Dequeue());
            if (node is null || !seen.Add(id))
            {
                continue;
            }
            found.Add((id, node));
            if (trace.Nodes.GetValueOrDefault(id)?.Stage != stage || WorkflowTrace.SamplerClasses.Contains($"{node["class_type"]}"))
            {
                continue;
            }
            foreach (JProperty input in ((JObject)node["inputs"]).Properties().Where(p => WorkflowTrace.IsLink(p.Value) && (!modelOnly || p.Name == "model")))
            {
                queue.Enqueue(input.Value);
            }
        }
        return found;
    }

    /// <summary>The segment record of every main model class that builds a segment with a confined LoRA, checked against an independent walk of its graph: the segment sampler's section,
    /// the nodes its model and positive chains list, the LoRA confined to the segment's section, and the text the segment's prompt record names.
    /// The classes with model-specific segment conditioning must build. Prints each class's chain signature: the records differ across model families.</summary>
    [Test]
    public void TestSegmentRecordsMatchTheGraphForEveryModelClass()
    {
        Lora("test_alien_lora");
        string prompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment} <lora:test_alien_lora>";
        string[] required = ["qwen-image-2.1", "qwen-image-edit", "qwen-image-edit-plus", "Flux.1-dev/kontext", "krea-2"];
        List<string> failures = [];
        Dictionary<string, List<string>> signatures = [];
        HashSet<string> builtClasses = [];
        int built = 0, rejected = 0;
        foreach (T2IModelClass modelClass in MainModelClasses())
        {
            foreach (bool diffusionModels in new[] { false, true })
            {
                string label = $"{modelClass.ID} [{(diffusionModels ? "diffusion_models" : "checkpoint")}]";
                Attempt attempt = TryGenerate(new() { Model = ModelFor(modelClass, diffusionModels), Prompt = prompt }, true);
                if (attempt.Error is not null)
                {
                    rejected++;
                    continue;
                }
                built++;
                builtClasses.Add(modelClass.ID);
                PromptRegion.Part part = new PromptRegion(attempt.Input.Get(T2IParamTypes.Prompt)).Parts.Single(p => p.Type == PromptRegion.PartType.Segment);
                string stage = $"seg{part.ContextID}";
                JObject workflow = attempt.Generator.Workflow;
                WorkflowTrace trace = attempt.Generator.Trace;
                JObject summary = (JObject)attempt.Input.ExtraMeta[WorkflowTrace.MetadataKey];
                List<string> samplerIds = [.. workflow.Properties().Where(p => WorkflowTrace.SamplerClasses.Contains($"{p.Value["class_type"]}") && trace.Nodes[p.Name].Stage == stage).Select(p => p.Name)];
                if (samplerIds.Count == 0)
                {
                    failures.Add($"{label}: no sampler at stage {stage}");
                    continue;
                }
                foreach (string samplerId in samplerIds)
                {
                    JObject inputs = (JObject)workflow[samplerId]["inputs"];
                    if (!trace.SamplerCalls.TryGetValue(samplerId, out JObject call) || call["section"]?.Value<int>() != part.ContextID)
                    {
                        failures.Add($"{label}: sampler {samplerId} has no CreateKSampler record for section {part.ContextID}: {call?.ToString(Formatting.None) ?? "none"}");
                    }
                    JObject entry = ((JArray)summary["samplers"]).OfType<JObject>().FirstOrDefault(s => $"{s["node"]}".StartsWith($"{samplerId}:"));
                    if (entry is null)
                    {
                        failures.Add($"{label}: sampler {samplerId} has no summary record");
                        continue;
                    }
                    // Guider samplers (SamplerCustomAdvanced) take the model and conditioning through their guider.
                    JToken guider = inputs["guider"];
                    JToken modelLink = inputs["model"] ?? (WorkflowTrace.IsLink(guider) ? workflow[$"{guider[0]}"]?["inputs"]?["model"] : null);
                    JToken positiveLink = inputs["positive"] ?? guider;
                    List<(string, JObject)> modelWalk = WorkflowTrace.IsLink(modelLink) ? StageWalk(workflow, modelLink, trace, stage, true) : [];
                    List<(string, JObject)> positiveWalk = WorkflowTrace.IsLink(positiveLink) ? StageWalk(workflow, positiveLink, trace, stage, false) : [];
                    foreach (JProperty input in inputs.Properties().Where(p => WorkflowTrace.IsLink(p.Value)))
                    {
                        if (workflow[$"{input.Value[0]}"] is null)
                        {
                            failures.Add($"{label}: sampler {samplerId} input '{input.Name}' links to node {input.Value[0]}, which is not in the workflow: {workflow[samplerId].ToString(Formatting.None)}");
                        }
                        List<string> listed = ChainIds(entry, input.Name);
                        List<(string, JObject)> walk = StageWalk(workflow, input.Value, trace, stage, false);
                        foreach ((string id, JObject node) in walk.Take(WorkflowTrace.MaxChainNodes))
                        {
                            if (!listed.Contains(id) || $"{summary["nodes"][id]?["class"]}" != $"{node["class_type"]}")
                            {
                                failures.Add($"{label}: sampler {samplerId} input '{input.Name}' reaches {id}:{node["class_type"]}, which its trace chain does not list with that class");
                            }
                        }
                    }
                    if (modelWalk.Count == 0 || positiveWalk.Count == 0)
                    {
                        failures.Add($"{label}: sampler {samplerId} has no model or conditioning input: {workflow[samplerId].ToString(Formatting.None)}");
                    }
                    if (!trace.Loras.Any(l => l.Lora == "test_alien_lora" && l.Section == part.ContextID && l.Stage == stage))
                    {
                        failures.Add($"{label}: no LoRA application recorded for section {part.ContextID} at {stage}: [{trace.Loras.JoinString("; ")}]");
                    }
                    foreach ((string loraId, JObject loraNode) in modelWalk.Where(n => $"{n.Item2["inputs"]?["lora_name"]}" == "test_alien_lora.safetensors"))
                    {
                        if (!trace.Loras.Any(l => l.Node == $"{loraId}:{loraNode["class_type"]}" && l.Section == part.ContextID))
                        {
                            failures.Add($"{label}: LoRA node {loraId}:{loraNode["class_type"]} on the segment model chain is not recorded for section {part.ContextID}");
                        }
                    }
                    List<string> encoded = [.. positiveWalk.SelectMany(n => ((JObject)n.Item2["inputs"]).Properties().Where(p => WorkflowTrace.PromptInputNames.Contains(p.Name) && p.Value.Type == JTokenType.String).Select(p => $"{p.Value}"))];
                    string segmentPrompt = $"{((JArray)summary["segments"]).Single()["prompt"]}";
                    if (!encoded.Any(text => segmentPrompt == $"@{stage}.pos#{MetadataHelper.Sha8(text)}"))
                    {
                        failures.Add($"{label}: the segment record names prompt {segmentPrompt}, which no encoder on sampler {samplerId}'s positive chain gets ({encoded.Count} prompt inputs found)");
                    }
                    string signature = $"model<-[{modelWalk.Select(n => $"{n.Item2["class_type"]}").JoinString(",")}] positive<-[{positiveWalk.Select(n => $"{n.Item2["class_type"]}").Distinct().JoinString(",")}]";
                    signatures.GetOrCreate(signature, () => []).Add(label);
                }
            }
        }
        foreach (string id in required.Where(id => !builtClasses.Contains(id)))
        {
            failures.Add($"{id}: required model class did not build a segment");
        }
        TestContext.Out.WriteLine($"{built} builds checked, {rejected} rejected by the generator, {signatures.Count} distinct segment chain signatures:");
        foreach ((string signature, List<string> labels) in signatures.OrderBy(s => s.Key))
        {
            TestContext.Out.WriteLine($"  {signature}\n      {labels.JoinString("\n      ")}");
        }
        Assert.That(signatures.Count, Is.GreaterThan(1), "segment records differ across model families");
        Assert.That(failures, Is.Empty, string.Join("\n", failures.Take(60)));
    }

    /// <summary>Changes made to the workflow outside CreateNode, the way extensions make them, are attributed to the step that made them, and to the innermost step when a model-generation step runs inside a step:
    /// an input edited in place (as SeedVR2Upscaler's RunOnNodesOfClass does), a node replaced by assignment, an input added inside a model load, and a node inserted by assignment, which the fitness check rejects.</summary>
    [Test]
    public void TestInPlaceEditsAndInsertionsAreAttributed()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        const string editStep = "trace test: edit sampler steps", replaceStep = "trace test: replace save node", insertStep = "trace test: insert preview", modelGenStep = "trace test: mark model";
        List<WorkflowGenerator.WorkflowGenStep> steps = [.. WorkflowGenerator.Steps], modelGenSteps = [.. WorkflowGenerator.ModelGenSteps];
        try
        {
            WorkflowGenerator.AddStep(g => g.RunOnNodesOfClass("SwarmKSampler", (_, n) => n["inputs"]["steps"] = 7), 9.5, editStep);
            WorkflowGenerator.AddStep(g => g.Workflow["9"] = g.Workflow["9"].DeepClone(), 10.5, replaceStep);
            WorkflowGenerator.AddStep(g => g.Workflow["900001"] = new JObject() { ["class_type"] = "PreviewImage", ["inputs"] = new JObject() { ["images"] = g.Workflow["9"]["inputs"]["images"].DeepClone() } }, 10.5, insertStep);
            WorkflowGenerator.AddModelGenStep(g => g.Workflow[$"{g.LoadingModel[0]}"]["inputs"]["trace_test_marker"] = 1, 1000, modelGenStep);
            Build plain = Generate(new() { Model = model, Prompt = PromptMarkerBase }, false);
            Build traced = Generate(new() { Model = model, Prompt = PromptMarkerBase }, true);
            Assert.That(SameWorkflow(traced, plain), Is.True, "tracing changed the workflow");
            WorkflowTrace trace = traced.Trace;
            string samplerId = traced.Workflow.Properties().Single(p => $"{p.Value["class_type"]}" == "SwarmKSampler").Name;
            WorkflowTrace.EditRecord stepsEdit = trace.Edits.Single(e => e.NodeId == samplerId);
            Assert.That((stepsEdit.Field, stepsEdit.Before.Value<int>(), stepsEdit.After.Value<int>(), trace.StepName(stepsEdit.Step)), Is.EqualTo(("steps", 20, 7, editStep)));
            WorkflowTrace.EditRecord markerEdit = trace.Edits.Single(e => e.Field == "trace_test_marker");
            Assert.That((markerEdit.Before, markerEdit.After.Value<int>(), trace.StepName(markerEdit.Step), trace.StepRecords[markerEdit.Step].IsModelGen), Is.EqualTo(((JToken)null, 1, modelGenStep, true)), "attributed to the model-generation step, not the step that loaded the model");
            WorkflowTrace.EditRecord replaced = trace.Edits.Single(e => e.NodeId == "9");
            Assert.That((replaced.Field, trace.StepName(replaced.Step)), Is.EqualTo(("(node)", replaceStep)));
            Assert.That(trace.StepName(trace.InsertedBy["900001"]), Is.EqualTo(insertStep));
            Assert.That(trace.Edits, Has.Count.EqualTo(3), string.Join("\n", trace.Edits.Select(e => trace.EditJson(e).ToString(Formatting.None))));
            JObject summary = (JObject)traced.Input.ExtraMeta[WorkflowTrace.MetadataKey];
            JObject[] expectedEdits =
            [
                new() { ["node"] = $"{samplerId}:SwarmKSampler", ["input"] = "steps", ["before"] = 20, ["after"] = 7, ["step"] = editStep },
                new() { ["node"] = $"{markerEdit.NodeId}:{markerEdit.ClassType}", ["input"] = "trace_test_marker", ["before"] = null, ["after"] = 1, ["step"] = modelGenStep },
                new() { ["node"] = "9:SwarmSaveImageWS", ["replaced"] = true, ["was"] = "SwarmSaveImageWS", ["step"] = replaceStep }
            ];
            foreach (JObject expected in expectedEdits)
            {
                Assert.That(((JArray)summary["edits"]).Any(e => JToken.DeepEquals(e, expected)), Is.True, $"edit {expected.ToString(Formatting.None)} in {summary["edits"].ToString(Formatting.None)}");
            }
            Assert.That(((JArray)summary["untracked_nodes"]).Any(u => JToken.DeepEquals(u, new JObject() { ["node"] = "900001:PreviewImage", ["inserted_by"] = insertStep })), Is.True, summary["untracked_nodes"].ToString(Formatting.None));
            JObject[] stepEntries = [.. ((JArray)summary["steps"]).OfType<JObject>()];
            JObject editEntry = stepEntries.Single(s => $"{s["name"]}" == editStep), insertEntry = stepEntries.Single(s => $"{s["name"]}" == insertStep);
            Assert.That($"{editEntry["source"]}", Does.StartWith("WorkflowTraceTests.cs:"));
            Assert.That((editEntry["edited"].Select(t => $"{t}"), insertEntry["inserted_without_createnode"].Select(t => $"{t}")), Is.EqualTo((new[] { samplerId }, new[] { "900001" })));
            List<string> failures = [];
            CheckTrace(traced, "extension steps", failures);
            Assert.That(failures, Does.Contain("extension steps: node 900001:PreviewImage has no provenance record"), string.Join("\n", failures));
        }
        finally
        {
            WorkflowGenerator.Steps = steps;
            WorkflowGenerator.ModelGenSteps = modelGenSteps;
        }
    }

    /// <summary>CreateNode reuse and overwrites are attributed: a node CreateNode returns again for an identical request lists the reusing call site and stage, and a node CreateNode replaces under the same ID counts the overwrite,
    /// in its provenance, in the image record's node table and in node_reuse.</summary>
    [Test]
    public void TestNodeReuseAndOverwritesAreAttributed()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        const string reuseStep = "trace test: reuse and overwrite nodes";
        List<WorkflowGenerator.WorkflowGenStep> steps = [.. WorkflowGenerator.Steps];
        try
        {
            string first = null, second = null;
            WorkflowGenerator.AddStep(g =>
            {
                first = g.CreateNode("PrimitiveInt", new JObject() { ["value"] = 3 });
                second = g.CreateNode("PrimitiveInt", new JObject() { ["value"] = 3 });
                g.CreateNode("PrimitiveInt", new JObject() { ["value"] = 4 }, "900002");
                g.CreateNode("PrimitiveInt", new JObject() { ["value"] = 5 }, "900002");
            }, 10.5, reuseStep);
            Build plain = Generate(new() { Model = model, Prompt = PromptMarkerBase }, false);
            Build traced = Generate(new() { Model = model, Prompt = PromptMarkerBase }, true);
            Assert.That(SameWorkflow(traced, plain), Is.True, "tracing changed the workflow");
            Assert.That(second, Is.EqualTo(first), "CreateNode returned the existing node for the identical request");
            WorkflowTrace trace = traced.Trace;
            WorkflowTrace.NodeRecord reused = trace.Nodes[first], overwritten = trace.Nodes["900002"];
            Assert.That(reused.ReusedBy, Has.Count.EqualTo(1));
            Assert.That(reused.ReusedBy[0].Origin, Does.StartWith($"{nameof(TestNodeReuseAndOverwritesAreAttributed)}@WorkflowTraceTests.cs:"));
            Assert.That(int.TryParse(reused.ReusedBy[0].Origin.AfterLast(':'), out _), Is.True, "the origin ends with the calling line");
            Assert.That(reused.ReusedBy[0].Stage, Is.EqualTo(reused.Stage));
            Assert.That((reused.Overwrites, overwritten.Overwrites, overwritten.ReusedBy.Count), Is.EqualTo((0, 1, 0)));
            JObject stored = (JObject)traced.Input.ExtraMeta[WorkflowTrace.MetadataKey];
            JObject reusedEntry = (JObject)stored["nodes"][first], overwrittenEntry = (JObject)stored["nodes"]["900002"];
            Assert.That(($"{reusedEntry["step"]}", reusedEntry["reused"]?.Value<int>(), reusedEntry["overwritten"]), Is.EqualTo((reuseStep, (int?)1, (JToken)null)));
            Assert.That(($"{overwrittenEntry["step"]}", overwrittenEntry["overwritten"]?.Value<int>(), overwrittenEntry["reused"]), Is.EqualTo((reuseStep, (int?)1, (JToken)null)));
            JObject reusedRecord = new() { ["node"] = $"{first}:PrimitiveInt", ["reused_by"] = new JArray(new JObject() { ["origin"] = reused.ReusedBy[0].Origin, ["stage"] = reused.Stage }), ["overwrites"] = 0 };
            JObject overwrittenRecord = new() { ["node"] = "900002:PrimitiveInt", ["reused_by"] = new JArray(), ["overwrites"] = 1 };
            foreach (JObject expected in new[] { reusedRecord, overwrittenRecord })
            {
                Assert.That(((JArray)stored["node_reuse"]).Any(r => JToken.DeepEquals(r, expected)), Is.True, $"{expected.ToString(Formatting.None)} in {stored["node_reuse"].ToString(Formatting.None)}");
            }
            List<string> failures = [];
            CheckTrace(traced, "node reuse", failures);
            Assert.That(failures, Is.Empty, string.Join("\n", failures));
        }
        finally
        {
            WorkflowGenerator.Steps = steps;
        }
    }

    /// <summary>Negative controls for the fitness check: it passes a clean traced build, and fails a build where a node has no provenance record, a record with no origin, a record of the wrong class, a node left the workflow with no step recorded as removing it,
    /// a link dangles, a sampler has no call record, or the image record is missing a node or a step.</summary>
    [Test]
    public void TestFitnessCheckRejectsMissingProvenance()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);
        Request request() => new() { Model = model, Prompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment}" };
        static string samplerOf(Build build) => build.Workflow.Properties().First(p => WorkflowTrace.SamplerClasses.Contains($"{p.Value["class_type"]}")).Name;
        List<string> clean = [];
        CheckTrace(Generate(request(), true), "clean", clean);
        Assert.That(clean, Is.Empty, string.Join("\n", clean));
        Dictionary<string, Func<Build, string>> breaks = new()
        {
            ["record removed"] = b =>
            {
                string id = samplerOf(b);
                b.Trace.Nodes.Remove(id);
                return $"node {id}:{b.Workflow[id]["class_type"]} has no provenance record";
            },
            ["origin missing"] = b =>
            {
                string id = samplerOf(b);
                b.Trace.Nodes[id].Origin = null;
                return $"node {id}:{b.Workflow[id]["class_type"]} provenance incomplete";
            },
            ["wrong class"] = b =>
            {
                string id = samplerOf(b);
                b.Trace.Nodes[id].ClassType = "NotTheClass";
                return $"node {id} recorded as NotTheClass but is {b.Workflow[id]["class_type"]}";
            },
            ["unrecorded removal"] = b =>
            {
                string id = samplerOf(b);
                string classType = $"{b.Workflow[id]["class_type"]}";
                b.Workflow.Remove(id);
                return $"node {id}:{classType} is gone from the workflow but no step is recorded as removing it";
            },
            ["dangling link"] = b =>
            {
                string id = samplerOf(b);
                b.Workflow[id]["inputs"]["model"] = new JArray("999999", 0);
                return "summary lists dangling links";
            },
            ["missing sampler call"] = b =>
            {
                string id = samplerOf(b);
                b.Trace.SamplerCalls.Remove(id);
                return $"sampler {id}:{b.Workflow[id]["class_type"]} has no call record";
            },
            ["node missing from the image record"] = b =>
            {
                string id = b.Workflow.Properties().First(p => !WorkflowTrace.SamplerClasses.Contains($"{p.Value["class_type"]}")).Name;
                ((JObject)((JObject)b.Input.ExtraMeta[WorkflowTrace.MetadataKey])["nodes"]).Remove(id);
                return $"node {id}:{b.Workflow[id]["class_type"]} has no entry with its class, stage and provenance in the image record";
            },
            ["step missing from the image record"] = b =>
            {
                ((JArray)((JObject)b.Input.ExtraMeta[WorkflowTrace.MetadataKey])["steps"]).RemoveAt(0);
                return $"{b.Trace.StepRecords.Count} steps ran but the image record lists {b.Trace.StepRecords.Count - 1}";
            },
            ["prompt text in the log detail"] = b =>
            {
                b.Input.InternalSet.ValuesInput["trace_test_text"] = "zqxleak a road";
                return "raw prompt text appears in the log detail";
            }
        };
        foreach ((string name, Func<Build, string> breakIt) in breaks)
        {
            Build build = Generate(request(), true);
            string expected = $"{name}: {breakIt(build)}";
            List<string> failures = [];
            CheckTrace(build, name, failures);
            Assert.That(failures.Any(f => f.StartsWith(expected)), Is.True, $"expected '{expected}', got:\n{string.Join("\n", failures)}");
        }
    }

    /// <summary>A trace that throws while finishing never fails the generation: the workflow equals the untraced one, the parameter tracker is detached, and the image record carries the trace error.</summary>
    [Test]
    public void TestTraceFailureNeverFailsGeneration()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        const string poisonStep = "trace test: poison the trace";
        List<WorkflowGenerator.WorkflowGenStep> steps = [.. WorkflowGenerator.Steps];
        try
        {
            WorkflowGenerator.AddStep(g => g.Trace?.Loads.Add(null), 10.5, poisonStep);
            Build plain = Generate(new() { Model = model, Prompt = PromptMarkerBase }, false);
            Build traced = Generate(new() { Model = model, Prompt = PromptMarkerBase }, true);
            Assert.That(SameWorkflow(traced, plain), Is.True, "the workflow is unaffected");
            Assert.That(traced.Input.ParamReadTracker, Is.Null);
            JObject record = (JObject)traced.Input.ExtraMeta[WorkflowTrace.MetadataKey];
            Assert.That($"{record["trace_error"]}", Does.StartWith("NullReferenceException"), record.ToString());
        }
        finally
        {
            WorkflowGenerator.Steps = steps;
        }
    }

    /// <summary>With Prompt Images, each sampler's conditioning record matches the graph: every text encoder on a sampler's positive chain that has an images input is listed with that link, and the prompt image has a media record with its hash and size.
    /// Qwen Image 2.1 and Qwen Image Edit Plus are built; at least one of them gives an encoder images, so the check is exercised.</summary>
    [Test]
    public void TestEncoderImageInputsMatchTheGraph()
    {
        int encodersWithImages = 0;
        foreach (string classId in new[] { "qwen-image-2.1", "qwen-image-edit-plus" })
        {
            T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses[classId], true);
            List<Image> promptImages = [PromptTestImage];
            Build traced = Generate(new() { Model = model, Prompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment}", Setters = { i => i.Set(T2IParamTypes.PromptImages, promptImages) } }, true);
            JObject workflow = traced.Workflow;
            WorkflowTrace trace = traced.Trace;
            JObject summary = trace.Build();
            int found = 0;
            foreach (JObject sampler in ((JArray)summary["samplers"]).OfType<JObject>())
            {
                string samplerId = $"{sampler["node"]}".Before(':');
                if (workflow[samplerId]?["inputs"]?["positive"] is not JArray positive)
                {
                    continue;
                }
                List<string> chain = ChainIds(sampler, "positive");
                foreach ((string id, JObject node) in StageWalk(workflow, positive, trace, trace.Nodes[samplerId].Stage, false))
                {
                    if (node["inputs"]?["images"] is JArray images && WorkflowTrace.IsLink(images))
                    {
                        found++;
                        Assert.That(chain, Does.Contain(id), $"{classId}: sampler {samplerId} chain lists encoder {id}");
                        Assert.That($"{summary["nodes"][id]["links"]["images"]}", Is.EqualTo($"{images[0]}:{images[1]}"), $"{classId}: sampler {samplerId} encoder {id} image input");
                    }
                }
            }
            JObject promptImage = ((JObject)summary["media"]).Properties().Select(p => (JObject)p.Value).Single(m => $"{m["param"]}" == "promptimages.0");
            Assert.That((promptImage["sha8"] is not null, promptImage["width"] is not null, promptImage["height"] is not null, $"{promptImage["mime"]}"), Is.EqualTo((true, true, true, "image/png")), $"{classId}: the prompt image's media record has its hash, size and type: {promptImage.ToString(Formatting.None)}");
            TestContext.Out.WriteLine($"{classId}: {found} text encoder image inputs on sampler conditioning chains");
            encodersWithImages += found;
        }
        Assert.That(encodersWithImages, Is.GreaterThan(0), "at least one model gives its text encoder the prompt image");
    }

    /// <summary>A model patch made by a model-generation step is in the image record, though no sampler input chain reaches the patched VAE: Seamless Tiling's model and VAE nodes are listed with the step that made them,
    /// the model load they patched lists them as a patch, and the decode reads the tiled VAE.</summary>
    [Test]
    public void TestModelPatchesAreInTheImageRecord()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        Build traced = Generate(new() { Model = model, Prompt = PromptMarkerBase, Setters = { i => i.Set(T2IParamTypes.SeamlessTileable, "true") } }, true);
        JObject record = (JObject)traced.Input.ExtraMeta[WorkflowTrace.MetadataKey];
        string vaeId = traced.Workflow.Properties().Single(p => $"{p.Value["class_type"]}" == "SwarmTileableVAE").Name;
        string tilingId = traced.Workflow.Properties().Single(p => $"{p.Value["class_type"]}" == "SwarmModelTiling").Name;
        string decodeId = traced.Workflow.Properties().Single(p => $"{p.Value["class_type"]}" == "VAEDecode").Name;
        Assert.That($"{traced.Workflow[decodeId]["inputs"]["vae"][0]}", Is.EqualTo(vaeId), "the decode reads the tiled VAE");
        Assert.That(($"{record["nodes"][vaeId]["step"]}", $"{record["nodes"][tilingId]["step"]}"), Is.EqualTo(("Seamless Tiling", "Seamless Tiling")));
        List<JObject> patches = [.. ((JArray)record["model_loads"]).OfType<JObject>().SelectMany(l => (l["patches"] as JArray ?? []).OfType<JObject>()).Where(p => $"{p["name"]}" == "Seamless Tiling")];
        Assert.That(patches, Has.Count.EqualTo(1), record["model_loads"].ToString(Formatting.None));
        Assert.That(patches[0]["created"].Select(t => $"{t}"), Is.EquivalentTo(new[] { tilingId, vaeId }));
        List<string> failures = [];
        CheckTrace(traced, "seamless", failures);
        Assert.That(failures, Is.Empty, string.Join("\n", failures));
    }

    /// <summary>The unsampler, a sampler made outside CreateKSampler, has a call record.</summary>
    [Test]
    public void TestUnsamplerHasCallRecord()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        Build traced = Generate(new() { Model = model, Prompt = PromptMarkerBase, Setters = { i => i.Set(T2IParamTypes.InitImage, InitTestImage), i => i.Set(T2IParamTypes.InitImageCreativity, 0.6), i => i.Set(T2IParamTypes.UnsamplerPrompt, "zqxunsample a road") } }, true);
        string unsamplerId = traced.Workflow.Properties().Single(p => $"{p.Value["class_type"]}" == "SwarmUnsampler").Name;
        JObject call = traced.Trace.SamplerCalls[unsamplerId];
        Assert.That(($"{call["kind"]}", call["steps"].Value<int>(), call["start_step"].Value<int>()), Is.EqualTo(("unsampler", traced.Workflow[unsamplerId]["inputs"]["steps"].Value<int>(), traced.Workflow[unsamplerId]["inputs"]["start_at_step"].Value<int>())));
        List<string> failures = [];
        CheckTrace(traced, "unsampler", failures);
        Assert.That(failures, Is.Empty, string.Join("\n", failures));
    }

    /// <summary>A workflow that did not come from the generator has no prompt labels, so its description hashes every string that could be prompt text whatever its input name, and keeps model file names and short identifiers readable.</summary>
    [Test]
    public void TestExternalWorkflowHashesFreeText()
    {
        JObject workflow = new()
        {
            ["1"] = new JObject() { ["class_type"] = "PrimitiveString", ["inputs"] = new JObject() { ["value"] = "zqxfree a cat on a road" } },
            ["2"] = new JObject() { ["class_type"] = "CheckpointLoaderSimple", ["inputs"] = new JObject() { ["ckpt_name"] = "test models/sdxl base.safetensors" } },
            ["3"] = new JObject() { ["class_type"] = "KSampler", ["inputs"] = new JObject() { ["sampler_name"] = "euler", ["model"] = new JArray("2", 0), ["positive"] = new JArray("1", 0) } }
        };
        JObject described = WorkflowTrace.DescribeExternalWorkflow("custom", workflow.ToString());
        Assert.That(described.ToString(Formatting.None), Does.Not.Contain("zqxfree"));
        JObject nodes = (JObject)described["nodes"];
        Assert.That($"{nodes["1"]["inputs"]["value"]}", Is.EqualTo($"#{MetadataHelper.Sha8("zqxfree a cat on a road")}(23)"));
        Assert.That($"{nodes["2"]["inputs"]["ckpt_name"]}", Is.EqualTo("test models/sdxl base.safetensors"));
        Assert.That(($"{nodes["3"]["class"]}", $"{nodes["3"]["inputs"]["sampler_name"]}", $"{nodes["3"]["links"]["model"]}"), Is.EqualTo(("KSampler", "euler", "2:0")));
    }

    /// <summary>Hashing text without copying it gives the same hash as hashing its UTF-8 bytes, at and around the cache threshold and the writer's buffer size, with surrogate pairs split across buffer boundaries and lone surrogates; long hashes are cached by reference.</summary>
    [Test]
    public void TestHashTextMatchesUtf8Hash()
    {
        foreach (int length in new[] { 0, 1, MetadataHelper.CachedHashMinLength - 1, MetadataHelper.CachedHashMinLength, MetadataHelper.CachedHashMinLength + 1, 3 * 4096 + 7 })
        {
            char[] chars = new char[length];
            for (int i = 0; i < length; i++)
            {
                chars[i] = (char)('a' + i % 26);
            }
            foreach (int at in new[] { 4095, 8191, length - 1 })
            {
                if (at > 0 && at < length)
                {
                    chars[at - 1] = '\uD83D';
                    chars[at] = '\uDE00';
                }
            }
            if (length > 100)
            {
                chars[50] = '\uD800';
                chars[99] = 'é';
            }
            string text = new(chars);
            Assert.That(MetadataHelper.HashText(text), Is.EqualTo(Utilities.HashSHA256(text.EncodeUTF8())), $"length {length}");
            Assert.That(MetadataHelper.Sha8(text), Is.EqualTo(Utilities.HashSHA256(text.EncodeUTF8())[..8]), $"length {length}");
        }
        string longText = new('x', MetadataHelper.CachedHashMinLength * 2);
        string hash = MetadataHelper.HashText(longText);
        Assert.That(MetadataHelper.LongHashes.TryGetValue(longText, out string cached) && cached == hash, Is.True, "a long text's hash is cached by reference");
        byte[] bytes = longText.EncodeUTF8();
        Assert.That(MetadataHelper.HashBytes(bytes), Is.EqualTo(Utilities.HashSHA256(bytes)));
    }

    /// <summary>An ExtraMeta value that cannot be serialized, which makes the image metadata fail to build.</summary>
    public class UnserializableValue
    {
        /// <summary>Throws when serialized.</summary>
        public string Value => throw new InvalidOperationException("trace test: unserializable");
    }

    /// <summary>Recording a custom workflow, the sent graph or the posted graph never fails the send: an error is logged, and the custom workflow's record says what failed.</summary>
    [Test]
    public void TestRecordingNeverFailsTheSend()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        T2IParamInput input = BuildInput(new() { Model = model, Prompt = PromptMarkerBase, Params = new() { ["imageformat"] = "JPG" } });
        input.ExtraMeta["trace_test_unserializable"] = new UnserializableValue();
        Assert.DoesNotThrow(() => WorkflowTrace.RecordExternalWorkflow(input, "custom", """{"1": {"class_type": "KSampler", "inputs": {}}}"""));
        Assert.That($"{((JObject)input.ExtraMeta[WorkflowTrace.MetadataKey])["trace_error"]}", Does.StartWith("JsonSerializationException").And.Contain(nameof(UnserializableValue)));
        ((JObject)input.ExtraMeta[WorkflowTrace.MetadataKey])["workflow"] = new JObject() { ["sent_sha256"] = "x" };
        Assert.DoesNotThrow(() => WorkflowTrace.RecordPosted(input, null, "TestBackend"), "a null posted workflow is logged, not thrown");
        WorkflowTrace trace = Generate(new() { Model = model, Prompt = PromptMarkerBase }, true).Trace;
        Assert.DoesNotThrow(() => trace.RecordSent(null, "{}"), "a null sent workflow is logged, not thrown");
    }

    /// <summary><see cref="WorkflowGenerator.WorkflowGenStep"/> keeps upstream's positional shape, so an extension that constructs or deconstructs it as (Action, Priority) still compiles; the step's name and source are not positional.</summary>
    [Test]
    public void TestWorkflowGenStepKeepsTheUpstreamShape()
    {
        Action<WorkflowGenerator> action = _ => { };
        WorkflowGenerator.WorkflowGenStep step = new(action, 5);
        (Action<WorkflowGenerator> deconstructed, double priority) = step;
        Assert.That((deconstructed, priority, step.Name, step.Source), Is.EqualTo((action, 5.0, (string)null, (string)null)));
        Assert.That(WorkflowGenerator.Steps.All(s => !string.IsNullOrEmpty(s.Source)), Is.True, "every registered step records its source");
    }

    /// <summary>A stop is attributed to the top-level step that set SkipFurtherSteps, not to a model-generation step it ran (which comes after it in the step records).</summary>
    [Test]
    public void TestStopIsAttributedToTheTopLevelStep()
    {
        WorkflowGenerator generator = new() { UserInput = BuildInput(new() { Model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false), Prompt = PromptMarkerBase }), Workflow = [] };
        WorkflowTrace trace = new(generator);
        trace.BeginStep(new(_ => { }, 0) { Name = "Stopping step" }, false);
        trace.BeginStep(new(_ => { }, 0) { Name = "Model generation inside it" }, true);
        trace.EndStep();
        trace.EndStep();
        trace.SkippedFurtherSteps();
        Assert.That(trace.StepRecords[^1].Name, Is.EqualTo("Model generation inside it"), "the last step record is the nested one");
        Assert.That(JToken.DeepEquals(trace.Events[^1], new JObject() { ["event"] = "stopped_further_steps", ["step"] = "Stopping step" }), Is.True, trace.Events[^1].ToString(Formatting.None));
    }

    /// <summary>The text of a prompt-view parameter is hashed under any input name: ACE-Step 1.5 writes the Text2Audio style to a "tags" input. A text that was not read from a prompt-view parameter, under an input name that is not a prompt name, is written as it is, which is the negative control.</summary>
    [Test]
    public void TestPromptViewParameterTextIsNeverRaw()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["ace-step-1_5"], false);
        Build traced = Generate(WithPromptParameterMarkers(new() { Model = model, Prompt = PromptMarkerBase }), true);
        List<JProperty> tagsNodes = [.. traced.Workflow.Properties().Where(p => $"{p.Value["inputs"]?["tags"]}" == PromptMarkerStyle)];
        Assert.That(tagsNodes, Is.Not.Empty, "the graph carries the style text in a tags input");
        JObject nodes = (JObject)((JObject)traced.Input.ExtraMeta[WorkflowTrace.MetadataKey])["nodes"];
        foreach (JProperty tagsNode in tagsNodes)
        {
            Assert.That($"{nodes[tagsNode.Name]["inputs"]["tags"]}", Is.EqualTo($"@{T2IParamTypes.Text2AudioStyle.Type.ID}#{MetadataHelper.Sha8(PromptMarkerStyle)}"), $"node {tagsNode.Name}");
        }
        JProperty tagsNode0 = tagsNodes[0];
        List<string> failures = [];
        CheckTrace(traced, "ace-step style", failures);
        Assert.That(failures, Is.Empty, string.Join("\n", failures));
        Assert.That(traced.Trace.FormatString(tagsNode0.Name, "unlisted_input", "zqxunread text"), Is.EqualTo("zqxunread text"), "negative control: unread text under a non-prompt input name is written as it is");
    }

    /// <summary>Placeholders are found as the backend's fill finds them (QuickSimpleTagFiller without subtags): each once, a default value kept in the tag, and an unclosed tag ignored.</summary>
    [Test]
    public void TestPlaceholdersMatchTheBackendFill()
    {
        string text = "{\"a\": \"${initimage}\", \"b\": \"${promptimages.0}\", \"c\": \"${initimage}\", \"d\": \"${steps:20}\", \"e\": \"${unclosed\"";
        List<string> found = WorkflowTrace.Placeholders(text);
        List<string> filled = [];
        StringConversionHelper.QuickSimpleTagFiller(text, "${", "}", tag => { filled.Add($"${{{tag}}}"); return null; }, false);
        Assert.That(found, Is.EqualTo(filled.Distinct()), "the same tags the fill reads, each once");
        Assert.That(found, Does.Contain("${steps:20}"));
    }

    /// <summary>Without ComfyUI object_info, the record says that link types, declared inputs and node classes were not checked, instead of listing no problems.</summary>
    [Test]
    public void TestMissingObjectInfoIsStated()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        JObject record = (JObject)Generate(new() { Model = model, Prompt = PromptMarkerBase }, true).Input.ExtraMeta[WorkflowTrace.MetadataKey];
        Assert.That($"{record["object_info"]}", Does.StartWith("not given"));
        Assert.That(record["type_errors"], Is.Null);
        Assert.That(WorkflowTraceTypingTests.BuildTypedSdxl().Trace.Build()["object_info"], Is.Null, "with object_info, the checks are listed instead");
    }

    /// <summary>Prompt text that is part of a registered prompt is labeled with the shortest registered prompt containing it, the most specific source, and with every label when several tie.</summary>
    [Test]
    public void TestPromptPartsGoToTheMostSpecificPrompt()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        WorkflowTrace trace = Generate(new() { Model = model, Prompt = PromptMarkerBase }, true).Trace;
        trace.LabelPrompt("zqxpart man|woman", "seg11.mask");
        trace.LabelPrompt("zqxpart woman|man", "seg12.mask");
        trace.LabelPrompt("zqxpart a man on a road with a woman", "test.pos");
        string part = trace.FormatString("1", "match_text", "man");
        Assert.That(part, Does.StartWith("@seg11.mask=@seg12.mask~part#").Or.StartWith("@seg12.mask=@seg11.mask~part#"), part);
        Assert.That(trace.FormatString("1", "text", "road with a woman"), Does.StartWith("@test.pos~part#"));
        Assert.That(trace.FormatString("1", "text", "a road"), Does.Not.StartWith("@test.pos"), "the base prompt is shorter than test.pos and also contains it");
    }

    /// <summary>A clone of an input does not record parameter reads into the original's trace.</summary>
    [Test]
    public void TestCloneDropsTheParameterReadTracker()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        T2IParamInput input = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
        int reads = 0;
        input.ParamReadTracker = (_, _, _, _) => reads++;
        T2IParamInput clone = input.Clone();
        Assert.That(clone.ParamReadTracker, Is.Null);
        clone.Get(T2IParamTypes.Steps);
        Assert.That(reads, Is.EqualTo(0));
        input.Get(T2IParamTypes.Steps);
        Assert.That(reads, Is.EqualTo(1));
    }

    /// <summary>A parameter read records its first value and source, the last when it differs, and with more than two distinct values, the read count and every distinct value up to the cap, which it says it reached.</summary>
    [Test]
    public void TestParameterReadsKeepDistinctValues()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        WorkflowTrace trace = Generate(new() { Model = model, Prompt = PromptMarkerBase }, true).Trace;
        foreach (int value in new[] { 11, 12, 11, 13 })
        {
            trace.OnParamRead(T2IParamTypes.Seed.Type, 7, (long)value, T2IParamInput.ParamReadSource.Section);
        }
        trace.OnParamRead(T2IParamTypes.Steps.Type, 7, 30, T2IParamInput.ParamReadSource.Section);
        trace.OnParamRead(T2IParamTypes.Steps.Type, 7, 30, T2IParamInput.ParamReadSource.Base);
        for (int value = 0; value < WorkflowTrace.MaxDistinctParamValues + 2; value++)
        {
            trace.OnParamRead(T2IParamTypes.CFGScale.Type, 7, (double)value, T2IParamInput.ParamReadSource.Section);
        }
        JObject reads = (JObject)trace.Build()["params_read"];
        JObject seed = new() { ["value"] = 11L, ["source"] = "section", ["last"] = 13L, ["last_source"] = "section", ["reads"] = 4, ["values"] = new JArray(11L, 12L, 13L) };
        JObject steps = new() { ["value"] = 30, ["source"] = "section", ["last"] = 30, ["last_source"] = "base" };
        Assert.That(JToken.DeepEquals(reads[$"{T2IParamTypes.Seed.Type.ID}@7"], seed), Is.True, reads[$"{T2IParamTypes.Seed.Type.ID}@7"].ToString(Formatting.None));
        Assert.That(JToken.DeepEquals(reads[$"{T2IParamTypes.Steps.Type.ID}@7"], steps), Is.True, "a change of source alone is recorded: " + reads[$"{T2IParamTypes.Steps.Type.ID}@7"].ToString(Formatting.None));
        JObject cfg = (JObject)reads[$"{T2IParamTypes.CFGScale.Type.ID}@7"];
        Assert.That((((JArray)cfg["values"]).Count, cfg["values_capped"]?.Value<bool>(), cfg["reads"].Value<int>(), cfg["last"].Value<double>()), Is.EqualTo((WorkflowTrace.MaxDistinctParamValues, (bool?)true, WorkflowTrace.MaxDistinctParamValues + 2, WorkflowTrace.MaxDistinctParamValues + 1.0)));
    }

    /// <summary>The graph sent and the graph a backend posts are recorded in the record the image metadata carries, for a PNG image (tier 0) and for a JPEG image whose other metadata already fills the limit (the last tier), and in the full summary; an input without a trace gets nothing.</summary>
    [Test]
    public void TestSentAndPostedAreRecordedInTheStoredRecord()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        bool priorDebug = Program.ServerSettings.AddDebugData;
        try
        {
            Program.ServerSettings.AddDebugData = true;
            foreach ((string format, bool notice) in new[] { ("PNG", false), ("JPG", true) })
            {
                T2IParamInput input = BuildInput(new() { Model = model, Prompt = PromptMarkerBase, Params = new() { ["imageformat"] = format } });
                if (notice)
                {
                    input.ExtraMeta["trace_test_padding"] = new string('x', MetadataHelper.MaxJpegExifChars);
                }
                string workflow = ComfyUIAPIAbstractBackend.CreateWorkflow(input, w => w, null, [.. FullFeatures]);
                JObject stored = (JObject)input.ExtraMeta[WorkflowTrace.MetadataKey];
                Assert.That(stored["omitted"] is not null, Is.EqualTo(notice), format);
                JObject info = (JObject)stored["workflow"];
                Assert.That($"{info["sent_sha256"]}", Is.EqualTo(Utilities.HashSHA256(workflow.EncodeUTF8())), format);
                Assert.That(info["rewritten_after_generate"].Value<bool>(), Is.False, format);
                Assert.That($"{info["posted"]}", Is.EqualTo(WorkflowTrace.NotPosted), format);
                WorkflowTrace.RecordPosted(input, workflow, "TestBackend");
                Assert.That(($"{info["posted"]["sha256"]}", $"{info["posted"]["backend"]}", info["posted"]["matches_sent"].Value<bool>()), Is.EqualTo(($"{info["sent_sha256"]}", "TestBackend", true)), format);
                JObject metadata = input.GenFullMetadataObject();
                Assert.That($"{metadata["sui_extra_data"][WorkflowTrace.MetadataKey]["workflow"]["posted"]["sha256"]}", Is.EqualTo($"{info["sent_sha256"]}"), $"{format}: the image metadata carries the posted record");
                WorkflowTrace.RecordPosted(input, $"{workflow} ", "TestBackend");
                Assert.That(info["posted"]["matches_sent"].Value<bool>(), Is.False, $"{format}: a different posted graph is flagged");
            }
            Program.ServerSettings.AddDebugData = false;
            T2IParamInput untraced = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
            WorkflowTrace.RecordPosted(untraced, ComfyUIAPIAbstractBackend.CreateWorkflow(untraced, w => w, null, [.. FullFeatures]), "TestBackend");
            Assert.That(untraced.ExtraMeta.ContainsKey(WorkflowTrace.MetadataKey), Is.False);
        }
        finally
        {
            Program.ServerSettings.AddDebugData = priorDebug;
        }
    }

    /// <summary>A JPEG image's saved metadata fits one EXIF segment, with model hashes on and prompts long enough that what is written after the trace finishes (the backend's copy of the prompt, the model hashes, the times, the posted workflow) overflows the tier picked when it finished.
    /// The record is refit on the save path (<see cref="SaveInput"/>); keeping the tier picked at Finish is the negative control, and must overflow for at least one prompt length.
    /// When the metadata without the record leaves no room for even the last tier (the prompt is saved three times: prompt, original_prompt and the backend's copy), the last tier is stored and the overflow is the rest of the metadata's.</summary>
    [Test]
    public void TestSavedJpegMetadataFitsOneExifSegment()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true, "-hashed");
        T2IModel lora = Lora("test_hashed_lora");
        model.Metadata.Hash = $"0x{new string('a', 64)}";
        lora.Metadata.Hash = $"0x{new string('b', 64)}";
        bool priorDebug = Program.ServerSettings.AddDebugData, priorHash = Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash;
        List<string> table = [];
        int overflowsWithoutRefit = 0, fullOthers = 0;
        try
        {
            Program.ServerSettings.AddDebugData = true;
            Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash = true;
            foreach (int promptChars in new[] { 2000, 4000, 6000, 8000, 10000 })
            {
                string filler = string.Concat(Enumerable.Repeat("a wide grassy field under a cloudy sky, ", promptChars / 40 + 1))[..promptChars];
                T2IParamInput input = BuildInput(new() { Model = model, Prompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment} <lora:test_hashed_lora> {filler}", Params = new() { ["imageformat"] = "JPG" } });
                string workflow = ComfyUIAPIAbstractBackend.CreateWorkflow(input, w => w, null, [.. FullFeatures]);
                JObject finished = (JObject)input.ExtraMeta[WorkflowTrace.MetadataKey];
                List<JObject> tiers = finished.Annotation<WorkflowTrace.RecordTierSet>()?.Tiers;
                Assert.That(tiers, Is.Not.Null, $"{promptChars}: a JPEG record carries its tiers");
                T2IParamInput saved = SaveInput(input, workflow);
                JObject fitted = (JObject)saved.ExtraMeta[WorkflowTrace.MetadataKey];
                JObject metadata = saved.GenFullMetadataObject();
                Assert.That(metadata["sui_models"]?.Select(m => $"{m["hash"]}"), Is.SupersetOf(new[] { model.Metadata.Hash, lora.Metadata.Hash }), $"{promptChars}: the saved metadata lists the model hashes");
                Assert.That($"{metadata["sui_extra_data"]["debug_backend"]["debug_internal_prompt"]}", Does.EndWith(filler), $"{promptChars}: the saved metadata holds the backend's copy of the prompt");
                Assert.That($"{metadata["sui_extra_data"][WorkflowTrace.MetadataKey]["workflow"]["posted"]["sha256"]}", Is.EqualTo(Utilities.HashSHA256(workflow.EncodeUTF8())), $"{promptChars}: the refit record keeps the posted workflow");
                int exif = ExifBytes(T2IParamInput.MetadataToString(metadata));
                int savedOther = WorkflowTrace.SavedMetadataChars(saved);
                saved.ExtraMeta[WorkflowTrace.MetadataKey] = finished;
                int exifWithoutRefit = ExifBytes(saved.GenRawMetadata());
                if (exifWithoutRefit > MaxSingleApp1ExifBytes)
                {
                    overflowsWithoutRefit++;
                }
                int finishedTier = tiers.FindIndex(t => ReferenceEquals(t, finished)), fittedTier = tiers.FindIndex(t => ReferenceEquals(t, fitted));
                string savedText = T2IParamInput.MetadataToString(metadata);
                int copies = (savedText.Length - savedText.Replace(filler, "").Length) / filler.Length;
                table.Add($"prompt {promptChars} chars, saved {copies} times ({string.Join(", ", metadata.Descendants().OfType<JValue>().Where(v => $"{v}".Contains(filler)).Select(v => v.Path))}): {savedOther} chars of other saved metadata; tier {finishedTier} at Finish, {exifWithoutRefit} EXIF bytes if kept; tier {fittedTier} refit, {exif} EXIF bytes");
                Assert.That(fittedTier, Is.GreaterThanOrEqualTo(finishedTier), $"{promptChars}: the metadata only grows after Finish, so the refit tier is never larger");
                if (savedOther + WorkflowTrace.RecordMetadataChars(tiers[^1]) <= MetadataHelper.MaxJpegExifChars)
                {
                    Assert.That(exif, Is.LessThanOrEqualTo(MaxSingleApp1ExifBytes), $"{promptChars}: the saved JPEG metadata fits one APP1 segment");
                }
                else
                {
                    fullOthers++;
                    Assert.That(fittedTier, Is.EqualTo(tiers.Count - 1), $"{promptChars}: the rest of the metadata is over the limit, so the last tier is stored");
                }
            }
        }
        finally
        {
            Program.ServerSettings.AddDebugData = priorDebug;
            Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash = priorHash;
            TestContext.Out.WriteLine(string.Join("\n", table));
        }
        Assert.That(overflowsWithoutRefit, Is.GreaterThan(0), "negative control: some prompt length overflows one APP1 segment when the tier picked at Finish is kept");
        Assert.That(fullOthers, Is.EqualTo(1), "only the longest prompt fills the metadata without the record");
    }

    /// <summary>Returns a segment-and-LoRA prompt with filler text appended to the given length, so its metadata is larger than one JPEG EXIF segment.</summary>
    public static string LongSegmentPrompt(int fillerChars)
    {
        Lora("test_alien_lora");
        string filler = string.Concat(Enumerable.Repeat("a wide grassy field under a cloudy sky, ", fillerChars / 40 + 1))[..fillerChars];
        return $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment} <lora:test_alien_lora> {filler}";
    }

    /// <summary>A WebP image keeps the full record: ImageSharp writes the whole EXIF profile as one RIFF chunk (BitWriterBase.WriteTrunksAfterData), and ImageSharp and the web UI's ExifReader read the chunk whole.
    /// A video's metadata is its track comment, which has no size limit, so the video of a request configured for JPEG is refit to tier 0 while that request's JPEG image gets a smaller tier.</summary>
    [Test]
    public void TestWebpAndVideoKeepTheFullRecord()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);
        bool priorDebug = Program.ServerSettings.AddDebugData;
        try
        {
            Program.ServerSettings.AddDebugData = true;
            T2IParamInput webp = BuildInput(new() { Model = model, Prompt = LongSegmentPrompt(6000), Params = new() { ["imageformat"] = "WEBP" } });
            string workflow = ComfyUIAPIAbstractBackend.CreateWorkflow(webp, w => w, null, [.. FullFeatures]);
            JObject stored = (JObject)webp.ExtraMeta[WorkflowTrace.MetadataKey];
            Assert.That(stored["omitted"], Is.Null, "a WebP record is the full record");
            Assert.That(stored.Annotation<WorkflowTrace.RecordTierSet>(), Is.Null, "a WebP record has no tiers to refit");
            string metadata = SaveInput(webp, workflow).GenRawMetadata();
            Assert.That(ExifBytes(metadata), Is.GreaterThan(MaxSingleApp1ExifBytes), "the metadata is larger than one JPEG segment");
            ImageFile image = InitTestImage;
            Assert.That(image.ConvertTo("WEBP", metadata, 0, 90).GetMetadata(), Is.EqualTo(metadata), "the WebP's EXIF chunk holds all of it");
            T2IParamInput jpg = BuildInput(new() { Model = model, Prompt = LongSegmentPrompt(6000), Params = new() { ["imageformat"] = "JPG" } });
            workflow = ComfyUIAPIAbstractBackend.CreateWorkflow(jpg, w => w, null, [.. FullFeatures]);
            List<JObject> tiers = ((JObject)jpg.ExtraMeta[WorkflowTrace.MetadataKey]).Annotation<WorkflowTrace.RecordTierSet>().Tiers;
            Assert.That(SaveInput(jpg, workflow, new VideoFile([0], MediaType.VideoMp4)).ExtraMeta[WorkflowTrace.MetadataKey], Is.SameAs(tiers[0]), "a video's track comment has no limit");
            Assert.That(tiers.IndexOf((JObject)SaveInput(jpg, workflow).ExtraMeta[WorkflowTrace.MetadataKey]), Is.GreaterThan(0), "the same request's JPEG image gets a smaller tier");
        }
        finally
        {
            Program.ServerSettings.AddDebugData = priorDebug;
        }
    }

    /// <summary>Returns the stealth metadata text EncodeStealthMetadata embedded in an image's alpha channel ("Alpha" mode), or null if the pixels do not hold a whole compressed copy.
    /// The bits run column by column, most significant bit first: the 15-byte signature, the payload length in bits (32 bits), then the gzip-compressed text.</summary>
    public static string DecodeStealthAlpha(ImageFile file)
    {
        using SixLabors.ImageSharp.Image<Rgba32> image = file.ToIS.CloneAs<Rgba32>();
        List<byte> bits = new(image.Width * image.Height);
        for (int x = 0; x < image.Width; x++)
        {
            for (int y = 0; y < image.Height; y++)
            {
                bits.Add((byte)(image[x, y].A & 1));
            }
        }
        byte[] bytesAt(int bitOffset, int count)
        {
            byte[] result = new byte[count];
            for (int i = 0; i < count * 8; i++)
            {
                result[i / 8] = (byte)((result[i / 8] << 1) | bits[bitOffset + i]);
            }
            return result;
        }
        if (bits.Count < 152 || Encoding.UTF8.GetString(bytesAt(0, 15)) != "stealth_pngcomp")
        {
            return null;
        }
        long payloadBits = BinaryPrimitives.ReadUInt32BigEndian(bytesAt(120, 4));
        if (152 + payloadBits > bits.Count)
        {
            return null;
        }
        try
        {
            using GZipStream gzip = new(new MemoryStream(bytesAt(152, (int)(payloadBits / 8))), CompressionMode.Decompress);
            using StreamReader reader = new(gzip, Encoding.UTF8);
            return reader.ReadToEnd();
        }
        catch (InvalidDataException)
        {
            return null;
        }
    }

    /// <summary>With stealth metadata on, the metadata ConvertTo writes into a PNG's pixels fits them: a small image gets the first tier whose compressed text fits, and its pixels decode back to the saved metadata, while a large image keeps tier 0.
    /// The tier before the one picked is the negative control: its stealth data does not fit, and the pixels hold a truncated copy.</summary>
    [Test]
    public void TestStealthMetadataFitsTheImage()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);
        Settings.User.FileFormatData fileFormat = Program.ServerSettings.DefaultUser.FileFormat;
        string priorStealth = fileFormat.StealthMetadata;
        bool priorDebug = Program.ServerSettings.AddDebugData;
        try
        {
            Program.ServerSettings.AddDebugData = true;
            fileFormat.StealthMetadata = "Alpha";
            T2IParamInput input = BuildInput(new() { Model = model, Prompt = LongSegmentPrompt(200), Params = new() { ["imageformat"] = "PNG" } });
            string workflow = ComfyUIAPIAbstractBackend.CreateWorkflow(input, w => w, null, [.. FullFeatures]);
            JObject finished = (JObject)input.ExtraMeta[WorkflowTrace.MetadataKey];
            List<JObject> tiers = finished.Annotation<WorkflowTrace.RecordTierSet>()?.Tiers;
            Assert.That(tiers, Is.Not.Null, "with stealth metadata on, a PNG record carries its tiers");
            Assert.That(finished, Is.SameAs(tiers[0]), "PNG text chunks have no limit, so Finish keeps tier 0");
            ImageFile small = TestImage(160, 160);
            T2IParamInput saved = SaveInput(input, workflow, small);
            int fitted = tiers.IndexOf((JObject)saved.ExtraMeta[WorkflowTrace.MetadataKey]);
            string metadata = saved.GenRawMetadata();
            TestContext.Out.WriteLine($"160x160 alpha holds {MetadataHelper.StealthCapacityBits(160, 160, "Alpha")} bits; tier {fitted} picked, {MetadataHelper.StealthPayloadBits(metadata, "Alpha")} bits");
            Assert.That(fitted, Is.GreaterThan(0), "a 160x160 image cannot hold tier 0");
            Assert.That(DecodeStealthAlpha(small.ConvertTo("PNG", metadata, 0, 100, "Alpha")), Is.EqualTo(metadata), "the pixels decode back to the saved metadata");
            saved.ExtraMeta[WorkflowTrace.MetadataKey] = tiers[fitted - 1];
            Assert.That(DecodeStealthAlpha(small.ConvertTo("PNG", saved.GenRawMetadata(), 0, 100, "Alpha")), Is.Null, "negative control: the tier before it does not fit, and the pixels hold a truncated copy");
            Assert.That(SaveInput(input, workflow, TestImage(1024, 1024)).ExtraMeta[WorkflowTrace.MetadataKey], Is.SameAs(tiers[0]), "a 1024x1024 image holds tier 0");
        }
        finally
        {
            fileFormat.StealthMetadata = priorStealth;
            Program.ServerSettings.AddDebugData = priorDebug;
        }
    }

    /// <summary>JPEG copies of an image's metadata (history thumbnails, model and workflow preview images) keep it in one EXIF segment: over the limit, the trace record is replaced by its notice, which keeps the log key and workflow identity, and every parameter is kept.
    /// The PNG's own metadata, too large for one segment, is the negative control. Metadata without a trace record passes through as it is.</summary>
    [Test]
    public void TestJpegCopiesOfMetadataFitOneExifSegment()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);
        bool priorDebug = Program.ServerSettings.AddDebugData;
        try
        {
            Program.ServerSettings.AddDebugData = true;
            T2IParamInput input = BuildInput(new() { Model = model, Prompt = LongSegmentPrompt(6000), Params = new() { ["imageformat"] = "PNG" } });
            string workflow = ComfyUIAPIAbstractBackend.CreateWorkflow(input, w => w, null, [.. FullFeatures]);
            string metadata = SaveInput(input, workflow).GenRawMetadata();
            Assert.That(ExifBytes(metadata), Is.GreaterThan(MaxSingleApp1ExifBytes), "negative control: the PNG's metadata as it is would split a JPEG copy's EXIF");
            ImageFile image = InitTestImage;
            ImageFile copy = image.ConvertTo("PNG", metadata).ToMetadataJpg();
            Assert.That(copy.ToIS.Metadata.ExifProfile.ToByteArray().Length, Is.LessThanOrEqualTo(MaxSingleApp1ExifBytes), "the JPEG copy's EXIF fits one segment");
            JObject original = metadata.ParseToJson(), copied = copy.GetMetadata().ParseToJson();
            JObject notice = (JObject)copied["sui_extra_data"][WorkflowTrace.MetadataKey];
            Assert.That($"{notice["log_key"]}", Is.EqualTo($"{original["sui_extra_data"][WorkflowTrace.MetadataKey]["log_key"]}"));
            Assert.That(JToken.DeepEquals(notice["workflow"], original["sui_extra_data"][WorkflowTrace.MetadataKey]["workflow"]), Is.True, "the notice keeps the workflow identity");
            Assert.That($"{notice["omitted"]}", Does.Contain("JPEG copy"));
            Assert.That(JToken.DeepEquals(copied["sui_image_params"], original["sui_image_params"]), Is.True, "every parameter is kept");
            string untraced = $"{{\"sui_image_params\": {{\"prompt\": \"{new string('a', 40000)}\"}}}}";
            Assert.That(MetadataHelper.FitTextToJpegExif(untraced), Is.SameAs(untraced), "metadata without a trace record passes through as it is");
        }
        finally
        {
            Program.ServerSettings.AddDebugData = priorDebug;
        }
    }

    /// <summary>The segment's redraw is pasted back into the image that reaches the final save, and the trace's downstream path says so.</summary>
    [Test]
    public void TestSegmentPasteBackReachesFinalSave()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);
        Request request = new() { Model = model, Prompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment}" };
        Build traced = Generate(request, true);
        JObject workflow = traced.Workflow;
        WorkflowTrace trace = traced.Trace;
        string samplerId = workflow.Properties().Single(p => WorkflowTrace.SamplerClasses.Contains($"{p.Value["class_type"]}") && trace.Nodes[p.Name].Stage == "seg11").Name;
        Dictionary<string, List<(string, string)>> consumers = [];
        foreach (JProperty property in workflow.Properties())
        {
            foreach (JProperty input in ((JObject)property.Value["inputs"]).Properties())
            {
                if (WorkflowTrace.IsLink(input.Value))
                {
                    consumers.GetOrCreate($"{input.Value[0]}", () => []).Add((property.Name, input.Name));
                }
            }
        }
        Dictionary<string, (string From, string Input)> parent = new() { [samplerId] = (null, null) };
        Queue<string> queue = new([samplerId]);
        while (queue.Count > 0 && !parent.ContainsKey("9"))
        {
            string current = queue.Dequeue();
            foreach ((string next, string inputName) in consumers.GetValueOrDefault(current, []))
            {
                if (parent.TryAdd(next, (current, inputName)))
                {
                    queue.Enqueue(next);
                }
            }
        }
        Assert.That(parent.ContainsKey("9"), Is.True, "the segment sampler reaches the final save");
        List<(string Id, string Input)> path = [];
        for (string at = "9"; at is not null; at = parent[at].From)
        {
            path.Insert(0, (at, parent[at].Input));
        }
        string pathText = string.Join(" > ", path.Select(p => $"{p.Id}:{workflow[p.Id]["class_type"]}"));
        (string compositeId, string compositeInput) = path.First(p => $"{workflow[p.Id]["class_type"]}".Contains("CompositeMasked"));
        Assert.That(compositeInput, Is.EqualTo("source"), $"the redraw is the composite source: {pathText}");
        (string destId, JObject dest) = Source(workflow, workflow[compositeId]["inputs"]["destination"]);
        Assert.That(trace.Nodes[destId].Stage, Is.Not.EqualTo("seg11"), "the redraw is pasted onto the pre-segment image");
        JObject summary = (JObject)traced.Input.ExtraMeta[WorkflowTrace.MetadataKey];
        JObject samplerEntry = (JObject)((JArray)summary["samplers"]).Single(s => $"{s["node"]}".StartsWith($"{samplerId}:"));
        List<List<string>> downstream = [.. ((JArray)samplerEntry["to_outputs"]).Select(p => p.Select(t => $"{t}").ToList())];
        string compositeRef = $"{compositeId}:{workflow[compositeId]["class_type"]}";
        Assert.That(downstream.Any(p => p[0] == $"{samplerId}:{workflow[samplerId]["class_type"]}" && p.Contains(compositeRef) && p[^1] == "9:SwarmSaveImageWS"), Is.True, $"a downstream path runs from the sampler through {compositeRef} to the final save: {samplerEntry["to_outputs"].ToString(Formatting.None)}");
        // The crop is recorded as metadata (no crop image is saved): its nodes, checked against the graph the composite reads.
        JObject crop = (JObject)((JArray)summary["segments"]).Single()["crop"];
        JObject compositeInputs = (JObject)workflow[compositeId]["inputs"];
        Assert.That($"{crop["bounds"]}", Is.EqualTo($"{compositeInputs["x"][0]}"), "the crop's bounds node gives the paste-back position");
        JObject bounds = (JObject)workflow[$"{crop["bounds"]}"];
        Assert.That($"{bounds["class_type"]}", Is.EqualTo("SwarmMaskBounds"));
        Assert.That(crop["pasted_by"].Select(t => $"{t}"), Does.Contain(compositeRef));
        (string thresholdId, JObject threshold) = Source(workflow, bounds["inputs"]["mask"]);
        JObject scaled = (JObject)workflow[$"{crop["scaled_image"]}"]["inputs"];
        JObject expected = new()
        {
            ["grow"] = bounds["inputs"]["grow"],
            ["aspect_x"] = bounds["inputs"]["aspect_x"],
            ["aspect_y"] = bounds["inputs"]["aspect_y"],
            ["threshold_min"] = threshold["inputs"]["min"],
            ["threshold_max"] = threshold["inputs"]["max"],
            ["scaled_width"] = scaled["width"],
            ["scaled_height"] = scaled["height"]
        };
        foreach (JProperty field in expected.Properties())
        {
            Assert.That(JToken.DeepEquals(crop[field.Name], field.Value), Is.True, $"crop {field.Name}: {crop[field.Name]} against the graph's {field.Value}");
        }
        Assert.That($"{workflow[$"{crop["cropped_mask"]}"]["class_type"]}", Is.EqualTo("CropMask"));
    }

    /// <summary>Different options must produce visibly different records: stages, loads, and samplers.</summary>
    [Test]
    public void TestRecordsReflectOptions()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);
        T2IModel alt = ModelFor(model.ModelClass, true, "-alt");
        Lora("test_alien_lora");
        string segmentPrompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment} <lora:test_alien_lora>";
        WorkflowTrace plain = Generate(new() { Model = model, Prompt = PromptMarkerBase }, true).Trace;
        WorkflowTrace segment = Generate(new() { Model = model, Prompt = segmentPrompt }, true).Trace;
        WorkflowTrace refiner = Generate(new() { Model = model, Prompt = segmentPrompt, Setters = { i => i.Set(T2IParamTypes.RefinerMethod, "PostApply"), i => i.Set(T2IParamTypes.RefinerControl, 0.4) } }, true).Trace;
        WorkflowTrace segmentModel = Generate(new() { Model = model, Prompt = segmentPrompt, Setters = { i => i.Set(T2IParamTypes.SegmentModel, alt) } }, true).Trace;
        Assert.That(plain.Nodes.Values.Select(n => n.Stage).Distinct(), Is.EquivalentTo(new[] { "base" }));
        Assert.That(segment.Nodes.Values.Select(n => n.Stage).Distinct(), Does.Contain("seg11"));
        Assert.That(refiner.Nodes.Values.Select(n => n.Stage).Distinct(), Is.SupersetOf(new[] { "base", "refiner", "seg11" }));
        Assert.That(refiner.Loads.Any(l => l.Type == "Refiner" && l.Section == T2IParamInput.SectionID_Refiner), Is.True);
        WorkflowTrace.ModelLoadRecord segmentLoad = segmentModel.Loads.Single(l => l.Model == alt.Name);
        Assert.That(segmentLoad.Type, Is.EqualTo("Refiner"));
        Assert.That(segmentLoad.Section, Is.EqualTo(11));
        Assert.That(segmentLoad.Stage, Is.EqualTo("segment"));
        Assert.That(segment.SegmentParts.Count, Is.EqualTo(1));
        Assert.That(plain.SegmentParts.Count, Is.EqualTo(0));
        Assert.That(segment.ParamReads.ContainsKey("steps@11"), Is.True, string.Join(", ", segment.ParamReads.Keys));
        Assert.That(segment.ParamReads["segmentmaskblur"].FirstSource, Is.EqualTo(T2IParamInput.ParamReadSource.Default));
    }

    /// <summary>The image record: the full trace goes to the log, and the image gets the record tier that fits its format.
    /// Every tier names the samplers with their calls and the workflow identity; tier 0 lists each sampler-chain node with its provenance; lower tiers drop node detail, then the node table, in that order, saying what was omitted and where the full trace is.
    /// PNG and WebP keep tier 0 at any size; JPEG, in any of its format values, keeps the first tier that fits the metadata limit. The format comes from the input's Image Format parameter, read without marking it used.</summary>
    [Test]
    public void TestImageRecordTiers()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);
        Lora("test_alien_lora");
        Build traced = Generate(new() { Model = model, Prompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment} <lora:test_alien_lora>" }, true);
        WorkflowTrace trace = traced.Trace;
        JObject summary = trace.Build();
        string logReference = trace.LogReference;
        List<JObject> tiers = trace.RecordTiers(summary);
        Assert.That(tiers, Has.Count.EqualTo(5));
        int samplers = ((JArray)summary["samplers"]).Count;
        string samplerId = $"{((JArray)summary["samplers"])[0]["node"]}".Before(':');
        foreach (JObject tier in tiers)
        {
            Assert.That(JToken.DeepEquals(tier["workflow"], summary["workflow"]), Is.True, "every tier keeps the workflow identity");
            Assert.That($"{tier["full_trace"]}", Does.Contain(logReference));
            Assert.That($"{tier["log_key"]}", Is.EqualTo(trace.LogKey), "every tier keeps the log key");
        }
        foreach (JObject tier in tiers.Take(4))
        {
            Assert.That(((JArray)tier["samplers"]).Count, Is.EqualTo(samplers), "every tier but the notice names every sampler");
            Assert.That(tier["samplers"][0]["call"], Is.Not.Null, "every tier but the notice keeps the sampler calls");
        }
        Assert.That(tiers[4]["samplers"].Value<int>(), Is.EqualTo(samplers), "the notice counts the samplers");
        Assert.That(WorkflowTrace.RecordMetadataChars(tiers[4]), Is.LessThan(2000), "the notice fits beside nearly any other metadata");
        Assert.That(tiers[0], Is.SameAs(summary), "tier 0 is the summary itself");
        Assert.That(tiers[0]["omitted"], Is.Null);
        JObject samplerNode = (JObject)tiers[0]["nodes"][samplerId];
        Assert.That((samplerNode["inputs"] is not null, samplerNode["links"] is not null), Is.EqualTo((true, true)), "tier 0 nodes carry their inputs and links");
        Assert.That($"{samplerNode["origin"]}", Does.Contain("@WorkflowGenerator"), "tier 0 nodes carry the creating call site");
        Assert.That(samplerNode["step"], Is.Not.Null, "tier 0 nodes carry the step that made them");
        JObject briefNode = (JObject)tiers[1]["nodes"][samplerId];
        Assert.That(briefNode.Properties().Select(p => p.Name), Is.SubsetOf(new[] { "class", "stage", "origin", "step", "priority", "load", "compat", "reused", "overwritten", "untracked", "inserted_by" }), "tier 1 nodes keep class, stage and provenance only");
        Assert.That(($"{briefNode["class"]}", $"{briefNode["origin"]}"), Is.EqualTo(($"{samplerNode["class"]}", $"{samplerNode["origin"]}")));
        Assert.That(tiers[2]["nodes"], Is.Null);
        Assert.That(tiers[2]["decision_counts"], Is.Not.Null);
        List<string> tier0Chain = [.. tiers[0]["samplers"][0]["inputs"]["model"].Select(v => $"{v}")], tier2Chain = [.. tiers[2]["samplers"][0]["inputs"]["model"].Select(v => $"{v}")];
        Assert.That(tier0Chain, Is.All.Matches<string>(id => tiers[0]["nodes"][id] is not null), "tier 0 chains name nodes of its node table");
        Assert.That(tier2Chain, Is.EqualTo(tier0Chain.Select(id => $"{id}:{traced.Workflow[id]["class_type"]}")), "tier 2 has no node table, so its chains name each node as id:Class");
        Assert.That(tiers.Skip(1).All(t => $"{t["omitted"]}".Contains(logReference)), Is.True, "lower tiers say what was omitted and where the full trace is");
        Assert.That(tiers.Select(WorkflowTrace.RecordMetadataChars), Is.Ordered.Descending, "each tier is smaller than the one before");
        int tier0 = WorkflowTrace.RecordMetadataChars(tiers[0]);
        Assert.That(WorkflowTrace.ForImageMetadata(tiers, "PNG", 0, 10), Is.SameAs(tiers[0]), "PNG keeps tier 0 over any limit");
        Assert.That(WorkflowTrace.ForImageMetadata(tiers, "JPG", 1000, tier0 + 1000), Is.SameAs(tiers[0]), "JPEG keeps tier 0 when it fits with the other metadata");
        Assert.That(WorkflowTrace.ForImageMetadata(tiers, "JPG", 1001, tier0 + 1000), Is.SameAs(tiers[1]), "the other metadata counts against the limit");
        Assert.That(WorkflowTrace.ForImageMetadata(tiers, "JPG75", 0, 10), Is.SameAs(tiers[4]), "the notice when nothing fits");
        foreach (string unlimited in new[] { "WEBP", "WEBP_LOSSLESS", "WEBP_90", "mp4", "wav" })
        {
            Assert.That(WorkflowTrace.ForImageMetadata(tiers, unlimited, 0, 10), Is.SameAs(tiers[0]), $"{unlimited} keeps tier 0 over any limit: its metadata is not in a JPEG EXIF segment");
        }
        Assert.That(JToken.DeepEquals(tiers[4], MetadataHelper.TraceNotice(tiers[0], $"{tiers[4]["omitted"]}")), Is.True, "the notice tier is the notice of tier 0");
        Assert.That(JToken.DeepEquals((JObject)traced.Input.ExtraMeta[WorkflowTrace.MetadataKey], tiers[0]), Is.True, "a PNG generation stores tier 0");
        Assert.That(summary["nodes"], Is.Not.Null, "the full trace keeps every node by stage");
        Assert.That(summary["params_read"], Is.Not.Null, "the full trace keeps every parameter read");
        Build jpegBuild = Generate(new() { Model = model, Prompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment} <lora:test_alien_lora>", Params = new() { ["imageformat"] = "JPG" } }, true);
        Assert.That(WorkflowTrace.ImageFormatOf(jpegBuild.Input), Is.EqualTo("JPG"));
        Assert.That(jpegBuild.Input.ParamsQueried.Contains(T2IParamTypes.ImageFormat.Type.ID), Is.False, "reading the format did not mark Image Format as used");
        Assert.That(WorkflowTrace.ImageFormatOf(traced.Input), Is.EqualTo("PNG"), "with no Image Format parameter, the default user's format");
    }

    /// <summary>Each feature variant reaches the generator step it exists to cover, and the trace attributes it (stage, step, media, prompt labels, loads), so the fitness sweep is not silently skipping features.</summary>
    [Test]
    public void TestFeatureVariantsReachTheirSteps()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        ControlNetFor(model.ModelClass);
        static IEnumerable<string> labels(WorkflowTrace t) => t.PromptLabels.Values.SelectMany(l => l);
        static bool byStep(WorkflowTrace t, string step) => t.Nodes.Values.Any(n => n.Step >= 0 && t.StepRecords[n.Step].Name == step);
        Dictionary<string, Func<WorkflowTrace, bool>> reached = new()
        {
            ["regions+objects"] = t => labels(t).Any(l => l.Contains(".region")) && labels(t).Any(l => l.Contains(".object")),
            ["initimage+mask"] = t => t.Media.Values.Any(m => m.Param == "maskimage"),
            ["controlnet"] = t => t.Media.Values.Any(m => m.Param == T2IParamTypes.Controlnets[0].Image.Type.ID) && byStep(t, "ControlNet"),
            ["refiner-upscale"] = t => t.ParamReads.Keys.Any(k => k.Before('@') == T2IParamTypes.RefinerUpscale.Type.ID) && t.Nodes.Values.Any(n => n.Stage == "refiner"),
            ["clear+variationseed"] = t => byStep(t, "Segment Clears") && t.ParamReads.Values.Any(r => r.Type == T2IParamTypes.VariationSeed.Type && $"{r.LastValue}" == "5"),
            ["image2video"] = t => t.Nodes.Values.Any(n => n.Stage == "video") && t.Loads.Any(l => l.Stage == "video"),
            ["image2video+extend"] = t => t.Nodes.Values.Any(n => n.Stage.StartsWith("extend")) && byStep(t, "Extend Video"),
            ["image2video+swap"] = t => t.Loads.Any(l => l.Stage == "videoswap" && l.Compat == VideoSwapTestModel.ModelClass.CompatClass.ID)
                && t.Gen.FinalLoadedModel?.ModelClass?.CompatClass?.ID != VideoSwapTestModel.ModelClass.CompatClass.ID
                && t.Nodes.Values.Count(n => n.Stage == "videoswap" && n.Load < 0) > 0
                && t.Nodes.Values.Where(n => n.Stage == "videoswap" && n.Load < 0).All(n => n.Compat == VideoSwapTestModel.ModelClass.CompatClass.ID),
            ["seedvr"] = t => t.Nodes.Values.Any(n => n.Stage == "seedvr") && t.Loads.Any(l => l.Type == "SeedVR2")
        };
        List<Variant> features = [.. Variants().Where(v => v.Feature)];
        Assert.That(features.Select(v => v.Name), Is.EquivalentTo(reached.Keys), "every feature variant has a reach check");
        List<string> missed = [];
        foreach (Variant variant in features)
        {
            WorkflowTrace trace = Generate(variant.Make(model), true).Trace;
            if (!reached[variant.Name](trace))
            {
                missed.Add($"{variant.Name}: stages [{trace.Nodes.Values.Select(n => n.Stage).Distinct().JoinString(", ")}], steps with nodes [{trace.Nodes.Values.Where(n => n.Step >= 0).Select(n => trace.StepRecords[n.Step].Name).Distinct().JoinString(", ")}], media [{trace.Media.Values.Select(m => m.Param).JoinString(", ")}], loads [{trace.Loads.Select(l => $"{l.Type}@{l.Stage}").JoinString(", ")}], labels [{labels(trace).JoinString(", ")}]");
            }
        }
        Assert.That(missed, Is.Empty, string.Join("\n", missed));
    }

    /// <summary>An object prompt's composition record says which crop of the base image each object pass inpainted, at what size, with what mask, and links the pass to its own workflow trace, without raw prompt text.</summary>
    [Test]
    public void TestObjectPassRecord()
    {
        PromptRegion.Part part = new PromptRegion($"{PromptMarkerBase} <object:0.5,0.25,0.5,0.5> zqxobject cat").Parts.Single(p => p.Type == PromptRegion.PartType.Object);
        // A 1024x1024 base: the object is 512,256 512x512, the crop (10% over-bound) is 460,204 564x614, inpainted at 2x.
        JObject pass = T2IMultiStepObjectBuilder.TraceObjectPass(part, 512, 256, 512, 512, 460, 204, 564, 614, 1128, 1228, 2.0, 2.0);
        Assert.That($"{pass["pass"]}", Is.EqualTo($"object{part.ContextID}"));
        Assert.That($"{pass["prompt"]}", Is.EqualTo($"@object{part.ContextID}{T2IMultiStepObjectBuilder.TraceHash(part.Prompt)}"));
        static JObject box(int x, int y, int width, int height) => new() { ["x"] = x, ["y"] = y, ["width"] = width, ["height"] = height };
        foreach ((string key, JObject expected) in new[] { ("object_px", box(512, 256, 512, 512)), ("crop_px", box(460, 204, 564, 614)), ("mask_px", box(104, 104, 1024, 1024)), ("inpaint_size", new JObject() { ["width"] = 1128, ["height"] = 1228 }) })
        {
            Assert.That(JToken.DeepEquals(pass[key], expected), Is.True, $"{key}: {pass[key]?.ToString(Formatting.None)}");
        }
        Assert.That(JToken.DeepEquals(pass["region"], new JObject() { ["x"] = part.X, ["y"] = part.Y, ["width"] = part.Width, ["height"] = part.Height }), Is.True, $"region: {pass["region"]?.ToString(Formatting.None)}");
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        T2IParamInput passInput = BuildInput(new() { Model = model, Prompt = part.Prompt });
        passInput.ExtraMeta[WorkflowTrace.MetadataKey] = new JObject() { ["workflow"] = new JObject() { ["sent_sha256"] = "feedface" } };
        T2IMultiStepObjectBuilder.TracePassResult(pass, passInput, TestImage(64, 32));
        Assert.That($"{pass["workflow"]["sent_sha256"]}", Is.EqualTo("feedface"));
        Assert.That(pass["seed"].Value<long>(), Is.EqualTo(1L));
        Assert.That($"{pass["model"]}", Is.EqualTo(T2IParamTypes.CleanModelName(model.Name)));
        JObject result = (JObject)pass["result"];
        Assert.That((result["width"].Value<int>(), result["height"].Value<int>(), $"{result["mime"]}", $"{result["raw_sha8"]}".Length), Is.EqualTo((64, 32, "image/png", 8)));
        Assert.That(pass.ToString(Formatting.None), Does.Not.Contain("zqx"));
        Assert.That(T2IMultiStepObjectBuilder.TraceMetadataKey, Is.Not.EqualTo(WorkflowTrace.MetadataKey), "the cleanup pass's own trace must not overwrite the composition record");
    }

    /// <summary>Tracing a generation that fails still balances its scopes and lets the original error through.</summary>
    [Test]
    public void TestFailedGenerationRethrows()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);
        T2IParamInput input = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
        input.Set(T2IParamTypes.Loras, ["test_missing_lora_that_does_not_exist"]);
        input.Set(T2IParamTypes.LoraWeights, ["1"]);
        WorkflowGenerator generator = new() { UserInput = input, Features = [.. FullFeatures] };
        generator.Trace = new(generator);
        SwarmUserErrorException caught = null;
        try
        {
            generator.Generate();
        }
        catch (SwarmUserErrorException ex)
        {
            caught = ex;
        }
        Assert.That(caught, Is.Not.Null);
        Assert.That(generator.Trace.Error, Does.Contain("SwarmUserErrorException"));
        Assert.That(input.ParamReadTracker, Is.Null);
    }

    /// <summary>Each trace logs under its own key, which the image record stores, so one image's log line is found even when the images of a batch share a request ID.
    /// The record says where that log line is kept under the current server log settings.</summary>
    [Test]
    public void TestLogKeyIsUniquePerTraceAndRetentionIsStated()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        T2IParamInput input = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
        List<WorkflowTrace> traces = [];
        foreach (T2IParamInput batchImage in new[] { input.Clone(), input.Clone() })
        {
            WorkflowGenerator generator = new() { UserInput = batchImage, Features = [.. FullFeatures] };
            generator.Trace = new(generator);
            generator.Generate();
            Assert.That($"{((JObject)batchImage.ExtraMeta[WorkflowTrace.MetadataKey])["log_key"]}", Is.EqualTo(generator.Trace.LogKey));
            traces.Add(generator.Trace);
        }
        Assert.That(traces[0].LogKey, Is.Not.EqualTo(traces[1].LogKey), "two images of one request log under different keys");
        Logs.LogLevel priorLevel = Logs.MinimumLevel;
        System.Collections.Concurrent.ConcurrentQueue<string> priorQueue = Logs.LogsToSave;
        try
        {
            Logs.MinimumLevel = Logs.LogLevel.Info;
            Logs.LogsToSave = null;
            Assert.That(WorkflowTrace.DebugLogRetention(), Does.Contain("only").And.Contain("Info"));
            Logs.MinimumLevel = Logs.LogLevel.Debug;
            Assert.That(WorkflowTrace.DebugLogRetention(), Does.Contain("not saved"));
            Logs.LogsToSave = new();
            Assert.That(WorkflowTrace.DebugLogRetention(), Does.EndWith("saved to the log file"));
        }
        finally
        {
            Logs.MinimumLevel = priorLevel;
            Logs.LogsToSave = priorQueue;
        }
    }

    /// <summary>A generator run outside <see cref="ComfyUIAPIAbstractBackend.CreateWorkflow"/>, as the model preload runs it, is untraced even with AddDebugData on.</summary>
    [Test]
    public void TestGenerateAloneIsUntraced()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        bool priorDebug = Program.ServerSettings.AddDebugData;
        try
        {
            Program.ServerSettings.AddDebugData = true;
            WorkflowGenerator generator = new() { UserInput = BuildInput(new() { Model = model, Prompt = PromptMarkerBase }), Features = [.. FullFeatures] };
            generator.Generate();
            Assert.That((generator.Trace, generator.UserInput.ExtraMeta.ContainsKey(WorkflowTrace.MetadataKey)), Is.EqualTo(((WorkflowTrace)null, false)));
        }
        finally
        {
            Program.ServerSettings.AddDebugData = priorDebug;
        }
    }

    /// <summary>The backend send path records what it sends, and a custom workflow is described without the generator.</summary>
    [Test]
    public void TestCreateWorkflowRecordsSentWorkflow()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        bool priorDebug = Program.ServerSettings.AddDebugData;
        try
        {
            Program.ServerSettings.AddDebugData = false;
            T2IParamInput plainInput = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
            string plain = ComfyUIAPIAbstractBackend.CreateWorkflow(plainInput, w => w, null, [.. FullFeatures]);
            Assert.That(plainInput.ExtraMeta.ContainsKey(WorkflowTrace.MetadataKey), Is.False);
            Program.ServerSettings.AddDebugData = true;
            T2IParamInput tracedInput = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
            string traced = ComfyUIAPIAbstractBackend.CreateWorkflow(tracedInput, w => w, null, [.. FullFeatures]);
            Assert.That(traced, Is.EqualTo(plain));
            JObject summary = (JObject)tracedInput.ExtraMeta[WorkflowTrace.MetadataKey];
            Assert.That($"{summary["workflow"]["sent_sha256"]}", Is.EqualTo(Utilities.HashSHA256(traced.EncodeUTF8())));
            Assert.That(summary["workflow"]["rewritten_after_generate"].Value<bool>(), Is.False);
            JObject metadata = tracedInput.GenFullMetadataObject();
            Assert.That(metadata["sui_extra_data"]?[WorkflowTrace.MetadataKey], Is.Not.Null, "the trace reaches the image metadata");
            JObject plainMetadata = plainInput.GenFullMetadataObject();
            Assert.That(metadata["sui_image_params"].ToString(), Is.EqualTo(plainMetadata["sui_image_params"].ToString()), "tracing changes no parameter metadata");
            Assert.That(metadata["sui_extra_data"]["unused_parameters"]?.ToString(), Is.EqualTo(plainMetadata["sui_extra_data"]?["unused_parameters"]?.ToString()), "tracing changes no unused-parameter detection");
            JObject external = WorkflowTrace.DescribeExternalWorkflow("custom", plain);
            Assert.That(((JObject)external["nodes"]).Count, Is.EqualTo(JObject.Parse(plain).Count));
            Assert.That($"{external["workflow"]["sent_sha256"]}", Is.EqualTo(Utilities.HashSHA256(plain.EncodeUTF8())));
            Assert.That(external.ToString(Formatting.None), Does.Not.Contain("zqx"), "custom workflow prompt text is hashed");
            T2IParamInput customInput = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
            WorkflowTrace.RecordExternalWorkflow(customInput, "custom", plain);
            JObject stored = (JObject)customInput.ExtraMeta[WorkflowTrace.MetadataKey];
            Assert.That($"{stored["log_key"]}", Has.Length.EqualTo(12), "the custom workflow record has its own log key");
            stored.Remove("log_key");
            Assert.That(stored.ToString(), Is.EqualTo(external.ToString()));
        }
        finally
        {
            Program.ServerSettings.AddDebugData = priorDebug;
        }
    }

    /// <summary>A custom workflow with one "${...}" placeholder per fill route of <see cref="ComfyUIAPIAbstractBackend.CreateWorkflow"/>: named tags, the image size, and a dynamic raw-workflow input.</summary>
    public const string ParamReadTemplate = """
        {
            "1": { "class_type": "KSampler", "inputs": { "seed": "${seed}", "steps": "${steps}", "cfg": "${cfg_scale}" } },
            "2": { "class_type": "EmptyLatentImage", "inputs": { "width": "${width}", "height": "${height}" } },
            "3": { "class_type": "CLIPTextEncode", "inputs": { "text": "${prompt}" } },
            "4": { "class_type": "CLIPTextEncode", "inputs": { "text": "${negative_prompt}" } },
            "5": { "class_type": "PrimitiveString", "inputs": { "value": "${comfyrawworkflowinputtextstyle:plain}" } }
        }
        """;

    /// <summary>A custom workflow's record lists the parameter reads behind every filled placeholder, plus the custom-workflow and ControlNet-preview checks, with prompt and free text hashed.
    /// A generated workflow's record lists the reads <see cref="ComfyUIAPIAbstractBackend.CreateWorkflow"/> made before generating under "params_read", which a generator traced alone does not make,
    /// and the reads made while preparing the text to send, and only those, under "workflow"."params_read".</summary>
    [Test]
    public void TestCreateWorkflowRecordsEveryParameterRead()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        string customId = ComfyUIBackendExtension.CustomWorkflowParam.Type.ID;
        bool priorDebug = Program.ServerSettings.AddDebugData;
        Func<string, T2IParamInput, T2IParamType> rawInputTypes = new ComfyUIBackendExtension().DynamicParamGenerator;
        T2IParamTypes.FakeTypeProviders.Add(rawInputTypes);
        try
        {
            Program.ServerSettings.AddDebugData = true;
            const string styleText = "zqxstyle custom words";
            T2IParamInput custom = BuildInput(new()
            {
                Model = model,
                Prompt = PromptMarkerBase,
                Setters =
                [
                    i => i.Set(ComfyUIBackendExtension.CustomWorkflowParam, $"PARSED%trace test%{ParamReadTemplate}"),
                    i => T2IParamTypes.ApplyParameter("comfyrawworkflowinputtextstyle", styleText, i)
                ]
            });
            string filled = ComfyUIAPIAbstractBackend.CreateWorkflow(custom, w => w, null, [.. FullFeatures]);
            Assert.That(filled, Does.Contain(styleText), "the dynamic input is filled from its parameter");
            JObject record = (JObject)custom.ExtraMeta[WorkflowTrace.MetadataKey];
            Assert.That($"{record["workflow"]["source"]}", Is.EqualTo("custom"));
            string[] behindPlaceholders = [T2IParamTypes.Seed.Type.ID, T2IParamTypes.Steps.Type.ID, T2IParamTypes.CFGScale.Type.ID, T2IParamTypes.Width.Type.ID, T2IParamTypes.Height.Type.ID,
                T2IParamTypes.Prompt.Type.ID, T2IParamTypes.NegativePrompt.Type.ID, "comfyrawworkflowinputtextstyle"];
            JObject reads = (JObject)record["params_read"];
            Assert.That(reads.Properties().Select(p => p.Name), Is.SupersetOf(behindPlaceholders.Append(customId).Append(T2IParamTypes.ControlNetPreviewOnly.Type.ID)), reads.ToString());
            Assert.That(record.ToString(), Does.Not.Contain("zqx"), "prompt text and free text are hashed");
            Assert.That($"{WorkflowTrace.FormatParamValue(null, styleText)}", Is.EqualTo(styleText), "negative control: without free-text hashing the style text is written raw");
            T2IParamInput generated = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
            ComfyUIAPIAbstractBackend.CreateWorkflow(generated, w => { generated.GetImageWidth(); return w; }, null, [.. FullFeatures]);
            JObject stored = (JObject)generated.ExtraMeta[WorkflowTrace.MetadataKey];
            Assert.That(((JObject)stored["params_read"]).ContainsKey(customId), Is.True, "the custom-workflow check before generation is recorded");
            Assert.That(((JObject)stored["workflow"]["params_read"]).Properties().Select(p => p.Name), Is.EquivalentTo(new[] { T2IParamTypes.RawResolution.Type.ID, T2IParamTypes.SideLength.Type.ID, T2IParamTypes.Width.Type.ID }));
            WorkflowGenerator alone = new() { UserInput = BuildInput(new() { Model = model, Prompt = PromptMarkerBase }), Features = [.. FullFeatures] };
            alone.Trace = new(alone);
            alone.Generate();
            Assert.That(((JObject)alone.Trace.Summary["params_read"]).ContainsKey(customId), Is.False, "negative control: the generator itself does not read the custom-workflow parameter");
        }
        finally
        {
            Program.ServerSettings.AddDebugData = priorDebug;
            T2IParamTypes.FakeTypeProviders.Remove(rawInputTypes);
        }
    }

    /// <summary>Image metadata keeps Swarm's indented JSON, except the trace record's value, which is written on one line: the text equals Newtonsoft's own indented serialization with the record given as raw compact JSON (non-ASCII escaped, as the metadata settings escape it), and parses back to the same object.
    /// The metadata around it stays indented, including what follows the record in its parent; a key of the same name elsewhere is indented as usual; metadata without a record is serialized exactly as before.</summary>
    [Test]
    public void TestMetadataWritesTheTraceRecordOnOneLine()
    {
        JObject record = new()
        {
            ["version"] = 1,
            ["nodes"] = new JObject() { ["4"] = new JObject() { ["class"] = "CheckpointLoaderSimple", ["inputs"] = new JObject() { ["ckpt_name"] = "café.safetensors" }, ["outputs"] = new JObject() { ["0"] = new JObject() { ["types"] = new JArray("MODEL") } } } },
            ["steps"] = new JArray(new JObject() { ["name"] = "Model Loader", ["created"] = new JArray() }, new JObject()),
            ["empty"] = new JObject()
        };
        JObject metadata()
        {
            return new()
            {
                ["sui_image_params"] = new JObject() { ["prompt"] = "a road", [WorkflowTrace.MetadataKey] = new JObject() { ["kept"] = new JArray(1, 2) } },
                ["sui_extra_data"] = new JObject() { ["before"] = "x", [WorkflowTrace.MetadataKey] = record.DeepClone(), ["after"] = new JObject() { ["nested"] = new JArray(1, new JObject() { ["deep"] = true }) } },
                ["sui_models"] = new JArray(new JObject() { ["name"] = "m" })
            };
        }
        string text = T2IParamInput.MetadataToString(metadata());
        JObject withRaw = metadata();
        withRaw["sui_extra_data"][WorkflowTrace.MetadataKey] = new JRaw(JsonConvert.SerializeObject(record, new JsonSerializerSettings() { StringEscapeHandling = StringEscapeHandling.EscapeNonAscii }));
        Assert.That(text, Is.EqualTo(JsonConvert.SerializeObject(withRaw, T2IParamInput.SafeSerializer).Replace("\r\n", "\n")));
        Assert.That(JToken.DeepEquals(JObject.Parse(text), metadata()), Is.True, "the text parses back to the metadata");
        Assert.That(text, Does.Contain("caf\\u00e9.safetensors"), "non-ASCII is escaped inside the record too");
        string indented = JsonConvert.SerializeObject(metadata(), T2IParamInput.SafeSerializer).Replace("\r\n", "\n");
        Assert.That(text.Length, Is.LessThan(indented.Length), "negative control: the indented text differs, and is longer");
        Assert.That(text.Split('\n').Count(l => l.Contains($"\"{WorkflowTrace.MetadataKey}\"")), Is.EqualTo(2), "the record is one line, and the key of the same name in the parameters keeps its own line");
        JObject untraced = metadata();
        ((JObject)untraced["sui_extra_data"]).Remove(WorkflowTrace.MetadataKey);
        Assert.That(T2IParamInput.MetadataToString(untraced), Is.EqualTo(JsonConvert.SerializeObject(untraced, T2IParamInput.SafeSerializer).Replace("\r\n", "\n")), "metadata without a record is serialized as before");
    }

    /// <summary>The server log form of a record has one line per top-level entry and per entry of a top-level table or list, with anything deeper on that line, and parses back to the record.</summary>
    [Test]
    public void TestLogJsonPutsEachEntryOnALine()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        JObject record = Generate(new() { Model = model, Prompt = PromptMarkerBase }, true).Trace.Build();
        string text = WorkflowTrace.LogJson(record);
        using JsonTextReader reader = new(new StringReader(text)) { DateParseHandling = DateParseHandling.None };
        Assert.That(JToken.ReadFrom(reader).ToString(Formatting.None), Is.EqualTo(record.ToString(Formatting.None)), "the log text reads back to the record");
        int expectedLines = 2 + record.Properties().Sum(p => p.Value is JContainer container && container.HasValues ? container.Count + 2 : 1);
        string[] lines = text.Split('\n');
        Assert.That(lines, Has.Length.EqualTo(expectedLines), text);
        foreach ((string id, JToken node) in (JObject)record["nodes"])
        {
            Assert.That(lines, Has.One.EqualTo($"    {JsonConvert.ToString(id)}: {node.ToString(Formatting.None)}{(id == ((JObject)record["nodes"]).Properties().Last().Name ? "" : ",")}"), $"node {id} is one line");
        }
    }

    /// <summary>Returns the bytes of a committed media fixture (Fixtures/media, made by the justfile there).</summary>
    public static byte[] MediaFixture(string name) => File.ReadAllBytes(Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "media", name));

    /// <summary>Media records and media parameter values carry what the file's header says: an image's size, an animation's frame count, and the duration of audio and video, with audio's sample rate and channels; an image the generator resized carries the size it was resized to.
    /// The fixture values are ffprobe's: a 0.5 s, 8000 Hz mono WAV and a 1 s, 64x48, 8-frame MP4. A video's frame size is not read, and bytes no reader recognizes give nothing.</summary>
    [Test]
    public void TestMediaHeadersGiveSizeFramesAndDuration()
    {
        AudioFile wav = new(MediaFixture("tone-0.5s-8khz-mono.wav"), MediaType.AudioWav);
        VideoFile mp4 = new(MediaFixture("pattern-1s-64x48-8fps.mp4"), MediaType.VideoMp4);
        static void same(JToken actual, JObject expected, string what) => Assert.That(JToken.DeepEquals(actual, expected), Is.True, $"{what}: {actual?.ToString(Formatting.None)}");
        same(WorkflowTrace.MediaFields(new("promptaudios.0", wav, null, null)), new() { ["duration_s"] = 0.5, ["sample_rate"] = 8000.0, ["channels"] = 1, ["mime"] = "audio/wav" }, "wav");
        same(WorkflowTrace.MediaFields(new("promptvideos.0", mp4, null, null)), new() { ["duration_s"] = 1.0, ["mime"] = "video/mp4" }, "mp4");
        same(WorkflowTrace.MediaFields(new("initimage", InitTestImage, 256, 192)), new() { ["width"] = 256, ["height"] = 192, ["resized"] = true, ["mime"] = "image/png" }, "resized image");
        Assert.That(WorkflowTrace.MediaHeader(new ImageFile() { RawData = mp4.RawData.ToArray(), Type = mp4.Type }), Is.EqualTo(new WorkflowTrace.MediaFacts(DurationSeconds: 1.0)), "a video that LoadVideo wraps in an ImageFile is read as video");
        same(WorkflowTrace.FormatParamValue(null, wav), new() { ["raw_sha8"] = MetadataHelper.Sha8(wav.RawData), ["duration_s"] = 0.5, ["sample_rate"] = 8000.0, ["channels"] = 1, ["mime"] = "audio/wav" }, "a media parameter value carries the header too");
        using SixLabors.ImageSharp.Image<Rgba32> animation = new(20, 10);
        animation.Frames.CreateFrame();
        animation.Frames.CreateFrame();
        using MemoryStream gif = new();
        SixLabors.ImageSharp.ImageExtensions.SaveAsGif(animation, gif);
        Assert.That(WorkflowTrace.MediaHeader(new Image(gif.ToArray(), MediaType.ImageGif)), Is.EqualTo(new WorkflowTrace.MediaFacts(20, 10, 3)));
        using MemoryStream png = new();
        SixLabors.ImageSharp.ImageExtensions.SaveAsPng(new SixLabors.ImageSharp.Image<Rgba32>(20, 10), png);
        Assert.That(WorkflowTrace.MediaHeader(new Image(png.ToArray(), MediaType.ImagePng)), Is.EqualTo(new WorkflowTrace.MediaFacts(20, 10)));
        Assert.That(WorkflowTrace.MediaHeader(new VideoFile([1, 2, 3, 4], MediaType.VideoMp4)), Is.EqualTo(WorkflowTrace.MediaFacts.None), "bytes no reader recognizes give nothing");
    }

    /// <summary>A text-to-image backend that is not a ComfyUI backend, for <see cref="TestBackendWithoutTraceLeavesANotice"/>.</summary>
    public class NonComfyBackend : AbstractT2IBackend
    {
        /// <inheritdoc/>
        public override IEnumerable<string> SupportedFeatures => [];

        /// <inheritdoc/>
        public override Task Init() => Task.CompletedTask;

        /// <inheritdoc/>
        public override Task Shutdown() => Task.CompletedTask;

        /// <inheritdoc/>
        public override Task<Image[]> Generate(T2IParamInput user_input) => Task.FromResult(Array.Empty<Image>());

        /// <inheritdoc/>
        public override Task<bool> LoadModel(T2IModel model, T2IParamInput input) => Task.FromResult(true);
    }

    /// <summary>With debug data on, an image from a backend that records no workflow trace carries a notice naming the backend; a ComfyUI backend's record replaces the notice; with debug data off there is neither.</summary>
    [Test]
    public void TestBackendWithoutTraceLeavesANotice()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        NonComfyBackend backend = new() { AbstractBackendData = new BackendHandler.T2IBackendData() { ID = 5, BackType = new("noncomfy", "Non-Comfy Test Backend", "", null, null, typeof(NonComfyBackend), null) } };
        bool priorDebug = Program.ServerSettings.AddDebugData;
        try
        {
            Program.ServerSettings.AddDebugData = true;
            T2IParamInput input = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
            T2IEngine.MarkTraceNotRecorded(input, backend);
            JObject metadata = input.GenFullMetadataObject();
            Assert.That($"{metadata["sui_extra_data"][WorkflowTrace.MetadataKey]["notice"]}", Does.StartWith("no workflow trace: backend 5 (Non-Comfy Test Backend) recorded none."));
            Assert.That(metadata["sui_extra_data"][WorkflowTrace.MetadataKey]["version"].Value<int>(), Is.EqualTo(WorkflowTrace.FormatVersion));
            ComfyUIAPIAbstractBackend.CreateWorkflow(input, w => w, null, [.. FullFeatures]);
            JObject record = (JObject)input.ExtraMeta[WorkflowTrace.MetadataKey];
            Assert.That((record["notice"], $"{record["log_key"]}".Length), Is.EqualTo(((JToken)null, 12)), "a ComfyUI backend's record replaces the notice");
            Program.ServerSettings.AddDebugData = false;
            T2IParamInput untraced = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
            T2IEngine.MarkTraceNotRecorded(untraced, backend);
            Assert.That(untraced.ExtraMeta.ContainsKey(WorkflowTrace.MetadataKey), Is.False);
        }
        finally
        {
            Program.ServerSettings.AddDebugData = priorDebug;
        }
    }
}
