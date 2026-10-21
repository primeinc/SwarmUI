using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SixLabors.ImageSharp.Metadata.Profiles.Exif;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
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

    /// <summary>A named request variant.</summary>
    /// <param name="Name">Variant name.</param>
    /// <param name="Make">Builds the request for a given main model.</param>
    /// <param name="CheckpointOnly">If true, only built with checkpoint-format main models: the variant exercises generator steps downstream of the loader, which the diffusion_models format does not change.</param>
    public record class Variant(string Name, Func<T2IModel, Request> Make, bool CheckpointOnly = false);

    /// <summary>The option variants every model class is built with.</summary>
    public static List<Variant> Variants()
    {
        Lora("test_alien_lora");
        string segmentPrompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment} <lora:test_alien_lora>";
        static void refiner(T2IParamInput input)
        {
            input.Set(T2IParamTypes.RefinerMethod, "PostApply");
            input.Set(T2IParamTypes.RefinerControl, 0.4);
        }
        return
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
            new("seedvr", m => new() { Model = m, Prompt = PromptMarkerBase, Setters = { i => i.Set(ComfyUIBackendExtension.SeedVRModel, SeedVRTestModel) } }, true)
        ];
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

    /// <summary>Checks the trace invariants for one traced build, appending violations to the list. Returns the stored summary's metadata length, and whether a JPEG save would store the use-PNG notice instead of the trace.</summary>
    public static (int SummaryChars, bool JpegNotice) CheckTrace(Build build, string label, List<string> failures)
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
            if (record.Step < 0 || string.IsNullOrEmpty(record.Origin) || string.IsNullOrEmpty(record.Stage))
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
        if (trace.StepStack.Count != 0 || trace.LoadStack.Count != 0 || trace.Stages.Count != 0)
        {
            failures.Add($"{label}: unbalanced trace scopes (steps={trace.StepStack.Count}, loads={trace.LoadStack.Count}, stages={trace.Stages.Count})");
        }
        if (build.Input.ExtraMeta.GetValueOrDefault(WorkflowTrace.MetadataKey) is not JObject stored)
        {
            failures.Add($"{label}: no {WorkflowTrace.MetadataKey} in ExtraMeta");
            return (0, false);
        }
        int storedLength = T2IParamInput.MetadataToString(stored).Length;
        string format = WorkflowTrace.ImageFormatOf(build.Input);
        if (format != "PNG")
        {
            failures.Add($"{label}: the harness saves as {format}, the checks below expect the PNG default");
        }
        if (!JToken.DeepEquals(stored, trace.Build()))
        {
            failures.Add($"{label}: the summary stored for a PNG image is not the full trace");
        }
        JObject jpeg = WorkflowTrace.ForImageMetadata(stored, "JPG", "[WorkflowTrace] request test");
        bool jpegNotice = jpeg["omitted"] is not null;
        build.Input.ExtraMeta[WorkflowTrace.MetadataKey] = jpeg;
        int exifBytes = ExifBytes(build.Input.GenRawMetadata());
        build.Input.ExtraMeta[WorkflowTrace.MetadataKey] = stored;
        if (exifBytes > MaxSingleApp1ExifBytes)
        {
            failures.Add($"{label}: JPEG image metadata EXIF is {exifBytes} bytes, over one APP1 segment ({MaxSingleApp1ExifBytes})");
        }
        if (stored.ToString(Formatting.None).Contains("zqx"))
        {
            failures.Add($"{label}: raw prompt text appears in the stored trace summary");
        }
        JObject summary = stored;
        if (summary["untracked_nodes"] is not null)
        {
            failures.Add($"{label}: summary lists untracked nodes {summary["untracked_nodes"].ToString(Formatting.None)}");
        }
        int listed = (summary["nodes"] as JObject)?.Properties().Sum(p => (p.Value as JArray)?.Count ?? 0) ?? 0;
        if (listed != build.Workflow.Count)
        {
            failures.Add($"{label}: summary lists {listed} nodes but the workflow has {build.Workflow.Count}");
        }
        int samplerNodes = build.Workflow.Properties().Count(p => WorkflowTrace.SamplerClasses.Contains($"{p.Value["class_type"]}"));
        int samplerEntries = (summary["samplers"] as JArray)?.Count(s => !$"{s["node"]}".Contains("(removed by")) ?? 0;
        if (samplerEntries < samplerNodes)
        {
            failures.Add($"{label}: {samplerNodes} sampler nodes but {samplerEntries} sampler records");
        }
        return (storedLength, jpegNotice);
    }

    /// <summary>Outcome of one fitness case.</summary>
    /// <param name="Label">Case label.</param>
    /// <param name="Rejected">The generator's readable rejection of the untraced build, if any.</param>
    /// <param name="Crash">A non-readable exception from the untraced build, if any.</param>
    /// <param name="Failures">Trace invariant violations.</param>
    /// <param name="PlainTicks">Untraced build time.</param>
    /// <param name="TracedTicks">Traced build time.</param>
    /// <param name="SummaryChars">Stored summary metadata length.</param>
    /// <param name="JpegNotice">True if a JPEG save would store the use-PNG notice instead of the trace.</param>
    public record class CaseOutcome(string Label, string Rejected, string Crash, List<string> Failures, long PlainTicks, long TracedTicks, int SummaryChars = 0, bool JpegNotice = false);

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
        (int summaryChars, bool jpegNotice) = CheckTrace(traced, label, failures);
        return new(label, null, null, failures, plainTicks, tracedTicks, summaryChars, jpegNotice);
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
                foreach (Variant variant in variants.Where(v => !diffusionModels || !v.CheckpointOnly))
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
        List<CaseOutcome> checkedOutcomes = [.. outcomes.Where(o => o.Rejected is null && o.Crash is null).OrderBy(o => o.SummaryChars)];
        if (checkedOutcomes.Count > 0)
        {
            CaseOutcome largest = checkedOutcomes[^1];
            TestContext.Out.WriteLine($"Stored summary size (PNG, full trace): median {checkedOutcomes[checkedOutcomes.Count / 2].SummaryChars} chars, max {largest.SummaryChars} chars ({largest.Label}); {checkedOutcomes.Count(o => o.JpegNotice)} over the {WorkflowTrace.MaxExifSummaryChars} char EXIF limit, which a JPEG or WebP save replaces with the use-PNG notice.");
        }
        foreach (CaseOutcome crash in upstreamCrashes)
        {
            TestContext.Out.WriteLine($"UPSTREAM CRASH (untraced build): {crash.Label}: {crash.Crash}");
        }
        foreach (IGrouping<string, CaseOutcome> reason in rejected.GroupBy(o => o.Rejected.Replace(o.Label.Before(' ').Replace('/', '_'), "{class}").Replace(o.Label.Before(' '), "{class}")))
        {
            TestContext.Out.WriteLine($"REJECTED x{reason.Count()}: {reason.Key}\n    {reason.Select(o => o.Label).JoinString("\n    ")}");
        }
        Assert.That(built, Is.GreaterThan(0));
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
        Assert.That(trace.Loras.Any(l => l.Contains("test_alien_lora") && l.Contains($"section={part.ContextID}") && l.Contains($"{loraId}:LoraLoaderModelOnly") && l.EndsWith($"@{stage}")), Is.True, string.Join("\n", trace.Loras));
        JObject call = trace.SamplerCalls[samplerId];
        Assert.That(call["start_step"].Value<int>(), Is.EqualTo(inputs["start_at_step"].Value<int>()));
        Assert.That(call["section"].Value<int>(), Is.EqualTo(part.ContextID));
        Assert.That(call["stage"].Value<string>(), Is.EqualTo(stage));
        JObject summary = (JObject)traced.Input.ExtraMeta[WorkflowTrace.MetadataKey];
        JObject samplerEntry = (JObject)((JArray)summary["samplers"]).Single(s => $"{s["node"]}".StartsWith($"{samplerId}:"));
        string modelChainText = samplerEntry["inputs"]["model"].ToString(Formatting.None);
        Assert.That(modelChainText, Does.Contain($"{loraId}:LoraLoaderModelOnly"));
        Assert.That(modelChainText, Does.Contain("lora_name=test_alien_lora.safetensors"));
        string positiveChainText = samplerEntry["inputs"]["positive"].ToString(Formatting.None);
        Assert.That(positiveChainText, Does.Contain($"{encId}:SwarmTextEncodeAdvanced"));
        Assert.That(positiveChainText, Does.Contain($"prompt=@{stage}.pos#{WorkflowTrace.Sha8(part.Prompt.Trim())}"));
        JObject segmentEntry = (JObject)((JArray)summary["segments"]).Single();
        Assert.That($"{segmentEntry["prompt"]}", Is.EqualTo($"@{stage}.pos#{WorkflowTrace.Sha8($"{encNode["inputs"]["prompt"]}")}"), "the segment record names the text its encoder node gets");
        Assert.That(summary["prompts"][$"{segmentEntry["prompt"]}"]?.Value<int>(), Is.EqualTo($"{encNode["inputs"]["prompt"]}".Length), "the segment prompt is listed with its length");
        Assert.That(positiveChainText, Does.Contain($"{refPixelsId}:SwarmImageScaleForMP"));
        Assert.That(summary.ToString(Formatting.None), Does.Not.Contain("zqx"));
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
        string current = samplerId;
        List<string> path = [current];
        string compositeId = null;
        while (current != "9")
        {
            (string next, string inputName) = consumers[current].First();
            if ($"{workflow[next]["class_type"]}".Contains("CompositeMasked"))
            {
                Assert.That(inputName, Is.EqualTo("source"), "the redraw is the composite source");
                compositeId = next;
            }
            current = next;
            path.Add(current);
            Assert.That(path.Count, Is.LessThan(20), string.Join(" > ", path));
        }
        Assert.That(compositeId, Is.Not.Null, string.Join(" > ", path));
        (string destId, JObject dest) = Source(workflow, workflow[compositeId]["inputs"]["destination"]);
        Assert.That(trace.Nodes[destId].Stage, Is.Not.EqualTo("seg11"), "the redraw is pasted onto the pre-segment image");
        JObject summary = (JObject)traced.Input.ExtraMeta[WorkflowTrace.MetadataKey];
        JObject samplerEntry = (JObject)((JArray)summary["samplers"]).Single(s => $"{s["node"]}".StartsWith($"{samplerId}:"));
        string downstream = samplerEntry["to_outputs"].ToString(Formatting.None);
        Assert.That(downstream, Does.Contain($"{compositeId}:"));
        Assert.That(downstream, Does.Contain("9:SwarmSaveImageWS"));
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
        Assert.That(segment.ParamReads["segmentmaskblur"].FirstSource, Is.EqualTo("default"));
    }

    /// <summary>The image metadata carries the full trace for PNG at any size, and for JPEG and WebP when it fits the EXIF limit; over the limit, a JPEG or WebP save carries a notice to save as PNG, where the server log has the trace, and the workflow and model identity, never part of a trace.
    /// The format comes from the input's Image Format parameter, read without marking it used.</summary>
    [Test]
    public void TestImageMetadataKeepsTheFullTraceOrSaysUsePng()
    {
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);
        Lora("test_alien_lora");
        Build traced = Generate(new() { Model = model, Prompt = $"{PromptMarkerBase} <segment:man> {PromptMarkerSegment} <lora:test_alien_lora>" }, true);
        JObject summary = traced.Trace.Build();
        int fullLength = T2IParamInput.MetadataToString(summary).Length;
        Assert.That(WorkflowTrace.ForImageMetadata(summary, "PNG", "[WorkflowTrace] request test", 10), Is.SameAs(summary), "PNG keeps the full trace over any limit");
        Assert.That(WorkflowTrace.ForImageMetadata(summary, "JPG", "[WorkflowTrace] request test", fullLength), Is.SameAs(summary), "JPEG keeps the full trace when it fits");
        foreach (string format in new[] { "JPG", "WEBP" })
        {
            JObject notice = WorkflowTrace.ForImageMetadata(summary, format, "[WorkflowTrace] request test", fullLength - 1);
            Assert.That(notice.Properties().Select(p => p.Name), Is.EquivalentTo(new[] { "omitted", "workflow", "model" }), format);
            Assert.That($"{notice["omitted"]["full_trace"]}", Does.Contain("save as PNG").And.Contain("[WorkflowTrace] request test"), format);
            Assert.That($"{notice["omitted"]["reason"]}", Does.Contain($"{fullLength} characters").And.Contain(format), format);
            Assert.That(JToken.DeepEquals(notice["workflow"], summary["workflow"]) && JToken.DeepEquals(notice["model"], summary["model"]), Is.True, format);
        }
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
            ["seedvr"] = t => t.Nodes.Values.Any(n => n.Stage == "seedvr") && t.Loads.Any(l => l.Type == "SeedVR2")
        };
        List<Variant> features = [.. Variants().Where(v => v.CheckpointOnly)];
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
        Assert.That($"{pass["object_px"]}", Is.EqualTo("512,256 512x512"));
        Assert.That($"{pass["crop_px"]}", Is.EqualTo("460,204 564x614"));
        Assert.That($"{pass["inpaint_size"]}", Is.EqualTo("1128x1228"));
        Assert.That($"{pass["mask_px"]}", Is.EqualTo("104,104 1024x1024"));
        T2IModel model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false);
        T2IParamInput passInput = BuildInput(new() { Model = model, Prompt = part.Prompt });
        passInput.ExtraMeta[WorkflowTrace.MetadataKey] = new JObject() { ["workflow"] = new JObject() { ["sent_sha256"] = "feedface" } };
        T2IMultiStepObjectBuilder.TracePassResult(pass, passInput, TestImage(64, 32));
        Assert.That($"{pass["workflow"]["sent_sha256"]}", Is.EqualTo("feedface"));
        Assert.That(pass["seed"].Value<long>(), Is.EqualTo(1L));
        Assert.That($"{pass["model"]}", Is.EqualTo(T2IParamTypes.CleanModelName(model.Name)));
        Assert.That($"{pass["result"]}", Does.StartWith("64x32 #").And.EndWith(" image/png"));
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
            Assert.That(((JArray)external["nodes"]).Count, Is.EqualTo(JObject.Parse(plain).Count));
            Assert.That($"{external["workflow"]["sent_sha256"]}", Is.EqualTo(Utilities.HashSHA256(plain.EncodeUTF8())));
            Assert.That(external.ToString(Formatting.None), Does.Not.Contain("zqx"), "custom workflow prompt text is hashed");
            T2IParamInput customInput = BuildInput(new() { Model = model, Prompt = PromptMarkerBase });
            WorkflowTrace.RecordExternalWorkflow(customInput, "custom", plain);
            Assert.That(customInput.ExtraMeta[WorkflowTrace.MetadataKey].ToString(), Is.EqualTo(external.ToString()));
        }
        finally
        {
            Program.ServerSettings.AddDebugData = priorDebug;
        }
    }
}
