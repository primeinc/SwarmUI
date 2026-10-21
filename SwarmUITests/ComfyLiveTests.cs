using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Backends;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using static SwarmUITests.NullModels;

namespace SwarmUITests;

/// <summary>Runs Swarm-generated workflows on a real ComfyUI with zero-weight models (<see cref="NullModels"/>) and a random-weight CLIPSeg: Swarm's ComfyUI API backend initializes from the live server's object_info and runs the job through its own <see cref="ComfyUIAPIAbstractBackend.GenerateLive"/>.
/// Needs the ComfyUI from SwarmUITests/ComfyLive/justfile; skipped unless SWARMUI_TEST_COMFY_URL and SWARMUI_TEST_COMFY_MODELS are set.</summary>
[Category("ComfyLive")]
public class ComfyLiveTests : SwarmUITest
{
    /// <summary>Base URL of the running ComfyUI.</summary>
    public static string Url => Environment.GetEnvironmentVariable("SWARMUI_TEST_COMFY_URL");

    /// <summary>The ComfyUI model folder (extra_model_paths base_path) the null models are written into.</summary>
    public static string ModelRoot => Environment.GetEnvironmentVariable("SWARMUI_TEST_COMFY_MODELS")?.Replace('\\', '/');

    /// <summary>Image width and height.</summary>
    public const int Size = 256;

    /// <summary>Name of the null LoRA.</summary>
    public const string LoraName = "qwen21-null";

    /// <summary>Longest a live job may take.</summary>
    public static readonly TimeSpan JobDeadline = TimeSpan.FromMinutes(20);

    /// <summary>The static inpaint mask rectangle, in pixels.</summary>
    public static readonly (int X, int Y, int Width, int Height) MaskRect = (48, 64, 64, 96);

    /// <summary>Init image pixel: red rises 0-60 left to right, green 190-250 top to bottom, blue 30, so no pixel is near the mid-gray a zero-weight VAE decodes to.</summary>
    public static Rgba32 InitPixel(int x, int y) => new((byte)(x * 60 / (Size - 1)), (byte)(190 + (y * 60 / (Size - 1))), 30, 255);

    /// <summary>Returns true if a pixel is inside <see cref="MaskRect"/>.</summary>
    public static bool InMaskRect(int x, int y) => x >= MaskRect.X && x < MaskRect.X + MaskRect.Width && y >= MaskRect.Y && y < MaskRect.Y + MaskRect.Height;

    /// <summary>Returns a <see cref="Size"/> square image of the given pixels.</summary>
    public static Image MakeImage(Func<int, int, Rgba32> pixel)
    {
        using SixLabors.ImageSharp.Image<Rgba32> img = new(Size, Size);
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                img[x, y] = pixel(x, y);
            }
        }
        return new Image(img);
    }

    /// <summary>The init image.</summary>
    public static Image InitImage;

    /// <summary>The static inpaint mask: white inside <see cref="MaskRect"/>, black elsewhere.</summary>
    public static Image MaskImage;

    /// <summary>The value of <see cref="Program.GlobalProgramCancel"/> before this fixture. The harness cancels it so nothing downloads; the live backend reads its websocket with it.</summary>
    public CancellationToken PriorCancel;

    /// <summary>Prepares the basics and writes the null models into the ComfyUI model folder.</summary>
    [OneTimeSetUp]
    public void PreInit()
    {
        if (string.IsNullOrWhiteSpace(Url) || string.IsNullOrWhiteSpace(ModelRoot))
        {
            Assert.Ignore("Set SWARMUI_TEST_COMFY_URL and SWARMUI_TEST_COMFY_MODELS, or run 'just test' in SwarmUITests/ComfyLive.");
        }
        Setup();
        WorkflowGeneratorHarness.Init();
        PriorCancel = Program.GlobalProgramCancel;
        Program.GlobalProgramCancel = CancellationToken.None;
        Program.ServerSettings.AddDebugData = true;
        InitImage = MakeImage(InitPixel);
        MaskImage = MakeImage((x, y) => InMaskRect(x, y) ? new Rgba32(255, 255, 255, 255) : new Rgba32(0, 0, 0, 255));
        T2IModel model = Qwen21Model();
        WriteQwen21DiT($"{ModelRoot}/diffusion_models/{model.Name}", Qwen21Dims.Micro);
        WriteQwen21Lora($"{ModelRoot}/loras/{LoraName}.safetensors", Qwen21Dims.Micro, 4);
        WriteQwen3VLTextEncoder($"{ModelRoot}/text_encoders/qwen3vl_8b_int8_convrot.safetensors", Qwen3VLDims.Null);
        WriteQwen21Vae($"{ModelRoot}/vae/QwenImage/qwen_image_2.1_vae_bf16.safetensors", Qwen21VaeDims.Null);
        Assert.That(File.Exists($"{ModelRoot}/clipseg/clipseg-rd64-refined-fp16-safetensors/model.safetensors"), "CLIPSeg model present (run 'just clipseg' in SwarmUITests/ComfyLive)");
    }

    /// <summary>Restores the global state the fixture changed.</summary>
    [OneTimeTearDown]
    public void PostTest()
    {
        if (Program.ServerSettings is not null)
        {
            Program.ServerSettings.AddDebugData = false;
        }
        Program.GlobalProgramCancel = PriorCancel;
    }

    /// <summary>The Qwen Image 2.1 main model, in diffusion_models format.</summary>
    public static T2IModel Qwen21Model() => WorkflowGeneratorHarness.ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true);

    /// <summary>Qwen Image 2.1 inpainting of the static mask, a LoRA, and a CLIPSeg segment refine whose mask is saved. Segment blur and grow are off, so the random CLIPSeg mask keeps its shape.</summary>
    public static WorkflowGeneratorHarness.Request InpaintSegmentRequest()
    {
        WorkflowGeneratorHarness.Lora(LoraName);
        return new()
        {
            Model = Qwen21Model(),
            Prompt = $"a cat on a road <lora:{LoraName}:1> <segment:cat,1,0.5> a cat face",
            Setters =
            [
                i => i.Set(T2IParamTypes.Width, Size),
                i => i.Set(T2IParamTypes.Height, Size),
                i => i.Set(T2IParamTypes.Steps, 2),
                i => i.Set(T2IParamTypes.CFGScale, 1.0),
                i => i.Set(T2IParamTypes.InitImage, InitImage),
                i => i.Set(T2IParamTypes.InitImageCreativity, 1.0),
                i => i.Set(T2IParamTypes.MaskImage, MaskImage),
                i => i.Set(T2IParamTypes.MaskShrinkGrow, 8),
                i => i.Set(T2IParamTypes.SegmentMaskBlur, 0),
                i => i.Set(T2IParamTypes.SegmentMaskGrow, 0),
                i => i.Set(T2IParamTypes.SaveSegmentMask, true)
            ]
        };
    }

    /// <summary>A Swarm ComfyUI API backend for <see cref="Url"/>, initialized the way Swarm initializes one: object_info, model lists, feature set.</summary>
    public static async Task<ComfyUIAPIBackend> InitBackend()
    {
        ComfyUIAPIBackend backend = new() { SettingsRaw = new ComfyUIAPIBackend.ComfyUIAPISettings() { Address = Url } };
        backend.BackendData = new BackendHandler.T2IBackendData() { ID = 0, BackType = Program.Backends.BackendTypes["comfyui_api"], Backend = backend };
        await backend.Init();
        Assert.That(backend.Status, Is.EqualTo(BackendStatus.RUNNING));
        Assert.That(backend.NodeTypes, Does.Contain("SwarmKSampler").And.Contain("SwarmClipSeg"), "Swarm's ExtraNodes loaded");
        return backend;
    }

    /// <summary>One live job: the websocket JSON events for its prompt, and the outputs Swarm took.</summary>
    /// <param name="Events">JSON websocket events of the job's prompt, in order.</param>
    /// <param name="Outputs">Image outputs, in order.</param>
    /// <param name="Input">The user input after the job.</param>
    public record class LiveJob(List<JObject> Events, List<T2IEngine.ImageOutput> Outputs, T2IParamInput Input);

    /// <summary>Runs a request through <see cref="ComfyUIAPIAbstractBackend.GenerateLive"/>, failing with a <see cref="TimeoutException"/> past <see cref="JobDeadline"/>.</summary>
    public static Task<LiveJob> RunLive(ComfyUIAPIBackend backend, WorkflowGeneratorHarness.Request request)
    {
        T2IParamInput input = WorkflowGeneratorHarness.BuildInput(request);
        return CaptureJob(input, take => backend.GenerateLive(input, "0", take));
    }

    /// <summary>Runs one job, capturing the websocket events of its prompt and the image outputs Swarm took, failing with a <see cref="TimeoutException"/> past <see cref="JobDeadline"/>.</summary>
    /// <param name="input">The user input the job runs for.</param>
    /// <param name="run">Starts the job, given the output callback.</param>
    public static async Task<LiveJob> CaptureJob(T2IParamInput input, Func<Action<object>, Task> run)
    {
        List<JObject> events = [];
        input.ReceiveRawBackendData = (_, data) =>
        {
            if (data.Length > 0 && data[0] == (byte)'{')
            {
                lock (events)
                {
                    events.Add(JObject.Parse(Encoding.UTF8.GetString(data)));
                }
            }
        };
        List<T2IEngine.ImageOutput> outputs = [];
        await run(output =>
        {
            if (output is T2IEngine.ImageOutput image)
            {
                lock (outputs)
                {
                    outputs.Add(image);
                }
            }
        }).WaitAsync(JobDeadline);
        string promptId = events.Where(e => $"{e["type"]}" == "execution_start").Select(e => $"{e["data"]["prompt_id"]}").Single();
        return new([.. events.Where(e => $"{e["data"]?["prompt_id"]}" == promptId)], outputs, input);
    }

    /// <summary>Returns the node IDs ComfyUI executed for a job, and the node IDs it served from its cache instead.</summary>
    public static (HashSet<string> Executed, HashSet<string> Cached) NodesRun(LiveJob job)
    {
        HashSet<string> executed = [], cached = [];
        foreach (JObject e in job.Events)
        {
            if ($"{e["type"]}" == "executing" && e["data"]["node"] is JValue node && node.Type == JTokenType.String)
            {
                executed.Add($"{node}");
            }
            else if ($"{e["type"]}" == "execution_cached" && e["data"]["nodes"] is JArray nodes)
            {
                cached.UnionWith(nodes.Select(n => $"{n}"));
            }
        }
        return (executed, cached);
    }

    /// <summary>Returns the RGB channels of a pixel.</summary>
    public static double[] Rgb(Rgba32 p) => [p.R, p.G, p.B];

    /// <summary>Fits, by least squares, pasted = k[c] + s * underneath[c] with one s shared by the three channels: the composite of a uniform RGBA source of color G and alpha A gives s = 1 - A and k = A * G.</summary>
    public static (double[] K, double S) FitBlend(List<(double[] Under, double[] Pasted)> samples)
    {
        double[] meanUnder = new double[3], meanPasted = new double[3];
        for (int c = 0; c < 3; c++)
        {
            meanUnder[c] = samples.Average(p => p.Under[c]);
            meanPasted[c] = samples.Average(p => p.Pasted[c]);
        }
        double covariance = 0, variance = 0;
        foreach ((double[] under, double[] pasted) in samples)
        {
            for (int c = 0; c < 3; c++)
            {
                covariance += (under[c] - meanUnder[c]) * (pasted[c] - meanPasted[c]);
                variance += (under[c] - meanUnder[c]) * (under[c] - meanUnder[c]);
            }
        }
        double s = covariance / variance;
        return ([.. Enumerable.Range(0, 3).Select(c => meanPasted[c] - (s * meanUnder[c]))], s);
    }

    /// <summary>Inputs the generator sends that the live ComfyUI does not declare, as Class.input; ComfyUI ignores them. SwarmKSampler.py has no control_after_generate (WorkflowGenerator.CreateKSampler sets it),
    /// and SwarmImageCompositeMaskedColorCorrecting has no resize_source (WorkflowGenerator's segment and mask composites set it). Any other undeclared input fails the live test.</summary>
    public static readonly string[] KnownUndeclaredInputs = ["SwarmKSampler.control_after_generate", "SwarmImageCompositeMaskedColorCorrecting.resize_source"];

    /// <summary>Qwen Image 2.1 with a LoRA, inpainting a static mask and refining a CLIPSeg segment, runs on real ComfyUI to the final save, and the saved image is the init image changed exactly where the static mask and the segment mask say:
    /// Swarm's crop bounds, cropped masks and paste-back coordinates put the generated pixels where the masks are. The model's RGBA output is alpha-composited (SwarmImageCompositeMaskedColorCorrecting multiplies the mask by the source alpha),
    /// so a pixel under one mask is A * G + (1 - A) * init, and under both masks the same blend applied twice. The trace records the graph that was sent, and the live object_info types every link.</summary>
    [Test]
    public async Task TestQwen21InpaintSegmentLoraRunsToTheFinalSave()
    {
        ComfyUIAPIBackend backend = await InitBackend();
        LiveJob job = await RunLive(backend, InpaintSegmentRequest());
        List<JObject> errors = [.. job.Events.Where(e => $"{e["type"]}" == "execution_error")];
        Assert.That(errors, Is.Empty, string.Join("\n", errors.Select(e => $"{e["data"]["node_id"]}:{e["data"]["node_type"]}: {e["data"]["exception_type"]}: {e["data"]["exception_message"]}")));
        JObject trace = job.Input.ExtraMeta.GetValueOrDefault("debug_trace") as JObject;
        Assert.That(trace, Is.Not.Null, "debug_trace recorded");
        T2IParamInput offlineInput = WorkflowGeneratorHarness.BuildInput(InpaintSegmentRequest());
        string sent = ComfyUIAPIAbstractBackend.CreateWorkflow(offlineInput, w => w, backend.ModelFolderFormat, [.. backend.SupportedFeatures], backend.RawObjectInfo);
        Assert.That($"{trace["workflow"]["sent_sha256"]}", Is.EqualTo(Utilities.HashSHA256(sent.EncodeUTF8())), "the trace's sent_sha256 is the graph regenerated from the same input");
        Assert.That(trace["workflow"]["rewritten_after_generate"].Value<bool>(), Is.False);
        JObject posted = trace["workflow"]["posted"] as JObject;
        Assert.That(posted, Is.Not.Null, $"the backend recorded what it posted: {trace["workflow"]["posted"]}");
        Assert.That($"{posted["sha256"]}", Is.EqualTo($"{trace["workflow"]["sent_sha256"]}"), "the graph posted to /prompt is the graph the trace describes");
        Assert.That($"{posted["backend"]}", Is.EqualTo(nameof(ComfyUIAPIBackend)));
        Assert.That(posted["matches_sent"].Value<bool>(), Is.True);
        JObject workflow = sent.ParseToJson();
        (HashSet<string> executed, HashSet<string> cached) = NodesRun(job);
        List<string> notExecuted = [.. workflow.Properties().Where(p => !executed.Contains(p.Name)).Select(p => $"{p.Name}:{p.Value["class_type"]}")];
        TestContext.Out.WriteLine($"{workflow.Count} nodes sent, {executed.Count} executed, {cached.Count} from cache; classes: {string.Join(", ", workflow.Properties().Select(p => $"{p.Value["class_type"]}").Distinct().Order())}");
        Assert.That(cached, Is.Empty, "nothing served from ComfyUI's cache (the justfile starts ComfyUI with --cache-none)");
        Assert.That(notExecuted, Is.Empty, "every node sent was executed in this job");
        // The segment record, against the graph that ran: its sampler is fed the text and the CLIPSeg mask it names.
        JObject segmentRecord = (JObject)((JArray)trace["segments"]).Single();
        string segmentStage = $"{segmentRecord["stage"]}";
        string samplerId = $"{((JArray)trace["samplers"]).Single(s => $"{s["call"]["stage"]}" == segmentStage)["node"]}".Before(':');
        JObject samplerInputs = (JObject)workflow[samplerId]["inputs"];
        Assert.That($"{workflow[samplerId]["class_type"]}", Is.EqualTo("SwarmKSampler"));
        Assert.That(executed, Does.Contain(samplerId));
        (string encoderId, JObject encoder) = WorkflowTraceTests.Source(workflow, samplerInputs["positive"]);
        while ($"{encoder["class_type"]}" == "ReferenceLatent")
        {
            (encoderId, encoder) = WorkflowTraceTests.Source(workflow, encoder["inputs"]["conditioning"]);
        }
        Assert.That($"{encoder["class_type"]}", Is.EqualTo("SwarmTextEncodeAdvanced"), encoderId);
        Assert.That($"{segmentRecord["prompt"]}", Is.EqualTo($"@{segmentStage}.pos#{MetadataHelper.Sha8($"{encoder["inputs"]["prompt"]}")}"), "the segment record names the text its encoder got");
        (string maskId, JObject maskNode) = WorkflowTraceTests.Source(workflow, samplerInputs["latent_image"]);
        while ($"{maskNode["class_type"]}" != "SwarmClipSeg" && maskNode["inputs"]["mask"] is JArray maskLink)
        {
            (maskId, maskNode) = WorkflowTraceTests.Source(workflow, maskLink);
        }
        Assert.That($"{maskNode["class_type"]}", Is.EqualTo("SwarmClipSeg"), maskId);
        Assert.That($"{segmentRecord["mask_text"]}", Is.EqualTo($"@{segmentStage}.mask#{MetadataHelper.Sha8($"{maskNode["inputs"]["match_text"]}")}"));
        Assert.That(segmentRecord["threshold"].Value<double>(), Is.EqualTo(maskNode["inputs"]["threshold"].Value<double>()));
        Assert.That(samplerInputs["start_at_step"].Value<int>(), Is.EqualTo((int)Math.Round(samplerInputs["steps"].Value<int>() * (1 - segmentRecord["creativity"].Value<double>()))));
        Assert.That(trace["type_errors"], Is.Empty, $"live object_info types every link: {trace["type_errors"]}");
        Assert.That(trace["classes_missing_from_object_info"], Is.Empty, $"{trace["classes_missing_from_object_info"]}");
        Assert.That(((JArray)trace["undeclared_inputs"]).Select(t => $"{t}".After(':')).Distinct(), Is.EquivalentTo(KnownUndeclaredInputs), $"the only inputs the live ComfyUI does not declare are the known ones: {trace["undeclared_inputs"]}");
        TestContext.Out.WriteLine($"outputs: {string.Join(", ", job.Outputs.Select(o => $"node {o.BackendInternalHint} real={o.IsReal} {o.File.Type.MimeType} {o.File.RawData.Length} bytes"))}");
        T2IEngine.ImageOutput final = job.Outputs.Single(o => o.IsReal);
        Assert.That(final.BackendInternalHint, Is.EqualTo("9"), "the final image came from the final save node");
        T2IEngine.ImageOutput segmentMask = job.Outputs.Single(o => !o.IsReal && int.TryParse(o.BackendInternalHint, out int id) && id >= 50000);
        string outDir = TestContext.CurrentContext.WorkDirectory.Replace('\\', '/');
        File.WriteAllBytes($"{outDir}/comfylive-final.png", final.File.RawData);
        File.WriteAllBytes($"{outDir}/comfylive-segment-mask.png", segmentMask.File.RawData);
        File.WriteAllBytes($"{outDir}/comfylive-init.png", InitImage.RawData);
        File.WriteAllText($"{outDir}/comfylive-workflow.json", workflow.ToString());
        File.WriteAllText($"{outDir}/comfylive-trace.json", trace.ToString());
        TestContext.Out.WriteLine($"saved comfylive-final.png, comfylive-segment-mask.png, comfylive-init.png, comfylive-workflow.json, comfylive-trace.json to {outDir}");
        using SixLabors.ImageSharp.Image<Rgba32> image = SixLabors.ImageSharp.Image.Load<Rgba32>(final.File.RawData);
        using SixLabors.ImageSharp.Image<Rgba32> mask = SixLabors.ImageSharp.Image.Load<Rgba32>(segmentMask.File.RawData);
        Assert.That((image.Width, image.Height, mask.Width, mask.Height), Is.EqualTo((Size, Size, Size, Size)));
        // SwarmMaskThreshold(0.01, 1) then ThresholdMask(0.001) keep segment mask values above ~0.011; the saved 8-bit mask is ambiguous at 2/255 and 3/255.
        List<(int X, int Y, int Covers)> pixels = [];
        int segmentOn = 0, ambiguous = 0;
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                byte m = mask[x, y].R;
                segmentOn += m >= 4 ? 1 : 0;
                if (m is 2 or 3)
                {
                    ambiguous++;
                    continue;
                }
                pixels.Add((x, y, (InMaskRect(x, y) ? 1 : 0) + (m >= 4 ? 1 : 0)));
            }
        }
        double coverage = segmentOn / (double)(Size * Size);
        List<(double[] Under, double[] Pasted)> once = [.. pixels.Where(p => p.Covers == 1).Select(p => (Rgb(InitPixel(p.X, p.Y)), Rgb(image[p.X, p.Y])))];
        (double[] k, double s) = FitBlend(once);
        double alpha = 1 - s;
        TestContext.Out.WriteLine($"segment mask covers {coverage:P1}, {ambiguous} pixels on the 8-bit threshold edge; {pixels.Count(p => p.Covers == 0)} untouched, {once.Count} under one mask, {pixels.Count(p => p.Covers == 2)} under both;"
            + $" fitted source alpha {alpha:0.0000}, color ({string.Join(", ", k.Select(v => $"{v / alpha:0.0}"))})");
        List<string> wrong = [];
        foreach ((int x, int y, int covers) in pixels)
        {
            double[] init = Rgb(InitPixel(x, y)), actual = Rgb(image[x, y]);
            double[] expected = covers switch
            {
                0 => init,
                1 => [.. Enumerable.Range(0, 3).Select(c => k[c] + (s * init[c]))],
                _ => [.. Enumerable.Range(0, 3).Select(c => k[c] + (s * k[c]) + (s * s * init[c]))]
            };
            double error = Enumerable.Range(0, 3).Max(c => Math.Abs(actual[c] - expected[c]));
            if (error > covers + 1)
            {
                wrong.Add($"({x},{y}) {image[x, y]} expected ({string.Join(", ", expected.Select(v => $"{v:0.0}"))}) under {covers} masks{(InMaskRect(x, y) ? " (static rect)" : "")}, segment mask {mask[x, y].R}");
            }
        }
        TestContext.Out.WriteLine($"{pixels.Count} pixels checked, {wrong.Count} wrong");
        Assert.That(coverage, Is.InRange(0.01, 0.99), "the random CLIPSeg mask is a partial shape");
        Assert.That(once.Count, Is.GreaterThan(1000), "pixels under exactly one mask to fit the blend from");
        Assert.That(alpha, Is.InRange(0.05, 1.0), "the generated source was pasted with visible alpha");
        Assert.That(wrong, Is.Empty, $"{wrong.Count} pixels wrong, first: {string.Join("; ", wrong.Take(20))}");
    }

    /// <summary>The offline typing fixture (<see cref="WorkflowTraceTypingTests.SdxlObjectInfo"/>) matches the live ComfyUI's object_info: for every fixture class, the same required and optional input names,
    /// the same type for every input that is not a choice list, a choice list wherever the fixture has one, and the same output types and output-node flag.</summary>
    [Test]
    public async Task TestTypingFixtureMatchesLiveObjectInfo()
    {
        ComfyUIAPIBackend backend = await InitBackend();
        static string inputType(JToken spec) => spec?[0] is JValue type && $"{type}" != "COMBO" ? $"{type}" : "(choice)";
        List<string> mismatches = [];
        foreach (JProperty fixtureClass in JObject.Parse(WorkflowTraceTypingTests.SdxlObjectInfo).Properties())
        {
            if (backend.RawObjectInfo[fixtureClass.Name] is not JObject live)
            {
                mismatches.Add($"{fixtureClass.Name}: not in the live object_info");
                continue;
            }
            JObject fixture = (JObject)fixtureClass.Value;
            foreach (string group in new[] { "required", "optional" })
            {
                JObject fixtureInputs = fixture["input"]?[group] as JObject ?? [], liveInputs = live["input"]?[group] as JObject ?? [];
                string fixtureNames = fixtureInputs.Properties().Select(p => p.Name).Order().JoinString(", "), liveNames = liveInputs.Properties().Select(p => p.Name).Order().JoinString(", ");
                if (fixtureNames != liveNames)
                {
                    mismatches.Add($"{fixtureClass.Name} {group} inputs: fixture [{fixtureNames}], live [{liveNames}]");
                }
                foreach (JProperty input in fixtureInputs.Properties())
                {
                    if (liveInputs[input.Name] is JToken liveSpec && inputType(input.Value) != inputType(liveSpec))
                    {
                        mismatches.Add($"{fixtureClass.Name}.{input.Name}: fixture {inputType(input.Value)}, live {inputType(liveSpec)}");
                    }
                }
            }
            foreach (string key in new[] { "output", "output_node" })
            {
                if (!JToken.DeepEquals(fixture[key], live[key]))
                {
                    mismatches.Add($"{fixtureClass.Name} {key}: fixture {fixture[key]?.ToString(Formatting.None)}, live {live[key]?.ToString(Formatting.None)}");
                }
            }
        }
        Assert.That(mismatches, Is.Empty, string.Join("\n", mismatches));
    }

    /// <summary>Color of the synthetic segment crop.</summary>
    public static readonly Rgba32 CropColor = new(250, 20, 200, 255);

    /// <summary>Qwen Image 2.1 refining a CLIPSeg segment of the init image, with the segment mask saved, blur and grow off.</summary>
    public static WorkflowGeneratorHarness.Request SegmentRequest() => new()
    {
        Model = Qwen21Model(),
        Prompt = "a cat on a road <segment:cat,1,0.5> a cat face",
        Setters =
        [
            i => i.Set(T2IParamTypes.Width, Size),
            i => i.Set(T2IParamTypes.Height, Size),
            i => i.Set(T2IParamTypes.Steps, 2),
            i => i.Set(T2IParamTypes.CFGScale, 1.0),
            i => i.Set(T2IParamTypes.InitImage, InitImage),
            i => i.Set(T2IParamTypes.InitImageCreativity, 1.0),
            i => i.Set(T2IParamTypes.SegmentMaskBlur, 0),
            i => i.Set(T2IParamTypes.SegmentMaskGrow, 0),
            i => i.Set(T2IParamTypes.SaveSegmentMask, true)
        ]
    };

    /// <summary>Swarm's paste-back nodes for a segment (ImageScale back to the crop bounds, ThresholdMask of the cropped mask, SwarmImageCompositeMaskedColorCorrecting at the bounds) run on real ComfyUI with a synthetic crop:
    /// the segment's decoded redraw is replaced by a solid-color image of another size and the composite's destination by the init image, so the final save must hold that color exactly where the segment mask passes the crop threshold, and the init image everywhere else.
    /// The trace records that the posted graph is not the generated one.</summary>
    [Test]
    public async Task TestSegmentPasteBackOfASyntheticCrop()
    {
        ComfyUIAPIBackend backend = await InitBackend();
        T2IParamInput input = WorkflowGeneratorHarness.BuildInput(SegmentRequest());
        JObject workflow = ComfyUIAPIAbstractBackend.CreateWorkflow(input, w => w, backend.ModelFolderFormat, [.. backend.SupportedFeatures], backend.RawObjectInfo).ParseToJson();
        JObject trace = (JObject)input.ExtraMeta["debug_trace"];
        JObject segment = (JObject)((JArray)trace["segments"]).Single();
        string segmentStage = $"{segment["stage"]}";
        string compositeId = $"{((JArray)segment["crop"]["pasted_by"]).Single()}".Before(':');
        JObject composite = (JObject)workflow[compositeId];
        Assert.That($"{composite["class_type"]}", Is.EqualTo("SwarmImageCompositeMaskedColorCorrecting"), "the segment record names the node that pastes its crop back");
        (string scaleId, JObject scale) = WorkflowTraceTests.Source(workflow, composite["inputs"]["source"]);
        Assert.That($"{scale["class_type"]}", Is.EqualTo("ImageScale"), "the redraw is scaled back to the crop bounds");
        (string decodeId, JObject decode) = WorkflowTraceTests.Source(workflow, scale["inputs"]["image"]);
        Assert.That($"{trace["nodes"][decodeId]}", Does.StartWith($"{decodeId}:{decode["class_type"]}@{segmentStage} "), $"{decodeId}:{decode["class_type"]} is the segment's decoded redraw, in the image record's node table");
        (string thresholdId, JObject threshold) = WorkflowTraceTests.Source(workflow, composite["inputs"]["mask"]);
        Assert.That($"{threshold["class_type"]}", Is.EqualTo("ThresholdMask"));
        using SixLabors.ImageSharp.Image<Rgba32> crop = new(37, 53, CropColor);
        workflow["990001"] = new JObject() { ["class_type"] = "SwarmLoadImageB64", ["inputs"] = new JObject() { ["image_base64"] = new Image(crop).AsBase64 } };
        workflow["990002"] = new JObject() { ["class_type"] = "SwarmLoadImageB64", ["inputs"] = new JObject() { ["image_base64"] = InitImage.AsBase64 } };
        scale["inputs"]["image"] = new JArray("990001", 0);
        composite["inputs"]["destination"] = new JArray("990002", 0);
        string posted = workflow.ToString();
        LiveJob job = await CaptureJob(input, take => backend.AwaitJobLive(posted, "0", take, input, CancellationToken.None));
        List<JObject> errors = [.. job.Events.Where(e => $"{e["type"]}" == "execution_error")];
        Assert.That(errors, Is.Empty, string.Join("\n", errors.Select(e => $"{e["data"]["node_id"]}:{e["data"]["node_type"]}: {e["data"]["exception_type"]}: {e["data"]["exception_message"]}")));
        (HashSet<string> executed, HashSet<string> cached) = NodesRun(job);
        Assert.That(cached, Is.Empty);
        Assert.That(executed, Is.SupersetOf(new[] { "990001", "990002", scaleId, thresholdId, compositeId, "9" }));
        Assert.That(trace["workflow"]["posted"]["matches_sent"].Value<bool>(), Is.False, "the trace flags a posted graph that differs from the generated one");
        Assert.That($"{trace["workflow"]["posted"]["sha256"]}", Is.EqualTo(Utilities.HashSHA256(posted.EncodeUTF8())));
        T2IEngine.ImageOutput final = job.Outputs.Single(o => o.IsReal);
        Assert.That(final.BackendInternalHint, Is.EqualTo("9"));
        T2IEngine.ImageOutput segmentMask = job.Outputs.Single(o => !o.IsReal && int.TryParse(o.BackendInternalHint, out int id) && id >= 50000);
        string outDir = TestContext.CurrentContext.WorkDirectory.Replace('\\', '/');
        File.WriteAllBytes($"{outDir}/comfylive-synthetic-final.png", final.File.RawData);
        File.WriteAllBytes($"{outDir}/comfylive-synthetic-segment-mask.png", segmentMask.File.RawData);
        File.WriteAllText($"{outDir}/comfylive-synthetic-workflow.json", posted);
        using SixLabors.ImageSharp.Image<Rgba32> image = SixLabors.ImageSharp.Image.Load<Rgba32>(final.File.RawData);
        using SixLabors.ImageSharp.Image<Rgba32> mask = SixLabors.ImageSharp.Image.Load<Rgba32>(segmentMask.File.RawData);
        Assert.That((image.Width, image.Height, mask.Width, mask.Height), Is.EqualTo((Size, Size, Size, Size)));
        // SwarmMaskThreshold(0.01, 1) then ThresholdMask(0.001) keep segment mask values above ~0.011; the saved 8-bit mask is ambiguous at 2/255 and 3/255.
        int pasted = 0, untouched = 0, ambiguous = 0;
        List<string> wrong = [];
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                byte m = mask[x, y].R;
                if (m is 2 or 3)
                {
                    ambiguous++;
                    continue;
                }
                Rgba32 expected = m >= 4 ? CropColor : InitPixel(x, y);
                Rgba32 actual = image[x, y];
                if (Math.Max(Math.Abs(actual.R - expected.R), Math.Max(Math.Abs(actual.G - expected.G), Math.Abs(actual.B - expected.B))) > 1)
                {
                    wrong.Add($"({x},{y}) {actual} expected {expected}, segment mask {m}");
                }
                pasted += m >= 4 ? 1 : 0;
                untouched += m >= 4 ? 0 : 1;
            }
        }
        TestContext.Out.WriteLine($"{pasted} pixels under the segment mask, {untouched} outside, {ambiguous} on the 8-bit threshold edge; {wrong.Count} wrong; saved comfylive-synthetic-*.png and the posted workflow to {outDir}");
        Assert.That(pasted, Is.GreaterThan(1000), "the random CLIPSeg mask covers enough pixels");
        Assert.That(untouched, Is.GreaterThan(1000), "the random CLIPSeg mask leaves enough pixels");
        Assert.That(wrong, Is.Empty, $"{wrong.Count} pixels wrong, first: {string.Join("; ", wrong.Take(20))}");
    }
}
