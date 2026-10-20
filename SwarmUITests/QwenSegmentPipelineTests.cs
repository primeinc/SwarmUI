using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>Exercises the registered Qwen 2.1 segment step and its runtime-proof graph. GPU execution is not simulated.</summary>
[NonParallelizable]
public class QwenSegmentPipelineTests : SwarmUITest
{
    private T2IModelHandler priorLoras;
    private bool priorHashSetting;

    [OneTimeSetUp]
    public void Initialize()
    {
        Setup();
        if (T2IParamTypes.Prompt is null)
        {
            T2IParamTypes.RegisterDefaults();
        }
        ComfyUIBackendExtension.SamplerParam ??= new(new T2IParamType("Sampler", "", "euler", ID: "sampler"));
        ComfyUIBackendExtension.SchedulerParam ??= new(new T2IParamType("Scheduler", "", "simple", ID: "scheduler"));
        ComfyUIBackendExtension.TextEncodedImage ??= new(new T2IParamType("Text Encoded Image", "", "auto", ID: "textencodedimage"));
        ComfyUIBackendExtension.EnableReferenceLatents ??= new(new T2IRegisteredParam<string>(new T2IParamType("Enable Reference Latents", "", "none", ID: "enablereferencelatents")).Type);
    }

    [SetUp]
    public void SetupModels()
    {
        Program.T2IModelSets.TryGetValue("LoRA", out priorLoras);
        priorHashSetting = Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash;
        Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash = false;
        T2IModelHandler handler = new() { ModelType = "LoRA" };
        foreach (string name in new[] { "face.safetensors", "hair.safetensors" })
        {
            handler.Models[name] = new T2IModel(handler, "/fixture", "/fixture/" + name, name);
        }
        Program.T2IModelSets["LoRA"] = handler;
    }

    [TearDown]
    public void RestoreModels()
    {
        if (priorLoras is null)
        {
            Program.T2IModelSets.Remove("LoRA");
        }
        else
        {
            Program.T2IModelSets["LoRA"] = priorLoras;
        }
        Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash = priorHashSetting;
    }

    private static WorkflowGenerator Prepare(bool qwen = true, double creativity = 1, bool secondSegment = false, bool withLora = true)
    {
        T2IModelCompatClass compat = qwen ? T2IModelClassSorter.CompatQwenImage21 : T2IModelClassSorter.CompatKrea2;
        T2IModel model = new(null, "/fixture", "/fixture/model.safetensors", "model.safetensors")
        {
            ModelClass = new() { ID = qwen ? "qwen-image-2.1" : "krea-2", CompatClass = compat, StandardWidth = 1024, StandardHeight = 1024 }
        };
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.Model, model);
        input.Set(T2IParamTypes.Prompt, $"base scene <segment:face,{creativity},0.8//cid=11>make the face blue" + (secondSegment ? "<segment:hair,1,0.8//cid=12>make the hair red" : ""));
        input.Set(T2IParamTypes.NegativePrompt, "global negative <segment:face,1,0.8//cid=11>segment negative");
        Assert.That(new PromptRegion(input.Get(T2IParamTypes.Prompt)).Parts.First().Strength2, Is.EqualTo(creativity).Within(0.00001));
        input.Set(T2IParamTypes.Width, 1024);
        input.Set(T2IParamTypes.Height, 1024);
        input.Set(T2IParamTypes.Steps, 20);
        input.Set(T2IParamTypes.CFGScale, 1.0);
        input.Set(T2IParamTypes.Seed, 42L);
        if (withLora)
        {
            input.Set(T2IParamTypes.Loras, secondSegment ? new List<string> { "face", "hair" } : new List<string> { "face" });
            input.Set(T2IParamTypes.LoraWeights, secondSegment ? new List<string> { "0.8", "0.6" } : new List<string> { "0.8" });
            input.Set(T2IParamTypes.LoraSectionConfinement, secondSegment ? new List<string> { "11", "12" } : new List<string> { "11" });
        }
        WorkflowGenerator g = new() { UserInput = input, Workflow = new(), FinalLoadedModel = model, ModelFolderFormat = "/", Features = ["variation_seed"] };
        g.CreateNode("UNETLoader", new JObject() { ["unet_name"] = "model.safetensors" }, "base-model");
        g.CreateNode("CLIPLoader", new JObject() { ["clip_name"] = "text.safetensors" }, "base-clip");
        g.CreateNode("VAELoader", new JObject() { ["vae_name"] = "vae.safetensors" }, "base-vae");
        g.CreateNode("LoadImage", new JObject() { ["image"] = "fixed-base.png" }, "base-image");
        g.CurrentModel = new(["base-model", 0], g, WGNodeData.DT_MODEL, compat);
        if (qwen)
        {
            // Synthetic fixture has no real CreateModelLoader call, so model-load anchor == base model.
            g.LoadedModelLoraAnchors["base-model:0"] = new JArray("base-model", 0);
        }
        g.CurrentTextEnc = new(["base-clip", 0], g, WGNodeData.DT_TEXTENC, compat);
        g.CurrentVae = new(["base-vae", 0], g, WGNodeData.DT_VAE, compat);
        g.CurrentMedia = new(["base-image", 0], g, WGNodeData.DT_IMAGE, compat);
        return g;
    }

    private static void RunSegment(WorkflowGenerator g, bool beforeRefiner = false)
    {
        g.UserInput.Set(T2IParamTypes.SegmentApplyAfter, beforeRefiner ? "Base" : "Refiner");
        WorkflowGenerator.Steps.Single(step => step.Priority == (beforeRefiner ? -4.5 : 5)).Action(g);
    }

    private static JProperty[] Nodes(WorkflowGenerator g, string type) => g.Workflow.Properties().Where(p => p.Value["class_type"]?.ToString() == type).ToArray();

    [TestCase(false)]
    [TestCase(true)]
    public void ReferenceEditUsesPlainCropLatentAndRuntimeDiagnostics(bool beforeRefiner)
    {
        WorkflowGenerator g = Prepare(withLora: false);
        RunSegment(g, beforeRefiner);

        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Is.Empty, "No dependency on the newer upstream node is allowed.");
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Has.Length.EqualTo(1));
        JProperty encoder = Nodes(g, "SwarmQwenImage21SegmentEncode").Single();
        Assert.That(encoder.Value["inputs"]!["image"], Is.EqualTo(new JArray(Nodes(g, "SwarmImageScaleForMP").Single().Name, 0)));
        Assert.That(encoder.Value["inputs"]!["prompt"]!.Value<string>()!.Trim(), Is.EqualTo("make the face blue"));
        Assert.That(encoder.Value["inputs"]!["negative_prompt"]!.Value<string>()!.Trim(), Is.EqualTo("segment negative"));

        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        Assert.That(sampler.Value["inputs"]!["positive"], Is.EqualTo(new JArray(encoder.Name, 0)));
        Assert.That(sampler.Value["inputs"]!["negative"], Is.EqualTo(new JArray(encoder.Name, 1)));
        JProperty[] vaeEncodes = Nodes(g, "VAEEncode");
        JProperty plainCropEncode = vaeEncodes.Single(p => p.Value["inputs"]!["pixels"] is JArray pixels
            && JToken.DeepEquals(pixels, new JArray(Nodes(g, "SwarmImageScaleForMP").Single().Name, 0)));
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.EqualTo(new JArray(plainCropEncode.Name, 0)));
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.Not.EqualTo(new JArray(encoder.Name, 2)));
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.Zero);
        Assert.That(Nodes(g, "DifferentialDiffusion"), Is.Empty, "Plain-latent Reference Edit must not use denoise-mask DifferentialDiffusion.");

        Assert.That(Nodes(g, "LoraLoaderModelOnly"), Is.Empty);

        JProperty diagnostic = Nodes(g, "SwarmQwen21SegmentDiagnostics").Single();
        Assert.That(diagnostic.Value["inputs"]!["generated"], Is.Not.Null);
        Assert.That(diagnostic.Value["inputs"]!["source"], Is.EqualTo(new JArray(Nodes(g, "SwarmImageScaleForMP").Single().Name, 0)));
        Assert.That(diagnostic.Value["inputs"]!["prompt"]!.Value<string>()!.Trim(), Is.EqualTo("make the face blue"));
        Assert.That(diagnostic.Value["inputs"]!["mode"]!.Value<string>(), Is.EqualTo("Reference Edit"));

        JProperty splitAlpha = Nodes(g, "SplitImageWithAlpha").Single();
        Assert.That(splitAlpha.Value["inputs"]!["image"], Is.EqualTo(new JArray(diagnostic.Name, 0)));
        JProperty scaleBack = Nodes(g, "ImageScale").Last();
        Assert.That(scaleBack.Value["inputs"]!["image"], Is.EqualTo(new JArray(splitAlpha.Name, 0)));
        JProperty composite = Nodes(g, "SwarmImageCompositeMaskedColorCorrecting").Single();
        Assert.That(composite.Value["inputs"]!["source"], Is.EqualTo(new JArray(scaleBack.Name, 0)));
        Assert.That(g.CurrentMedia.Path, Is.EqualTo(new JArray(composite.Name, 0)));
    }

    [Test]
    public void ReferenceEditUsesCreativityAsEditControl()
    {
        WorkflowGenerator g = Prepare(creativity: 0.8, withLora: false);
        RunSegment(g);
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.EqualTo(4));
        Assert.That(Nodes(g, "DifferentialDiffusion"), Is.Empty);
        Assert.That(Nodes(g, "SetLatentNoiseMask"), Has.Length.EqualTo(1), "The crop helper may build its legacy masked latent, but Reference Edit must not sample it.");
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.Not.EqualTo(new JArray(Nodes(g, "SetLatentNoiseMask").Single().Name, 0)));
    }

    [Test]
    public void ReferenceEditDoesNotRequireNativeComfyEncoderFeature()
    {
        WorkflowGenerator g = Prepare(withLora: false);
        Assert.That(g.Features, Does.Not.Contain("qwen21_native_conditioning"));
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Has.Length.EqualTo(1));
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Is.Empty);
    }

    [Test]
    public void TextOnlySegmentStillUsesQwenEncoderWithoutLora()
    {
        WorkflowGenerator g = Prepare(withLora: false);
        RunSegment(g);
        Assert.That(Nodes(g, "LoraLoaderModelOnly"), Is.Empty);
        JProperty encoder = Nodes(g, "SwarmQwenImage21SegmentEncode").Single();
        Assert.That(encoder.Value["inputs"]!["prompt"]!.Value<string>()!.Trim(), Is.EqualTo("make the face blue"));
    }

    [Test]
    public void ExplicitSameModelOverrideStillUsesScopedLora()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.SegmentModel, g.FinalLoadedModel);
        g.NodeHelpers["modelloader_model.safetensors_Refiner"] = "base-model:0:base-clip:0:base-vae:0";
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Is.Empty);
        Assert.That(Nodes(g, "SwarmTextEncodeAdvanced"), Has.Length.EqualTo(2));
        Assert.That(Nodes(g, "LoraLoaderModelOnly"), Has.Length.EqualTo(1));
    }

    [Test]
    public void ZeroCreativityDoesNotRegenerate()
    {
        WorkflowGenerator g = Prepare(creativity: 0);
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmKSampler"), Is.Empty);
        Assert.That(Nodes(g, "SwarmQwen21SegmentDiagnostics"), Is.Empty);
        Assert.That(g.CurrentMedia.Path, Is.EqualTo(new JArray("base-image", 0)));
    }

    [Test]
    public void MaskedModeKeepsPartialDenoiseAndDifferentialDiffusion()
    {
        WorkflowGenerator g = Prepare(creativity: 0.6, withLora: false);
        g.UserInput.InternalSet.ValuesInput[T2IParamTypes.CleanTypeName("Segment Qwen 2.1 Sampling")] = "Masked Img2Img";
        RunSegment(g);
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.EqualTo(8));
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.EqualTo(new JArray(Nodes(g, "SetLatentNoiseMask").Single().Name, 0)));
        Assert.That(Nodes(g, "DifferentialDiffusion"), Has.Length.EqualTo(1));
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Has.Length.EqualTo(1));
        Assert.That(Nodes(g, "SwarmQwen21SegmentDiagnostics"), Has.Length.EqualTo(1));
    }

    [Test]
    public void TwoSegmentsHaveIndependentEncodersPromptsAndLoraModelPaths()
    {
        WorkflowGenerator g = Prepare(secondSegment: true);
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Is.Empty);
        Assert.That(Nodes(g, "SwarmQwen21SegmentDiagnostics"), Has.Length.EqualTo(2));

        JProperty[] encoders = Nodes(g, "SwarmTextEncodeAdvanced");
        Assert.That(encoders, Has.Length.EqualTo(4));
        Assert.That(encoders.Select(e => e.Value["inputs"]!["prompt"]!.Value<string>()!.Trim()),
            Does.Contain("make the face blue").And.Contain("make the hair red"));

        JProperty[] samplers = Nodes(g, "SwarmKSampler");
        JToken firstModel = g.Workflow[samplers[0].Value["inputs"]!["model"]![0]!.Value<string>()!]!;
        JToken secondModel = g.Workflow[samplers[1].Value["inputs"]!["model"]![0]!.Value<string>()!]!;
        Assert.That(firstModel["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("face.safetensors"));
        Assert.That(secondModel["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("hair.safetensors"));
        Assert.That(secondModel["inputs"]!["model"], Is.EqualTo(firstModel["inputs"]!["model"]));
    }

    [Test]
    public void ScopedQwenLoraCutsReferenceConditioningAndUsesTextOnlyImg2Img()
    {
        WorkflowGenerator g = Prepare();
        RunSegment(g);

        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Is.Empty,
            "A confined Qwen LoRA must not compete with source-subject reference conditioning.");
        Assert.That(Nodes(g, "ReferenceLatent"), Is.Empty);
        JProperty[] textEncoders = Nodes(g, "SwarmTextEncodeAdvanced");
        Assert.That(textEncoders, Has.Length.EqualTo(2));
        Assert.That(textEncoders.Any(e => e.Value["inputs"]!["prompt"]!.Value<string>()!.Trim() == "make the face blue"), Is.True);
        Assert.That(textEncoders.All(e => e.Value["inputs"]!["images"] is null), Is.True);

        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        JProperty plainCropEncode = Nodes(g, "VAEEncode").Single(p =>
            p.Value["inputs"]!["pixels"] is JArray pixels
            && JToken.DeepEquals(pixels, new JArray(Nodes(g, "SwarmImageScaleForMP").Single().Name, 0)));
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.EqualTo(new JArray(plainCropEncode.Name, 0)));

        JProperty lora = Nodes(g, "LoraLoaderModelOnly").Single();
        Assert.That(sampler.Value["inputs"]!["model"], Is.EqualTo(new JArray(lora.Name, 0)));
        Assert.That(lora.Value["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("face.safetensors"));

        JProperty diagnostic = Nodes(g, "SwarmQwen21SegmentDiagnostics").Single();
        Assert.That(diagnostic.Value["inputs"]!["mode"]!.Value<string>(), Is.EqualTo("LoRA Img2Img"));
    }

    [Test]
    public void ScopedQwenLoraIsInsertedInsideExistingWrappersWithoutMutatingBaseChain()
    {
        WorkflowGenerator g = Prepare();
        string compile = g.CreateNode("TorchCompileModel", new JObject()
        {
            ["model"] = new JArray("base-model", 0),
            ["backend"] = "inductor"
        });
        string outer = g.CreateNode("DifferentialDiffusion", new JObject()
        {
            ["model"] = new JArray(compile, 0)
        });
        g.LoadedModelLoraAnchors[$"{compile}:0"] = new JArray("base-model", 0);

        (JArray variant, JArray clip) = g.LoadLorasForConfinementInsideWrappers(11, new JArray(outer, 0), new JArray("base-clip", 0));

        JObject outerClone = (JObject)g.Workflow[variant[0]!.ToString()]!;
        Assert.That(outerClone["class_type"]!.Value<string>(), Is.EqualTo("DifferentialDiffusion"));
        JArray compiledPath = (JArray)outerClone["inputs"]!["model"]!;
        JObject compileClone = (JObject)g.Workflow[compiledPath[0]!.ToString()]!;
        Assert.That(compileClone["class_type"]!.Value<string>(), Is.EqualTo("TorchCompileModel"));
        JArray loraPath = (JArray)compileClone["inputs"]!["model"]!;
        JObject lora = (JObject)g.Workflow[loraPath[0]!.ToString()]!;
        Assert.That(lora["class_type"]!.Value<string>(), Is.EqualTo("LoraLoaderModelOnly"));
        Assert.That(lora["inputs"]!["model"], Is.EqualTo(new JArray("base-model", 0)));
        Assert.That(lora["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("face.safetensors"));
        Assert.That(lora["inputs"]!["strength_model"]!.Value<double>(), Is.EqualTo(0.8).Within(0.0001));
        Assert.That(clip, Is.EqualTo(new JArray("base-clip", 0)));

        // Original shared model chain must remain untouched for the base pass.
        Assert.That(g.Workflow[compile]!["inputs"]!["model"], Is.EqualTo(new JArray("base-model", 0)));
        Assert.That(g.Workflow[outer]!["inputs"]!["model"], Is.EqualTo(new JArray(compile, 0)));

        (JArray cached, _) = g.LoadLorasForConfinementInsideWrappers(11, new JArray(outer, 0), new JArray("base-clip", 0));
        Assert.That(cached, Is.EqualTo(variant), "Repeated use of the same section/model chain should reuse its scoped variant.");
    }

    [Test]
    public void KreaKeepsItsExistingSegmentPath()
    {
        WorkflowGenerator g = Prepare(qwen: false, creativity: 0.6);
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Is.Empty);
        Assert.That(Nodes(g, "SwarmQwen21SegmentDiagnostics"), Is.Empty);
        Assert.That(Nodes(g, "SwarmKSampler").Single().Value["inputs"]!["start_at_step"]!.Value<int>(), Is.EqualTo(8));
        Assert.That(Nodes(g, "SetLatentNoiseMask"), Has.Length.EqualTo(1));
        Assert.That(Nodes(g, "LoraLoader"), Has.Length.EqualTo(1));
    }
}
