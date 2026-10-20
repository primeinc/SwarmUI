using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>Exercises the registered segment step, not an isolated replacement helper. GPU model execution is not simulated.</summary>
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
        ComfyUIBackendExtension.EnableReferenceLatents ??= new(new T2IParamType("Enable Reference Latents", "", "none", ID: "enablereferencelatents"));
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

    private static WorkflowGenerator Prepare(bool qwen = true, double creativity = 1, bool secondSegment = false)
    {
        T2IModelCompatClass compat = qwen ? T2IModelClassSorter.CompatQwenImage21 : T2IModelClassSorter.CompatKrea2;
        T2IModel model = new(null, "/fixture", "/fixture/model.safetensors", "model.safetensors")
        {
            ModelClass = new() { ID = qwen ? "qwen-image-2.1" : "krea-2", CompatClass = compat, StandardWidth = 1024, StandardHeight = 1024 }
        };
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.Model, model);
        input.Set(T2IParamTypes.Prompt, $"base scene <segment:face,0.8,{creativity}//cid=11>make the face blue" + (secondSegment ? "<segment:hair,0.8,1//cid=12>make the hair red" : ""));
        input.Set(T2IParamTypes.NegativePrompt, "global negative <segment:face,0.8,1//cid=11>segment negative");
        input.Set(T2IParamTypes.Width, 1024);
        input.Set(T2IParamTypes.Height, 1024);
        input.Set(T2IParamTypes.Steps, 20);
        input.Set(T2IParamTypes.CFGScale, 1.0);
        input.Set(T2IParamTypes.Seed, 42L);
        input.Set(T2IParamTypes.Loras, secondSegment ? new List<string> { "face", "hair" } : new List<string> { "face" });
        input.Set(T2IParamTypes.LoraWeights, secondSegment ? new List<string> { "0.8", "0.6" } : new List<string> { "0.8" });
        input.Set(T2IParamTypes.LoraSectionConfinement, secondSegment ? new List<string> { "11", "12" } : new List<string> { "11" });
        WorkflowGenerator g = new() { UserInput = input, Workflow = new(), FinalLoadedModel = model, ModelFolderFormat = "/", Features = ["variation_seed", "qwen21_native_conditioning"] };
        g.CreateNode("UNETLoader", new JObject() { ["unet_name"] = "model.safetensors" }, "base-model");
        g.CreateNode("CLIPLoader", new JObject() { ["clip_name"] = "text.safetensors" }, "base-clip");
        g.CreateNode("VAELoader", new JObject() { ["vae_name"] = "vae.safetensors" }, "base-vae");
        g.CreateNode("LoadImage", new JObject() { ["image"] = "fixed-base.png" }, "base-image");
        g.CurrentModel = new(["base-model", 0], g, WGNodeData.DT_MODEL, compat);
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
    public void NativeSegmentUsesCropPairedConditioningAndFreshLatent(bool beforeRefiner)
    {
        WorkflowGenerator g = Prepare();
        RunSegment(g, beforeRefiner);
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Has.Length.EqualTo(1), "QWEN_NATIVE_SEGMENT_MISSING");
        JProperty encoder = Nodes(g, "TextEncodeQwenImage21").Single();
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        JObject inputs = (JObject)encoder.Value["inputs"];
        Assert.That(inputs["images.image_1"], Is.EqualTo(new JArray(Nodes(g, "SwarmImageScaleForMP").Single().Name, 0)));
        Assert.That(inputs["resolution"]!.Value<int>(), Is.Zero);
        Assert.That(inputs["prompt"]!.Value<string>()!.Trim(), Is.EqualTo("make the face blue"));
        Assert.That(inputs["negative_prompt"]!.Value<string>()!.Trim(), Is.EqualTo("segment negative"));
        Assert.That(sampler.Value["inputs"]!["positive"], Is.EqualTo(new JArray(encoder.Name, 0)));
        Assert.That(sampler.Value["inputs"]!["negative"], Is.EqualTo(new JArray(encoder.Name, 1)));
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.EqualTo(new JArray(encoder.Name, 2)));
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.Zero);
        Assert.That(Nodes(g, "ReferenceLatent"), Is.Empty, "Native references must not be appended again by CreateKSampler");
        Assert.That(Nodes(g, "SwarmTextEncodeAdvanced"), Is.Empty);
        JProperty lora = Nodes(g, "LoraLoaderModelOnly").Single();
        Assert.That(sampler.Value["inputs"]!["model"], Is.EqualTo(new JArray(lora.Name, 0)));
        Assert.That(lora.Value["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("face.safetensors"));
        Assert.That(lora.Value["inputs"]!["strength_model"]!.Value<double>(), Is.EqualTo(0.8).Within(0.0001));
        JProperty composite = Nodes(g, "SwarmImageCompositeMaskedColorCorrecting").Single();
        Assert.That(g.CurrentMedia.Path, Is.EqualTo(new JArray(composite.Name, 0)));
        Assert.That(composite.Value["inputs"]!["destination"], Is.EqualTo(new JArray("base-image", 0)));
        Assert.That(composite.Value["inputs"]!["mask"], Is.Not.Null);
    }

    [Test]
    public void NativeSegmentCarriesTextWithoutAnyLora()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Remove(T2IParamTypes.Loras);
        RunSegment(g);
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Has.Length.EqualTo(1), "QWEN_NATIVE_SEGMENT_MISSING");
        Assert.That(Nodes(g, "LoraLoaderModelOnly"), Is.Empty);
        Assert.That(Nodes(g, "TextEncodeQwenImage21")[0].Value["inputs"]!["prompt"]!.Value<string>()!.Trim(), Is.EqualTo("make the face blue"));
    }

    [Test]
    public void ExplicitSameModelOverrideUsesSegmentVaeAndScopedLora()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.SegmentModel, g.FinalLoadedModel);
        g.NodeHelpers["modelloader_model.safetensors_Refiner"] = "base-model:0:base-clip:0:base-vae:0";
        RunSegment(g);
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Has.Length.EqualTo(1), "QWEN_NATIVE_SEGMENT_MISSING");
        Assert.That(Nodes(g, "TextEncodeQwenImage21")[0].Value["inputs"]!["vae"], Is.EqualTo(g.CurrentVae.Path));
        Assert.That(Nodes(g, "LoraLoaderModelOnly"), Has.Length.EqualTo(1));
    }

    [Test]
    public void MissingNativeBackendNodeFailsExplicitly()
    {
        WorkflowGenerator g = Prepare();
        g.Features.Remove("qwen21_native_conditioning");
        Assert.Throws<SwarmUserErrorException>(() => RunSegment(g));
    }

    [Test]
    public void ZeroCreativityDoesNotRegenerate()
    {
        WorkflowGenerator g = Prepare(creativity: 0);
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmKSampler"), Is.Empty);
        Assert.That(g.CurrentMedia.Path, Is.EqualTo(new JArray("base-image", 0)));
    }

    [Test]
    public void MaskedModeKeepsPartialDenoiseAndMaskedCrop()
    {
        WorkflowGenerator g = Prepare(creativity: 0.6);
        g.UserInput.InternalSet.ValuesInput["segmentqwen21sampling"] = "Masked Img2Img";
        RunSegment(g);
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Has.Length.EqualTo(1), "QWEN_NATIVE_SEGMENT_MISSING");
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.EqualTo(8));
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.EqualTo(new JArray(Nodes(g, "SetLatentNoiseMask").Single().Name, 0)));
        Assert.That(Nodes(g, "ReferenceLatent"), Is.Empty);
    }

    [Test]
    public void TwoSegmentsHaveIndependentPromptsAndLoraModelPaths()
    {
        WorkflowGenerator g = Prepare(secondSegment: true);
        RunSegment(g);
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Has.Length.EqualTo(2), "QWEN_NATIVE_SEGMENT_MISSING");
        JProperty[] samplers = Nodes(g, "SwarmKSampler");
        JToken firstModel = g.Workflow[samplers[0].Value["inputs"]!["model"]![0]!.Value<string>()!]!;
        JToken secondModel = g.Workflow[samplers[1].Value["inputs"]!["model"]![0]!.Value<string>()!]!;
        Assert.That(firstModel["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("face.safetensors"));
        Assert.That(secondModel["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("hair.safetensors"));
        Assert.That(secondModel["inputs"]!["model"], Is.EqualTo(firstModel["inputs"]!["model"]), "Second segment must not inherit the first segment adapter");
        Assert.That(Nodes(g, "ReferenceLatent"), Is.Empty);
    }

    [Test]
    public void KreaKeepsItsExistingSegmentPath()
    {
        WorkflowGenerator g = Prepare(qwen: false, creativity: 0.6);
        RunSegment(g);
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Is.Empty);
        Assert.That(Nodes(g, "SwarmKSampler").Single().Value["inputs"]!["start_at_step"]!.Value<int>(), Is.EqualTo(8));
        Assert.That(Nodes(g, "SetLatentNoiseMask"), Has.Length.EqualTo(1));
        Assert.That(Nodes(g, "LoraLoader"), Has.Length.EqualTo(1));
    }
}
