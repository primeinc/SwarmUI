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
    private bool hadPriorQwen21VaeInfo;
    private CommonModels.ModelInfo priorQwen21VaeInfo;

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
        // Scoped-Qwen tests reload the model through the full Comfy model-generation pipeline.
        // Initialize every extension param that pipeline reads while leaving each feature at its default/off state.
        ComfyUIBackendExtension.SetClipDevice ??= new(new T2IParamType("Set CLIP Device", "", "cpu", ID: "setclipdevice"));
        ComfyUIBackendExtension.SelfAttentionGuidanceScale ??= new(new T2IParamType("Self-Attention Guidance Scale", "", "0.5", ID: "selfattentionguidancescale"));
        ComfyUIBackendExtension.SelfAttentionGuidanceSigmaBlur ??= new(new T2IParamType("Self-Attention Guidance Sigma Blur", "", "2", ID: "selfattentionguidancesigmablur"));
        ComfyUIBackendExtension.PerturbedAttentionGuidanceScale ??= new(new T2IParamType("Perturbed Attention Guidance Scale", "", "3", ID: "perturbedattentionguidancescale"));
        ComfyUIBackendExtension.RescaleCFGMultiplier ??= new(new T2IParamType("Rescale CFG Multiplier", "", "0.7", ID: "rescalecfgmultiplier"));
        ComfyUIBackendExtension.RenormCFG ??= new(new T2IParamType("Renorm CFG", "", "0", ID: "renormcfg"));
        ComfyUIBackendExtension.UseCfgZeroStar ??= new(new T2IParamType("Use CFG Zero Star", "", "false", ID: "usecfgzerostar"));
        ComfyUIBackendExtension.UseTCFG ??= new(new T2IParamType("Use TCFG", "", "false", ID: "usetcfg"));
        ComfyUIBackendExtension.NormalizedAttentionGuidanceScale ??= new(new T2IParamType("Normalized Attention Guidance Scale", "", "0", ID: "normalizedattentionguidancescale"));
        ComfyUIBackendExtension.NormalizedAttentionGuidanceAlpha ??= new(new T2IParamType("Normalized Attention Guidance Alpha", "", "0.5", ID: "normalizedattentionguidancealpha"));
        ComfyUIBackendExtension.NormalizedAttentionGuidanceTau ??= new(new T2IParamType("Normalized Attention Guidance Tau", "", "1.5", ID: "normalizedattentionguidancetau"));
        ComfyUIBackendExtension.TeaCacheMode ??= new(new T2IParamType("TeaCache Mode", "", "disabled", ID: "teacachemode"));
        ComfyUIBackendExtension.TeaCacheThreshold ??= new(new T2IParamType("TeaCache Threshold", "", "0.25", ID: "teacachethreshold"));
        ComfyUIBackendExtension.TeaCacheStart ??= new(new T2IParamType("TeaCache Start", "", "0", ID: "teacachestart"));
        ComfyUIBackendExtension.EasyCacheMode ??= new(new T2IParamType("EasyCache Mode", "", "disabled", ID: "easycachemode"));
        ComfyUIBackendExtension.EasyCacheThreshold ??= new(new T2IParamType("EasyCache Threshold", "", "0", ID: "easycachethreshold"));
        ComfyUIBackendExtension.EasyCacheStart ??= new(new T2IParamType("EasyCache Start", "", "0", ID: "easycachestart"));
        ComfyUIBackendExtension.EasyCacheEnd ??= new(new T2IParamType("EasyCache End", "", "1", ID: "easycacheend"));
        ComfyUIBackendExtension.ModelAttentionBackend ??= new(new T2IParamType("Model Attention Backend", "", "pytorch attention", ID: "modelattentionbackend"));
        ComfyUIBackendExtension.UseSparseAttention ??= new(new T2IParamType("Use Sparse Attention", "", "None", ID: "usesparseattention"));
    }

    [SetUp]
    public void SetupModels()
    {
        Program.T2IModelSets.TryGetValue("LoRA", out priorLoras);
        priorHashSetting = Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash;
        Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash = false;
        hadPriorQwen21VaeInfo = CommonModels.Known.TryGetValue("qwen-image-2.1-vae", out priorQwen21VaeInfo);
        CommonModels.Known["qwen-image-2.1-vae"] = new(
            "qwen-image-2.1-vae",
            "Qwen Image 2.1 VAE",
            "test fixture",
            "https://example.invalid/qwen_image_2.1_vae_bf16.safetensors",
            new string('0', 64),
            "VAE",
            "QwenImage/qwen_image_2.1_vae_bf16.safetensors");
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
        if (hadPriorQwen21VaeInfo)
        {
            CommonModels.Known["qwen-image-2.1-vae"] = priorQwen21VaeInfo;
        }
        else
        {
            CommonModels.Known.TryRemove("qwen-image-2.1-vae", out _);
        }
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
        if (qwen)
        {
            T2IModel qwenClip = new(null, "/fixture", "/fixture/qwen3vl_8b_int8_convrot.safetensors", "qwen3vl_8b_int8_convrot.safetensors");
            T2IModel qwenVae = new(null, "/fixture", "/fixture/qwen_image_2.1_vae_bf16.safetensors", "qwen_image_2.1_vae_bf16.safetensors");
            input.Set(T2IParamTypes.QwenModel, qwenClip);
            input.Set(T2IParamTypes.VAE, qwenVae);
        }
        input.Set(T2IParamTypes.Prompt, $"base scene <segment:face,{creativity},0.8//cid=11>make the face blue" + (secondSegment ? "<segment:hair,1,0.8//cid=12>make the hair red" : ""));
        input.Set(T2IParamTypes.NegativePrompt, "global negative <segment:face,1,0.8//cid=11>segment negative");
        Assert.That(new PromptRegion(input.Get(T2IParamTypes.Prompt)).Parts.First().Strength2, Is.EqualTo(creativity).Within(0.00001));
        input.Set(T2IParamTypes.Width, 1024);
        input.Set(T2IParamTypes.Height, 1024);
        input.Set(T2IParamTypes.Steps, 20);
        input.Set(T2IParamTypes.CFGScale, 1.0);
        input.Set(T2IParamTypes.Seed, 42L);
        input.Set(T2IParamTypes.Loras, secondSegment ? new List<string> { "face", "hair" } : new List<string> { "face" });
        input.Set(T2IParamTypes.LoraWeights, secondSegment ? new List<string> { "0.8", "0.6" } : new List<string> { "0.8" });
        input.Set(T2IParamTypes.LoraSectionConfinement, secondSegment ? new List<string> { "11", "12" } : new List<string> { "11" });
        WorkflowGenerator g = new() { UserInput = input, Workflow = new(), FinalLoadedModel = model, ModelFolderFormat = "/", Features = ["variation_seed"] };
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
    public void ReferenceEditUsesNativeFreshLatentAndRuntimeDiagnostics(bool beforeRefiner)
    {
        WorkflowGenerator g = Prepare();
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
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.EqualTo(new JArray(encoder.Name, 2)),
            "Qwen 2.1 Reference Edit must sample from the encoder's native empty latent, not re-encode the original crop.");
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.Zero,
            "Native Qwen reference edits use full denoising; partial denoising belongs to Masked Img2Img.");
        Assert.That(Nodes(g, "DifferentialDiffusion"), Is.Empty, "Fresh-latent Reference Edit must not use denoise-mask DifferentialDiffusion.");

        JProperty lora = Nodes(g, "LoraLoaderModelOnly").Single();
        Assert.That(sampler.Value["inputs"]!["model"], Is.EqualTo(new JArray(lora.Name, 0)));
        Assert.That(lora.Value["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("face.safetensors"));
        Assert.That(lora.Value["inputs"]!["strength_model"]!.Value<double>(), Is.EqualTo(0.8).Within(0.0001));

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
    public void ReferenceEditUsesNativeFullDenoiseForPartialSegmentStrength()
    {
        WorkflowGenerator g = Prepare(creativity: 0.8);
        RunSegment(g);
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        JProperty encoder = Nodes(g, "SwarmQwenImage21SegmentEncode").Single();
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.Zero,
            "Reference Edit uses Qwen's fresh-latent generation, so partial denoise would discard initial noise steps.");
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.EqualTo(new JArray(encoder.Name, 2)));
        Assert.That(Nodes(g, "DifferentialDiffusion"), Is.Empty);
        Assert.That(Nodes(g, "SetLatentNoiseMask"), Has.Length.EqualTo(1), "The crop helper may build its legacy masked latent, but Reference Edit must not sample it.");
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.Not.EqualTo(new JArray(Nodes(g, "SetLatentNoiseMask").Single().Name, 0)));
    }

    [Test]
    public void ReferenceEditDoesNotRequireNativeComfyEncoderFeature()
    {
        WorkflowGenerator g = Prepare();
        Assert.That(g.Features, Does.Not.Contain("qwen21_native_conditioning"));
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Has.Length.EqualTo(1));
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Is.Empty);
    }

    [Test]
    public void TextOnlySegmentStillUsesQwenEncoderWithoutLora()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Remove(T2IParamTypes.Loras);
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
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Has.Length.EqualTo(1));
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
        WorkflowGenerator g = Prepare(creativity: 0.6);
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
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Has.Length.EqualTo(2));
        Assert.That(Nodes(g, "SwarmQwen21SegmentDiagnostics"), Has.Length.EqualTo(2));

        JProperty[] encoders = Nodes(g, "SwarmQwenImage21SegmentEncode");
        Assert.That(encoders.Select(e => e.Value["inputs"]!["prompt"]!.Value<string>()!.Trim()), Is.EquivalentTo(new[] { "make the face blue", "make the hair red" }));

        JProperty[] samplers = Nodes(g, "SwarmKSampler");
        JToken firstModel = g.Workflow[samplers[0].Value["inputs"]!["model"]![0]!.Value<string>()!]!;
        JToken secondModel = g.Workflow[samplers[1].Value["inputs"]!["model"]![0]!.Value<string>()!]!;
        Assert.That(firstModel["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("face.safetensors"));
        Assert.That(secondModel["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("hair.safetensors"));
    }

    [Test]
    public void SegmentScopedLoraIsAppliedBeforeTorchCompile()
    {
        WorkflowGenerator g = Prepare(creativity: 0.8);
        g.UserInput.Set(T2IParamTypes.TorchCompile, "inductor");
        RunSegment(g);

        JProperty lora = Nodes(g, "LoraLoaderModelOnly").Single();
        JProperty compile = Nodes(g, "TorchCompileModel").Single();
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();

        Assert.That(compile.Value["inputs"]!["model"], Is.EqualTo(new JArray(lora.Name, 0)),
            "Scoped Qwen LoRA must be inside the compiled model graph, matching ordinary model-load LoRA ordering.");
        Assert.That(sampler.Value["inputs"]!["model"], Is.EqualTo(new JArray(compile.Name, 0)),
            "Segment sampler must consume the compiled model whose input already contains the confined LoRA.");
        Assert.That(sampler.Value["inputs"]!["latent_image"], Is.EqualTo(new JArray(Nodes(g, "SwarmQwenImage21SegmentEncode").Single().Name, 2)),
            "The scoped LoRA and Qwen-native fresh reference latent must reach the same sampler.");
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.Zero);
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
