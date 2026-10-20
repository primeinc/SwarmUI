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
        ComfyUIBackendExtension.RefinerUpscaleMethod ??= new(new T2IParamType("Refiner Upscale Method", "", "None", ID: "refinerupscalemethod"));
        ComfyUIBackendExtension.RefinerSamplerParam ??= new(new T2IParamType("Refiner Sampler", "", "euler", ID: "refinersampler"));
        ComfyUIBackendExtension.RefinerSchedulerParam ??= new(new T2IParamType("Refiner Scheduler", "", "simple", ID: "refinerscheduler"));
        ComfyUIBackendExtension.RefinerHyperTile ??= new(new T2IParamType("Refiner HyperTile", "", "128", ID: "refinerhypertile"));
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

    private static WorkflowGenerator Prepare(bool qwen = true, double creativity = 1, bool secondSegment = false, bool withLora = false)
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
        WorkflowGenerator g = Prepare();
        RunSegment(g, beforeRefiner);
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Is.Empty);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Is.Empty);
        JProperty[] encoders = Nodes(g, "SwarmTextEncodeAdvanced");
        Assert.That(encoders, Has.Length.EqualTo(2));
        JArray crop = new(Nodes(g, "SwarmImageScaleForMP").Single().Name, 0);
        Assert.That(encoders.All(e => JToken.DeepEquals(e.Value["inputs"]!["images"], crop)), Is.True,
            "Qwen segment must use the crop in the exact conditioner used by normal editing.");
        Assert.That(encoders.Select(e => e.Value["inputs"]!["prompt"]!.Value<string>()!.Trim()),
            Is.EquivalentTo(new[] { "make the face blue", "segment negative" }));
        Assert.That(Nodes(g, "ReferenceLatent"), Has.Length.EqualTo(2),
            "Positive and negative conditioning must both include a crop VAE reference.");
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        Assert.That(sampler.Value["inputs"]!["positive"]![0]!.Value<string>(),
            Is.EqualTo(Nodes(g, "ReferenceLatent")[0].Name));
        string latentId = sampler.Value["inputs"]!["latent_image"]![0]!.Value<string>()!;
        JProperty latent = g.Workflow.Property(latentId)!;
        Assert.That(latent.Value["class_type"]!.Value<string>(), Is.EqualTo("VAEEncode"));
        Assert.That(latent.Value["inputs"]!["pixels"], Is.EqualTo(crop));
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.Zero);
        Assert.That(sampler.Value["inputs"]!["steps"]!.Value<int>(), Is.EqualTo(20));
        Assert.That(Nodes(g, "DifferentialDiffusion"), Is.Empty);
        Assert.That(Nodes(g, "LoraLoaderModelOnly"), Is.Empty);
        Assert.That(Nodes(g, "SwarmQwen21SegmentDiagnostics"), Is.Empty);
        Assert.That(Nodes(g, "ThresholdMask"), Is.Empty,
            "A blurred Qwen segment edge must not be binarized by the 0.001 threshold.");
        JProperty composite = Nodes(g, "SwarmImageCompositeMaskedColorCorrecting").Single();
        Assert.That(g.CurrentMedia.Path, Is.EqualTo(new JArray(composite.Name, 0)));
    }

    [Test]
    public void ReferenceEditUsesCreativityAsEditControl()
    {
        WorkflowGenerator g = Prepare(creativity: 0.8);
        RunSegment(g);
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        Assert.That(sampler.Value["inputs"]!["steps"]!.Value<int>(), Is.EqualTo(25));
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.EqualTo(5));
        Assert.That(sampler.Value["inputs"]!["steps"]!.Value<int>() -
            sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.EqualTo(20),
            "A 20-step Qwen segment must really execute 20 denoising passes even with partial creativity.");
        Assert.That(Nodes(g, "DifferentialDiffusion"), Is.Empty);
        Assert.That(Nodes(g, "SetLatentNoiseMask"), Has.Length.EqualTo(1));
        Assert.That(sampler.Value["inputs"]!["latent_image"],
            Is.Not.EqualTo(new JArray(Nodes(g, "SetLatentNoiseMask").Single().Name, 0)));
    }

    [TestCase(true)]
    [TestCase(false)]
    public void ReferenceEditOriginalReferenceSwitchMustReachEncoder(bool enabled)
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.SegmentQwenUseOriginalReference, enabled);
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmTextEncodeAdvanced"), Has.Length.EqualTo(2));
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Is.Empty);
        JArray crop = new(Nodes(g, "SwarmImageScaleForMP").Single().Name, 0);
        if (enabled)
        {
            Assert.That(Nodes(g, "SwarmTextEncodeAdvanced").All(e =>
                JToken.DeepEquals(e.Value["inputs"]!["images"], crop)), Is.True);
            Assert.That(Nodes(g, "ReferenceLatent"), Has.Length.EqualTo(2));
        }
        else
        {
            Assert.That(Nodes(g, "SwarmTextEncodeAdvanced").All(e =>
                e.Value["inputs"]!["images"] is null ||
                e.Value["inputs"]!["images"]!.Type == JTokenType.Null), Is.True);
            Assert.That(Nodes(g, "ReferenceLatent"), Is.Empty);
        }
    }

    [Test]
    public void ReferenceEditDoesNotRequireNativeComfyEncoderFeature()
    {
        WorkflowGenerator g = Prepare();
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmTextEncodeAdvanced"), Has.Length.EqualTo(2));
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Is.Empty);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Is.Empty);
    }

    [Test]
    public void TextOnlySegmentStillUsesQwenEncoderWithoutLora()
    {
        WorkflowGenerator g = Prepare();
        RunSegment(g);
        Assert.That(Nodes(g, "LoraLoaderModelOnly"), Is.Empty);
        JProperty[] encoders = Nodes(g, "SwarmTextEncodeAdvanced");
        Assert.That(encoders, Has.Length.EqualTo(2));
        Assert.That(encoders.Any(e =>
            e.Value["inputs"]!["prompt"]!.Value<string>()!.Trim() == "make the face blue"), Is.True);
    }

    [Test]
    public void ExplicitSameModelOverrideStillUsesScopedLora()
    {
        WorkflowGenerator g = Prepare(withLora: true);
        g.UserInput.Set(T2IParamTypes.SegmentModel, g.FinalLoadedModel);
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmTextEncodeAdvanced"), Has.Length.EqualTo(2));
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Is.Empty);
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
        Assert.That(sampler.Value["inputs"]!["steps"]!.Value<int>(), Is.EqualTo(34));
        Assert.That(sampler.Value["inputs"]!["start_at_step"]!.Value<int>(), Is.EqualTo(14));
        Assert.That(sampler.Value["inputs"]!["latent_image"],
            Is.EqualTo(new JArray(Nodes(g, "SetLatentNoiseMask").Single().Name, 0)));
        Assert.That(Nodes(g, "DifferentialDiffusion"), Has.Length.EqualTo(1));
        Assert.That(Nodes(g, "SwarmTextEncodeAdvanced"), Has.Length.EqualTo(2));
        Assert.That(Nodes(g, "SwarmQwen21SegmentDiagnostics"), Is.Empty);
    }

    [Test]
    public void TwoSegmentsHaveIndependentEncodersPromptsAndLoraModelPaths()
    {
        WorkflowGenerator g = Prepare(secondSegment: true, withLora: true);
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Is.Empty);
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Is.Empty);
        JProperty[] encoders = Nodes(g, "SwarmTextEncodeAdvanced");
        Assert.That(encoders, Has.Length.EqualTo(4));
        Assert.That(encoders.Select(e => e.Value["inputs"]!["prompt"]!.Value<string>()!.Trim()),
            Does.Contain("make the face blue").And.Contain("make the hair red"));
        JProperty[] samplers = Nodes(g, "SwarmKSampler");
        Assert.That(samplers, Has.Length.EqualTo(2));
        JToken firstModel = g.Workflow[samplers[0].Value["inputs"]!["model"]![0]!.Value<string>()!]!;
        JToken secondModel = g.Workflow[samplers[1].Value["inputs"]!["model"]![0]!.Value<string>()!]!;
        Assert.That(firstModel["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("face.safetensors"));
        Assert.That(secondModel["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("hair.safetensors"));
    }

    [TestCase(true)]
    [TestCase(false)]
    public void RealPromptPreparsePreservesSelectedSectionLoraOnSegmentSampler(bool matchesSection)
    {
        WorkflowGenerator g = Prepare(withLora: true);
        g.UserInput.Set(T2IParamTypes.Prompt, "base scene <segment:face,1,0.8>portrait change");
        g.UserInput.Set(T2IParamTypes.NegativePrompt, "global negative <segment:face,1,0.8>segment negative");
        g.UserInput.Set(T2IParamTypes.LoraSectionConfinement,
            new List<string> { matchesSection ? "11" : "12" });
        // Production input uses PreparsePromptLikes to supply the //cid; the earlier
        // fixture inserted it by hand and could never catch a request-path mismatch.
        g.UserInput.PreparsePromptLikes();
        PromptRegion.Part part = new PromptRegion(g.UserInput.Get(T2IParamTypes.Prompt)).Parts
            .Single(p => p.Type == PromptRegion.PartType.Segment);
        Assert.That(part.ContextID, Is.EqualTo(11));
        WorkflowGenerator.Steps.Single(step => step.Priority == -15).Action(g);
        Assert.That(Nodes(g, "LoraLoaderModelOnly"), Is.Empty,
            "The base model cannot silently consume an adapter assigned to a segment.");
        RunSegment(g);
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        JArray modelPath = (JArray)sampler.Value["inputs"]!["model"]!;
        JProperty modelNode = g.Workflow.Property(modelPath[0]!.Value<string>()!)!;
        if (matchesSection)
        {
            Assert.That(modelNode.Value["class_type"]!.Value<string>(), Is.EqualTo("LoraLoaderModelOnly"));
            Assert.That(modelNode.Value["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("face.safetensors"));
            Assert.That(modelNode.Value["inputs"]!["strength_model"]!.Value<double>(), Is.EqualTo(0.8).Within(0.0001));
        }
        else
        {
            Assert.That(Nodes(g, "LoraLoaderModelOnly"), Is.Empty,
                "Negative control: a mismatch between the prompt's generated cid and selected scope must not apply the LoRA.");
        }
    }

    [TestCase(true)]
    [TestCase(false)]
    public void RealBaseLoaderToSegmentHandoffKeepsSectionLoRaDistinct(bool matchesSection)
    {
        // Unlike Prepare(withLora: true)'s synthetic base-model node, invoke the production
        // initial loader before creating the segmented image workflow.
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.Loras, new List<string> { "hair", "face" });
        g.UserInput.Set(T2IParamTypes.LoraWeights, new List<string> { "0.35", "0.8" });
        g.UserInput.Set(T2IParamTypes.LoraSectionConfinement,
            new List<string> { "-1", matchesSection ? "11" : "12" });
        WorkflowGenerator.Steps.Single(step => step.Priority == -15).Action(g);

        JArray initialModel = g.CurrentModel.Path;
        JProperty initialLora = g.Workflow.Property(initialModel[0]!.Value<string>()!)!;
        Assert.That(initialLora.Value["class_type"]!.Value<string>(), Is.EqualTo("LoraLoaderModelOnly"));
        Assert.That(initialLora.Value["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("hair.safetensors"));

        RunSegment(g);
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        JArray samplerModel = (JArray)sampler.Value["inputs"]!["model"]!;
        if (!matchesSection)
        {
            Assert.That(samplerModel, Is.EqualTo(initialModel),
                "Negative control: a LoRA assigned to a different section must not appear in this segment.");
            return;
        }
        Assert.That(samplerModel, Is.Not.EqualTo(initialModel),
            "Segment sampler must not silently reuse the globally loaded model without its section LoRA.");
        JProperty scoped = g.Workflow.Property(samplerModel[0]!.Value<string>()!)!;
        Assert.That(scoped.Value["class_type"]!.Value<string>(), Is.EqualTo("LoraLoaderModelOnly"));
        Assert.That(scoped.Value["inputs"]!["lora_name"]!.Value<string>(), Is.EqualTo("face.safetensors"));
        JArray scopedParent = (JArray)scoped.Value["inputs"]!["model"]!;
        Assert.That(scopedParent, Is.EqualTo(initialModel),
            "Section LoRA must extend the actual global loader's model object, not a separately rebuilt unpatched base.");
    }

    [Test]
    public void SegmentScopedLoraIsAppliedBeforeTorchCompile()
    {
        WorkflowGenerator g = Prepare(withLora: true);
        g.UserInput.Set(T2IParamTypes.TorchCompile, "inductor");
        RunSegment(g);

        JProperty lora = Nodes(g, "LoraLoaderModelOnly").Single();
        JProperty compile = Nodes(g, "TorchCompileModel").Single();
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();

        Assert.That(compile.Value["inputs"]!["model"], Is.EqualTo(new JArray(lora.Name, 0)),
            "Scoped Qwen LoRA must be inside the compiled model graph, matching ordinary model-load LoRA ordering.");
        Assert.That(sampler.Value["inputs"]!["model"], Is.EqualTo(new JArray(compile.Name, 0)),
            "Segment sampler must consume the compiled model whose input already contains the confined LoRA.");
    }

    [Test]
    public void NativeQwenSegmentPreservesConditioningPostTransforms()
    {
        WorkflowGenerator g = Prepare(withLora: true);
        g.UserInput.Set(T2IParamTypes.ConditioningMultiplier, 0.7);
        g.UserInput.Set(T2IParamTypes.NegativeConditioningMultiplier, 0.5);
        g.UserInput.Set(T2IParamTypes.NegativePrompt, "");
        g.UserInput.Set(T2IParamTypes.ZeroNegative, true);
        RunSegment(g);
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        JProperty posRef = g.Workflow.Property(sampler.Value["inputs"]!["positive"]![0]!.Value<string>()!)!;
        Assert.That(posRef.Value["class_type"]!.Value<string>(), Is.EqualTo("ReferenceLatent"));
        JProperty positive = g.Workflow.Property(posRef.Value["inputs"]!["conditioning"]![0]!.Value<string>()!)!;
        Assert.That(positive.Value["class_type"]!.Value<string>(), Is.EqualTo("ConditioningMultiply"));
        Assert.That(positive.Value["inputs"]!["multiplier"]!.Value<double>(), Is.EqualTo(0.7).Within(0.001));
        JProperty negRef = g.Workflow.Property(sampler.Value["inputs"]!["negative"]![0]!.Value<string>()!)!;
        Assert.That(negRef.Value["class_type"]!.Value<string>(), Is.EqualTo("ReferenceLatent"));
        JProperty zero = g.Workflow.Property(negRef.Value["inputs"]!["conditioning"]![0]!.Value<string>()!)!;
        Assert.That(zero.Value["class_type"]!.Value<string>(), Is.EqualTo("ConditioningZeroOut"));
        JProperty negative = g.Workflow.Property(zero.Value["inputs"]!["conditioning"]![0]!.Value<string>()!)!;
        Assert.That(negative.Value["class_type"]!.Value<string>(), Is.EqualTo("ConditioningMultiply"));
        Assert.That(negative.Value["inputs"]!["multiplier"]!.Value<double>(), Is.EqualTo(0.5).Within(0.001));
    }

    [Test]
    public void BatchedQwenSegmentsMustProcessEachImageIndependently()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.BatchSize, 2);
        RunSegment(g);
        Assert.That(Nodes(g, "ImageFromBatch"), Has.Length.EqualTo(2));
        JProperty[] encoders = Nodes(g, "SwarmTextEncodeAdvanced");
        Assert.That(encoders, Has.Length.EqualTo(4));
        Assert.That(Nodes(g, "SwarmKSampler"), Has.Length.EqualTo(2));
        JProperty finalBatch = Nodes(g, "ImageBatch").Single();
        Assert.That(g.CurrentMedia.Path, Is.EqualTo(new JArray(finalBatch.Name, 0)));
        string[] cropIds = encoders.Select(e =>
            e.Value["inputs"]!["images"]![0]!.Value<string>()!).Distinct().ToArray();
        Assert.That(cropIds, Has.Length.EqualTo(2), "Each batch item must use its own crop.");
    }

    [TestCase("small", true)]
    [TestCase("none", false)]
    public void QwenSegmentOptionalPromptImagesRespectTextImageSetting(string setting, bool expected)
    {
        WorkflowGenerator g = Prepare();
        using SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> source = new(640, 320);
        // The first prompt image is the original full image. Segment edit replaces
        // that vision slot with the crop, retaining explicit additional references.
        g.UserInput.Set(T2IParamTypes.PromptImages, new List<Image> { new(source), new(source) });
        g.UserInput.Set(ComfyUIBackendExtension.TextEncodedImage, setting);
        RunSegment(g);
        JProperty[] encoders = Nodes(g, "SwarmTextEncodeAdvanced");
        Assert.That(encoders, Has.Length.EqualTo(2));
        JArray crop = new(Nodes(g, "SwarmImageScaleForMP").Single().Name, 0);
        Assert.That(Nodes(g, "ReferenceLatent").Length, Is.EqualTo(expected ? 4 : 2));
        if (expected)
        {
            JProperty batched = Nodes(g, "ImageBatch").Single();
            Assert.That(encoders.All(e => JToken.DeepEquals(
                e.Value["inputs"]!["images"], new JArray(batched.Name, 0))), Is.True);
            Assert.That(batched.Value["inputs"]!["image1"], Is.EqualTo(crop));
            JProperty[] slices = Nodes(g, "ImageFromBatch");
            Assert.That(slices, Has.Length.EqualTo(2),
                "The VAE and Qwen vision encoder must consume the same batched reference geometry.");
            string[] sliceIds = slices.Select(node => node.Name).ToArray();
            Assert.That(Nodes(g, "VAEEncode").Count(node =>
                node.Value["inputs"]!["pixels"] is JArray path &&
                sliceIds.Contains(path[0]!.Value<string>()!)), Is.EqualTo(2),
                "Each reference latent must encode the aligned image, not an unresized separate tensor.");
        }
        else
        {
            Assert.That(encoders.All(e =>
                JToken.DeepEquals(e.Value["inputs"]!["images"], crop)), Is.True);
            Assert.That(Nodes(g, "ImageFromBatch"), Is.Empty);
        }
    }

    [Test]
    public void QwenSegmentRejectsPromptImagesBeyondAvailableReferenceSlots()
    {
        WorkflowGenerator g = Prepare();
        using SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> source = new(64, 64);
        Image image = new(source);
        // Crop replaces Prompt Images[0], leaving 16 total Qwen vision slots.
        g.UserInput.Set(T2IParamTypes.PromptImages, Enumerable.Repeat(image, 17).ToList());
        Assert.Throws<SwarmUserErrorException>(() => RunSegment(g));
    }

    [Test]
    public void NativeQwenSegmentPassesDynamicLoraHooksIntoRealEncoder()
    {
        WorkflowGenerator g = Prepare(withLora: true);
        g.UserInput.DynamicLoraIndices.Add(0);
        g.UserInput.Set(T2IParamTypes.Prompt,
            "base scene <segment:face,1,0.8//cid=11>portrait <lora:face//hook=0> <fromto[0.5]:first,second>");
        RunSegment(g);
        JProperty hooks = Nodes(g, "CreateHookLora").Single();
        JProperty[] encoders = Nodes(g, "SwarmTextEncodeAdvanced");
        Assert.That(encoders, Has.Length.GreaterThanOrEqualTo(2));
        Assert.That(encoders.Any(p =>
            JToken.DeepEquals(p.Value["inputs"]!["lora_hooks"], new JArray(hooks.Name, 0))), Is.True,
            "Scheduled dynamic LoRA hooks must use the same real Swarm conditioner as normal generation.");
        Assert.That(encoders.All(p => p.Value["inputs"]!["steps"]!.Value<int>() == 20), Is.True);
        Assert.That(Nodes(g, "LoraLoaderModelOnly"), Is.Empty);
    }

    [Test]
    public void ZeroCreativityStillSavesRequestedSegmentMask()
    {
        WorkflowGenerator g = Prepare(creativity: 0);
        g.UserInput.Set(T2IParamTypes.SaveSegmentMask, true);
        RunSegment(g);

        Assert.That(Nodes(g, "MaskToImage"), Has.Length.EqualTo(1),
            "A zero-creativity Qwen segment still promises to save its detected mask.");
        Assert.That(Nodes(g, "SwarmKSampler"), Is.Empty);
        Assert.That(g.CurrentMedia.Path, Is.EqualTo(new JArray("base-image", 0)));
    }

    [Test]
    public void PostRefinerConfinedLoraRebuildPreservesRefinerOnlyLoras()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.Loras, new List<string> { "hair", "face", "hair" });
        g.UserInput.Set(T2IParamTypes.LoraWeights, new List<string> { "0.2", "0.4", "0.8" });
        g.UserInput.Set(T2IParamTypes.LoraSectionConfinement, new List<string> { "5", "1", "11" });
        g.UserInput.Set(T2IParamTypes.RefinerMethod, "StepSwapNoisy");
        g.UserInput.Set(T2IParamTypes.RefinerControl, 1.0);
        WorkflowGenerator.Steps.Single(step => step.Priority == -4).Action(g);
        RunSegment(g);

        JProperty sampler = Nodes(g, "SwarmKSampler").Last();
        string modelNodeId = sampler.Value["inputs"]!["model"]![0]!.Value<string>()!;
        List<(string name, double strength)> loras = [];
        for (int i = 0; i < 20; i++)
        {
            JToken node = g.Workflow[modelNodeId]!;
            if (node["class_type"]?.ToString() == "LoraLoaderModelOnly")
            {
                loras.Add((node["inputs"]!["lora_name"]!.Value<string>()!,
                    node["inputs"]!["strength_model"]!.Value<double>()));
            }
            if (node["inputs"]?["model"] is not JArray next)
            {
                break;
            }
            modelNodeId = next[0]!.Value<string>()!;
        }

        Assert.That(loras, Has.Count.EqualTo(2),
            "A post-refiner segment should carry its confined LoRA and refiner-only LoRAs, not base-only LoRAs.");
        Assert.That(loras.Any(p => p.name == "hair.safetensors" && System.Math.Abs(p.strength - 0.8) < 0.0001), Is.True);
        Assert.That(loras.Any(p => p.name == "face.safetensors" && System.Math.Abs(p.strength - 0.4) < 0.0001), Is.True);
    }

    [TestCase(false)]
    [TestCase(true)]
    public void UserReportedFourStepReplacementRetainsEveryDenoisingPass(bool withLora)
    {
        WorkflowGenerator g = Prepare(withLora: withLora);
        g.UserInput.Set(T2IParamTypes.Prompt,
            "<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes");
        g.UserInput.Set(T2IParamTypes.Steps, 4);
        g.UserInput.Set(T2IParamTypes.CFGScale, 1.3);
        using SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> source = new(768, 1344);
        g.UserInput.Set(T2IParamTypes.PromptImages, new List<Image> { new(source) });
        g.UserInput.PreparsePromptLikes();
        PromptRegion.Part part = new PromptRegion(g.UserInput.Get(T2IParamTypes.Prompt)).Parts
            .Single(p => p.Type == PromptRegion.PartType.Segment);
        Assert.That(part.ContextID, Is.EqualTo(11));
        Assert.That(part.DataText, Is.EqualTo("girl"));
        Assert.That(part.Strength2, Is.EqualTo(0.8).Within(0.00001));
        Assert.That(part.Strength, Is.EqualTo(0.1).Within(0.00001));
        RunSegment(g);
        JProperty sampler = Nodes(g, "SwarmKSampler").Single();
        int schedule = sampler.Value["inputs"]!["steps"]!.Value<int>();
        int start = sampler.Value["inputs"]!["start_at_step"]!.Value<int>();
        Assert.That(schedule - start, Is.EqualTo(4));
        Assert.That(schedule, Is.EqualTo(5));
        Assert.That(start, Is.EqualTo(1));
        JProperty[] encoders = Nodes(g, "SwarmTextEncodeAdvanced");
        Assert.That(encoders, Has.Length.EqualTo(2));
        Assert.That(encoders.Any(e =>
            e.Value["inputs"]!["prompt"]!.Value<string>()!.Trim() ==
                "Edit the image, replace the girl with a black boy with blue eyes"), Is.True);
        JArray crop = new(Nodes(g, "SwarmImageScaleForMP").Single().Name, 0);
        Assert.That(encoders.All(e =>
            JToken.DeepEquals(e.Value["inputs"]!["images"], crop)), Is.True,
            "The global full-image reference must be replaced by the actual segment crop.");
        Assert.That(Nodes(g, "ReferenceLatent"), Has.Length.EqualTo(2));
        Assert.That(Nodes(g, "ImageBatch"), Is.Empty,
            "Do not send the same edit source as both crop and full-frame reference.");
        Assert.That(Nodes(g, "ThresholdMask"), Is.Empty);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Is.Empty);
        Assert.That(Nodes(g, "TextEncodeQwenImage21"), Is.Empty);
        Assert.That(Nodes(g, "LoraLoaderModelOnly").Length, Is.EqualTo(withLora ? 1 : 0));
    }

    [Test]
    public void KreaKeepsItsExistingSegmentPath()
    {
        WorkflowGenerator g = Prepare(qwen: false, creativity: 0.6, withLora: true);
        RunSegment(g);
        Assert.That(Nodes(g, "SwarmQwenImage21SegmentEncode"), Is.Empty);
        Assert.That(Nodes(g, "SwarmQwen21SegmentDiagnostics"), Is.Empty);
        Assert.That(Nodes(g, "SwarmKSampler").Single().Value["inputs"]!["start_at_step"]!.Value<int>(), Is.EqualTo(8));
        Assert.That(Nodes(g, "SetLatentNoiseMask"), Has.Length.EqualTo(1));
        Assert.That(Nodes(g, "LoraLoader"), Has.Length.EqualTo(1));
    }
}
