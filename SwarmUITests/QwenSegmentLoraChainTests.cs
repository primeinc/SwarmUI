using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>
/// Targeted negative controls for the real C# graph check that runs before
/// Qwen GPU inference. Passing these tests proves missing LoRA wiring fails
/// early; it does NOT prove that a rendered image responds to a real LoRA.
/// </summary>
public class QwenSegmentLoraChainTests : SwarmUITest
{
    [OneTimeSetUp]
    public static void PreInit()
    {
        Setup();
        // Unlike the full application, the NUnit process does not register
        // T2I parameters at startup. Without this, every LoRA-chain control
        // crashes before inspecting a graph.
        if (T2IParamTypes.Loras is null)
        {
            T2IParamTypes.RegisterDefaults();
        }
    }

    [Test]
    public static void PromptLoRAIdentityPreservesAuthoredLinuxFilenameCase()
    {
        // Case-colliding filenames may coexist on a Linux ComfyUI backend.
        // A lowercase preprocessing step previously selected the first
        // catalog entry even when the user explicitly chose the other one.
        string[] catalog =
        [
            "subjects/Portrait.safetensors",
            "subjects/portrait.safetensors"
        ];
        Assert.That(T2IPromptHandling.MatchPromptLoraName("subjects/portrait", catalog),
            Is.EqualTo("subjects/portrait.safetensors"));
        Assert.That(T2IPromptHandling.MatchPromptLoraName("subjects/Portrait", catalog),
            Is.EqualTo("subjects/Portrait.safetensors"));
        Assert.That(T2IPromptHandling.MatchPromptLoraName(
            @"subjects\Portrait.safetensors", catalog),
            Is.EqualTo("subjects/Portrait.safetensors"),
            "Windows separators and optional suffix must still resolve the exact adapter.");
    }

    [Test]
    public static void AmbiguousCaseInsensitivePromptLoRAMatchMustNotPickWrongAdapter()
    {
        string[] catalog =
        [
            "subjects/Portrait.safetensors",
            "subjects/portrait.safetensors"
        ];
        SwarmUserErrorException error = Assert.Throws<SwarmUserErrorException>(() =>
            T2IPromptHandling.MatchPromptLoraName("subjects/PORTRAIT", catalog));
        Assert.That(error.Message, Does.Contain("ambiguous"));
        Assert.That(error.Message, Does.Contain("case-sensitive"));
    }

    [Test]
    public static void PromptLoRAResolutionKeepsLegacySingleMatchAndFuzzyNames()
    {
        string[] catalog = ["subjects/Portrait.safetensors"];
        Assert.That(T2IPromptHandling.MatchPromptLoraName("subjects/portrait", catalog),
            Is.EqualTo("subjects/Portrait.safetensors"),
            "Existing case-insensitive input remains accepted when unambiguous.");
        Assert.That(T2IPromptHandling.MatchPromptLoraName(
            @"subjects\\portrait.SAFETENSORS", catalog),
            Is.EqualTo("subjects/Portrait.safetensors"),
            "Doubly escaped Windows separator and case-insensitive suffix remain unambiguous.");
        Assert.That(T2IPromptHandling.MatchPromptLoraName("Portrait", catalog),
            Is.EqualTo("subjects/Portrait.safetensors"),
            "Legacy shorthand selection remains supported.");
        Assert.That(T2IPromptHandling.MatchPromptLoraName("unrelated", catalog), Is.Null);
    }

    [Test]
    public static void UISelectedLoRAListUsesExactCatalogCaseAndRejectsAmbiguity()
    {
        // Validate the actual HTTP input normalization branch, not only the
        // helper that the prompt-tag processor calls. The UI dropdown may
        // submit two LoRA files with names differing only in Linux case.
        T2IParamType selectedLoras = new(
            "LoRAs", "Selected LoRA catalog inputs", "",
            GetValues: _ => new List<string>
            {
                "subjects/Portrait",
                "subjects/portrait"
            },
            Type: T2IParamDataType.LIST,
            Subtype: "LoRA"
        );
        Assert.That(T2IParamTypes.ValidateParam(
            selectedLoras, "subjects/portrait", null), Is.EqualTo("subjects/portrait"));
        Assert.That(T2IParamTypes.ValidateParam(
            selectedLoras, "subjects/Portrait", null), Is.EqualTo("subjects/Portrait"));
        Assert.That(T2IParamTypes.ValidateParam(
            selectedLoras, "subjects/Portrait,subjects/portrait", null),
            Is.EqualTo("subjects/Portrait\n|||\nsubjects/portrait"));
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            T2IParamTypes.ValidateParam(selectedLoras, "subjects/PORTRAIT", null))
            .Message, Does.Contain("ambiguous"));
    }

    private static T2IModel QwenPromptModel()
    {
        T2IModelClass modelClass = new()
        {
            ID = T2IModelClassSorter.CompatQwenImage21.ID,
            CompatClass = T2IModelClassSorter.CompatQwenImage21,
            Name = "Qwen Image 2.1",
            StandardWidth = 1024,
            StandardHeight = 1024
        };
        return new T2IModel(null, "/models/diffusion_models",
            "/models/diffusion_models/qwen-test.safetensors", "qwen-test.safetensors")
        {
            ModelClass = modelClass,
            StandardWidth = 1024,
            StandardHeight = 1024
        };
    }

    [Test]
    public static void ExplicitMissingQwenSegmentLoRAMustFailAtActualPromptProcessor()
    {
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.Model, QwenPromptModel());
        T2IPromptHandling.PromptTagContext context = new()
        {
            Input = input,
            SectionID = 11,
            Loras = ["installed/real.safetensors"]
        };
        SwarmUserErrorException error = Assert.Throws<SwarmUserErrorException>(() =>
            T2IPromptHandling.PromptTagPostProcessors["lora"]("subjects/not-installed", context));
        Assert.That(error.Message, Does.Contain("Qwen segment 11"));
        Assert.That(error.Message, Does.Contain("not-installed"));
        Assert.That(error.Message, Does.Contain("Refusing to generate"));
        Assert.That(input.Get(T2IParamTypes.Loras), Is.Null,
            "An unresolved authored LoRA must not be promoted as if installed.");
    }

    [Test]
    public static void MissingQwenSegmentLoRAMustFailWithNonQwenBaseModelOverride()
    {
        T2IModelClass sdClass = new()
        {
            ID = T2IModelClassSorter.CompatSdv1.ID,
            CompatClass = T2IModelClassSorter.CompatSdv1,
            Name = "SDv1"
        };
        T2IModel sdModel = new(null, "/models/Stable-Diffusion",
            "/models/Stable-Diffusion/test.safetensors", "test.safetensors")
        {
            ModelClass = sdClass
        };
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.Model, sdModel);
        input.Set(T2IParamTypes.SegmentModel, QwenPromptModel());
        T2IPromptHandling.PromptTagContext context = new()
        {
            Input = input,
            SectionID = 11,
            Loras = ["installed/real.safetensors"]
        };
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            T2IPromptHandling.PromptTagPostProcessors["lora"]("missing/adapter", context))
            .Message, Does.Contain("Qwen segment 11"));
    }

    [Test]
    public static void UnrelatedGlobalUnresolvedLoRAStillUsesLegacyWarningPath()
    {
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.Model, QwenPromptModel());
        T2IPromptHandling.PromptTagContext context = new()
        {
            Input = input,
            SectionID = 0,
            Loras = ["installed/real.safetensors"]
        };
        Assert.DoesNotThrow(() =>
            T2IPromptHandling.PromptTagPostProcessors["lora"]("missing/global", context));
        Assert.That(input.ExtraMeta.ContainsKey("parser_warnings"), Is.True,
            "Only explicit segmented Qwen adapters must fail closed.");
    }

    private static WorkflowGenerator Prepare(int segmentConfinement = 11, double selectedStrength = 1)
    {
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.Loras, new List<string> { "subjects/portrait" });
        input.Set(T2IParamTypes.LoraWeights, new List<string> { selectedStrength.ToString("0.00") });
        input.Set(T2IParamTypes.LoraSectionConfinement, new List<string> { $"{segmentConfinement}" });
        WorkflowGenerator g = new() { UserInput = input, Workflow = new JObject() };
        g.Workflow["raw"] = new JObject
        {
            ["class_type"] = "UNETLoader",
            ["inputs"] = new JObject { ["unet_name"] = "test-model.safetensors" }
        };
        g.Workflow["patched"] = new JObject
        {
            ["class_type"] = "LoraLoaderModelOnly",
            ["inputs"] = new JObject
            {
                ["model"] = new JArray("raw", 0),
                ["lora_name"] = "subjects/portrait.safetensors",
                ["strength_model"] = 1.0
            }
        };
        return g;
    }

    [Test]
    public static void MatchingSectionLoRAMustBeOnTheActualSamplerModelPath()
    {
        WorkflowGenerator g = Prepare();
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void MissingLoRAMustFailEvenWhenAValidLoaderNodeExistsElsewhereInGraph()
    {
        WorkflowGenerator g = Prepare();
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("raw", 0), 11, false));
    }

    [Test]
    public static void ModelOnlyLoRALoaderOutputOneCannotCountAsModelChain()
    {
        WorkflowGenerator g = Prepare();
        // LoraLoaderModelOnly has only MODEL port 0. A name/weight-only
        // verifier used to accept this invalid output port.
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
                g, new JArray("patched", 1), 11, false)).Message,
            Does.Contain("MODEL must use output 0"));
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void ComfyLoRALoaderClipOutputCannotCountAsModelChain()
    {
        WorkflowGenerator g = Prepare();
        ((JObject)g.Workflow["patched"])["class_type"] = "LoraLoader";
        // Comfy's LoraLoader RETURN_TYPES are (MODEL, CLIP). Only port 0
        // can feed the sampler's MODEL input, even with a matching LoRA name.
        ((JObject)g.Workflow["patched"]["inputs"])["clip"] = new JArray("clip", 0);
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
                g, new JArray("patched", 1), 11, false)).Message,
            Does.Contain("MODEL must use output 0"));
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void IndirectClipOutputCannotPassThroughModelWrapper()
    {
        WorkflowGenerator g = Prepare();
        g.Workflow["sampling"] = new JObject
        {
            ["class_type"] = "ModelSamplingDiscrete",
            ["inputs"] = new JObject { ["model"] = new JArray("patched", 1) }
        };
        Assert.Throws<SwarmUserErrorException>(() =>
            WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
                g, new JArray("sampling", 0), 11, false));
    }

    [Test]
    public static void IncorrectPatchStrengthMustFailBeforeGpu()
    {
        WorkflowGenerator g = Prepare(selectedStrength: 0.7);
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void LoRAFromAnotherSegmentMustNotContaminateThisSection()
    {
        WorkflowGenerator g = Prepare(segmentConfinement: 12);
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("raw", 0), 11, false));
    }

    [Test]
    public static void MaskedLanPaintMustRejectDifferentialDiffusionOnSamplerModelPath()
    {
        WorkflowGenerator g = Prepare();
        g.Workflow["diff"] = new JObject
        {
            ["class_type"] = "DifferentialDiffusion",
            ["inputs"] = new JObject { ["model"] = new JArray("patched", 0) }
        };
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("diff", 0), 11, false));
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void AnotherSegmentLoRAOnSamplerPathMustFail()
    {
        WorkflowGenerator g = Prepare(segmentConfinement: 12);
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void SameBasenameDifferentDirectoryMustNotCount()
    {
        WorkflowGenerator g = Prepare();
        ((JObject)g.Workflow["patched"]["inputs"])["lora_name"] = "other/portrait.safetensors";
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void SameDirectoryDifferentCaseMustNotCountOnCaseSensitiveBackend()
    {
        WorkflowGenerator g = Prepare();
        ((JObject)g.Workflow["patched"]["inputs"])["lora_name"] = "subjects/Portrait.safetensors";
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void BackslashAndForwardSlashRemainEquivalentForSameLoRA()
    {
        WorkflowGenerator g = Prepare();
        ((JObject)g.Workflow["patched"]["inputs"])["lora_name"] = "subjects\\portrait.safetensors";
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void SameFolderAndBasenameWithDifferentCaseMustNotCountOnLinux()
    {
        WorkflowGenerator g = Prepare();
        ((JObject)g.Workflow["patched"]["inputs"])["lora_name"] =
            "subjects/Portrait.safetensors";
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
                g, new JArray("patched", 0), 11, false)).Message,
            Does.Contain("NOT on the actual sampler model chain"));
        ((JObject)g.Workflow["patched"]["inputs"])["lora_name"] =
            "subjects\\portrait.safetensors";
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void ZeroWeightLoRAIsAnInertAndValidControl()
    {
        WorkflowGenerator g = Prepare();
        ((JObject)g.Workflow["patched"]["inputs"])["strength_model"] = 0.0;
        g.UserInput.Set(T2IParamTypes.LoraWeights, new List<string> { "0" });
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void DuplicateLoRAOnSamplerPathMustFail()
    {
        WorkflowGenerator g = Prepare();
        g.Workflow["patched2"] = new JObject
        {
            ["class_type"] = "LoraLoaderModelOnly",
            ["inputs"] = new JObject
            {
                ["model"] = new JArray("patched", 0),
                ["lora_name"] = "subjects/portrait.safetensors",
                ["strength_model"] = 1.0
            }
        };
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched2", 0), 11, false));
    }

    [Test]
    public static void NoSelectedLoRARejectsAnUnexpectedActiveSamplerLoader()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.Loras, new List<string>());
        g.UserInput.Set(T2IParamTypes.LoraWeights, new List<string>());
        g.UserInput.Set(T2IParamTypes.LoraSectionConfinement, new List<string>());
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void NoSelectedLoRAMustStillRejectMissingSamplerModelNode()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.Loras, new List<string>());
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
                g, new JArray("missing_loader", 0), 11, false)).Message,
            Does.Contain("missing model node"));
    }

    [Test]
    public static void MatchingLoRAMustRejectSelfReferentialModelCycle()
    {
        WorkflowGenerator g = Prepare();
        ((JObject)g.Workflow["patched"]["inputs"])["model"] = new JArray("patched", 0);
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
                g, new JArray("patched", 0), 11, false)).Message,
            Does.Contain("cyclic sampler model chain"));
    }

    [Test]
    public static void MatchingLoRAMustRejectDanglingOrAbsentUpstreamModel()
    {
        WorkflowGenerator g = Prepare();
        ((JObject)g.Workflow["patched"]["inputs"])["model"] = new JArray("lost_source", 0);
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
                g, new JArray("patched", 0), 11, false)).Message,
            Does.Contain("missing model node"));
        ((JObject)g.Workflow["patched"]["inputs"]).Remove("model");
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
                g, new JArray("patched", 0), 11, false)).Message,
            Does.Contain("no upstream MODEL connection"));
    }

    [Test]
    public static void NoSelectedLoRAAllowsAPlainModelPath()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.Loras, new List<string>());
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("raw", 0), 11, false));
    }

    [Test]
    public static void NoSelectedLoRAMustStillRejectDifferentialDiffusion()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.Loras, new List<string>());
        g.Workflow["diff"] = new JObject
        {
            ["class_type"] = "DifferentialDiffusion",
            ["inputs"] = new JObject { ["model"] = new JArray("raw", 0) }
        };
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("diff", 0), 11, false));
    }

    [Test]
    public static void NoSelectedLoRAAllowsZeroStrengthLoaderAsInert()
    {
        WorkflowGenerator g = Prepare();
        g.UserInput.Set(T2IParamTypes.Loras, new List<string>());
        ((JObject)g.Workflow["patched"]["inputs"])["strength_model"] = 0.0;
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }

    [Test]
    public static void GlobalLoRAIsExpectedInASegment()
    {
        WorkflowGenerator g = Prepare(segmentConfinement: 0);
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("raw", 0), 11, false));
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }
}
