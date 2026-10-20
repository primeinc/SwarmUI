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
    public static void PreInit() => Setup();

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
    public static void GlobalLoRAIsExpectedInASegment()
    {
        WorkflowGenerator g = Prepare(segmentConfinement: 0);
        Assert.Throws<SwarmUserErrorException>(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("raw", 0), 11, false));
        Assert.DoesNotThrow(() => WorkflowGeneratorSteps.VerifyQwenSegmentLoraChain(
            g, new JArray("patched", 0), 11, false));
    }
}
