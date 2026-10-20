using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>
/// Checks the actual native Qwen conditioner with the exact segment syntax
/// saved in a user's image. Swarm weights must not leak into native CLIP text.
/// </summary>
public class QwenSegmentNativePromptTests : SwarmUITest
{
    [OneTimeSetUp]
    public static void PreInit()
    {
        Setup();
        if (T2IParamTypes.Loras is null)
        {
            T2IParamTypes.RegisterDefaults();
        }
    }

    [Test]
    public static void NativeEncoderMustReceiveRealSegmentLoRATriggerInsteadOfWeightMarkup()
    {
        const string savedPrompt = "sunrise, a 12 year old girl is walking down the street looking at the viewer "
            + "<segment:girl,0.75,0.15//cid=11> <weight[1.2]:HannahLynn>";
        PromptRegion.Part part = new PromptRegion(savedPrompt).Parts.Single();
        Assert.That(part.ContextID, Is.EqualTo(11));
        Assert.That(part.Prompt.Trim(), Is.EqualTo("<weight[1.2]:HannahLynn>"));

        WorkflowGenerator g = new()
        {
            UserInput = new T2IParamInput(null),
            Workflow = new JObject()
        };
        WGNodeData clip = new(new JArray("clip", 0), g, WGNodeData.DT_TEXTENC, null);
        WGNodeData vae = new(new JArray("vae", 0), g, WGNodeData.DT_VAE, null);
        g.CreateQwen21SegmentConditioning(
            part.Prompt, "<weight[0.5]:blurry>", clip, vae,
            new JArray("crop", 1), new JArray("crop", 2), part.ContextID, part.DataText);

        JObject encoder = g.Workflow.Properties()
            .Select(p => p.Value as JObject)
            .Single(node => node["class_type"]?.ToString() == "TextEncodeQwenImage21");
        Assert.That(encoder["inputs"]?["prompt"]?.ToString(), Is.EqualTo(
            "Replace the girl in <image1> with HannahLynn. Keep the rest of the image unchanged."));
        Assert.That(encoder["inputs"]?["negative_prompt"]?.ToString(), Is.EqualTo("blurry"));
        Assert.That(encoder["inputs"]?["images.image_1"]?.ToString(),
            Is.EqualTo(new JArray("crop", 1).ToString()));
    }

    [TestCase(true, true, 0)]
    [TestCase(true, false, 1)]
    [TestCase(false, true, 0)]
    [TestCase(false, false, 0)]
    public static void QwenMustKeepIndependentPromptImageReferences(
        bool useOriginal, bool hasInitImage, int expectedFirstIndex)
    {
        Assert.That(WorkflowGenerator.FirstQwen21AdditionalReferenceIndex(
            useOriginal, hasInitImage), Is.EqualTo(expectedFirstIndex));
    }

    [Test]
    public static void NativeQwenPreservesExplicitReplacementInstructions()
    {
        Assert.That(WorkflowGenerator.BuildQwen21SegmentEditInstruction(
            "Edit the image, replace the girl with a black boy with blue eyes", "girl"),
            Is.EqualTo("Edit the image, replace the girl with a black boy with blue eyes"));
    }

    [Test]
    public static void NativeQwenMustNotSendStandaloneLoRATriggerAsEditInstruction()
    {
        Assert.That(WorkflowGenerator.BuildQwen21SegmentEditInstruction(
            "<weight[1.2]:HannahLynn>", "girl"),
            Is.EqualTo("Replace the girl in <image1> with HannahLynn. Keep the rest of the image unchanged."));
    }

    [Test]
    public static void NativeQwenTextFlattenerPreservesNestedSubjectText()
    {
        Assert.That(T2IPromptHandling.ProcessPromptLikeForLength(
            "<weight[1.2]:portrait of <weight[1.5]:HannahLynn>>"),
            Is.EqualTo("portrait of HannahLynn"));
        Assert.That(T2IPromptHandling.ProcessPromptLikeForLength(
            "Edit the image, replace the girl with a black boy with blue eyes"),
            Is.EqualTo("Edit the image, replace the girl with a black boy with blue eyes"));
    }
}
