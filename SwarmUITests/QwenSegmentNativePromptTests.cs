using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>
/// The segment parser owns Swarm directives; the native Qwen encoder must
/// receive exactly the remaining author-supplied text, not a generated
/// replacement instruction. Graph tests do not prove rendered edit success.
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

    private static JObject NativeEncoderFor(
        string prompt, string negativePrompt = "", bool useOriginalReference = true)
    {
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.SegmentQwenUseOriginalReference, useOriginalReference, sectionId: 11);
        WorkflowGenerator generator = new()
        {
            UserInput = input,
            Workflow = new JObject()
        };
        WGNodeData clip = new(new JArray("clip", 0), generator, WGNodeData.DT_TEXTENC, null);
        WGNodeData vae = new(new JArray("vae", 0), generator, WGNodeData.DT_VAE, null);
        generator.CreateQwen21SegmentConditioning(
            prompt, negativePrompt, clip, vae,
            new JArray("crop", 1), new JArray("crop", 2), 11);
        return generator.Workflow.Properties()
            .Select(p => p.Value as JObject)
            .Single(node => node["class_type"]?.ToString() == "TextEncodeQwenImage21");
    }

    [Test]
    public static void NativeEncoderMustOnlyFlattenSwarmWeightTags()
    {
        const string savedPrompt = "sunrise, a girl is walking down the street "
            + "<segment:girl,0.75,0.15//cid=11> <weight[1.2]:HannahLynn>";
        PromptRegion.Part segment = new PromptRegion(savedPrompt).Parts.Single();
        Assert.That(segment.ContextID, Is.EqualTo(11));
        Assert.That(segment.Prompt, Is.EqualTo(" <weight[1.2]:HannahLynn>"));
        JObject encoder = NativeEncoderFor(segment.Prompt, "<weight[0.5]:blurry>");
        Assert.That(encoder["inputs"]?["prompt"]?.ToString(), Is.EqualTo(" HannahLynn"));
        Assert.That(encoder["inputs"]?["negative_prompt"]?.ToString(), Is.EqualTo("blurry"));
        Assert.That(encoder["inputs"]?["images.image_1"]?.ToString(),
            Is.EqualTo(new JArray("crop", 1).ToString()));
    }

    [TestCase("HannahLynn")]
    [TestCase("Replace")]
    [TestCase("Add")]
    [TestCase("Remove")]
    [TestCase("Replace the girl with a black boy with blue eyes")]
    [TestCase("Add sunglasses to the girl")]
    [TestCase("Put a hat on the subject")]
    [TestCase("Give the girl blue eyes")]
    [TestCase("Draw a star on the girl's shirt")]
    [TestCase("Make the subject smile")]
    [TestCase("In this picture, replace the girl with a black boy with blue eyes")]
    [TestCase("  Preserve this punctuation, and spacing!  ")]
    [TestCase("Replace the girl in <image1> with the boy in <image2>.")]
    public static void NativeQwenMustPreserveLiteralAuthorText(string prompt)
    {
        Assert.That(WorkflowGenerator.FlattenQwen21SegmentPrompt(prompt), Is.EqualTo(prompt));
        JObject encoder = NativeEncoderFor(prompt);
        Assert.That(encoder["inputs"]?["prompt"]?.ToString(), Is.EqualTo(prompt));
    }

    [Test]
    public static void NestedSwarmWeightTagsAreFlattenedWithoutWritingAnInstruction()
    {
        const string prompt = "<weight[1.2]:portrait of <weight[1.5]:HannahLynn>>";
        const string expected = "portrait of HannahLynn";
        Assert.That(WorkflowGenerator.FlattenQwen21SegmentPrompt(prompt), Is.EqualTo(expected));
        Assert.That(NativeEncoderFor(prompt)["inputs"]?["prompt"]?.ToString(), Is.EqualTo(expected));
    }

    [Test]
    public static void WeightedMultilineInstructionIsPreservedAfterMarkupProcessing()
    {
        const string prompt = "<weight[1.2]:Add sunglasses to the girl>";
        Assert.That(NativeEncoderFor(prompt)["inputs"]?["prompt"]?.ToString(),
            Is.EqualTo("Add sunglasses to the girl"));
    }

    [TestCase(true)]
    [TestCase(false)]
    public static void CropReferenceSwitchMustNeverInventPromptText(bool cropEnabled)
    {
        JObject encoder = NativeEncoderFor("<weight[1.2]:HannahLynn>", useOriginalReference: cropEnabled);
        Assert.That(encoder["inputs"]?["prompt"]?.ToString(), Is.EqualTo("HannahLynn"));
        Assert.That(encoder["inputs"]?["prompt"]?.ToString(), Does.Not.Contain("<image1>"));
        Assert.That(encoder["inputs"]?["images.image_1"] is not null, Is.EqualTo(cropEnabled));
    }

    [Test]
    public static void AuthorSuppliedImageTokenMustSurviveWithoutRewriting()
    {
        JObject encoder = NativeEncoderFor("Replace the subject in <image1>.", useOriginalReference: false);
        Assert.That(encoder["inputs"]?["prompt"]?.ToString(),
            Is.EqualTo("Replace the subject in <image1>."));
    }

    [Test]
    public static void MissingSegmentTextMustFailWithoutInvention()
    {
        Assert.Throws<SwarmUserErrorException>(
            () => WorkflowGenerator.FlattenQwen21SegmentPrompt("  "));
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
}
