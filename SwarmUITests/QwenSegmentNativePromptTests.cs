using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>
/// The production Qwen segment path uses SwarmTextEncodeAdvanced for Swarm
/// syntax and upstream ReferenceLatent for Qwen edit references. These are
/// graph-contract tests, not Qwen inference or image-space LoRA acceptance.
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

    private static (JObject Positive, JObject Negative, JObject Graph, JArray Latent) ConditionersFor(
        string prompt, string negativePrompt = "", bool useOriginalReference = true,
        bool referenceEdit = false)
    {
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.SegmentQwenUseOriginalReference,
            useOriginalReference, sectionId: 11);
        if (referenceEdit)
        {
            input.Set(T2IParamTypes.SegmentQwen21Sampling, "Reference Edit", sectionId: 11);
        }
        WorkflowGenerator generator = new()
        {
            UserInput = input,
            Workflow = new JObject()
        };
        WGNodeData clip = new(new JArray("clip", 0), generator, WGNodeData.DT_TEXTENC, null);
        WGNodeData vae = new(new JArray("vae", 0), generator, WGNodeData.DT_VAE, null);
        var result = generator.CreateQwen21SegmentConditioning(
            prompt, negativePrompt, clip, vae,
            new JArray("crop", 1), new JArray("crop", 2), 11, steps: 4);
        JObject[] encoders = generator.Workflow.Properties()
            .Select(p => p.Value as JObject)
            .Where(node => node?["class_type"]?.ToString() == "SwarmTextEncodeAdvanced")
            .ToArray();
        Assert.That(encoders, Has.Length.EqualTo(2),
            "One authored positive and one authored negative text encoder are required");
        Assert.That(generator.Workflow.Properties()
            .Any(p => p.Value?["class_type"]?.ToString() == "TextEncodeQwenImage21"),
            Is.False, "Do not accidentally mix the two conditioning contracts");
        Assert.That(encoders[0]["inputs"]?["prompt"]?.ToString(), Is.EqualTo(prompt));
        Assert.That(encoders[1]["inputs"]?["prompt"]?.ToString(), Is.EqualTo(negativePrompt));
        return (encoders[0], encoders[1], generator.Workflow, result.Latent);
    }

    [Test]
    public static void RegionParserRemovesOnlyTheSegmentDirective()
    {
        const string saved = "sunrise, a girl on the street <segment:girl,0.75,0.15//cid=11> <weight[1.2]:HannahLynn>";
        PromptRegion region = new(saved);
        PromptRegion.Part part = region.Parts.Single();
        Assert.That(part.Prompt, Is.EqualTo(" <weight[1.2]:HannahLynn>"));
        var (positive, negative, _, _) = ConditionersFor(part.Prompt, "<weight[0.5]:blurry>");
        Assert.That(positive["inputs"]?["prompt"]?.ToString(), Is.EqualTo(part.Prompt));
        Assert.That(negative["inputs"]?["prompt"]?.ToString(), Is.EqualTo("<weight[0.5]:blurry>"));
    }

    [TestCase("HannahLynn")]
    [TestCase("Replace")]
    [TestCase("Add")]
    [TestCase("Remove.")]
    [TestCase("删除女孩")]
    [TestCase("Replace the girl with a black boy with blue eyes")]
    [TestCase("Add sunglasses to the girl")]
    [TestCase("Put a hat on the subject")]
    [TestCase("Give the girl blue eyes")]
    [TestCase("Draw a star on the girl's shirt")]
    [TestCase("  Preserve this punctuation, and spacing!  ")]
    [TestCase("Replace the girl in <image1> with the boy in <image2>.")]
    [TestCase("This is <alternative> literal text, not a Swarm tag.")]
    public static void AuthoredInstructionsReachNativeSwarmEncoderVerbatim(string prompt)
    {
        var (positive, _, _, _) = ConditionersFor(prompt);
        Assert.That(positive["inputs"]?["prompt"]?.ToString(), Is.EqualTo(prompt));
    }

    [TestCase("<weight[1.2]:HannahLynn>")]
    [TestCase("<weight[1.2]:portrait of <weight[1.5]:HannahLynn>>")]
    [TestCase("<alternate:hat,glasses>")]
    [TestCase("<fromto[0.5]:before,after>")]
    [TestCase("<break>")]
    [TestCase("<embed:portrait>")]
    public static void SwarmTagsReachTheirActualProcessorRatherThanLengthEstimator(string prompt)
    {
        var (positive, negative, _, _) = ConditionersFor(prompt, prompt + " negative");
        Assert.That(positive["inputs"]?["prompt"]?.ToString(), Is.EqualTo(prompt));
        Assert.That(negative["inputs"]?["prompt"]?.ToString(), Is.EqualTo(prompt + " negative"));
        Assert.That(positive["inputs"]?["steps"]?.ToObject<int>(), Is.EqualTo(4));
    }

    [Test]
    public static void OriginalReferenceCreatesVisionAndReferenceLatentBranches()
    {
        var (positive, negative, graph, latent) = ConditionersFor("Replace the subject.", "blurry");
        Assert.That(positive["inputs"]?["images"]?.ToString(),
            Is.EqualTo(new JArray("crop", 1).ToString()));
        Assert.That(negative["inputs"]?["images"]?.ToString(),
            Is.EqualTo(new JArray("crop", 1).ToString()));
        JObject[] referenceNodes = graph.Properties()
            .Select(p => p.Value as JObject)
            .Where(n => n?["class_type"]?.ToString() == "ReferenceLatent").ToArray();
        Assert.That(referenceNodes, Has.Length.EqualTo(2),
            "Both Qwen positive and negative conditioning must carry the source reference latent");
        Assert.That(graph[latent[0]?.ToString()]?["class_type"]?.ToString(),
            Is.EqualTo("LanPaint_ImageEncode"), "Masked Img2Img uses the native masked encoder");
    }

    private static void RequirePairedQwenReferenceLinks(JObject graph)
    {
        JProperty[] encoders = [.. graph.Properties().Where(p =>
            p.Value?["class_type"]?.ToString() == "SwarmTextEncodeAdvanced")];
        Assert.That(encoders, Has.Length.EqualTo(2));
        JProperty cropVae = graph.Properties().Single(p =>
            p.Value?["class_type"]?.ToString() == "VAEEncode"
            && p.Value?["inputs"]?["pixels"]?.ToString()
                == new JArray("crop", 1).ToString());
        foreach (JProperty encoder in encoders)
        {
            JArray expectedText = new(encoder.Name, 0);
            JArray expectedLatent = new(cropVae.Name, 0);
            JProperty[] attached = [.. graph.Properties().Where(p =>
                p.Value?["class_type"]?.ToString() == "ReferenceLatent"
                && p.Value?["inputs"]?["conditioning"]?.ToString() == expectedText.ToString()
                && p.Value?["inputs"]?["latent"]?.ToString() == expectedLatent.ToString())];
            Assert.That(attached, Has.Length.EqualTo(1),
                "Every authored text encoder needs its own native ReferenceLatent pointing at the exact crop VAE");
        }
    }

    [Test]
    public static void ReferenceConnectionsAreCheckedAndBrokenLinksFail()
    {
        var (_, _, graph, _) = ConditionersFor("Replace the subject.", "blurry");
        Assert.DoesNotThrow(() => RequirePairedQwenReferenceLinks(graph));
        JProperty negativeEncoder = graph.Properties().Single(p =>
            p.Value?["class_type"]?.ToString() == "SwarmTextEncodeAdvanced"
            && p.Value?["inputs"]?["prompt"]?.ToString() == "blurry");
        JProperty negativeReference = graph.Properties().Single(p =>
            p.Value?["class_type"]?.ToString() == "ReferenceLatent"
            && p.Value?["inputs"]?["conditioning"]?.ToString()
                == new JArray(negativeEncoder.Name, 0).ToString());
        negativeReference.Value["inputs"]["latent"] = new JArray("disconnected_crop", 0);
        Assert.Throws<AssertionException>(() => RequirePairedQwenReferenceLinks(graph));
    }

    [Test]
    public static void DisablingOriginalReferenceDoesNotInventAnImageOrTextToken()
    {
        var (positive, negative, graph, _) = ConditionersFor("HannahLynn", useOriginalReference: false);
        Assert.That(positive["inputs"]?["prompt"]?.ToString(), Is.EqualTo("HannahLynn"));
        Assert.That(positive["inputs"]?["images"]?.Type == JTokenType.Array, Is.False);
        Assert.That(negative["inputs"]?["images"]?.Type == JTokenType.Array, Is.False);
        Assert.That(graph.Properties().Count(p =>
            p.Value?["class_type"]?.ToString() == "ReferenceLatent"), Is.Zero);
    }

    [Test]
    public static void ReferenceEditStartsFromNativeZeroedCropLatent()
    {
        var (_, _, graph, latent) = ConditionersFor(
            "Replace the girl.", referenceEdit: true);
        JObject empty = graph[latent[0]?.ToString()] as JObject;
        Assert.That(empty?["class_type"]?.ToString(), Is.EqualTo("LatentMultiply"));
        Assert.That(empty?["inputs"]?["multiplier"]?.ToObject<double>(), Is.Zero);
        string baseLatent = empty?["inputs"]?["samples"]?[0]?.ToString();
        Assert.That(graph[baseLatent]?["class_type"]?.ToString(), Is.EqualTo("VAEEncode"));
        Assert.That(graph.Properties().Any(p =>
            p.Value?["class_type"]?.ToString() == "LanPaint_ImageEncode"), Is.False);
    }
    [TestCase(80, 96, 64, 96)]
    [TestCase(128, 128, 128, 128)]
    [TestCase(15, 20, 32, 32)]
    [TestCase(48, 49, 64, 64)]
    public static void NativeQwenReferencesUseIndependentRoundToEvenGrid(
        int sourceWidth, int sourceHeight, int expectedWidth, int expectedHeight)
    {
        (int width, int height) =
            WorkflowGenerator.NativeQwen21ReferenceDimensions(sourceWidth, sourceHeight);
        Assert.That(width, Is.EqualTo(expectedWidth));
        Assert.That(height, Is.EqualTo(expectedHeight));
    }

}
