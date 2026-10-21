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
        // Upstream TextEncodeQwenImage21 uses prevent_empty_text=True on both
        // conditionings; this option must not rewrite either authored string.
        Assert.That(encoders[0]["inputs"]?["qwen21_prevent_empty_text"]?.ToObject<bool>(), Is.True);
        Assert.That(encoders[1]["inputs"]?["qwen21_prevent_empty_text"]?.ToObject<bool>(), Is.True);
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
    public static void EmptyNegativeTextKeepsNativeQwenTokenizerFlagWithoutInventedInstructions()
    {
        var (positive, negative, _, _) = ConditionersFor("Replace the subject.", "");
        Assert.That(positive["inputs"]?["prompt"]?.ToString(), Is.EqualTo("Replace the subject."));
        Assert.That(negative["inputs"]?["prompt"]?.ToString(), Is.Empty);
        Assert.That(negative["inputs"]?["qwen21_prevent_empty_text"]?.ToObject<bool>(), Is.True);
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

    [TestCase(true, false, 1)]
    [TestCase(false, false, 1)]
    [TestCase(true, true, 0)]
    [TestCase(false, true, 0)]
    public static void PromptImageSourceIsNotReaddedAsIndependentReference(
        bool useOriginalReference, bool hasInitImage, int expectedFirstIndependentIndex)
    {
        // Without Init Image, PromptImages[0] IS the original edit source.
        // Disabling the original-reference option must not sneak that source
        // back into Qwen conditioning as an independent Prompt Image.
        Assert.That(WorkflowGenerator.FirstQwen21AdditionalReferenceIndex(
            useOriginalReference, hasInitImage), Is.EqualTo(expectedFirstIndependentIndex));
    }

    [Test]
    public static void PromptImageOnlySourceWithDisabledReferenceDoesNotLeakIntoVision()
    {
        // A user-provided PromptImages[0] becomes the source when there is no
        // Init Image. Disabling the original reference must also exclude it
        // from the list of independent visual references.
        using SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24> pixels = new(96, 96);
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.PromptImages,
            new System.Collections.Generic.List<SwarmUI.Utils.Image> { new(pixels) });
        input.Set(T2IParamTypes.SegmentQwenUseOriginalReference, false, sectionId: 11);
        WorkflowGenerator generator = new()
        {
            UserInput = input,
            Workflow = new JObject(),
            Features = new() { "comfy_loadimage_b64" }
        };
        WGNodeData clip = new(new JArray("clip", 0), generator, WGNodeData.DT_TEXTENC, null);
        WGNodeData vae = new(new JArray("vae", 0), generator, WGNodeData.DT_VAE, null);
        generator.CreateQwen21SegmentConditioning("Keep the scene.", "", clip, vae,
            new JArray("crop", 1), new JArray("crop", 2), 11, steps: 4);
        JObject[] encoders = [.. generator.Workflow.Properties()
            .Where(p => p.Value?["class_type"]?.ToString() == "SwarmTextEncodeAdvanced")
            .Select(p => (JObject)p.Value)];
        Assert.That(encoders, Has.Length.EqualTo(2));
        foreach (JObject encoder in encoders)
        {
            JObject inputs = (JObject)encoder["inputs"];
            Assert.That(inputs["images"]?.Type == JTokenType.Array, Is.False,
                "Do not smuggle the explicitly disabled edit source into Qwen vision");
            Assert.That(inputs["image_1"], Is.Null,
                "The original Prompt Image cannot return as an independent reference");
        }
        Assert.That(generator.Workflow.Properties().Count(p =>
            p.Value?["class_type"]?.ToString() == "ReferenceLatent"), Is.Zero);
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
    [Test]
    public static void DynamicLoRAHookReachesBothNativeSwarmQwenConditioners()
    {
        // The parser CPU test proves the //hook marker survives. This real
        // generator test proves that its CreateHookLora node is wired into
        // BOTH the positive and negative SwarmTextEncodeAdvanced nodes.
        // It does not claim that Qwen responds to the LoRA in image space.
        bool hadHandler = SwarmUI.Core.Program.T2IModelSets.TryGetValue(
            "LoRA", out T2IModelHandler previousHandler);
        try
        {
            T2IModelHandler handler = new() { ModelType = "LoRA" };
            handler.Models["Audit/HannahLynn.safetensors"] = new T2IModel(
                handler, "/models/loras", "/models/loras/Audit/HannahLynn.safetensors",
                "Audit/HannahLynn.safetensors");
            SwarmUI.Core.Program.T2IModelSets["LoRA"] = handler;

            T2IParamInput input = new(null);
            input.Set(T2IParamTypes.Loras,
                new System.Collections.Generic.List<string> { "Audit/HannahLynn" });
            input.Set(T2IParamTypes.LoraWeights,
                new System.Collections.Generic.List<string> { "0.8" });
            input.DynamicLoraIndices.Add(0);

            WorkflowGenerator generator = new()
            {
                UserInput = input,
                Workflow = new JObject()
            };
            WGNodeData clip = new(new JArray("clip", 0), generator, WGNodeData.DT_TEXTENC, null);
            WGNodeData vae = new(new JArray("vae", 0), generator, WGNodeData.DT_VAE, null);
            const string authored = "HannahLynn <lora:Audit/HannahLynn//hook=0>";
            generator.CreateQwen21SegmentConditioning(authored, "", clip, vae,
                new JArray("crop", 1), new JArray("crop", 2), 11, steps: 4);

            JProperty hook = generator.Workflow.Properties().Single(p =>
                p.Value?["class_type"]?.ToString() == "CreateHookLora");
            Assert.That(hook.Value?["inputs"]?["lora_name"]?.ToString(),
                Does.Contain("Audit/HannahLynn"));
            Assert.That(hook.Value?["inputs"]?["strength_model"]?.ToObject<double>(),
                Is.EqualTo(0.8).Within(0.00001));
            JArray hookPath = new(hook.Name, 0);
            JProperty[] encoders = [.. generator.Workflow.Properties().Where(p =>
                p.Value?["class_type"]?.ToString() == "SwarmTextEncodeAdvanced")];
            Assert.That(encoders, Has.Length.EqualTo(2));
            foreach (JProperty encoder in encoders)
            {
                Assert.That(encoder.Value?["inputs"]?["lora_hooks"]?.ToString(),
                    Is.EqualTo(hookPath.ToString()),
                    "Both Qwen conditioning branches need the same dynamic LoRA hook group");
            }
            Assert.That(encoders.Any(p => p.Value?["inputs"]?["prompt"]?.ToString() == authored),
                Is.True, "The authored LoRA hook marker must not be removed or rewritten");
        }
        finally
        {
            if (hadHandler)
            {
                SwarmUI.Core.Program.T2IModelSets["LoRA"] = previousHandler;
            }
            else
            {
                SwarmUI.Core.Program.T2IModelSets.Remove("LoRA");
            }
        }
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
