using NUnit.Framework;
using System.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>Exercises the same source selector used by the production graph.</summary>
public class QwenSegmentOriginalSourceTests
{
    [Test]
    public void SegmentOverrideUsesConfiguredModelNotEarlierBaseModel()
    {
        string qwen = T2IModelClassSorter.CompatQwenImage21.ID;
        Assert.That(WorkflowGeneratorSteps.UsesQwen21SegmentModel(
            baseIsQwen21: false, hasSegmentOverride: true, segmentModelCompat: qwen), Is.True);
        Assert.That(WorkflowGeneratorSteps.UsesQwen21SegmentModel(
            baseIsQwen21: true, hasSegmentOverride: true, segmentModelCompat: "stable-diffusion-xl"), Is.False);
        Assert.That(WorkflowGeneratorSteps.UsesQwen21SegmentModel(
            baseIsQwen21: true, hasSegmentOverride: false, segmentModelCompat: null), Is.True);
        Assert.That(WorkflowGeneratorSteps.UsesQwen21SegmentModel(
            baseIsQwen21: false, hasSegmentOverride: false, segmentModelCompat: qwen), Is.False);
    }

    [Test]
    public void ReplicatedSourceBatchUsesDistinctDeterministicSegmentSeeds()
    {
        long seed = 20261020;
        Assert.That(WorkflowGeneratorSteps.Qwen21BatchSegmentSeed(seed, 0), Is.EqualTo(seed));
        Assert.That(WorkflowGeneratorSteps.Qwen21BatchSegmentSeed(seed, 1), Is.EqualTo(seed + 1));
        Assert.That(WorkflowGeneratorSteps.Qwen21BatchSegmentSeed(seed, 2), Is.EqualTo(seed + 2));
        Assert.That(WorkflowGeneratorSteps.Qwen21BatchSegmentSeed(long.MaxValue, 1), Is.EqualTo(0));
    }

    [TestCase("<segment:girl,0.8,0.1> Edit the image", true, false, true, true)]
    [TestCase("sunrise, a girl on the street <segment:girl,0.75,0.15> <weight[1.2]:HannahLynn>", true, false, true, true)]
    [TestCase("sunrise, a girl on the street <segment:girl,0.75,0.15> HannahLynn", true, false, false, false)]
    [TestCase("<segment:girl,0.8,0.1> Edit the image", true, false, false, true)]
    [TestCase("sunrise, a girl <segment:girl,0.8,0.1> HannahLynn", false, false, true, false)]
    [TestCase("sunrise, a girl <segment:girl,0.8,0.1> HannahLynn", true, true, true, false)]
    [TestCase("sunrise, a girl on the street", true, false, true, false)]
    public void OriginalImageSelectionMustNotDependOnWhetherGlobalPromptIsEmpty(
        string text, bool isQwen21, bool isSingleImage, bool hasExplicitSource, bool expectOriginal)
    {
        PromptRegion prompt = new(text);
        Assert.That(WorkflowGeneratorSteps.ShouldQwenSegmentUseOriginalSource(
            prompt, isQwen21, isSingleImage, hasExplicitSource), Is.EqualTo(expectOriginal));
    }

    /// <summary>Executes the real Swarm generator, not a synthetic node-graph fixture.</summary>
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes")]
    [TestCase("<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes")]
    public void ProductionQwenGraphUsesOriginalSourceWithoutRewritingInstruction(string prompt)
    {
        if (T2IParamTypes.Model is null)
        {
            T2IParamTypes.RegisterDefaults();
        }
        T2IModelClass modelClass = new()
        {
            ID = T2IModelClassSorter.CompatQwenImage21.ID,
            Name = "Qwen Image 2.1",
            CompatClass = T2IModelClassSorter.CompatQwenImage21,
            StandardWidth = 1024,
            StandardHeight = 1024
        };
        T2IModel model = new(null, "/models/diffusion_models", "/models/diffusion_models/qwen-test.safetensors", "qwen-test.safetensors")
        {
            ModelClass = modelClass, StandardWidth = 1024, StandardHeight = 1024
        };
        T2IModel vae = new(null, "/models/vae", "/models/vae/vae.safetensors", "vae.safetensors");
        T2IModel clip = new(null, "/models/text_encoders", "/models/text_encoders/clip.safetensors", "clip.safetensors");
        using var pixels = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24>(128, 128);
        var image = new SwarmUI.Utils.Image(pixels);
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.Model, model);
        input.Set(T2IParamTypes.VAE, vae);
        input.Set(T2IParamTypes.QwenModel, clip);
        input.Set(T2IParamTypes.InitImage, image);
        input.Set(T2IParamTypes.Prompt, prompt);
        input.Set(T2IParamTypes.Seed, 20261020L);
        input.Set(T2IParamTypes.Steps, 12);
        input.Set(T2IParamTypes.Width, 1024);
        input.Set(T2IParamTypes.Height, 1024);
        input.PreparsePromptLikes();
        WorkflowGenerator generator = new() { UserInput = input };
        Newtonsoft.Json.Linq.JObject graph = generator.Generate();
        var nodes = graph.Properties().Select(p => (Id: p.Name, Node: (Newtonsoft.Json.Linq.JObject)p.Value)).ToArray();
        var crop = nodes.Single(x => x.Node["class_type"]?.ToString() == "InpaintCropImproved");
        var detector = nodes.Single(x => x.Node["class_type"]?.ToString() == "SwarmClipSeg");
        var sampler = nodes.Single(x => x.Node["class_type"]?.ToString() == "LanPaint_KSampler");
        var encoder = nodes.Single(x => x.Node["class_type"]?.ToString() == "TextEncodeQwenImage21");
        var stitch = nodes.Single(x => x.Node["class_type"]?.ToString() == "InpaintStitchImproved");
        var mask = nodes.Single(x => x.Node["class_type"]?.ToString() == "ThresholdMask");
        string sourceId = crop.Node["inputs"]?["image"]?[0]?.ToString();
        Assert.That(sourceId, Is.Not.Null);
        Assert.That(graph[sourceId]?["class_type"]?.ToString(), Is.EqualTo("LoadImage"),
            "Explicit original photo must be the segment canvas with or without global scene text");
        Assert.That(detector.Node["inputs"]?["images"]?.ToString(), Is.EqualTo(crop.Node["inputs"]?["image"]?.ToString()));
        Assert.That(crop.Node["inputs"]?["mask"]?.ToString(),
            Is.EqualTo(new Newtonsoft.Json.Linq.JArray(mask.Id, 0).ToString()));
        Assert.That(encoder.Node["inputs"]?["prompt"]?.ToString(),
            Is.EqualTo(" Edit the image, replace the girl with a black boy with blue eyes"));
        Assert.That(sampler.Node["inputs"]?["denoise"]?.ToObject<double>(), Is.EqualTo(0.8).Within(0.000001));
        Assert.That(sampler.Node["inputs"]?["LanPaint_Info"]?.ToString(), Is.EqualTo("LanPaint KSampler."));
        Assert.That(sampler.Node["inputs"]?["Inpainting_mode"]?.ToString(), Is.EqualTo("🖼️ Image Inpainting"));
        Assert.That(stitch.Node["inputs"]?["stitcher"]?.ToString(),
            Is.EqualTo(new Newtonsoft.Json.Linq.JArray(crop.Id, 0).ToString()));
        Assert.That(nodes.Count(x => x.Node["class_type"]?.ToString() is "SaveImage" or "SwarmSaveImageWS"), Is.EqualTo(1));
        Assert.That(nodes.Count(x => x.Node["class_type"]?.ToString() == "PreviewImage"), Is.Zero);
    }

}
