using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;

namespace SwarmUITests;

/// <summary>
/// Exercise the real SwarmUI EmptyImage dispatch for Qwen Image 2.1.
/// Generic EmptyLatentImage has four channels and is NOT valid for Qwen.
/// A CPU test separately executes the registered 64-channel Python node.
/// </summary>
public class Qwen21EmptyLatentTests : SwarmUITest
{
    [OneTimeSetUp]
    public static void PreInit() => Setup();

    [TestCase(1024, 768, 2)]
    [TestCase(1024, 1024, 1)]
    public void NativeGenerationRequiresTheQwenEmptyLatent(int width, int height, int batchSize)
    {
        T2IModelClass cls = new()
        {
            ID = T2IModelClassSorter.CompatQwenImage21.ID,
            Name = "Qwen Image 2.1",
            CompatClass = T2IModelClassSorter.CompatQwenImage21,
            StandardWidth = 1024,
            StandardHeight = 1024
        };
        T2IModel model = new(null, "/models/diffusion_models",
            "/models/diffusion_models/qwen-test.safetensors", "qwen-test.safetensors")
        {
            ModelClass = cls,
            StandardWidth = 1024,
            StandardHeight = 1024
        };
        WorkflowGenerator gen = new()
        {
            UserInput = new T2IParamInput(null),
            FinalLoadedModel = model,
            Workflow = new JObject()
        };
        WGNodeData latent = gen.EmptyImage(width, height, batchSize);
        JObject node = (JObject)gen.Workflow[latent.Path[0].ToString()];
        Assert.That(node["class_type"]?.ToString(), Is.EqualTo("SwarmEmptyQwenImage21Latent"),
            "Qwen Image 2.1 must not sample a four-channel SD empty latent");
        Assert.That(node["inputs"]?["width"]?.ToObject<int>(), Is.EqualTo(width),
            "Do not halve the text-to-image canvas width");
        Assert.That(node["inputs"]?["height"]?.ToObject<int>(), Is.EqualTo(height),
            "Do not halve the text-to-image canvas height");
        Assert.That(node["inputs"]?["batch_size"]?.ToObject<int>(), Is.EqualTo(batchSize));
        Assert.That(latent.Width, Is.EqualTo(width));
        Assert.That(latent.Height, Is.EqualTo(height));
    }
    [Test]
    public void MainQwenImage21ReferencesMustKeepIndependentGeometryAndVisionSockets()
    {
        T2IModelClass cls = new()
        {
            ID = T2IModelClassSorter.CompatQwenImage21.ID,
            Name = "Qwen Image 2.1",
            CompatClass = T2IModelClassSorter.CompatQwenImage21,
            StandardWidth = 1024,
            StandardHeight = 1024
        };
        T2IModel model = new(null, "/models/diffusion_models",
            "/models/diffusion_models/qwen-test.safetensors", "qwen-test.safetensors")
        {
            ModelClass = cls, StandardWidth = 1024, StandardHeight = 1024
        };
        using var pixels1 = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24>(80, 96);
        using var pixels2 = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24>(96, 80);
        var images = new System.Collections.Generic.List<SwarmUI.Utils.Image> {
            new(pixels1), new(pixels2)
        };
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.PromptImages, images);
        WorkflowGenerator gen = new()
        {
            UserInput = input,
            FinalLoadedModel = model,
            Workflow = new JObject(),
            Features = new() { "comfy_loadimage_b64" }
        };
        gen.CreateConditioningDirect("Preserve both objects.", new JArray("clip", 0),
            model, true, steps: 4);
        var encoders = gen.Workflow.Properties().Where(p =>
            p.Value?["class_type"]?.ToString() == "SwarmTextEncodeAdvanced").ToArray();
        Assert.That(encoders, Has.Length.EqualTo(1));
        JObject inputs = (JObject)encoders[0].Value["inputs"];
        Assert.That(inputs["image_1"], Is.Not.Null,
            "Qwen must provide independently sized reference 1");
        Assert.That(inputs["image_2"], Is.Not.Null,
            "Qwen must provide independently sized reference 2");
        Assert.That(inputs["images"]?.Type == JTokenType.Array, Is.False,
            "ImageBatch resizes heterogeneous references; Qwen uses separate native image_N sockets");
        Assert.That(gen.Workflow.Properties().Any(p =>
            p.Value?["class_type"]?.ToString() == "ImageBatch"), Is.False);
        Assert.That(inputs["qwen21_prevent_empty_text"]?.ToObject<bool>(), Is.True,
            "The native Qwen tokenizer requires prevent_empty_text for both normal and segment conditioning");
        var nativeImages = gen.LoadPromptImagesForMainRef(images);
        Assert.That(nativeImages, Has.Count.EqualTo(2));
        // Native Qwen rounds each reference to its own 32-pixel grid, not
        // to a common canvas or the unrelated generator's 1024-pixel size.
        for (int i = 0; i < 2; i++)
        {
            int expectedWidth = i == 0 ? 64 : 96;
            int expectedHeight = i == 0 ? 96 : 64;
            var textImage = (JArray)inputs[$"image_{i + 1}"];
            foreach (JArray imagePath in new[] { textImage, nativeImages[i] })
            {
                JObject imageNode = gen.Workflow[imagePath[0].ToString()] as JObject;
                Assert.That(imageNode?["class_type"]?.ToString(), Is.EqualTo("ImageScale"));
                Assert.That(imageNode["inputs"]?["width"]?.ToObject<int>(), Is.EqualTo(expectedWidth));
                Assert.That(imageNode["inputs"]?["height"]?.ToObject<int>(), Is.EqualTo(expectedHeight));
            }
        }
    }

}
