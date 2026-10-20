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
}
