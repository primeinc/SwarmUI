using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;

namespace SwarmUITests;

/// <summary>Checks Comfy workflow graph generation without requiring a GPU backend.</summary>
public class ComfyWorkflowGeneratorTests : SwarmUITest
{
    [OneTimeSetUp]
    public static void PreInit()
    {
        Setup();
        if (T2IParamTypes.Prompt is null)
        {
            T2IParamTypes.RegisterDefaults();
        }
    }

    private static WorkflowGenerator NewGenerator()
    {
        return new WorkflowGenerator()
        {
            UserInput = new T2IParamInput(null),
            Workflow = new JObject()
        };
    }

    [Test]
    public static void TestQwenReferenceLatentUsesScopedSegmentVae()
    {
        WorkflowGenerator generator = NewGenerator();
        T2IModelCompatClass compat = T2IModelClassSorter.CompatQwenImage21;
        generator.FinalLoadedModel = new T2IModel(null, "/models/diffusion_models", null, "qwen-21")
        {
            ModelClass = new T2IModelClass() { ID = "qwen-image-2.1", CompatClass = compat }
        };
        generator.CurrentVae = new WGNodeData(["base-vae", 0], generator, WGNodeData.DT_VAE, compat);
        WGNodeData segmentVae = new(["segment-vae", 0], generator, WGNodeData.DT_VAE, compat);

        generator.WithCurrentVae(segmentVae, () => generator.BuildInputImageHandling([["cropped-segment", 0]], ["positive", 0], ["negative", 0], ["latent", 0]));

        JProperty encoded = generator.Workflow.Properties().Single(p => p.Value["class_type"]?.ToString() == "VAEEncode");
        Assert.That(encoded.Value["inputs"]["vae"][0].ToString(), Is.EqualTo("segment-vae"));
        Assert.That(generator.CurrentVae.Path[0].ToString(), Is.EqualTo("base-vae"));
        Assert.That(generator.Workflow.Properties().Count(p => p.Value["class_type"]?.ToString() == "ReferenceLatent"), Is.EqualTo(2));
    }

    [Test]
    public static void TestKreaDoesNotAutoAddReferenceLatents()
    {
        WorkflowGenerator generator = NewGenerator();
        T2IModelCompatClass compat = T2IModelClassSorter.CompatKrea2;
        generator.FinalLoadedModel = new T2IModel(null, "/models/diffusion_models", null, "krea-2")
        {
            ModelClass = new T2IModelClass() { ID = "krea-2", CompatClass = compat }
        };
        generator.CurrentVae = new WGNodeData(["base-vae", 0], generator, WGNodeData.DT_VAE, compat);

        generator.BuildInputImageHandling([["cropped-segment", 0]], ["positive", 0], ["negative", 0], ["latent", 0]);

        Assert.That(generator.Workflow.Properties().Any(p => p.Value["class_type"]?.ToString() == "ReferenceLatent"), Is.False);
    }

    [Test]
    public static void TestScopedVaeRestoredOnException()
    {
        WorkflowGenerator generator = NewGenerator();
        generator.CurrentVae = new WGNodeData(["base-vae", 0], generator, WGNodeData.DT_VAE, null);
        WGNodeData segmentVae = new(["segment-vae", 0], generator, WGNodeData.DT_VAE, null);
        Assert.Throws<System.InvalidOperationException>(() => generator.WithCurrentVae(segmentVae, () => throw new System.InvalidOperationException("test")));
        Assert.That(generator.CurrentVae.Path[0].ToString(), Is.EqualTo("base-vae"));
    }

    [Test]
    public static void TestSegmentCompositeCanIgnoreGeneratedAlpha()
    {
        WorkflowGenerator generator = NewGenerator();
        JArray output = generator.RecompositeCropped("bounds", ["mask", 0], ["original", 0], ["sampled", 0], stripSourceAlpha: true);

        JProperty split = generator.Workflow.Properties().Single(p => p.Value["class_type"]?.ToString() == "SplitImageWithAlpha");
        Assert.That(split.Value["inputs"]["image"][0].ToString(), Is.EqualTo("sampled"));

        JProperty resized = generator.Workflow.Properties().Single(p => p.Value["class_type"]?.ToString() == "ImageScale");
        Assert.That(resized.Value["inputs"]["image"][0].ToString(), Is.EqualTo(split.Name));
        Assert.That(resized.Value["inputs"]["image"][1].ToString(), Is.EqualTo("0"));

        JToken composite = generator.Workflow[output[0].ToString()];
        Assert.That(composite["inputs"]["source"][0].ToString(), Is.EqualTo(resized.Name));
        Assert.That(composite["inputs"]["destination"][0].ToString(), Is.EqualTo("original"));
        Assert.That(composite["inputs"]["mask"], Is.Not.Null);
    }

    [Test]
    public static void TestSegmentCompositePreservesExistingBehaviorByDefault()
    {
        WorkflowGenerator generator = NewGenerator();
        generator.RecompositeCropped("bounds", ["mask", 0], ["original", 0], ["sampled", 0]);

        Assert.That(generator.Workflow.Properties().Any(p => p.Value["class_type"]?.ToString() == "SplitImageWithAlpha"), Is.False);
        JProperty resized = generator.Workflow.Properties().Single(p => p.Value["class_type"]?.ToString() == "ImageScale");
        Assert.That(resized.Value["inputs"]["image"][0].ToString(), Is.EqualTo("sampled"));
    }
}
