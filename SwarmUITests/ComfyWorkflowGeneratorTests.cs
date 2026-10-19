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
