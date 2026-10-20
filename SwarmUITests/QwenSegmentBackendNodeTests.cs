using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;

namespace SwarmUITests;

/// <summary>
/// Real exported-graph node availability is checked against the selected
/// backend's /object_info inventory, not a global aggregate feature list.
/// These CPU-only controls do NOT claim successful Qwen inference.
/// </summary>
public class QwenSegmentBackendNodeTests : SwarmUITest
{
    [OneTimeSetUp]
    public static void PreInit() => Setup();

    private static JObject Node(string classType) => new()
    {
        ["class_type"] = classType,
        ["inputs"] = new JObject()
    };

    private static JObject NativeGraph() => new()
    {
        ["crop"] = Node("InpaintCropImproved"),
        ["encoder"] = Node("TextEncodeQwenImage21"),
        ["mask"] = Node("SwarmRequireNonEmptyMask"),
        ["threshold"] = Node("ThresholdMask"),
        ["sampler"] = Node("LanPaint_KSampler"),
        ["stitch"] = Node("InpaintStitchImproved"),
        ["save"] = Node("SaveImage")
    };

    [Test]
    public static void InstalledNativeNodesPass()
    {
        JObject graph = NativeGraph();
        HashSet<string> types = [
            "InpaintCropImproved", "TextEncodeQwenImage21",
            "SwarmRequireNonEmptyMask", "ThresholdMask",
            "LanPaint_KSampler", "InpaintStitchImproved", "SaveImage"
        ];
        Assert.DoesNotThrow(() => ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(graph, types));
    }

    [Test]
    public static void SelectedBackendWithoutLanPaintFailsBeforeSubmittingWorkflow()
    {
        JObject graph = NativeGraph();
        HashSet<string> types = [
            "InpaintCropImproved", "TextEncodeQwenImage21",
            "SwarmRequireNonEmptyMask", "ThresholdMask", "InpaintStitchImproved"
        ];
        SwarmUserErrorException error = Assert.Throws<SwarmUserErrorException>(
            () => ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(graph, types));
        Assert.That(error.Message, Does.Contain("LanPaint_KSampler"));
        Assert.That(error.Message, Does.Contain("THIS backend"));
    }

    [Test]
    public static void MissingAllBackendNodeInventoryFailsForNativeSegments()
    {
        Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(NativeGraph(), []));
    }

    [Test]
    public static void OrdinaryGenerationDoesNotRequireQwenNodes()
    {
        JObject graph = new() { ["save"] = Node("SaveImage") };
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(graph, []));
    }

    [Test]
    public static void ReferenceEditDoesNotRequireLanPaintSamplerWhenNotUsed()
    {
        JObject graph = NativeGraph();
        graph["sampler"] = Node("KSampler");
        HashSet<string> types = [
            "InpaintCropImproved", "TextEncodeQwenImage21",
            "SwarmRequireNonEmptyMask", "ThresholdMask",
            "InpaintStitchImproved"
        ];
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(graph, types));
    }
}
