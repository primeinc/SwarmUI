using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Utils;

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
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(NativeGraph(), new HashSet<string>()));
    }

    [Test]
    public static void OrdinaryGenerationDoesNotRequireQwenNodes()
    {
        JObject graph = new() { ["save"] = Node("SaveImage") };
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(graph, new HashSet<string>()));
    }

    [TestCase("SwarmTextEncodeAdvanced")]
    [TestCase("ReferenceLatent")]
    [TestCase("VAEEncode")]
    [TestCase("ImageBatch")]
    [TestCase("GetImageSize")]
    public static void MissingConditioningOrReferenceNodesFailBeforeInference(string missingClass)
    {
        JObject graph = NativeGraph();
        graph["extra"] = Node(missingClass);
        HashSet<string> available = [
            "InpaintCropImproved", "TextEncodeQwenImage21",
            "SwarmRequireNonEmptyMask", "ThresholdMask",
            "LanPaint_KSampler", "InpaintStitchImproved", "SaveImage"
        ];
        Assert.That(
            Assert.Throws<SwarmUserErrorException>(() =>
                ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(graph, available))
            .Message, Does.Contain(missingClass));
        available.Add(missingClass);
        Assert.DoesNotThrow(() => ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(graph, available));
    }

    [Test]
    public static void StaleExternalSwarmTextSchemaRejectsIndependentQwenReferences()
    {
        JObject graph = NativeGraph();
        graph["vision"] = Node("SwarmTextEncodeAdvanced");
        JObject inputs = (JObject)graph["vision"]["inputs"];
        inputs["image_1"] = new JArray("crop", 1);
        inputs["image_2"] = new JArray("extra", 0);

        // Older SwarmText has the legacy IMAGE batch input and the correct
        // class name, but lacks both independent Qwen image_N sockets.
        JObject oldBackend = new()
        {
            ["SwarmTextEncodeAdvanced"] = new JObject
            {
                ["input"] = new JObject
                {
                    ["optional"] = new JObject
                    {
                        ["images"] = new JArray("IMAGE")
                    }
                }
            }
        };
        SwarmUserErrorException first = Assert.Throws<SwarmUserErrorException>(
            () => ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(
                graph, oldBackend));
        Assert.That(first.Message, Does.Contain("image_1"));
        Assert.That(first.Message, Does.Contain("THIS backend"));

        JObject optional = (JObject)oldBackend["SwarmTextEncodeAdvanced"]["input"]["optional"];
        optional["image_1"] = new JArray("IMAGE");
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, oldBackend))
            .Message, Does.Contain("image_2"));

        optional["image_2"] = new JArray("IMAGE");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, oldBackend));
        // The preexisting single-reference batch input remains compatible.
        graph["vision"]["inputs"] = new JObject { ["images"] = new JArray("crop", 1) };
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, new JObject()));
    }

    [Test]
    public static void ReferenceEditDoesNotRequireLanPaintSamplerWhenNotUsed()
    {
        JObject graph = NativeGraph();
        graph["sampler"] = Node("KSampler");
        HashSet<string> types = [
            "InpaintCropImproved", "TextEncodeQwenImage21",
            "SwarmRequireNonEmptyMask", "ThresholdMask",
            "InpaintStitchImproved", "KSampler", "SaveImage"
        ];
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(graph, types));
    }
    [Test]
    public static void RawQwenPreflightAcceptsUnquotedImagePlaceholdersWhenNodesExist()
    {
        // Saved custom workflows can use an unquoted ${initimage} token,
        // which is not parseable JSON until its placeholder is expanded.
        // Validation must precede any remote upload, however.
        const string raw = """
            {
                "image": {"class_type":"LoadImage","inputs":{"image":${initimage}}},
                "crop": {"class_type":"InpaintCropImproved","inputs":{"image":["image",0]}},
                "sampler": {"class_type":"LanPaint_KSampler","inputs":{"seed":${seed}}}
            }
            """;
        HashSet<string> installed = ["InpaintCropImproved", "LanPaint_KSampler", "LoadImage"];
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, installed));
    }

    [Test]
    public static void RawQwenPreflightRejectsMissingNativeNodeBeforeImageUpload()
    {
        const string raw = """
            {
                "image": {"class_type":"LoadImage","inputs":{"image":${initimage}}},
                "crop": {"class_type":"InpaintCropImproved","inputs":{"image":["image",0]}},
                "sampler": {"class_type":"LanPaint_KSampler","inputs":{"seed":${seed}}}
            }
            """;
        HashSet<string> missingSampler = ["InpaintCropImproved", "LoadImage"];
        SwarmUserErrorException error = Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, missingSampler));
        Assert.That(error.Message, Does.Contain("LanPaint_KSampler"));
    }

    [Test]
    public static void RawQwenPreflightRejectsDynamicClassTypeBeforeUpload()
    {
        const string raw = """
            {
                "image": {"class_type":"LoadImage","inputs":{"image":${initimage}}},
                "crop": {"class_type":"InpaintCropImproved","inputs":{"image":["image",0]}},
                "future": {"class_type":"${unresolved_node_type}","inputs":{"seed":${seed}}}
            }
            """;
        SwarmUserErrorException error = Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(
                raw, new HashSet<string> { "InpaintCropImproved", "LoadImage" }));
        Assert.That(error.Message, Does.Contain("class_type"));
        Assert.That(error.Message, Does.Contain("before image upload"));
    }

    [TestCase("Inpaint${node_kind}")]
    [TestCase("Vendor${node_kind}Sampler")]
    public static void RawQwenPreflightRejectsEmbeddedClassTypeMacroBeforeUpload(string classType)
    {
        // A wholly dynamic class_type is already rejected. Partial expansion
        // is equally unsafe: replacing ${node_kind} with null must NOT turn
        // "Inpaint${node_kind}" into an apparently static "Inpaintnull".
        const string template = """
            {
                "source": {"class_type":"LoadImage","inputs":{"image":${initimage}}},
                "dynamic": {"class_type":"__NODE_TYPE__","inputs":{"seed":${seed}}}
            }
            """;
        string raw = template.Replace("__NODE_TYPE__", classType);
        SwarmUserErrorException error = Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(
                raw, new HashSet<string> { "LoadImage" }));
        Assert.That(error.Message, Does.Contain("class_type"));
        Assert.That(error.Message, Does.Contain("before image upload"));
    }

}
