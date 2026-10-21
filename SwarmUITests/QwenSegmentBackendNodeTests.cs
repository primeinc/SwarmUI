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
    public static void RawQwenPreflightMustEnforceActualLanPaintRequiredFields()
    {
        // The exact regression class that originally bypassed C# and CPU
        // checks: Python sampler defaults do not satisfy Comfy workflow JSON.
        JObject backend = new()
        {
            ["LanPaint_KSampler"] = new JObject
            {
                ["input"] = new JObject
                {
                    ["required"] = new JObject
                    {
                        ["LanPaint_NumSteps"] = new JArray("INT"),
                        ["LanPaint_PromptMode"] = new JArray("STRING"),
                        ["LanPaint_Info"] = new JArray("STRING"),
                        ["Inpainting_mode"] = new JArray("STRING")
                    }
                }
            }
        };
        JObject graph = new()
        {
            ["crop"] = Node("InpaintCropImproved"),
            ["sampler"] = new JObject
            {
                ["class_type"] = "LanPaint_KSampler",
                ["inputs"] = new JObject
                {
                    ["LanPaint_NumSteps"] = 5,
                    ["LanPaint_PromptMode"] = "Image First",
                    ["LanPaint_Info"] = "LanPaint KSampler.",
                    ["Inpainting_mode"] = "Image Inpainting"
                }
            }
        };
        HashSet<string> available = ["InpaintCropImproved", "LanPaint_KSampler"];
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(graph.ToString(), available, backend));
        foreach (string field in new[] { "LanPaint_Info", "Inpainting_mode", "LanPaint_NumSteps" })
        {
            JObject absent = (JObject)graph.DeepClone();
            ((JObject)absent["sampler"]["inputs"]).Remove(field);
            SwarmUserErrorException failure = Assert.Throws<SwarmUserErrorException>(() =>
                ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(absent.ToString(), available, backend),
                $"Comfy's required '{field}' must fail before sending the workflow");
            Assert.That(failure.Message, Does.Contain(field));
        }
        // Other models do not inherit Qwen-only native contract checks.
        JObject ordinary = new() { ["sampler"] = graph["sampler"].DeepClone() };
        ((JObject)ordinary["sampler"]["inputs"]).Remove("LanPaint_Info");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(ordinary.ToString(), available, backend));
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

        optional["image_2"] = new JArray("MASK");
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, oldBackend))
            .Message, Does.Contain("IMAGE"),
            "A name-matching but MASK-typed image socket must not pass preflight");
        optional["image_2"] = new JArray("IMAGE");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, oldBackend));
        // The preexisting single-reference batch input remains compatible.
        graph["vision"]["inputs"] = new JObject { ["images"] = new JArray("crop", 1) };
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, new JObject()));
    }


    [Test]
    public static void LegacyRemoteMaskGuardWithoutCoverageSocketMustFailBeforeUpload()
    {
        JObject graph = new()
        {
            ["crop"] = Node("InpaintCropImproved"),
            ["guard"] = Node("SwarmRequireNonEmptyMask")
        };
        graph["guard"]["inputs"]["mask"] = new JArray("crop", 2);
        graph["guard"]["inputs"]["minimum_confidence"] = 0.0;
        graph["guard"]["inputs"]["maximum_coverage"] = 0.99;

        JObject backend = new()
        {
            ["SwarmRequireNonEmptyMask"] = new JObject
            {
                ["input"] = new JObject
                {
                    ["required"] = new JObject
                    {
                        ["mask"] = new JArray("MASK"),
                        ["minimum_confidence"] = new JArray("FLOAT")
                    }
                }
            }
        };
        SwarmUserErrorException missing = Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend));
        Assert.That(missing.Message, Does.Contain("maximum_coverage"));
        Assert.That(missing.Message, Does.Contain("THIS backend"));

        JObject required = (JObject)backend["SwarmRequireNonEmptyMask"]["input"]["required"];
        required["maximum_coverage"] = new JArray("IMAGE");
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend))
            .Message, Does.Contain("FLOAT"),
            "A name-matching IMAGE socket cannot validate a coverage fraction");

        required["maximum_coverage"] = new JArray("FLOAT");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend));

        const string raw = """
            {
                "crop": {"class_type":"InpaintCropImproved","inputs":{}},
                "guard": {"class_type":"SwarmRequireNonEmptyMask",
                    "inputs":{"mask":["crop",2],"minimum_confidence":0.0,"maximum_coverage":0.99}}
            }
            """;
        HashSet<string> types = ["InpaintCropImproved", "SwarmRequireNonEmptyMask"];
        required.Remove("maximum_coverage");
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, types, backend))
            .Message, Does.Contain("maximum_coverage"),
            "Raw external Qwen workflows must reject stale backends before any uploads");
        required["maximum_coverage"] = new JArray("FLOAT");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, types, backend));
    }

    [Test]
    public static void QwenCoverageGuardDoesNotAffectPlainTextToImage()
    {
        JObject graph = new()
        {
            ["base"] = Node("SwarmEmptyQwenImage21Latent"),
            ["vision"] = Node("SwarmTextEncodeAdvanced")
        };
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, new JObject()));
    }


    [Test]
    public static void QwenOnlyRawImageEditWithoutSegmentOrEmptyLatentMustPreflightBackend()
    {
        // A valid custom Qwen image-edit graph can supply its own starting
        // latent, so neither InpaintCropImproved nor SwarmEmptyQwenImage21Latent
        // is a reliable sentinel for newly required Swarm encoder inputs.
        JObject graph = new()
        {
            ["source"] = Node("LoadImage"),
            ["vision"] = Node("SwarmTextEncodeAdvanced")
        };
        graph["vision"]["inputs"]["image_1"] = new JArray("source", 0);
        graph["vision"]["inputs"]["qwen21_prevent_empty_text"] = true;
        HashSet<string> available = ["LoadImage"];
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(graph, available))
            .Message, Does.Contain("SwarmTextEncodeAdvanced"),
            "A Qwen encoder-only graph must not bypass selected-backend node inventory");

        available.Add("SwarmTextEncodeAdvanced");
        JObject backend = new()
        {
            ["SwarmTextEncodeAdvanced"] = new JObject
            {
                ["input"] = new JObject
                {
                    ["optional"] = new JObject { ["images"] = new JArray("IMAGE") }
                }
            }
        };
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend))
            .Message, Does.Contain("image_1"),
            "A Qwen raw edit has no crop/empty-latent sentinel but still needs typed image slots");

        JObject optional = (JObject)backend["SwarmTextEncodeAdvanced"]["input"]["optional"];
        optional["image_1"] = new JArray("IMAGE");
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend))
            .Message, Does.Contain("qwen21_prevent_empty_text"));
        optional["qwen21_prevent_empty_text"] = new JArray("BOOLEAN");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend));
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(graph, available));

        const string raw = """
            {
                "source":{"class_type":"LoadImage","inputs":{"image":${initimage}}},
                "vision":{"class_type":"SwarmTextEncodeAdvanced",
                    "inputs":{"image_1":["source",0],"qwen21_prevent_empty_text":true}}
            }
            """;
        optional.Remove("image_1");
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, available, backend))
            .Message, Does.Contain("image_1"),
            "Raw edits must fail before the image-upload placeholder is resolved");
        optional["image_1"] = new JArray("IMAGE");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, available, backend));
    }


    [Test]
    public static void NativeQwen21RawEditWithoutCropMustRequireBuiltinQwenNodeBeforeUpload()
    {
        // The shipped upstream TextEncodeQwenImage21 outputs conditioning
        // and a native empty latent. Raw upstream edit graphs do not require
        // either SwarmEmptyQwenImage21Latent or InpaintCropImproved.
        JObject native = new()
        {
            ["source"] = Node("LoadImage"),
            ["qwen"] = Node("TextEncodeQwenImage21")
        };
        HashSet<string> installed = ["LoadImage"];
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(native, installed))
            .Message, Does.Contain("TextEncodeQwenImage21"),
            "A raw upstream Qwen node must be recognized even without Swarm-specific markers");

        const string raw = """
            {
                "source":{"class_type":"LoadImage","inputs":{"image":${initimage}}},
                "qwen":{"class_type":"TextEncodeQwenImage21","inputs":{"prompt":"Replace person"}}
            }
            """;
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, installed))
            .Message, Does.Contain("TextEncodeQwenImage21"));
        installed.Add("TextEncodeQwenImage21");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(native, installed));
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, installed));
    }


    [TestCase("image_0")]
    [TestCase("image_17")]
    [TestCase("image_999999999999999999999")]
    [TestCase("image_unknown")]
    public static void RawQwenInvalidVisionSocketNameCannotEvadeBackendPreflight(string socket)
    {
        JObject graph = new()
        {
            ["source"] = Node("LoadImage"),
            ["vision"] = Node("SwarmTextEncodeAdvanced")
        };
        graph["vision"]["inputs"][socket] = new JArray("source", 0);
        HashSet<string> available = ["LoadImage", "SwarmTextEncodeAdvanced"];
        JObject backend = new()
        {
            ["SwarmTextEncodeAdvanced"] = new JObject
            {
                ["input"] = new JObject
                {
                    ["optional"] = new JObject { ["image_16"] = new JArray("IMAGE") }
                }
            }
        };
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend))
            .Message, Does.Contain(socket));

        const string template = """
            {
                "source":{"class_type":"LoadImage","inputs":{"image":${initimage}}},
                "vision":{"class_type":"SwarmTextEncodeAdvanced",
                    "inputs":{"__SOCKET__":["source",0]}}
            }
            """;
        string raw = template.Replace("__SOCKET__", socket);
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, available, backend))
            .Message, Does.Contain(socket),
            "Invalid independent Qwen reference input must fail before image upload");

        ((JObject)graph["vision"]["inputs"]).Remove(socket);
        graph["vision"]["inputs"]["image_16"] = new JArray("source", 0);
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend));
    }

    [Test]
    public static void RawQwenPreflightRejectsStaleIndependentVisionSocketsBeforeUpload()
    {
        const string raw = """
            {
                "source": {"class_type":"LoadImage","inputs":{"image":${initimage}}},
                "crop": {"class_type":"InpaintCropImproved","inputs":{"image":["source",0]}},
                "vision": {"class_type":"SwarmTextEncodeAdvanced",
                    "inputs":{"image_1":["crop",1],"image_2":["source",0]}}
            }
            """;
        HashSet<string> installed = ["LoadImage", "InpaintCropImproved", "SwarmTextEncodeAdvanced"];
        JObject backend = new()
        {
            ["SwarmTextEncodeAdvanced"] = new JObject
            {
                ["input"] = new JObject
                {
                    ["optional"] = new JObject { ["images"] = new JArray("IMAGE") }
                }
            }
        };
        // The old class-only preflight passed this graph despite the remote
        // encoder silently lacking both new image_N inputs.
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, installed, backend))
            .Message, Does.Contain("image_1"));
        JObject optional = (JObject)backend["SwarmTextEncodeAdvanced"]["input"]["optional"];
        optional["image_1"] = new JArray("IMAGE");
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, installed, backend))
            .Message, Does.Contain("image_2"));
        optional["image_2"] = new JArray("IMAGE");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, installed, backend));
    }


    [Test]
    public static void StaleQwenTextEncoderMissingEmptyTextOptionFailsBeforeImageUpload()
    {
        JObject graph = NativeGraph();
        graph["vision"] = Node("SwarmTextEncodeAdvanced");
        graph["vision"]["inputs"]["images"] = new JArray("crop", 1);
        graph["vision"]["inputs"]["qwen21_prevent_empty_text"] = true;
        JObject backend = new()
        {
            ["SwarmTextEncodeAdvanced"] = new JObject
            {
                ["input"] = new JObject
                {
                    ["optional"] = new JObject { ["images"] = new JArray("IMAGE") }
                }
            }
        };
        // Even with one image (no image_N inputs), old SwarmText.py has
        // no Qwen native empty-text option. Validate actual /object_info.
        var error = Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend));
        Assert.That(error.Message, Does.Contain("qwen21_prevent_empty_text"));
        Assert.That(error.Message, Does.Contain("THIS backend"));

        const string raw = """
            {
                "source": {"class_type":"LoadImage","inputs":{"image":${initimage}}},
                "crop": {"class_type":"InpaintCropImproved","inputs":{"image":["source",0]}},
                "vision": {"class_type":"SwarmTextEncodeAdvanced",
                    "inputs":{"images":["crop",1],"qwen21_prevent_empty_text":true}}
            }
            """;
        HashSet<string> installed = ["LoadImage", "InpaintCropImproved", "SwarmTextEncodeAdvanced"];
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, installed, backend))
            .Message, Does.Contain("qwen21_prevent_empty_text"));

        ((JObject)backend["SwarmTextEncodeAdvanced"]["input"]["optional"])
            ["qwen21_prevent_empty_text"] = new JArray("IMAGE");
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend))
            .Message, Does.Contain("BOOLEAN"),
            "A socket advertised as IMAGE cannot accept Qwen's boolean tokenizer flag");
        ((JObject)backend["SwarmTextEncodeAdvanced"]["input"]["optional"])
            ["qwen21_prevent_empty_text"] = new JArray("BOOLEAN");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backend));
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(raw, installed, backend));
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

    [Test]
    public static void QwenIndexedVisionSocketsAboveNativeLimitFailBeforeUpload()
    {
        // The real SwarmText.py advertises and reads image_1..image_16 only.
        // A raw Qwen graph with image_17 must not be silently treated as a
        // non-Qwen workflow or accepted on an older /object_info schema.
        JObject graph = new()
        {
            ["vision"] = Node("SwarmTextEncodeAdvanced")
        };
        JObject visionInputs = (JObject)graph["vision"]["inputs"];
        visionInputs["image_17"] = new JArray("source", 0);
        JObject backendInfo = new()
        {
            ["SwarmTextEncodeAdvanced"] = new JObject
            {
                ["input"] = new JObject
                {
                    ["optional"] = new JObject { ["image_16"] = new JArray("IMAGE") }
                }
            }
        };
        HashSet<string> available = ["SwarmTextEncodeAdvanced"];
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backendInfo))
            .Message, Does.Contain("image_17"));
        Assert.That(Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(
                graph.ToString(), available, backendInfo))
            .Message, Does.Contain("image_17"));

        visionInputs.Property("image_17").Remove();
        visionInputs["image_16"] = new JArray("source", 0);
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backendInfo));
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.PreflightRawQwenSegmentNodes(
                graph.ToString(), available, backendInfo));
    }

    [Test]
    public static void Qwen21TextToImageMustRejectMissing64ChannelLatentNode()
    {
        // Qwen 2.1 plain text-to-image has no InpaintCropImproved node,
        // but now requires this Swarm-specific 64-channel latent generator.
        JObject qwenTextGraph = new()
        {
            ["latent"] = Node("SwarmEmptyQwenImage21Latent"),
            ["save"] = Node("SaveImage")
        };
        HashSet<string> unavailable = ["SaveImage"];
        SwarmUserErrorException missing = Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(qwenTextGraph, unavailable));
        Assert.That(missing.Message, Does.Contain("SwarmEmptyQwenImage21Latent"));
        unavailable.Add("SwarmEmptyQwenImage21Latent");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(qwenTextGraph, unavailable));
        // Other models must not inherit Qwen-only node requirements.
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateNativeQwenSegmentNodes(
                new JObject { ["save"] = Node("SaveImage") }, new HashSet<string>()));
    }

    [Test]
    public static void PlainQwen21TextToImageRejectsLegacySwarmEncoderOptionSchema()
    {
        // This is a plain Qwen T2I workflow, without InpaintCropImproved.
        // The new 64-channel latent and native empty-text option both require
        // backend compatibility preflight before Comfy workflow submission.
        JObject graph = new()
        {
            ["latent"] = Node("SwarmEmptyQwenImage21Latent"),
            ["vision"] = Node("SwarmTextEncodeAdvanced"),
            ["save"] = Node("SaveImage")
        };
        graph["vision"]["inputs"]["qwen21_prevent_empty_text"] = true;
        JObject backendInfo = new()
        {
            ["SwarmTextEncodeAdvanced"] = new JObject
            {
                ["input"] = new JObject
                {
                    ["optional"] = new JObject { ["images"] = new JArray("IMAGE") }
                }
            }
        };
        SwarmUserErrorException missing = Assert.Throws<SwarmUserErrorException>(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backendInfo));
        Assert.That(missing.Message, Does.Contain("qwen21_prevent_empty_text"));
        ((JObject)backendInfo["SwarmTextEncodeAdvanced"]["input"]["optional"])
            ["qwen21_prevent_empty_text"] = new JArray("BOOLEAN");
        Assert.DoesNotThrow(() =>
            ComfyUIAPIAbstractBackend.ValidateQwen21IndependentVisionSockets(graph, backendInfo));
    }

}
