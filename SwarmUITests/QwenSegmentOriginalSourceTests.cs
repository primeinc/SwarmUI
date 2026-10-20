using NUnit.Framework;
using System.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>Exercises the same source selector used by the production graph.</summary>
public class QwenSegmentOriginalSourceTests
{
    // Tests create placeholder model catalog entries for graph construction.
    // Restore the process-global catalogs after EACH test, including failures,
    // so other NUnit fixtures cannot mistake synthetic VAE/LoRA files for real
    // registered models. A passing Qwen test must not contaminate the next one.
    private bool hadVaeHandler, hadLoraHandler, hadCommonQwenVae;
    private T2IModelHandler originalVaeHandler, originalLoraHandler;
    private T2IModel originalVaeModel, originalLoraModel;
    private bool hadVaeModel, hadLoraModel;
    private CommonModels.ModelInfo originalCommonQwenVae;

    [SetUp]
    public void SnapshotModelCatalogs()
    {
        hadVaeHandler = SwarmUI.Core.Program.T2IModelSets.TryGetValue("VAE", out originalVaeHandler);
        hadLoraHandler = SwarmUI.Core.Program.T2IModelSets.TryGetValue("LoRA", out originalLoraHandler);
        hadVaeModel = hadVaeHandler && originalVaeHandler.Models.TryGetValue(
            "qwen-vae.safetensors", out originalVaeModel);
        hadLoraModel = hadLoraHandler && originalLoraHandler.Models.TryGetValue(
            "Audit/HannahLynn.safetensors", out originalLoraModel);
        hadCommonQwenVae = CommonModels.Known.TryGetValue(
            "qwen-image-2.1-vae", out originalCommonQwenVae);
    }

    [TearDown]
    public void RestoreModelCatalogs()
    {
        if (hadVaeHandler)
        {
            SwarmUI.Core.Program.T2IModelSets["VAE"] = originalVaeHandler;
            if (hadVaeModel)
            {
                originalVaeHandler.Models["qwen-vae.safetensors"] = originalVaeModel;
            }
            else
            {
                originalVaeHandler.Models.TryRemove("qwen-vae.safetensors", out _);
            }
        }
        else
        {
            SwarmUI.Core.Program.T2IModelSets.Remove("VAE");
        }

        if (hadLoraHandler)
        {
            SwarmUI.Core.Program.T2IModelSets["LoRA"] = originalLoraHandler;
            if (hadLoraModel)
            {
                originalLoraHandler.Models["Audit/HannahLynn.safetensors"] = originalLoraModel;
            }
            else
            {
                originalLoraHandler.Models.TryRemove("Audit/HannahLynn.safetensors", out _);
            }
        }
        else
        {
            SwarmUI.Core.Program.T2IModelSets.Remove("LoRA");
        }
        if (hadCommonQwenVae)
        {
            CommonModels.Known["qwen-image-2.1-vae"] = originalCommonQwenVae;
        }
        else
        {
            CommonModels.Known.TryRemove("qwen-image-2.1-vae", out _);
        }
    }
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

    [Test]
    public void OrdinaryInitImageWorkflowMustKeepExistingResizeBehavior()
    {
        if (T2IParamTypes.InitImage is null)
        {
            T2IParamTypes.RegisterDefaults();
        }
        using var pixels = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24>(128, 128);
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.InitImage, new SwarmUI.Utils.Image(pixels));
        Assert.That(T2IParamTypes.InitImage.Type.ImageShouldResize, Is.True);
        Newtonsoft.Json.Linq.JObject graph = new()
        {
            ["4"] = new Newtonsoft.Json.Linq.JObject
            {
                ["class_type"] = "LoadImage",
                ["inputs"] = new Newtonsoft.Json.Linq.JObject { ["image"] = "${initimage}" }
            }
        };
        Assert.That(ComfyUIAPIAbstractBackend.ShouldPreserveQwenOriginalInitUpload(graph, input), Is.False);
    }

    /// <summary>Executes the real Swarm generator, not a synthetic node-graph fixture.</summary>
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", false, true, -1, false, 1, false, false)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", true, true, -1, false, 1, false, false)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", true, false, -1, false, 1, false, false)]
    [TestCase("<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", false, true, -1, false, 1, false, false)]
    [TestCase("<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", true, true, -1, false, 1, false, false)]
    [TestCase("<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", true, false, -1, false, 1, false, false)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", false, true, 0.0, false, 1, false, false)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", false, true, 1.0, false, 1, false, false)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", false, true, -1, true, 1, false, false)]
    [TestCase("<segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", true, false, -1, true, 1, false, false)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", false, true, -1, false, 2, false, false)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", true, false, -1, true, 2, false, false)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", false, true, -1, false, 1, true, false)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", true, true, -1, true, 1, true, false)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", false, true, -1, true, 1, false, true)]
    [TestCase("sunrise, a girl walks down the street <segment:girl,0.8,0.1> Edit the image, replace the girl with a black boy with blue eyes", true, false, -1, true, 1, false, true)]
    public void ProductionQwenGraphUsesOriginalSourceWithoutRewritingInstruction(string prompt, bool independentPromptImage, bool hasInitImage, double sectionLoraWeight, bool useBase64Images, int batchSize, bool useSegmentOverride, bool referenceEdit)
    {
        if (T2IParamTypes.Model is null)
        {
            T2IParamTypes.RegisterDefaults();
        }
        // Normal application startup registers ComfyUI extension parameters.
        // NUnit's source-only test process does not run extension startup.
        // Register only the parameter handles exercised by this graph.
        static void ensureComfyParam(ref T2IRegisteredParam<string> param, string name, string defaultValue)
        {
            param ??= T2IParamTypes.Register<string>(
                new("Qwen Smoke " + name, "Test-only Comfy graph parameter", defaultValue));
        }
        ensureComfyParam(ref ComfyUIBackendExtension.PreferredDType, "Preferred DType", "automatic");
        ensureComfyParam(ref ComfyUIBackendExtension.SamplerParam, "Sampler", "euler");
        ensureComfyParam(ref ComfyUIBackendExtension.SchedulerParam, "Scheduler", "simple");
        ensureComfyParam(ref ComfyUIBackendExtension.TextEncodedImage, "Text Encoded Image", "auto");
        // The real application registers its known model catalog at startup.
        // Graph construction reads its Qwen VAE metadata even with a user VAE
        // selected; this placeholder is never downloaded or sampled.
        CommonModels.Known.TryAdd("qwen-image-2.1-vae",
            new CommonModels.ModelInfo("qwen-image-2.1-vae", "Qwen Test VAE",
                "Graph construction only", "https://example.invalid/qwen-vae.safetensors",
                new string('0', 64), "VAE", "qwen-vae.safetensors"));
        // Runtime registers the VAE model set before graph generation.
        // An in-memory entry prevents the source-only test from downloading
        // a model when the segment override deliberately changes compat class.
        if (!SwarmUI.Core.Program.T2IModelSets.TryGetValue("VAE", out T2IModelHandler vaeHandler))
        {
            vaeHandler = new T2IModelHandler() { ModelType = "VAE" };
            SwarmUI.Core.Program.T2IModelSets["VAE"] = vaeHandler;
        }
        vaeHandler.Models["qwen-vae.safetensors"] = new T2IModel(
            vaeHandler, "/models/vae", "/models/vae/qwen-vae.safetensors",
            "qwen-vae.safetensors");
        // Source-only NUnit does not initialize the application extension.
        // Supply inert typed handles for optional extension controls so the
        // real generator can run. Nothing is downloaded or inferred here.
        var registry = typeof(T2IParamTypes).GetMethods(System.Reflection.BindingFlags.Public
                | System.Reflection.BindingFlags.Static)
            .Single(m => m.Name == "Register" && m.IsGenericMethodDefinition);
        foreach (var field in typeof(ComfyUIBackendExtension).GetFields(
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static))
        {
            if (field.GetValue(null) is not null || !field.FieldType.IsGenericType
                || field.FieldType.GetGenericTypeDefinition() != typeof(T2IRegisteredParam<>))
            {
                continue;
            }
            System.Type valueType = field.FieldType.GenericTypeArguments[0];
            string defaultValue = valueType == typeof(bool) ? "false"
                : (valueType == typeof(int) || valueType == typeof(double)) ? "0"
                : valueType == typeof(string) ? "" : null;
            T2IParamType registration = new("Qwen Smoke " + field.Name,
                "Isolated workflow generator regression test handle", defaultValue);
            field.SetValue(null, registry.MakeGenericMethod(valueType).Invoke(null, [registration]));
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
            ModelClass = modelClass,
            StandardWidth = 1024,
            StandardHeight = 1024
        };
        T2IModel vae = new(null, "/models/vae", "/models/vae/vae.safetensors", "vae.safetensors");
        T2IModel clip = new(null, "/models/text_encoders", "/models/text_encoders/clip.safetensors", "clip.safetensors");
        using var pixels = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24>(128, 128);
        var image = new SwarmUI.Utils.Image(pixels);
        T2IParamInput input = new(null);
        if (useSegmentOverride)
        {
            // The base checkpoint is deliberately not Qwen 2.1; the effective
            // native segment model must still choose the original photograph.
            T2IModelClass baseClass = new()
            {
                ID = "stable-diffusion-v1",
                Name = "Base SDv1",
                CompatClass = T2IModelClassSorter.CompatSdv1,
                StandardWidth = 512,
                StandardHeight = 512
            };
            T2IModel baseModel = new(null, "/models/Stable-Diffusion",
                "/models/Stable-Diffusion/base-test.safetensors", "base-test.safetensors")
            {
                ModelClass = baseClass,
                StandardWidth = 512,
                StandardHeight = 512
            };
            input.Set(T2IParamTypes.Model, baseModel);
            input.Set(T2IParamTypes.SegmentModel, model);
        }
        else
        {
            input.Set(T2IParamTypes.Model, model);
        }
        input.Set(T2IParamTypes.VAE, vae);
        input.Set(T2IParamTypes.QwenModel, clip);
        if (hasInitImage)
        {
            input.Set(T2IParamTypes.InitImage, image);
        }
        using var independentPixels = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24>(80, 96);
        var independentReference = new SwarmUI.Utils.Image(independentPixels);
        if (independentPromptImage)
        {
            input.Set(T2IParamTypes.PromptImages, new System.Collections.Generic.List<SwarmUI.Utils.Image>
            {
                independentReference
            });
        }
        input.Set(T2IParamTypes.Prompt, prompt);
        input.Set(T2IParamTypes.Seed, 20261020L);
        input.Set(T2IParamTypes.BatchSize, batchSize);
        input.Set(T2IParamTypes.Steps, 12);
        input.Set(T2IParamTypes.Width, 1024);
        input.Set(T2IParamTypes.Height, 1024);
        input.PreparsePromptLikes();
        if (referenceEdit)
        {
            int segment = new PromptRegion(input.Get(T2IParamTypes.Prompt)).Parts
                .Single(p => p.Type == PromptRegion.PartType.Segment).ContextID;
            input.Set(T2IParamTypes.SegmentQwen21Sampling, "Reference Edit", sectionId: segment);
        }
        // Production prompt parsing assigns the segment's confinement context.
        // A model-only LoRA OFF and ON still must reach this segment sampler.
        if (sectionLoraWeight >= 0)
        {
            int section = new PromptRegion(input.Get(T2IParamTypes.Prompt)).Parts
                .Single(p => p.Type == PromptRegion.PartType.Segment).ContextID;
            Assert.That(section, Is.GreaterThan(0));
            const string path = "Audit/HannahLynn.safetensors";
            if (!SwarmUI.Core.Program.T2IModelSets.TryGetValue("LoRA", out T2IModelHandler loraHandler))
            {
                loraHandler = new T2IModelHandler() { ModelType = "LoRA" };
                SwarmUI.Core.Program.T2IModelSets["LoRA"] = loraHandler;
            }
            loraHandler.Models[path] = new T2IModel(loraHandler, "/models/loras",
                "/models/loras/" + path, path);
            input.Set(T2IParamTypes.Loras, new System.Collections.Generic.List<string> { "Audit/HannahLynn" });
            input.Set(T2IParamTypes.LoraWeights, new System.Collections.Generic.List<string>
                { sectionLoraWeight.ToString(System.Globalization.CultureInfo.InvariantCulture) });
            input.Set(T2IParamTypes.LoraTencWeights, new System.Collections.Generic.List<string> { "0" });
            input.Set(T2IParamTypes.LoraSectionConfinement,
                new System.Collections.Generic.List<string> { section.ToString() });
        }
        WorkflowGenerator generator = new()
        {
            UserInput = input,
            Features = useBase64Images ? new() { "comfy_loadimage_b64" } : new()
        };
        Newtonsoft.Json.Linq.JObject graph = generator.Generate();
        Assert.That(ComfyUIAPIAbstractBackend.ShouldPreserveQwenOriginalInitUpload(graph, input),
            Is.EqualTo(hasInitImage), "Native Qwen Init Image must bypass external canvas resampling");
        var nodes = graph.Properties().Select(p => (Id: p.Name, Node: (Newtonsoft.Json.Linq.JObject)p.Value)).ToArray();
        string authoredRegionText = new PromptRegion(input.Get(T2IParamTypes.Prompt)).Parts
            .Single(p => p.Type == PromptRegion.PartType.Segment).Prompt;
        var positiveEncoders = nodes.Where(x => x.Node["class_type"]?.ToString() == "SwarmTextEncodeAdvanced"
            && x.Node["inputs"]?["prompt"]?.ToString() == authoredRegionText).ToArray();
        Assert.That(nodes.Any(x => x.Node["class_type"]?.ToString() == "TextEncodeQwenImage21"), Is.False,
            "The Qwen path delegates Swarm weight/schedule tags to SwarmTextEncodeAdvanced");
        if (referenceEdit)
        {
            var referenceCrop = nodes.Single(x => x.Node["class_type"]?.ToString() == "InpaintCropImproved");
            var referenceEncoder = positiveEncoders.Single();
            var decoder = nodes.Single(x => x.Node["class_type"]?.ToString() == "LanPaint_ImageDecode");
            string samplerId = decoder.Node["inputs"]?["samples"]?[0]?.ToString();
            var referenceSampler = (Newtonsoft.Json.Linq.JObject)graph[samplerId];
            Assert.That(referenceSampler["class_type"]?.ToString(), Is.EqualTo("KSampler"));
            Assert.That(referenceSampler["inputs"]?["denoise"]?.ToObject<double>(), Is.EqualTo(1));
            string emptyId = referenceSampler["inputs"]?["latent_image"]?[0]?.ToString();
            var emptyLatent = graph[emptyId] as Newtonsoft.Json.Linq.JObject;
            Assert.That(emptyLatent?["class_type"]?.ToString(), Is.EqualTo("LatentMultiply"));
            Assert.That(emptyLatent?["inputs"]?["multiplier"]?.ToObject<double>(), Is.Zero);
            string encodedId = emptyLatent?["inputs"]?["samples"]?[0]?.ToString();
            Assert.That(graph[encodedId]?["class_type"]?.ToString(), Is.EqualTo("VAEEncode"));
            Assert.That(graph[encodedId]?["inputs"]?["pixels"]?.ToString(),
                Is.EqualTo(new Newtonsoft.Json.Linq.JArray(referenceCrop.Id, 1).ToString()));
            Assert.That(nodes.Count(x => x.Node["class_type"]?.ToString() == "LanPaint_ImageEncode"),
                Is.Zero, "Reference Edit must use Qwen's own empty edit latent");
            Assert.That(nodes.Count(x => x.Node["class_type"]?.ToString() == "LanPaint_KSampler"),
                Is.Zero, "Reference Edit must not mix native Qwen and LanPaint sampler contracts");
            Assert.That(nodes.Single(x => x.Node["class_type"]?.ToString() == "InpaintStitchImproved")
                .Node["inputs"]?["stitcher"]?.ToString(),
                Is.EqualTo(new Newtonsoft.Json.Linq.JArray(referenceCrop.Id, 0).ToString()));
            Assert.That(referenceEncoder.Node["inputs"]?["prompt"]?.ToString(),
                Is.EqualTo(authoredRegionText));
            Assert.That(nodes.Count(x => x.Node["class_type"]?.ToString() is "SaveImage" or "SwarmSaveImageWS"),
                Is.EqualTo(1));
            return;
        }
        if (batchSize > 1)
        {
            var cropNodes = nodes.Where(x => x.Node["class_type"]?.ToString() == "InpaintCropImproved").ToArray();
            var detectors = nodes.Where(x => x.Node["class_type"]?.ToString() == "SwarmClipSeg").ToArray();
            var samplers = nodes.Where(x => x.Node["class_type"]?.ToString() == "LanPaint_KSampler").ToArray();
            var encoders = positiveEncoders;
            var stitches = nodes.Where(x => x.Node["class_type"]?.ToString() == "InpaintStitchImproved").ToArray();
            var splitNodes = nodes.Where(x => x.Node["class_type"]?.ToString() == "ImageFromBatch").ToArray();
            Assert.That(cropNodes, Has.Length.EqualTo(batchSize), "Each batch item needs its own crop");
            Assert.That(detectors, Has.Length.EqualTo(batchSize), "Each batch item needs its own detector");
            Assert.That(samplers, Has.Length.EqualTo(batchSize), "Each batch item needs independent inference");
            Assert.That(encoders, Has.Length.EqualTo(batchSize), "Each batch item needs its own Qwen reference");
            Assert.That(stitches, Has.Length.EqualTo(batchSize), "Each batch item needs its matching stitch");
            Assert.That(splitNodes, Has.Length.EqualTo(batchSize), "Split source batch before detection");
            var indexes = splitNodes.Select(x => x.Node["inputs"]?["batch_index"]?.ToObject<int>()).ToArray();
            Assert.That(indexes, Is.EquivalentTo(new int?[] { 0, 1 }));
            for (int i = 0; i < batchSize; i++)
            {
                string batchSourceId = cropNodes[i].Node["inputs"]?["image"]?[0]?.ToString();
                Assert.That(graph[batchSourceId]?["class_type"]?.ToString(), Is.EqualTo("ImageFromBatch"));
                Assert.That(detectors[i].Node["inputs"]?["images"]?.ToString(),
                    Is.EqualTo(cropNodes[i].Node["inputs"]?["image"]?.ToString()));
                Assert.That(encoders[i].Node["inputs"]?["prompt"]?.ToString(),
                    Is.EqualTo(authoredRegionText));
                Assert.That(encoders[i].Node["inputs"]?["images"]?.ToString(),
                    Is.EqualTo(new Newtonsoft.Json.Linq.JArray(cropNodes[i].Id, 1).ToString()));
                Assert.That(stitches[i].Node["inputs"]?["stitcher"]?.ToString(),
                    Is.EqualTo(new Newtonsoft.Json.Linq.JArray(cropNodes[i].Id, 0).ToString()));
            }
            var seeds = samplers.Select(x => x.Node["inputs"]?["seed"]?.ToObject<long>()).ToArray();
            Assert.That(seeds.Distinct().Count(), Is.EqualTo(batchSize), "Batch copies require distinct sampler noise");
            Assert.That(nodes.Count(x => x.Node["class_type"]?.ToString() is "SaveImage" or "SwarmSaveImageWS"),
                Is.EqualTo(1), "One final batch output, not per-crop diagnostics");
            return;
        }
        var crop = nodes.Single(x => x.Node["class_type"]?.ToString() == "InpaintCropImproved");
        var detector = nodes.Single(x => x.Node["class_type"]?.ToString() == "SwarmClipSeg");
        var sampler = nodes.Single(x => x.Node["class_type"]?.ToString() == "LanPaint_KSampler");
        var encoder = positiveEncoders.Single();
        var stitch = nodes.Single(x => x.Node["class_type"]?.ToString() == "InpaintStitchImproved");
        var mask = nodes.Single(x => x.Node["class_type"]?.ToString() == "ThresholdMask");
        string sourceId = crop.Node["inputs"]?["image"]?[0]?.ToString();
        Assert.That(sourceId, Is.Not.Null);
        Assert.That(graph[sourceId]?["class_type"]?.ToString(),
            Is.EqualTo(useBase64Images ? "SwarmLoadImageB64" : "LoadImage"),
            "The edit canvas must load the original image, not the sampled base frame");
        if (useBase64Images)
        {
            string expectedBytes = (hasInitImage ? image : independentReference).AsBase64;
            Assert.That(graph[sourceId]?["inputs"]?["image_base64"]?.ToString(), Is.EqualTo(expectedBytes),
                "The native detector must receive the exact supplied image bytes, without resampling");
        }
        Assert.That(detector.Node["inputs"]?["images"]?.ToString(), Is.EqualTo(crop.Node["inputs"]?["image"]?.ToString()));
        Assert.That(crop.Node["inputs"]?["mask"]?.ToString(),
            Is.EqualTo(new Newtonsoft.Json.Linq.JArray(mask.Id, 0).ToString()));
        // Multiple Qwen reference images keep their own pixel geometry.
        // Do NOT feed differently sized photos through Comfy ImageBatch,
        // which resizes later images to the first reference's dimensions.
        if (hasInitImage && independentPromptImage)
        {
            Assert.That(encoder.Node["inputs"]?["images"]?.Type
                == Newtonsoft.Json.Linq.JTokenType.Array, Is.False);
            Assert.That(encoder.Node["inputs"]?["image_1"]?.ToString(),
                Is.EqualTo(new Newtonsoft.Json.Linq.JArray(crop.Id, 1).ToString()));
            string independentId = encoder.Node["inputs"]?["image_2"]?[0]?.ToString();
            Assert.That(independentId, Is.Not.Null.And.Not.EqualTo(sourceId));
            // Native Qwen rounds 80x96 to 64x96 for BOTH vision and VAE.
            // Each reference remains independent; it is never forced to
            // the crop's 1024x1024 size by ImageBatch.
            var scaled = graph[independentId] as Newtonsoft.Json.Linq.JObject;
            Assert.That(scaled?["class_type"]?.ToString(), Is.EqualTo("ImageScale"));
            Assert.That(scaled?["inputs"]?["width"]?.ToObject<int>(), Is.EqualTo(64));
            Assert.That(scaled?["inputs"]?["height"]?.ToObject<int>(), Is.EqualTo(96));
            Assert.That(scaled?["inputs"]?["upscale_method"]?.ToString(), Is.EqualTo("lanczos"));
            string independentLoaderId = scaled?["inputs"]?["image"]?[0]?.ToString();
            Assert.That(graph[independentLoaderId]?["class_type"]?.ToString(),
                Is.EqualTo(useBase64Images ? "SwarmLoadImageB64" : "LoadImage"));
            var secondVae = nodes.Single(x => x.Node["class_type"]?.ToString() == "VAEEncode"
                && x.Node["inputs"]?["pixels"]?.ToString()
                    == new Newtonsoft.Json.Linq.JArray(independentId, 0).ToString());
            Assert.That(nodes.Count(x => x.Node["class_type"]?.ToString() == "ReferenceLatent"
                && x.Node["inputs"]?["latent"]?.ToString()
                    == new Newtonsoft.Json.Linq.JArray(secondVae.Id, 0).ToString()), Is.EqualTo(2),
                "The same normalized second image must be used by both Swarm vision and VAE references");
        }
        else
        {
            var visionPath = encoder.Node["inputs"]?["images"] as Newtonsoft.Json.Linq.JArray;
            Assert.That(visionPath?.ToString(),
                Is.EqualTo(new Newtonsoft.Json.Linq.JArray(crop.Id, 1).ToString()),
                "The original Qwen crop must reach the Swarm vision encoder");
        }
        Assert.That(encoder.Node["inputs"]?["prompt"]?.ToString(),
            Is.EqualTo(authoredRegionText));
        Assert.That(sampler.Node["inputs"]?["denoise"]?.ToObject<double>(), Is.EqualTo(0.8).Within(0.000001));
        if (sectionLoraWeight >= 0)
        {
            var loraLoaders = nodes.Where(x => x.Node["class_type"]?.ToString()
                is "LoraLoaderModelOnly" or "LoraLoader").ToArray();
            Assert.That(loraLoaders, Has.Length.EqualTo(1),
                "Only one confined LoRA loader may exist in a single-segment graph");
            var loader = loraLoaders.Single();
            Assert.That(loader.Node["inputs"]?["lora_name"]?.ToString().Replace('\\', '/'),
                Does.EndWith("Audit/HannahLynn.safetensors"));
            Assert.That(loader.Node["inputs"]?["strength_model"]?.ToObject<double>(),
                Is.EqualTo(sectionLoraWeight).Within(0.000001));
            string cursor = sampler.Node["inputs"]?["model"]?[0]?.ToString();
            bool samplerUsesLoader = false;
            for (int hop = 0; hop < 20 && cursor is not null; hop++)
            {
                if (cursor == loader.Id)
                {
                    samplerUsesLoader = true;
                    break;
                }
                cursor = graph[cursor]?["inputs"]?["model"]?[0]?.ToString();
            }
            Assert.That(samplerUsesLoader, Is.True,
                "Confined LoRA must be on the sampler model path, not an orphan loader");
        }
        Assert.That(sampler.Node["inputs"]?["LanPaint_Info"]?.ToString(), Is.EqualTo("LanPaint KSampler."));
        Assert.That(sampler.Node["inputs"]?["Inpainting_mode"]?.ToString(), Is.EqualTo("🖼️ Image Inpainting"));
        Assert.That(stitch.Node["inputs"]?["stitcher"]?.ToString(),
            Is.EqualTo(new Newtonsoft.Json.Linq.JArray(crop.Id, 0).ToString()));
        Assert.That(nodes.Count(x => x.Node["class_type"]?.ToString() is "SaveImage" or "SwarmSaveImageWS"), Is.EqualTo(1));
        Assert.That(nodes.Count(x => x.Node["class_type"]?.ToString() == "PreviewImage"), Is.Zero);
    }

}
