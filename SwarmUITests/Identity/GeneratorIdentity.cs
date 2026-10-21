using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using FreneticUtilities.FreneticExtensions;
using FreneticUtilities.FreneticToolkit;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Builtin_DynamicThresholding;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests.Identity;

/// <summary>Builds every main model class x option variant with only APIs the upstream commit and the trace branch share, and writes one line per case:
/// label, sha256 of the generated workflow's text and of the exact text <see cref="ComfyUIAPIAbstractBackend.CreateWorkflow"/> sends (only the random SaveImage prefix replaced), and, for the input CreateWorkflow sent, sha256 of the image parameter metadata after unused parameters are removed together with the unused-parameter list (swarm_version left out) and its ExtraMeta keys other than debug_trace; or the error.
/// Fails if any error is not a generator rejection or fewer than 80% of the cases build. IDENTITY_OUT names the output file; IDENTITY_DEBUG=1 turns AddDebugData on. Skipped without IDENTITY_OUT; SwarmUITests/Identity/justfile runs it in this tree and in the upstream tree and compares the dumps.</summary>
[Category("Identity")]
public class GeneratorIdentity : SwarmUITest
{
    /// <summary>Folder the fake models are registered under.</summary>
    public static string Root;

    /// <summary>Registers a fake model of a class in a model set.</summary>
    public static T2IModel Add(string type, string name, T2IModelClass modelClass, bool diffusionModels = false)
    {
        T2IModelHandler handler = Program.T2IModelSets[type];
        string folder = diffusionModels ? $"{Root}/diffusion_models" : $"{Root}/{type}";
        T2IModel model = new(handler, folder, $"{folder}/{name}", name) { ModelClass = modelClass, StandardWidth = modelClass?.StandardWidth ?? 0, StandardHeight = modelClass?.StandardHeight ?? 0, Metadata = new() { ModelName = name, ModelClassType = modelClass?.ID } };
        handler.Models[name] = model;
        return model;
    }

    /// <summary>Returns the fake main model of a class in checkpoint or diffusion_models format, registering it once.</summary>
    public static T2IModel ModelFor(T2IModelClass modelClass, bool diffusionModels, string suffix = "")
    {
        string name = $"test/{(diffusionModels ? "dm" : "ckpt")}/{modelClass.ID.Replace('/', '_')}{suffix}.safetensors";
        return Program.T2IModelSets["Stable-Diffusion"].Models.TryGetValue(name, out T2IModel existing) ? existing : Add("Stable-Diffusion", name, modelClass, diffusionModels);
    }

    /// <summary>Returns a fake ControlNet for a main model class: its compat class's ControlNet class if one exists, otherwise an unclassed one.</summary>
    public static T2IModel ControlNetFor(T2IModelClass modelClass)
    {
        T2IModelClass controlClass = T2IModelClassSorter.ModelClasses.Values.Where(c => c.ID.EndsWith("/controlnet") && c.CompatClass?.ID == modelClass.CompatClass?.ID).OrderBy(c => c.ID).FirstOrDefault();
        string name = $"test/cn/{(controlClass?.ID ?? "generic").Replace('/', '_')}.safetensors";
        return Program.T2IModelSets["ControlNet"].Models.TryGetValue(name, out T2IModel existing) ? existing : Add("ControlNet", name, controlClass);
    }

    /// <summary>Returns a solid-color PNG image.</summary>
    public static Image TestImage(int width, int height)
    {
        using SixLabors.ImageSharp.Image<Rgba32> img = new(width, height, new Rgba32(120, 140, 160, 255));
        return new Image(img);
    }

    /// <summary>Marks every model file name literal in the generator's IL as already present, so the generator never tries to download one.</summary>
    public static void MarkModelFileLiteralsPresent()
    {
        Module module = typeof(WorkflowGenerator).Module;
        BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;
        foreach (MethodBase method in module.GetTypes().SelectMany(t => t.GetMethods(flags).Cast<MethodBase>().Concat(t.GetConstructors(flags))))
        {
            byte[] il;
            try
            {
                il = method.GetMethodBody()?.GetILAsByteArray();
            }
            catch (Exception)
            {
                continue;
            }
            for (int i = 0; il is not null && i + 4 < il.Length; i++)
            {
                if (il[i] != 0x72 || il[i + 4] != 0x70)
                {
                    continue;
                }
                try
                {
                    string str = module.ResolveString(BitConverter.ToInt32(il, i + 1));
                    if (!str.Contains("://") && (str.EndsWith(".safetensors") || str.EndsWith(".pth") || str.EndsWith(".pt") || str.EndsWith(".bin") || str.EndsWith(".gguf")))
                    {
                        WorkflowGenerator.ClipModelsValid[str] = str;
                        WorkflowGenerator.VisionModelsValid[str] = str;
                        WorkflowGenerator.IPAdapterModelsValid[str] = str;
                        WorkflowGenerator.IPAdapterModelsValid[$"LORA-{str}"] = str;
                    }
                }
                catch (Exception)
                {
                }
            }
        }
    }

    /// <summary>Returns workflow JSON text with each random SaveImage filename prefix (read from the parsed workflow) replaced by a fixed one, every other byte as it was.</summary>
    public static string MaskedText(string text)
    {
        foreach (JProperty property in JObject.Parse(text).Properties())
        {
            if (property.Value["inputs"]?["filename_prefix"] is JValue prefix && $"{prefix}".StartsWith("SwarmUI_"))
            {
                text = text.Replace($"\"{prefix}\"", "\"SwarmUI_(random)_\"");
            }
        }
        return text;
    }

    /// <summary>Writes the identity dump to IDENTITY_OUT.</summary>
    [Test]
    public void DumpWorkflowIdentity()
    {
        string outPath = Environment.GetEnvironmentVariable("IDENTITY_OUT");
        if (string.IsNullOrWhiteSpace(outPath))
        {
            Assert.Ignore("Set IDENTITY_OUT, or run 'just run' in SwarmUITests/Identity.");
        }
        if (ComfyUIBackendExtension.SamplerParam is not null)
        {
            Assert.Ignore("Another fixture already ran ComfyUIBackendExtension.OnInit, and running it again would register its parameters twice; run 'just run' in SwarmUITests/Identity, which runs this test alone.");
        }
        Setup();
        Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash = false;
        Program.ServerSettings.AddDebugData = Environment.GetEnvironmentVariable("IDENTITY_DEBUG") == "1";
        Program.GlobalProgramCancel = new CancellationToken(true);
        Root = Path.Combine(Path.GetTempPath(), "swarmui-identity").Replace('\\', '/');
        T2IModelClassSorter.Init();
        foreach (string type in new[] { "Stable-Diffusion", "VAE", "LoRA", "Embedding", "ControlNet", "Clip", "ClipVision" })
        {
            Program.T2IModelSets[type] = new() { ModelType = type, DownloadFolderPath = $"{Root}/{type}" };
        }
        CommonModels.RegisterCoreSet();
        T2IParamTypes.RegisterDefaults();
        Program.Backends = new();
        new ComfyUIBackendExtension().OnInit();
        new DynamicThresholdingExtension().OnInit();
        foreach (CommonModels.ModelInfo info in CommonModels.Known.Values)
        {
            Add(info.FolderType, info.FileName, null);
        }
        MarkModelFileLiteralsPresent();
        Add("LoRA", "test_alien_lora.safetensors", null);
        Image init = TestImage(512, 384), promptImage = TestImage(256, 256), mask = TestImage(512, 384);
        T2IModel video = ModelFor(T2IModelClassSorter.ModelClasses["wan-2_1-image2video-14b"], true), seedvr = ModelFor(T2IModelClassSorter.ModelClasses["seedvr2"], true);
        string basePrompt = "zqxbase man on a road", segmentPrompt = "zqxbase man on a road <segment:man> zqxseg green alien <lora:test_alien_lora>";
        static void refiner(T2IParamInput i)
        {
            i.Set(T2IParamTypes.RefinerMethod, "PostApply");
            i.Set(T2IParamTypes.RefinerControl, 0.4);
        }
        void videoSet(T2IParamInput i)
        {
            i.Set(T2IParamTypes.VideoModel, video);
            i.Set(T2IParamTypes.VideoFrames, 17);
        }
        List<(string Name, Func<T2IModel, (string, List<Action<T2IParamInput>>)> Make, bool CheckpointOnly)> variants =
        [
            ("plain", m => (basePrompt, []), false),
            ("segment-lora", m => (segmentPrompt, []), false),
            ("refiner+segment", m => (segmentPrompt, [refiner]), false),
            ("refinermodel+segment", m => (segmentPrompt, [refiner, i => i.Set(T2IParamTypes.RefinerModel, ModelFor(m.ModelClass, m.IsDiffusionModelsFormat, "-alt"))]), false),
            ("segmentmodel", m => (segmentPrompt, [i => i.Set(T2IParamTypes.SegmentModel, ModelFor(m.ModelClass, m.IsDiffusionModelsFormat, "-alt"))]), false),
            ("applyafterbase+refiner", m => (segmentPrompt, [refiner, i => i.Set(T2IParamTypes.SegmentApplyAfter, "Base")]), false),
            ("initimage+promptimage+segment", m => (segmentPrompt, [i => i.Set(T2IParamTypes.InitImage, init), i => i.Set(T2IParamTypes.InitImageCreativity, 0.6), i => i.Set(T2IParamTypes.PromptImages, [promptImage])]), false),
            ("regions+objects", m => ($"{basePrompt} <region:0,0,0.5,1> zqxregion left side <object:0.5,0.25,0.5,0.5> zqxobject cat", []), true),
            ("initimage+mask", m => (basePrompt, [i => i.Set(T2IParamTypes.InitImage, init), i => i.Set(T2IParamTypes.MaskImage, mask), i => i.Set(T2IParamTypes.InitImageCreativity, 0.6)]), true),
            ("controlnet", m => (basePrompt, [i => i.Set(T2IParamTypes.Controlnets[0].Strength, 0.8), i => i.Set(T2IParamTypes.Controlnets[0].Image, init), i => i.Set(T2IParamTypes.Controlnets[0].Model, ControlNetFor(m.ModelClass))]), true),
            ("refiner-upscale", m => (basePrompt, [refiner, i => i.Set(T2IParamTypes.RefinerUpscale, 1.5)]), true),
            ("clear+variationseed", m => ($"{basePrompt} <clear:background>", [i => i.Set(T2IParamTypes.VariationSeed, 5L), i => i.Set(T2IParamTypes.VariationSeedStrength, 0.3)]), true),
            ("image2video", m => (basePrompt, [videoSet]), true),
            ("image2video+extend", m => ($"{basePrompt} <extend:17> zqxextend walking on", [videoSet, i => i.Set(T2IParamTypes.VideoExtendModel, video)]), true),
            ("seedvr", m => (basePrompt, [i => i.Set(ComfyUIBackendExtension.SeedVRModel, seedvr)]), true)
        ];
        string[] features = ["comfyui", "refiners", "controlnet", "endstepsearly", "seamless", "video", "variation_seed", "freeu", "yolov8", "comfy_loadimage_b64", "comfy_saveimage_ws", "comfy_saveaudio_ws", "comfy_just_load_model", "comfy_latent_blend_masked", "set_clip_device", "gguf", "nunchaku", "bnb_nf4", "teacache", "dynamic_thresholding"];
        string[] markers = ["/lora", "/vae", "/audio-vae", "/controlnet", "/control-", "/embedding", "/textual-inversion"];
        List<(string Label, Func<T2IParamInput> Make)> cases = [];
        foreach (T2IModelClass modelClass in T2IModelClassSorter.ModelClasses.Values.Where(c => !markers.Any(c.ID.Contains)).OrderBy(c => c.ID))
        {
            ControlNetFor(modelClass);
            foreach (bool diffusionModels in new[] { false, true })
            {
                T2IModel model = ModelFor(modelClass, diffusionModels);
                ModelFor(modelClass, diffusionModels, "-alt");
                foreach ((string Name, Func<T2IModel, (string, List<Action<T2IParamInput>>)> Make, bool CheckpointOnly) variant in variants.Where(v => !diffusionModels || !v.CheckpointOnly))
                {
                    cases.Add(($"{modelClass.ID} [{(diffusionModels ? "diffusion_models" : "checkpoint")}] {variant.Name}", () =>
                    {
                        (string prompt, List<Action<T2IParamInput>> setters) = variant.Make(model);
                        T2IParamInput input = new(null);
                        input.Set(T2IParamTypes.Model, model);
                        input.Set(T2IParamTypes.Prompt, prompt);
                        input.Set(T2IParamTypes.NegativePrompt, "");
                        input.Set(T2IParamTypes.Width, 1024);
                        input.Set(T2IParamTypes.Height, 1024);
                        input.Set(T2IParamTypes.Steps, 20);
                        input.Set(T2IParamTypes.CFGScale, 4.0);
                        input.Set(T2IParamTypes.Seed, 1L);
                        input.Set(T2IParamTypes.Images, 1);
                        foreach (Action<T2IParamInput> setter in setters)
                        {
                            setter(input);
                        }
                        input.ApplySpecialLogic();
                        input.ApplyLateSpecialLogic();
                        return input;
                    }
                    ));
                }
            }
        }
        string[] lines = new string[cases.Count];
        List<string> crashes = [];
        int traced = 0;
        Parallel.For(0, cases.Count, i =>
        {
            T2IParamInput input = null;
            try
            {
                input = cases[i].Make();
                WorkflowGenerator generator = new() { UserInput = input, Features = [.. features] };
                JObject workflow = generator.Generate();
                T2IParamInput sendInput = cases[i].Make();
                string sent = ComfyUIAPIAbstractBackend.CreateWorkflow(sendInput, w => w, null, [.. features]);
                if (sendInput.ExtraMeta.ContainsKey("debug_trace"))
                {
                    Interlocked.Increment(ref traced);
                }
                JObject metadata = sendInput.GenFullMetadataObject();
                ((JObject)metadata["sui_image_params"]).Remove("swarm_version");
                string meta = Utilities.HashSHA256($"{metadata["sui_image_params"]}\n{metadata["sui_extra_data"]?["unused_parameters"]}".EncodeUTF8());
                string extra = sendInput.ExtraMeta.Keys.Where(k => k != "debug_trace").Order().JoinString(",");
                lines[i] = $"{cases[i].Label}\t{Utilities.HashSHA256(MaskedText(workflow.ToString()).EncodeUTF8())}\t{Utilities.HashSHA256(MaskedText(sent).EncodeUTF8())}\t{meta}\t{extra}";
            }
            catch (Exception ex)
            {
                lines[i] = $"{cases[i].Label}\tERROR {ex.GetType().Name}: {ex.Message.Replace('\n', ' ')}";
                if (ex is not SwarmReadableErrorException)
                {
                    lock (crashes)
                    {
                        crashes.Add($"{cases[i].Label}: {ex}");
                    }
                }
            }
        });
        File.WriteAllLines(outPath, lines);
        List<string[]> built = [.. lines.Where(l => !l.Contains("\tERROR ")).Select(l => l.Split('\t'))];
        TestContext.Out.WriteLine($"IDENTITY {cases.Count} cases, {cases.Count - built.Count} errors, debug={Program.ServerSettings.AddDebugData}, {traced} builds carry debug_trace, {built.Select(l => l[1]).Distinct().Count()} distinct workflow hashes, {built.Select(l => l[2]).Distinct().Count()} distinct sent hashes, {built.Select(l => l[3]).Distinct().Count()} distinct metadata hashes, written to {outPath}");
        Assert.That(crashes, Is.Empty, $"every error is a generator rejection (SwarmReadableErrorException); first: {crashes.FirstOrDefault()}");
        Assert.That(built.Count, Is.GreaterThanOrEqualTo(cases.Count * 8 / 10), "at least 80% of the cases build, so identical dumps compare real workflows");
    }
}
