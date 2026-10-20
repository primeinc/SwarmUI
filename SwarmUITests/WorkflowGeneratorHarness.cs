using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using FreneticUtilities.FreneticToolkit;
using Newtonsoft.Json.Linq;
using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Builtin_DynamicThresholding;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>Builds real ComfyUI workflows with the <see cref="WorkflowGenerator"/> in-process: no server, backend, model files, network, or GPU.</summary>
public static class WorkflowGeneratorHarness
{
    /// <summary>Lock for <see cref="Init"/>.</summary>
    public static readonly LockObject InitLock = new();

    /// <summary>True once <see cref="Init"/> completed.</summary>
    public static bool Initialized = false;

    /// <summary>Root folder the fake model sets point at. Nothing is written to it.</summary>
    public static string FakeModelRoot;

    /// <summary>Feature set of a ComfyUI backend with the Swarm nodes and common node packs installed.</summary>
    public static readonly string[] FullFeatures = ["comfyui", "refiners", "controlnet", "endstepsearly", "seamless", "video", "variation_seed", "freeu", "yolov8", "comfy_loadimage_b64", "comfy_saveimage_ws", "comfy_saveaudio_ws", "comfy_just_load_model", "comfy_latent_blend_masked", "set_clip_device", "gguf", "nunchaku", "bnb_nf4", "teacache", "dynamic_thresholding"];

    /// <summary>Feature set of a bare ComfyUI backend without the Swarm nodes.</summary>
    public static readonly string[] BareFeatures = ["comfyui", "refiners", "controlnet", "endstepsearly", "seamless", "video"];

    /// <summary>Model class ID fragments that are not main generation models.</summary>
    public static readonly string[] NonMainClassMarkers = ["/lora", "/vae", "/audio-vae", "/controlnet", "/control-", "/embedding", "/textual-inversion"];

    /// <summary>Prepares the static server state the generator reads. Safe to call repeatedly.</summary>
    public static void Init()
    {
        lock (InitLock)
        {
            if (Initialized)
            {
                return;
            }
            SpecialTools.Internationalize();
            Program.ServerSettings.Metadata.ImageMetadataIncludeModelHash = false;
            Program.ServerSettings.AddDebugData = false;
            Program.GlobalProgramCancel = new CancellationToken(true);
            FakeModelRoot = Path.Combine(Path.GetTempPath(), "swarmui-workflow-tests").Replace('\\', '/');
            T2IModelClassSorter.Init();
            foreach (string type in new[] { "Stable-Diffusion", "VAE", "LoRA", "Embedding", "ControlNet", "Clip", "ClipVision" })
            {
                Program.T2IModelSets[type] = new() { ModelType = type, DownloadFolderPath = $"{FakeModelRoot}/{type}" };
            }
            CommonModels.RegisterCoreSet();
            T2IParamTypes.RegisterDefaults();
            Program.Backends = new();
            new ComfyUIBackendExtension().OnInit();
            new DynamicThresholdingExtension().OnInit();
            foreach (CommonModels.ModelInfo info in CommonModels.Known.Values)
            {
                AddModel(info.FolderType, info.FileName, null);
            }
            MarkModelFileLiteralsPresent();
            Initialized = true;
        }
    }

    /// <summary>Marks every model filename string literal compiled into SwarmUI as already present, so the generator never starts a download.
    /// Reads 'ldstr' operands from method IL, which covers every literal the generator can pass to its download helpers.</summary>
    public static void MarkModelFileLiteralsPresent()
    {
        Module module = typeof(WorkflowGenerator).Module;
        HashSet<string> names = [];
        BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;
        foreach (Type type in module.GetTypes())
        {
            IEnumerable<MethodBase> methods = type.GetMethods(flags).Cast<MethodBase>().Concat(type.GetConstructors(flags));
            foreach (MethodBase method in methods)
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
                if (il is null)
                {
                    continue;
                }
                for (int i = 0; i + 4 < il.Length; i++)
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
                            names.Add(str);
                        }
                    }
                    catch (Exception)
                    {
                        // Not a string token at this offset.
                    }
                }
            }
        }
        foreach (string name in names)
        {
            WorkflowGenerator.ClipModelsValid[name] = name;
            WorkflowGenerator.VisionModelsValid[name] = name;
            WorkflowGenerator.IPAdapterModelsValid[name] = name;
            WorkflowGenerator.IPAdapterModelsValid[$"LORA-{name}"] = name;
        }
    }

    /// <summary>Registers a fake model file in a model set and returns it.</summary>
    /// <param name="type">Model set type, eg "Stable-Diffusion" or "LoRA".</param>
    /// <param name="name">Model name including extension.</param>
    /// <param name="modelClass">The model class, or null.</param>
    /// <param name="diffusionModelsFolder">If true, the model sits in a 'diffusion_models' folder (backbone-only format).</param>
    public static T2IModel AddModel(string type, string name, T2IModelClass modelClass, bool diffusionModelsFolder = false)
    {
        T2IModelHandler handler = Program.T2IModelSets[type];
        string folder = diffusionModelsFolder ? $"{FakeModelRoot}/diffusion_models" : $"{FakeModelRoot}/{type}";
        T2IModel model = new(handler, folder, $"{folder}/{name}", name)
        {
            ModelClass = modelClass,
            StandardWidth = modelClass?.StandardWidth ?? 0,
            StandardHeight = modelClass?.StandardHeight ?? 0,
            Metadata = new() { ModelName = name, ModelClassType = modelClass?.ID }
        };
        handler.Models[name] = model;
        return model;
    }

    /// <summary>Returns every registered main generation model class, ordered by ID.</summary>
    public static List<T2IModelClass> MainModelClasses()
    {
        Init();
        return [.. T2IModelClassSorter.ModelClasses.Values.Where(c => !NonMainClassMarkers.Any(c.ID.Contains)).OrderBy(c => c.ID)];
    }

    /// <summary>Returns a fake main model of the given class, registering it on first use.</summary>
    public static T2IModel ModelFor(T2IModelClass modelClass, bool diffusionModelsFolder, string suffix = "")
    {
        Init();
        string name = $"test/{(diffusionModelsFolder ? "dm" : "ckpt")}/{modelClass.ID.Replace('/', '_')}{suffix}.safetensors";
        if (Program.T2IModelSets["Stable-Diffusion"].Models.TryGetValue(name, out T2IModel existing))
        {
            return existing;
        }
        return AddModel("Stable-Diffusion", name, modelClass, diffusionModelsFolder);
    }

    /// <summary>Returns a fake LoRA, registering it on first use.</summary>
    public static T2IModel Lora(string name)
    {
        Init();
        if (Program.T2IModelSets["LoRA"].Models.TryGetValue($"{name}.safetensors", out T2IModel existing))
        {
            return existing;
        }
        return AddModel("LoRA", $"{name}.safetensors", null);
    }

    /// <summary>Returns a solid-color PNG image of the given size.</summary>
    public static Image TestImage(int width, int height)
    {
        using SixLabors.ImageSharp.Image<Rgba32> img = new(width, height, new Rgba32(120, 140, 160, 255));
        return new Image(img);
    }

    /// <summary>A generation request for the harness.</summary>
    public class Request
    {
        /// <summary>Main model.</summary>
        public T2IModel Model;

        /// <summary>Main prompt, Swarm prompt syntax.</summary>
        public string Prompt = "a photo of a man on a road";

        /// <summary>Negative prompt.</summary>
        public string NegativePrompt = "";

        /// <summary>Extra raw parameters, by parameter ID, applied as user text input.</summary>
        public Dictionary<string, string> Params = [];

        /// <summary>Extra typed parameter setters, applied after <see cref="Params"/>.</summary>
        public List<Action<T2IParamInput>> Setters = [];

        /// <summary>Backend feature set.</summary>
        public string[] Features = FullFeatures;
    }

    /// <summary>Builds the user input for a request, running Swarm's normal prompt parsing.</summary>
    public static T2IParamInput BuildInput(Request request)
    {
        Init();
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.Model, request.Model);
        input.Set(T2IParamTypes.Prompt, request.Prompt);
        input.Set(T2IParamTypes.NegativePrompt, request.NegativePrompt);
        input.Set(T2IParamTypes.Width, 1024);
        input.Set(T2IParamTypes.Height, 1024);
        input.Set(T2IParamTypes.Steps, 20);
        input.Set(T2IParamTypes.CFGScale, 4.0);
        input.Set(T2IParamTypes.Seed, 1L);
        input.Set(T2IParamTypes.Images, 1);
        foreach ((string key, string value) in request.Params)
        {
            T2IParamTypes.ApplyParameter(key, value, input);
        }
        foreach (Action<T2IParamInput> setter in request.Setters)
        {
            setter(input);
        }
        input.ApplySpecialLogic();
        input.ApplyLateSpecialLogic();
        return input;
    }

    /// <summary>Result of one harness build.</summary>
    /// <param name="Input">The user input after generation.</param>
    /// <param name="Workflow">The generated workflow.</param>
    /// <param name="WorkflowText">The generated workflow as text.</param>
    /// <param name="Trace">The trace, if traced.</param>
    public record class Build(T2IParamInput Input, JObject Workflow, string WorkflowText, WorkflowTrace Trace);

    /// <summary>Generates a workflow for a request, optionally traced.</summary>
    public static Build Generate(Request request, bool traced)
    {
        T2IParamInput input = BuildInput(request);
        WorkflowGenerator generator = new() { UserInput = input, Features = [.. request.Features] };
        if (traced)
        {
            generator.Trace = new(generator);
        }
        JObject workflow = generator.Generate();
        return new(input, workflow, workflow.ToString(), generator.Trace);
    }

    /// <summary>Returns true if two builds of the same request produced the same workflow. The only per-build randomness the generator emits, the bare-ComfyUI SaveImage 'filename_prefix' ('SwarmUI_{random}_'), is masked.</summary>
    public static bool SameWorkflow(Build a, Build b)
    {
        return a.WorkflowText == b.WorkflowText || ComparableText(a.Workflow) == ComparableText(b.Workflow);
    }

    /// <summary>Returns the workflow text with the random SaveImage 'filename_prefix' masked.</summary>
    public static string ComparableText(JObject workflow)
    {
        JObject copy = (JObject)workflow.DeepClone();
        foreach (JProperty property in copy.Properties())
        {
            if (property.Value["inputs"] is JObject inputs && inputs["filename_prefix"] is JValue prefix && $"{prefix}".StartsWith("SwarmUI_"))
            {
                inputs["filename_prefix"] = "SwarmUI_(random)_";
            }
        }
        return copy.ToString();
    }
}
