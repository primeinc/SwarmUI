using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUI.Builtin_ComfyUIBackend;

/// <summary>Records how a <see cref="WorkflowGenerator"/> built its workflow, without altering the workflow.
/// Covers the steps that ran, the stage and origin of every node, the Swarm type of every node output, every model load and LoRA, every parameter read, prompt and media placeholders, node removals and rewires, and the input chains of every sampler.
/// Exists only for generations where <see cref="Settings.AddDebugData"/> is enabled, or where a caller attaches one explicitly.</summary>
public class WorkflowTrace(WorkflowGenerator gen)
{
    /// <summary>Version of the trace output format.</summary>
    public const int FormatVersion = 1;

    /// <summary>ExtraMeta key the trace summary is stored under.</summary>
    public const string MetadataKey = "debug_trace";

    /// <summary>Node classes treated as samplers for chain analysis.</summary>
    public static HashSet<string> SamplerClasses = ["KSampler", "KSamplerAdvanced", "SwarmKSampler", "SamplerCustom", "SamplerCustomAdvanced", "SwarmUnsampler"];

    /// <summary>Node input names that carry prompt text. Text in these inputs is never written raw.</summary>
    public static HashSet<string> PromptInputNames = ["text", "prompt", "text_g", "text_l", "lyrics", "caption", "match_text", "style"];

    /// <summary>Maximum number of nodes listed per sampler input chain.</summary>
    public static int MaxChainNodes = 48;

    /// <summary>Longest summary, as <see cref="T2IParamInput.MetadataToString(JObject)"/> characters, stored in images whose format keeps metadata in EXIF (every format but PNG).
    /// EXIF metadata is a UTF-16 UserComment that has to fit one 64 KiB JPEG APP1 segment together with the parameter metadata; PNG text chunks have no such limit.</summary>
    public static int MaxExifSummaryChars = 20_000;

    /// <summary>The generator being traced.</summary>
    public WorkflowGenerator Gen = gen;

    /// <summary>Optional ComfyUI object_info data, used to add Comfy output types and to check link types.</summary>
    public JObject ObjectInfo;

    /// <summary>A generator step or model-generation step that ran.</summary>
    public class StepRecord
    {
        /// <summary>Order index of this record.</summary>
        public int Index;

        /// <summary>Step name, or its registration source when unnamed.</summary>
        public string Name;

        /// <summary>Registration priority.</summary>
        public double Priority;

        /// <summary>Registration source, as file:line.</summary>
        public string Source;

        /// <summary>True for a model-generation step, which runs inside a model load.</summary>
        public bool IsModelGen;

        /// <summary>Index of the model load this step ran inside, or -1.</summary>
        public int Load = -1;

        /// <summary>Number of nodes created while this was the innermost step.</summary>
        public int Created;

        /// <summary>Node IDs that left the workflow during this step.</summary>
        public List<string> Removed = [];

        /// <summary>Typed state of the generator's trackers after the step, for top-level steps.</summary>
        public JObject State;
    }

    /// <summary>A node written into the workflow.</summary>
    public class NodeRecord
    {
        /// <summary>Workflow node ID.</summary>
        public string Id;

        /// <summary>Comfy node class.</summary>
        public string ClassType;

        /// <summary>Index of the innermost step that created it.</summary>
        public int Step;

        /// <summary>Generation stage it was created in.</summary>
        public string Stage;

        /// <summary>Creating code location, as member@file:line.</summary>
        public string Origin;

        /// <summary>Compat class ID of the model loaded when it was created.</summary>
        public string Compat;

        /// <summary>Index of the model load it was created inside, or -1.</summary>
        public int Load = -1;

        /// <summary>Number of times CreateNode replaced an existing node with this ID.</summary>
        public int Overwrites;

        /// <summary>Code locations that received this node from the CreateNode dedup instead of a new node.</summary>
        public List<string> ReusedBy = [];

        /// <summary>Name of the step that removed it from the workflow, if removed.</summary>
        public string RemovedBy;
    }

    /// <summary>A call to <see cref="WorkflowGenerator.CreateModelLoader(T2IModel, string, string, bool, int)"/>.</summary>
    public class ModelLoadRecord
    {
        /// <summary>Order index of this record.</summary>
        public int Index;

        /// <summary>Index of the step it ran in.</summary>
        public int Step;

        /// <summary>Generation stage it ran in.</summary>
        public string Stage;

        /// <summary>Loader type argument, eg 'Base', 'Refiner', 'image2video'.</summary>
        public string Type;

        /// <summary>Requested model name.</summary>
        public string Model;

        /// <summary>Model name actually loaded, if it differs from the request.</summary>
        public string LoadedModel;

        /// <summary>Model class ID.</summary>
        public string ModelClass;

        /// <summary>Model compat class ID.</summary>
        public string Compat;

        /// <summary>Section ID argument.</summary>
        public int Section;

        /// <summary>Requested loader node ID, if any.</summary>
        public string RequestedId;

        /// <summary>True if the generator returned an earlier load of the same model and type.</summary>
        public bool Cached;

        /// <summary>Output paths (model, text encoder, VAE).</summary>
        public string ModelOut, ClipOut, VaeOut;

        /// <summary>LoRAs applied during this load.</summary>
        public List<string> Loras = [];

        /// <summary>True once the load returned.</summary>
        public bool Finished;
    }

    /// <summary>A parameter read by the generator.</summary>
    public class ParamReadRecord
    {
        /// <summary>The parameter type.</summary>
        public T2IParamType Type;

        /// <summary>Section ID of the read.</summary>
        public int Section;

        /// <summary>First value returned.</summary>
        public object FirstValue;

        /// <summary>Source of the first value: "section", "base", or "default".</summary>
        public string FirstSource;

        /// <summary>Last value returned.</summary>
        public object LastValue;

        /// <summary>Source of the last value.</summary>
        public string LastSource;

        /// <summary>Number of reads.</summary>
        public int Count;
    }

    /// <summary>A media input embedded into a node.</summary>
    /// <param name="Param">Source parameter name, eg 'initimage' or 'promptimages.0'.</param>
    /// <param name="Media">The media file.</param>
    /// <param name="Width">Embedded width, if resized.</param>
    /// <param name="Height">Embedded height, if resized.</param>
    public record class MediaRecord(string Param, MediaFile Media, int? Width, int? Height);

    /// <summary>Scope for <see cref="EnterStage(string)"/>.</summary>
    public class StageScope(WorkflowTrace trace) : IDisposable
    {
        /// <inheritdoc/>
        public void Dispose()
        {
            trace.Stages.RemoveAt(trace.Stages.Count - 1);
        }
    }

    /// <summary>Steps that ran, in order.</summary>
    public List<StepRecord> StepRecords = [];

    /// <summary>Stack of currently running step indices.</summary>
    public List<int> StepStack = [];

    /// <summary>Stack of explicit stage names.</summary>
    public List<string> Stages = [];

    /// <summary>Node records by ID.</summary>
    public Dictionary<string, NodeRecord> Nodes = [];

    /// <summary>Node IDs currently believed present, for removal detection.</summary>
    public HashSet<string> LiveNodes = [];

    /// <summary>Model loads, in order.</summary>
    public List<ModelLoadRecord> Loads = [];

    /// <summary>Stack of currently running model load indices.</summary>
    public List<int> LoadStack = [];

    /// <summary>All LoRA applications, in order.</summary>
    public List<string> Loras = [];

    /// <summary>Parameter reads by "id" or "id@section".</summary>
    public Dictionary<string, ParamReadRecord> ParamReads = [];

    /// <summary>Registered prompt texts, mapped to their labels.</summary>
    public Dictionary<string, List<string>> PromptLabels = [];

    /// <summary>Media inputs by node ID.</summary>
    public Dictionary<string, MediaRecord> Media = [];

    /// <summary>Every <see cref="WGNodeData"/> created, for output typing.</summary>
    public List<WGNodeData> Data = [];

    /// <summary>Arguments of every CreateKSampler result, by node ID.</summary>
    public Dictionary<string, JObject> SamplerCalls = [];

    /// <summary>Segment parts processed, in order.</summary>
    public JArray SegmentParts = [];

    /// <summary>Cache hits inside the generator (conditioning, VAE loaders, model loaders).</summary>
    public List<string> CacheHits = [];

    /// <summary>Connection rewires, in order.</summary>
    public List<string> Rewires = [];

    /// <summary>Free-form events, such as early stops.</summary>
    public List<string> Events = [];

    /// <summary>Error that stopped generation, if any.</summary>
    public string Error;

    /// <summary>The built summary, once finished. Shared with <see cref="T2IParamInput.ExtraMeta"/>.</summary>
    public JObject Summary;

    /// <summary>The generation stage nodes are currently attributed to.</summary>
    public string CurrentStage => Stages.Count > 0 ? Stages[^1]
        : Gen.IsImageToVideoSwap ? "videoswap"
        : Gen.IsImageToVideo ? "video"
        : Gen.IsPixelDecoderStage ? "pixeldecoder"
        : Gen.IsRefinerStage ? "refiner"
        : "base";

    /// <summary>Returns the first 8 hex chars of the SHA-256 of some text.</summary>
    public static string Sha8(string text) => Utilities.HashSHA256((text ?? "").EncodeUTF8())[..8];

    /// <summary>Returns the first 8 hex chars of the SHA-256 of some bytes.</summary>
    public static string Sha8(byte[] data) => Utilities.HashSHA256(data ?? [])[..8];

    /// <summary>Formats a workflow path (eg ["4", 0]) as "4:0".</summary>
    public static string PathText(JArray path) => path is null ? null : path.Count == 2 ? $"{path[0]}:{path[1]}" : path.ToString(Formatting.None);

    /// <summary>Begins tracing a generation.</summary>
    public void Begin()
    {
        Gen.UserInput.ParamReadTracker = OnParamRead;
    }

    /// <summary>Records a parameter read.</summary>
    public void OnParamRead(T2IParamType type, int section, object value, string source)
    {
        string key = section == 0 ? type.ID : $"{type.ID}@{section}";
        if (!ParamReads.TryGetValue(key, out ParamReadRecord record))
        {
            record = new() { Type = type, Section = section, FirstValue = value, FirstSource = source };
            ParamReads[key] = record;
        }
        record.LastValue = value;
        record.LastSource = source;
        record.Count++;
    }

    /// <summary>Records a <see cref="WGNodeData"/>, read when the trace finishes.</summary>
    public void TrackData(WGNodeData data)
    {
        Data.Add(data);
    }

    /// <summary>Records that a step began.</summary>
    public void BeginStep(WorkflowGenerator.WorkflowGenStep step, bool isModelGen)
    {
        StepRecord record = new()
        {
            Index = StepRecords.Count,
            Name = step.Name ?? step.Source ?? "(unnamed)",
            Priority = step.Priority,
            Source = step.Source,
            IsModelGen = isModelGen,
            Load = LoadStack.Count > 0 ? LoadStack[^1] : -1
        };
        StepRecords.Add(record);
        StepStack.Add(record.Index);
    }

    /// <summary>Records that the innermost step ended.</summary>
    public void EndStep()
    {
        StepRecord record = StepRecords[StepStack[^1]];
        StepStack.RemoveAt(StepStack.Count - 1);
        if (StepStack.Count > 0)
        {
            return;
        }
        foreach (string id in LiveNodes.ToList())
        {
            if (!Gen.Workflow.ContainsKey(id))
            {
                LiveNodes.Remove(id);
                record.Removed.Add(id);
                if (Nodes.TryGetValue(id, out NodeRecord node))
                {
                    node.RemovedBy = record.Name;
                }
            }
        }
        record.State = new()
        {
            ["model"] = DescribeData(Gen.CurrentModel),
            ["textenc"] = DescribeData(Gen.CurrentTextEnc),
            ["vae"] = DescribeData(Gen.CurrentVae),
            ["media"] = DescribeData(Gen.CurrentMedia),
            ["input_image"] = DescribeData(Gen.BasicInputImage),
            ["loaded_model"] = Gen.FinalLoadedModel?.Name,
            ["mask"] = PathText(Gen.FinalMask),
            ["mask_crop"] = Gen.MaskShrunkInfo?.BoundsNode is null ? null : $"bounds={Gen.MaskShrunkInfo.BoundsNode} mask={Gen.MaskShrunkInfo.CroppedMask} latent={Gen.MaskShrunkInfo.MaskedLatent} image={Gen.MaskShrunkInfo.ScaledImage}",
            ["differential"] = Gen.IsDifferentialDiffusion
        };
    }

    /// <summary>Records that a step stopped all further steps.</summary>
    public void SkippedFurtherSteps()
    {
        Events.Add($"step '{StepRecords[^1].Name}' stopped further steps");
    }

    /// <summary>Records a node written into the workflow.</summary>
    public void NodeCreated(string id, string classType, bool overwrote, string member, string file, int line)
    {
        int step = StepStack.Count > 0 ? StepStack[^1] : -1;
        if (step >= 0)
        {
            StepRecords[step].Created++;
        }
        int overwrites = 0;
        if (overwrote && Nodes.TryGetValue(id, out NodeRecord prior))
        {
            overwrites = prior.Overwrites + 1;
        }
        Nodes[id] = new()
        {
            Id = id,
            ClassType = classType,
            Step = step,
            Stage = CurrentStage,
            Origin = Origin(member, file, line),
            Compat = Gen.FinalLoadedModel?.ModelClass?.CompatClass?.ID,
            Load = LoadStack.Count > 0 ? LoadStack[^1] : -1,
            Overwrites = overwrites
        };
        LiveNodes.Add(id);
    }

    /// <summary>Records that CreateNode returned an existing identical node.</summary>
    public void NodeReused(string id, string member, string file, int line)
    {
        if (Nodes.TryGetValue(id, out NodeRecord node))
        {
            node.ReusedBy.Add($"{Origin(member, file, line)}@{CurrentStage}");
        }
    }

    /// <summary>Formats a code location as member@file:line.</summary>
    public static string Origin(string member, string file, int line)
    {
        return $"{member}@{(file ?? "").Replace('\\', '/').AfterLast('/')}:{line}";
    }

    /// <summary>Records a cache hit inside the generator.</summary>
    public void CacheHit(string kind, string result)
    {
        CacheHits.Add($"{kind} -> {result} @{CurrentStage}");
    }

    /// <summary>Records the start of a model load.</summary>
    public void BeginModelLoad(T2IModel model, string type, string requestedId, int sectionId)
    {
        ModelLoadRecord record = new()
        {
            Index = Loads.Count,
            Step = StepStack.Count > 0 ? StepStack[^1] : -1,
            Stage = CurrentStage,
            Type = type,
            Model = model?.Name,
            ModelClass = model?.ModelClass?.ID,
            Compat = model?.ModelClass?.CompatClass?.ID,
            Section = sectionId,
            RequestedId = requestedId
        };
        Loads.Add(record);
        LoadStack.Add(record.Index);
    }

    /// <summary>Records the end of a model load.</summary>
    public void EndModelLoad(T2IModel model, JArray modelOut, JArray clipOut, JArray vaeOut)
    {
        ModelLoadRecord record = Loads[LoadStack[^1]];
        LoadStack.RemoveAt(LoadStack.Count - 1);
        if (model?.Name != record.Model)
        {
            record.LoadedModel = model?.Name;
        }
        record.ModelOut = PathText(modelOut);
        record.ClipOut = PathText(clipOut);
        record.VaeOut = PathText(vaeOut);
        record.Finished = true;
    }

    /// <summary>Records a model load answered from the generator's loader cache.</summary>
    public void ModelLoadCached(T2IModel model, string type, int sectionId, JArray modelOut, JArray clipOut, JArray vaeOut)
    {
        Loads.Add(new()
        {
            Index = Loads.Count,
            Step = StepStack.Count > 0 ? StepStack[^1] : -1,
            Stage = CurrentStage,
            Type = type,
            Model = model?.Name,
            ModelClass = model?.ModelClass?.ID,
            Compat = model?.ModelClass?.CompatClass?.ID,
            Section = sectionId,
            Cached = true,
            ModelOut = PathText(modelOut),
            ClipOut = PathText(clipOut),
            VaeOut = PathText(vaeOut),
            Finished = true
        });
        CacheHits.Add($"modelloader {model?.Name} {type} -> {PathText(modelOut)} @{CurrentStage}");
    }

    /// <summary>Records a LoRA application.</summary>
    /// <param name="confinement">Section confinement ID the LoRA was applied for.</param>
    /// <param name="lora">The LoRA model.</param>
    /// <param name="weight">Model weight.</param>
    /// <param name="tencWeight">Text encoder weight.</param>
    /// <param name="nodeId">Node that applies it.</param>
    /// <param name="kind">How it is applied: "loader", "hook", or "dynamic-hook".</param>
    public void LoraApplied(int confinement, T2IModel lora, float weight, float tencWeight, string nodeId, string kind)
    {
        string classType = Gen.Workflow.TryGetValue(nodeId, out JToken node) ? $"{node["class_type"]}" : "?";
        string text = $"{T2IParamTypes.CleanModelName(lora?.Name ?? "?")} weight={weight} tenc={tencWeight} section={confinement} {kind} {nodeId}:{classType} @{CurrentStage}";
        Loras.Add(text);
        if (LoadStack.Count > 0)
        {
            Loads[LoadStack[^1]].Loras.Add(text);
        }
    }

    /// <summary>Registers a prompt text that will reach a text encoder node, under a label.</summary>
    public void LabelPrompt(string text, string label)
    {
        if (text is null)
        {
            return;
        }
        List<string> labels = PromptLabels.GetOrCreate(text, () => []);
        if (!labels.Contains(label))
        {
            labels.Add(label);
        }
    }

    /// <summary>Registers a media input embedded into a node.</summary>
    public void LabelMedia(string nodeId, string param, MediaFile media, int? width = null, int? height = null)
    {
        param = (param ?? "").Replace("${", "").Replace("}", "");
        Media[nodeId] = new(param, media, width, height);
    }

    /// <summary>Records the arguments of a sampler created by CreateKSampler.</summary>
    public void SamplerCall(string nodeId, JObject args)
    {
        args["stage"] = CurrentStage;
        args["step"] = StepStack.Count > 0 ? StepRecords[StepStack[^1]].Name : null;
        SamplerCalls[nodeId] = args;
    }

    /// <summary>Records a segment part being processed.</summary>
    public void SegmentPart(PromptRegion.Part part, int index)
    {
        SegmentParts.Add(new JObject()
        {
            ["index"] = index,
            ["stage"] = CurrentStage,
            ["section"] = part.ContextID,
            ["mask_text"] = $"@{CurrentStage}.mask#{Sha8(part.DataText)}",
            ["threshold"] = part.Strength,
            ["creativity"] = part.Strength2,
            ["prompt"] = $"@{CurrentStage}.pos#{Sha8(part.Prompt)}"
        });
    }

    /// <summary>Records a connection rewire.</summary>
    public void Rewired(JArray from, JArray to, int count)
    {
        string step = StepStack.Count > 0 ? StepRecords[StepStack[^1]].Name : "?";
        Rewires.Add($"{PathText(from)} -> {PathText(to)} ({count} inputs) by {step}");
    }

    /// <summary>Marks nodes created inside the returned scope as belonging to the given stage.</summary>
    public StageScope EnterStage(string stage)
    {
        Stages.Add(stage);
        return new(this);
    }

    /// <summary>Records that generation failed, and logs what was traced up to the failure.</summary>
    public void Fail(Exception ex)
    {
        Gen.UserInput.ParamReadTracker = null;
        Error = $"{ex.GetType().Name}: {ex.Message}";
        try
        {
            Logs.Debug($"[WorkflowTrace] generation failed, partial trace:\n{Build().ToString(Formatting.Indented)}\n[WorkflowTrace] detail:\n{BuildDetail().ToString(Formatting.Indented)}");
        }
        catch (Exception traceEx)
        {
            Logs.Debug($"[WorkflowTrace] generation failed and the partial trace could not be built: {traceEx.ReadableString()}");
        }
    }

    /// <summary>Finishes the trace after generation: logs the full summary plus detail, and stores the summary for the image format (<see cref="ForImageMetadata"/>) into the input's ExtraMeta.</summary>
    public void Finish()
    {
        Gen.UserInput.ParamReadTracker = null;
        Summary = Build();
        string logReference = $"[WorkflowTrace] request {Gen.UserInput.UserRequestId}";
        Logs.Debug($"{logReference}:\n{Summary.ToString(Formatting.Indented)}\n[WorkflowTrace] detail:\n{BuildDetail().ToString(Formatting.Indented)}");
        Gen.UserInput.ExtraMeta[MetadataKey] = ForImageMetadata(Summary, ImageFormatOf(Gen.UserInput), logReference);
    }

    /// <summary>Returns the format images of this input are saved in, as Session.ApplyMetadata picks it, without marking any parameter as used.</summary>
    public static string ImageFormatOf(T2IParamInput input)
    {
        if (input.InternalSet.TryGet(T2IParamTypes.ImageFormat, out string format) && !string.IsNullOrWhiteSpace(format))
        {
            return format;
        }
        return input.SourceSession?.User?.Settings?.FileFormat?.ImageFormat ?? Program.ServerSettings?.DefaultUser?.FileFormat?.ImageFormat ?? "PNG";
    }

    /// <summary>Returns what to store in the image metadata for a summary: the full summary for PNG, and for EXIF formats when it fits <see cref="MaxExifSummaryChars"/>.
    /// Otherwise a notice saying to save as PNG for the full trace and where the server log has it, with the workflow and model identity, so the image can still be matched to the log.</summary>
    /// <param name="summary">The full summary.</param>
    /// <param name="format">Image format, as <see cref="T2IParamTypes.ImageFormat"/> values ("PNG", "JPG", "WEBP", ...).</param>
    /// <param name="logReference">How the server debug log labels this trace.</param>
    /// <param name="maxExifChars">Limit for EXIF formats, or -1 for <see cref="MaxExifSummaryChars"/>.</param>
    public static JObject ForImageMetadata(JObject summary, string format, string logReference, int maxExifChars = -1)
    {
        if (maxExifChars < 0)
        {
            maxExifChars = MaxExifSummaryChars;
        }
        if (format == "PNG")
        {
            return summary;
        }
        int length = T2IParamInput.MetadataToString(summary).Length;
        if (length <= maxExifChars)
        {
            return summary;
        }
        return new JObject()
        {
            ["omitted"] = new JObject()
            {
                ["reason"] = $"the trace is {length} characters; {format} images keep metadata in EXIF, which holds {maxExifChars}",
                ["full_trace"] = $"save as PNG to keep the full trace in the image; the server debug log has it as '{logReference}'"
            },
            ["workflow"] = summary["workflow"]?.DeepClone(),
            ["model"] = summary["model"]?.DeepClone()
        };
    }

    /// <summary>Records the workflow text actually sent to ComfyUI.</summary>
    /// <param name="workflow">The final workflow JSON text.</param>
    /// <param name="changedAfterGenerate">True if anything rewrote the workflow text after the generator returned it.</param>
    public void RecordSent(string workflow, bool changedAfterGenerate)
    {
        if (Summary?["workflow"] is not JObject info)
        {
            return;
        }
        info["sent_sha256"] = Utilities.HashSHA256(workflow.EncodeUTF8());
        info["sent_length"] = workflow.Length;
        info["rewritten_after_generate"] = changedAfterGenerate;
        Logs.Debug($"[WorkflowTrace] request {Gen.UserInput.UserRequestId}: sent workflow sha256={info["sent_sha256"]} length={workflow.Length} rewritten_after_generate={changedAfterGenerate}");
    }

    /// <summary>Records a workflow that did not come from the generator (custom or raw workflows): logs its description, and stores it for the image format (<see cref="ForImageMetadata"/>) into the input's ExtraMeta.</summary>
    public static void RecordExternalWorkflow(T2IParamInput input, string source, string workflow)
    {
        JObject summary = DescribeExternalWorkflow(source, workflow);
        string logReference = $"[WorkflowTrace] request {input.UserRequestId}";
        Logs.Debug($"{logReference} ({source} workflow):\n{summary.ToString(Formatting.Indented)}");
        input.ExtraMeta[MetadataKey] = ForImageMetadata(summary, ImageFormatOf(input), logReference);
    }

    /// <summary>Builds a trace summary for a workflow that did not come from the generator (custom or raw workflows).</summary>
    public static JObject DescribeExternalWorkflow(string source, string workflow)
    {
        JObject result = new()
        {
            ["version"] = FormatVersion,
            ["workflow"] = new JObject()
            {
                ["source"] = source,
                ["sent_sha256"] = Utilities.HashSHA256((workflow ?? "").EncodeUTF8()),
                ["sent_length"] = workflow?.Length ?? 0
            }
        };
        try
        {
            JObject parsed = workflow.ParseToJson();
            WorkflowTrace reader = new(null);
            JArray nodes = [];
            foreach (JProperty property in parsed.Properties())
            {
                if (property.Value is JObject node)
                {
                    nodes.Add(reader.DescribeNode(property.Name, node, null));
                }
            }
            result["nodes"] = nodes;
        }
        catch (Exception ex)
        {
            result["parse_error"] = ex.Message;
        }
        return result;
    }

    /// <summary>Describes a <see cref="WGNodeData"/> as "node:slot TYPE compat WxH".</summary>
    public static string DescribeData(WGNodeData data)
    {
        if (data is null)
        {
            return null;
        }
        string result = $"{PathText(data.Path)} {data.DataType}";
        if (data.Compat is not null)
        {
            result += $" {data.Compat.ID}";
        }
        if (data.Width is not null || data.Height is not null)
        {
            result += $" {data.Width?.ToString() ?? "?"}x{data.Height?.ToString() ?? "?"}";
        }
        if (data.Frames is not null)
        {
            result += $" frames={data.Frames}";
        }
        if (data.FPS is not null)
        {
            result += $" fps={data.FPS.ToString(Formatting.None)}";
        }
        if (data.MayHaveAlpha)
        {
            result += " alpha";
        }
        if (data.AttachedAudio is not null)
        {
            result += $" audio={PathText(data.AttachedAudio.Path)}";
        }
        return result;
    }

    /// <summary>Typing of a single node output slot, merged from every <see cref="WGNodeData"/> that referenced it.</summary>
    public class SlotType
    {
        /// <summary>Swarm data types seen for this slot.</summary>
        public List<string> Types = [];

        /// <summary>Compat class IDs seen for this slot.</summary>
        public List<string> Compat = [];

        /// <summary>Known image-space width, height, and frame count.</summary>
        public int? Width, Height, Frames;

        /// <summary>True if any reference marked the slot as possibly having alpha.</summary>
        public bool Alpha;

        /// <summary>Formats the slot type.</summary>
        public override string ToString()
        {
            string result = Types.JoinString("|");
            if (Compat.Count > 0)
            {
                result += $"/{Compat.JoinString("|")}";
            }
            if (Width is not null || Height is not null)
            {
                result += $"/{Width?.ToString() ?? "?"}x{Height?.ToString() ?? "?"}";
            }
            if (Frames is not null)
            {
                result += $"/{Frames}f";
            }
            if (Alpha)
            {
                result += "/alpha";
            }
            return result;
        }
    }

    /// <summary>Merged typing of every node output slot, by "node:slot".</summary>
    public Dictionary<string, SlotType> SlotTypes = [];

    /// <summary>Number of output slots typed in <see cref="SlotTypes"/>, by node ID (highest typed slot index plus one).</summary>
    public Dictionary<string, int> TypedSlotCounts = [];

    /// <summary>Builds <see cref="SlotTypes"/> and <see cref="TypedSlotCounts"/> from <see cref="Data"/>.</summary>
    public void BuildSlotTypes()
    {
        SlotTypes = [];
        TypedSlotCounts = [];
        foreach (WGNodeData data in Data)
        {
            if (data.Path is null || data.Path.Count != 2 || data.Path[1].Type != JTokenType.Integer)
            {
                continue;
            }
            string nodeId = $"{data.Path[0]}";
            TypedSlotCounts[nodeId] = Math.Max(TypedSlotCounts.GetValueOrDefault(nodeId, 0), data.Path[1].Value<int>() + 1);
            SlotType slot = SlotTypes.GetOrCreate(PathText(data.Path), () => new());
            if (data.DataType is not null && !slot.Types.Contains(data.DataType))
            {
                slot.Types.Add(data.DataType);
            }
            if (data.Compat is not null && !slot.Compat.Contains(data.Compat.ID))
            {
                slot.Compat.Add(data.Compat.ID);
            }
            slot.Width = data.Width ?? slot.Width;
            slot.Height = data.Height ?? slot.Height;
            slot.Frames = data.Frames ?? slot.Frames;
            slot.Alpha |= data.MayHaveAlpha;
        }
    }

    /// <summary>Returns the Comfy output types for a node class, from <see cref="ObjectInfo"/>.</summary>
    public JArray ComfyOutputTypes(string classType)
    {
        if (ObjectInfo is null || !ObjectInfo.TryGetValue(classType, out JToken info))
        {
            return null;
        }
        return info["output"] as JArray;
    }

    /// <summary>Returns the Comfy declared type of an input of a node class, from <see cref="ObjectInfo"/>, or null if unknown or a choice list.</summary>
    public string ComfyInputType(string classType, string input)
    {
        if (ObjectInfo is null || !ObjectInfo.TryGetValue(classType, out JToken info) || info["input"] is not JObject inputs)
        {
            return null;
        }
        foreach (string group in new[] { "required", "optional" })
        {
            if (inputs[group] is JObject set && set.TryGetValue(input, out JToken decl) && decl is JArray declArr && declArr.Count > 0 && declArr[0].Type == JTokenType.String)
            {
                return $"{declArr[0]}";
            }
        }
        return null;
    }

    /// <summary>Returns true if <see cref="ObjectInfo"/> lists the node class and it declares the input as required, optional or hidden. Returns null if the class is not listed.</summary>
    public bool? ComfyDeclaresInput(string classType, string input)
    {
        if (ObjectInfo is null || !ObjectInfo.TryGetValue(classType, out JToken info) || info["input"] is not JObject inputs)
        {
            return null;
        }
        return new[] { "required", "optional", "hidden" }.Any(group => inputs[group] is JObject set && set.ContainsKey(input));
    }

    /// <summary>Returns true if the value is a link to another node, ie ["id", slot].</summary>
    public static bool IsLink(JToken value)
    {
        return value is JArray arr && arr.Count == 2 && arr[0].Type == JTokenType.String && arr[1].Type == JTokenType.Integer;
    }

    /// <summary>Formats a string node input with placeholders for prompts, media, and long text.</summary>
    public string FormatString(string nodeId, string key, string value)
    {
        if (key.EndsWith("_base64"))
        {
            if (Media.TryGetValue(nodeId, out MediaRecord media))
            {
                return $"@{media.Param}#{Sha8(value)}{MediaSize(media)}";
            }
            return $"b64#{Sha8(value)}({value.Length})";
        }
        if (PromptLabels.TryGetValue(value, out List<string> labels))
        {
            return $"{labels.Select(l => $"@{l}").JoinString("=")}#{Sha8(value)}";
        }
        if (PromptInputNames.Contains(key))
        {
            foreach ((string text, List<string> textLabels) in PromptLabels)
            {
                if (value.Length > 0 && text.Contains(value))
                {
                    return $"{textLabels.Select(l => $"@{l}").JoinString("=")}~part#{Sha8(value)}";
                }
            }
            return $"#{Sha8(value)}({value.Length})";
        }
        if (value.Length > 120)
        {
            return $"#{Sha8(value)}({value.Length})";
        }
        return value;
    }

    /// <summary>Returns " WxH mime" for a media record, reading the image size if it was not resized.</summary>
    public static string MediaSize(MediaRecord media)
    {
        int? width = media.Width, height = media.Height;
        if ((width is null || height is null) && media.Media is ImageFile img && img.Type.MetaType == MediaMetaType.Image)
        {
            try
            {
                (int imgWidth, int imgHeight) = img.GetResolution();
                width = imgWidth;
                height = imgHeight;
            }
            catch (Exception)
            {
                width = null;
                height = null;
            }
        }
        string size = width is null || height is null ? "" : $" {width}x{height}";
        return $"{size} {media.Media?.Type?.MimeType ?? "?"}";
    }

    /// <summary>Formats a parameter value with placeholders for prompts, models, and media.</summary>
    public static JToken FormatParamValue(T2IParamType type, object value)
    {
        if (value is null)
        {
            return null;
        }
        if (type?.ViewType == ParamViewType.PROMPT && value is string prompt)
        {
            return $"@{type.ID}#{Sha8(prompt)}({prompt.Length})";
        }
        return value switch
        {
            string str => str.Length > 120 ? $"#{Sha8(str)}({str.Length})" : str,
            bool or int or long or float or double => JToken.FromObject(value),
            T2IModel model => $"{T2IParamTypes.CleanModelName(model.Name)} [{model.ModelClass?.ID ?? "?"}]",
            MediaFile media => $"media#{Sha8(media.RawData)} {media.Type?.MimeType ?? "?"}",
            List<string> list => new JArray(list.Select(s => (JToken)(s.Length > 120 ? $"#{Sha8(s)}({s.Length})" : s))),
            List<T2IModel> models => new JArray(models.Select(m => (JToken)T2IParamTypes.CleanModelName(m.Name))),
            JToken tok => tok.ToString(Formatting.None),
            System.Collections.IEnumerable items => new JArray(items.Cast<object>().Select(i => i is MediaFile m ? (JToken)$"media#{Sha8(m.RawData)} {m.Type?.MimeType ?? "?"}" : (JToken)$"{i}")),
            _ => $"{value}"
        };
    }

    /// <summary>Describes one node as a single line: "id:Class@stage &lt;origin&gt; [scalars] {links} =&gt; outputs".</summary>
    /// <param name="id">Node ID.</param>
    /// <param name="node">Node JSON.</param>
    /// <param name="relativeStage">If non-null, the stage tag is only shown when the node's stage differs from this, and the origin is omitted.</param>
    public string DescribeNode(string id, JObject node, string relativeStage)
    {
        string classType = $"{node["class_type"]}";
        StringBuilder sb = new();
        sb.Append($"{id}:{classType}");
        NodeRecord record = Nodes.GetValueOrDefault(id);
        if (record is not null && record.Stage != relativeStage)
        {
            sb.Append($"@{record.Stage}");
        }
        if (record is not null && relativeStage is null)
        {
            sb.Append($" <{record.Origin}>");
        }
        else if (record is null && Gen is not null)
        {
            sb.Append(" <untracked>");
        }
        List<string> scalars = [];
        List<string> links = [];
        if (node["inputs"] is JObject inputs)
        {
            foreach (JProperty input in inputs.Properties())
            {
                if (IsLink(input.Value))
                {
                    links.Add($"{input.Name}<{input.Value[0]}:{input.Value[1]}");
                }
                else if (input.Value.Type == JTokenType.Null)
                {
                    continue;
                }
                else if (input.Value.Type == JTokenType.String)
                {
                    scalars.Add($"{input.Name}={FormatString(id, input.Name, $"{input.Value}")}");
                }
                else
                {
                    string raw = input.Value.ToString(Formatting.None);
                    scalars.Add($"{input.Name}={(raw.Length > 120 ? $"#{Sha8(raw)}({raw.Length})" : raw)}");
                }
            }
        }
        if (scalars.Count > 0)
        {
            sb.Append($" [{scalars.JoinString(", ")}]");
        }
        if (links.Count > 0)
        {
            sb.Append($" {{{links.JoinString(", ")}}}");
        }
        List<string> outputs = [];
        JArray comfyTypes = ComfyOutputTypes(classType);
        int slotCount = Math.Max(comfyTypes?.Count ?? 0, TypedSlotCounts.GetValueOrDefault(id, 0));
        for (int slot = 0; slot < slotCount; slot++)
        {
            string swarmType = SlotTypes.TryGetValue($"{id}:{slot}", out SlotType slotType) ? slotType.ToString() : null;
            string comfyType = comfyTypes is not null && slot < comfyTypes.Count ? $"{comfyTypes[slot]}" : null;
            if (swarmType is not null || comfyType is not null)
            {
                outputs.Add($"{slot}:{(comfyType is not null && swarmType is not null ? $"{comfyType}={swarmType}" : swarmType ?? comfyType)}");
            }
        }
        if (outputs.Count > 0)
        {
            sb.Append($" => {outputs.JoinString(", ")}");
        }
        return sb.ToString();
    }

    /// <summary>Returns the upstream nodes feeding a link, nearest first. Recurses only through nodes of the given stage; nodes of other stages and other samplers are listed as boundaries without recursing.</summary>
    public JArray Chain(JArray link, string stage, JObject workflow)
    {
        JArray result = [];
        HashSet<string> seen = [];
        Queue<string> queue = new();
        queue.Enqueue($"{link[0]}");
        while (queue.Count > 0 && result.Count < MaxChainNodes)
        {
            string id = queue.Dequeue();
            if (!seen.Add(id) || workflow[id] is not JObject node)
            {
                continue;
            }
            result.Add(DescribeNode(id, node, stage));
            NodeRecord record = Nodes.GetValueOrDefault(id);
            bool sameStage = record is null || record.Stage == stage;
            if (!sameStage || SamplerClasses.Contains($"{node["class_type"]}"))
            {
                continue;
            }
            if (node["inputs"] is JObject inputs)
            {
                foreach (JProperty input in inputs.Properties())
                {
                    if (IsLink(input.Value))
                    {
                        queue.Enqueue($"{input.Value[0]}");
                    }
                }
            }
        }
        if (queue.Any(id => !seen.Contains(id)))
        {
            result.Add($"... (truncated at {MaxChainNodes} nodes)");
        }
        return result;
    }

    /// <summary>Returns true if the node class is an output node (sends results back to Swarm or saves them).</summary>
    public bool IsOutputNode(string classType)
    {
        if (ObjectInfo is not null && ObjectInfo.TryGetValue(classType, out JToken info) && info["output_node"] is JToken outFlag && outFlag.Type == JTokenType.Boolean)
        {
            return outFlag.Value<bool>();
        }
        return classType.Contains("Save") || classType == "PreviewImage";
    }

    /// <summary>Returns the shortest downstream path from a node to every output node it reaches.</summary>
    public JArray Downstream(string startId, JObject workflow, Dictionary<string, List<string>> consumers)
    {
        JArray result = [];
        Dictionary<string, string> parent = new() { [startId] = null };
        Queue<string> queue = new();
        queue.Enqueue(startId);
        while (queue.Count > 0)
        {
            string id = queue.Dequeue();
            if (id != startId && IsOutputNode($"{workflow[id]?["class_type"]}"))
            {
                List<string> path = [];
                for (string cur = id; cur is not null; cur = parent[cur])
                {
                    path.Add($"{cur}:{workflow[cur]?["class_type"]}");
                }
                path.Reverse();
                result.Add(path.JoinString(" > "));
                continue;
            }
            foreach (string next in consumers.GetValueOrDefault(id, []))
            {
                if (!parent.ContainsKey(next))
                {
                    parent[next] = id;
                    queue.Enqueue(next);
                }
            }
        }
        return result;
    }

    /// <summary>Builds the trace summary.</summary>
    public JObject Build()
    {
        JObject workflow = Gen.Workflow ?? [];
        BuildSlotTypes();
        Dictionary<string, List<string>> consumers = [];
        foreach (JProperty property in workflow.Properties())
        {
            if (property.Value["inputs"] is JObject inputs)
            {
                foreach (JProperty input in inputs.Properties())
                {
                    if (IsLink(input.Value))
                    {
                        List<string> list = consumers.GetOrCreate($"{input.Value[0]}", () => []);
                        if (!list.Contains(property.Name))
                        {
                            list.Add(property.Name);
                        }
                    }
                }
            }
        }
        T2IModel mainModel = Gen.UserInput.InternalSet.ValuesInput.GetValueOrDefault(T2IParamTypes.Model.Type.ID) as T2IModel;
        JObject result = new()
        {
            ["version"] = FormatVersion,
            ["workflow"] = new JObject()
            {
                ["source"] = "generated",
                ["sha256"] = Utilities.HashSHA256(workflow.ToString().EncodeUTF8()),
                ["nodes"] = workflow.Count,
                ["nodes_created"] = Nodes.Count,
                ["nodes_removed"] = Nodes.Values.Count(n => n.RemovedBy is not null),
                ["features"] = new JArray(Gen.Features.OrderBy(f => f).Select(f => (JToken)f))
            },
            ["model"] = mainModel is null ? null : $"{T2IParamTypes.CleanModelName(mainModel.Name)} [{mainModel.ModelClass?.ID ?? "?"} / {mainModel.ModelClass?.CompatClass?.ID ?? "?"}]"
        };
        if (Error is not null)
        {
            result["error"] = Error;
        }
        JArray steps = [];
        foreach (StepRecord step in StepRecords)
        {
            string text = $"{(step.IsModelGen ? $"  load#{step.Load} modelgen" : "step")} {step.Priority} {step.Name} ({step.Source}) created={step.Created}";
            if (step.Removed.Count > 0)
            {
                text += $" removed=[{step.Removed.JoinString(",")}]";
            }
            steps.Add(text);
        }
        result["steps"] = steps;
        if (Events.Count > 0)
        {
            result["events"] = new JArray(Events.Select(e => (JToken)e));
        }
        JArray loads = [];
        foreach (ModelLoadRecord load in Loads)
        {
            JObject loadObj = new()
            {
                ["index"] = load.Index,
                ["stage"] = load.Stage,
                ["type"] = load.Type,
                ["model"] = T2IParamTypes.CleanModelName(load.Model ?? "?"),
                ["class"] = load.ModelClass,
                ["compat"] = load.Compat,
                ["section"] = load.Section,
                ["cached"] = load.Cached,
                ["out"] = $"model={load.ModelOut} textenc={load.ClipOut} vae={load.VaeOut}"
            };
            if (load.LoadedModel is not null)
            {
                loadObj["loaded_model"] = T2IParamTypes.CleanModelName(load.LoadedModel);
            }
            if (load.RequestedId is not null)
            {
                loadObj["requested_id"] = load.RequestedId;
            }
            if (load.Loras.Count > 0)
            {
                loadObj["loras"] = new JArray(load.Loras.Select(l => (JToken)l));
            }
            if (!load.Finished)
            {
                loadObj["finished"] = false;
            }
            if (load.Step >= 0)
            {
                loadObj["step"] = StepRecords[load.Step].Name;
            }
            loads.Add(loadObj);
        }
        result["model_loads"] = loads;
        result["loras"] = new JArray(Loras.Select(l => (JToken)l));
        JArray samplers = [];
        foreach (JProperty property in workflow.Properties())
        {
            string classType = $"{property.Value["class_type"]}";
            if (!SamplerClasses.Contains(classType) && !SamplerCalls.ContainsKey(property.Name))
            {
                continue;
            }
            JObject node = property.Value as JObject;
            string stage = Nodes.GetValueOrDefault(property.Name)?.Stage;
            JObject samplerObj = new()
            {
                ["node"] = DescribeNode(property.Name, node, null)
            };
            if (SamplerCalls.TryGetValue(property.Name, out JObject args))
            {
                samplerObj["call"] = args;
            }
            JObject chains = [];
            if (node["inputs"] is JObject inputs)
            {
                foreach (JProperty input in inputs.Properties())
                {
                    if (IsLink(input.Value))
                    {
                        chains[input.Name] = Chain(input.Value as JArray, stage, workflow);
                    }
                }
            }
            samplerObj["inputs"] = chains;
            JArray downstream = Downstream(property.Name, workflow, consumers);
            samplerObj["to_outputs"] = downstream.Count > 0 ? downstream : "(reaches no output node)";
            samplers.Add(samplerObj);
        }
        foreach ((string id, JObject args) in SamplerCalls)
        {
            if (!workflow.ContainsKey(id))
            {
                samplers.Add(new JObject() { ["node"] = $"{id} (removed by {Nodes.GetValueOrDefault(id)?.RemovedBy ?? "?"})", ["call"] = args });
            }
        }
        result["samplers"] = samplers;
        if (SegmentParts.Count > 0)
        {
            result["segments"] = SegmentParts;
        }
        JObject nodesByStage = [];
        foreach (JProperty property in workflow.Properties())
        {
            if (property.Value is not JObject node)
            {
                continue;
            }
            string stage = Nodes.GetValueOrDefault(property.Name)?.Stage ?? "untracked";
            if (nodesByStage[stage] is not JArray stageNodes)
            {
                stageNodes = [];
                nodesByStage[stage] = stageNodes;
            }
            stageNodes.Add(DescribeNode(property.Name, node, null));
        }
        result["nodes"] = nodesByStage;
        JArray removed = [];
        foreach (NodeRecord node in Nodes.Values.Where(n => n.RemovedBy is not null))
        {
            removed.Add($"{node.Id}:{node.ClassType}@{node.Stage} <{node.Origin}> removed by {node.RemovedBy}");
        }
        if (removed.Count > 0)
        {
            result["removed"] = removed;
        }
        if (Rewires.Count > 0)
        {
            result["rewires"] = new JArray(Rewires.Select(r => (JToken)r));
        }
        if (CacheHits.Count > 0)
        {
            result["cache_hits"] = new JArray(CacheHits.Select(c => (JToken)c));
        }
        JArray untracked = [];
        foreach (JProperty property in workflow.Properties())
        {
            if (!Nodes.ContainsKey(property.Name))
            {
                untracked.Add($"{property.Name}:{property.Value["class_type"]}");
            }
        }
        if (untracked.Count > 0)
        {
            result["untracked_nodes"] = untracked;
        }
        if (ObjectInfo is not null)
        {
            JArray typeErrors = [];
            JArray undeclared = [];
            JArray unknownClasses = [];
            foreach (JProperty property in workflow.Properties())
            {
                string classType = $"{property.Value["class_type"]}";
                if (!ObjectInfo.ContainsKey(classType))
                {
                    unknownClasses.Add($"{property.Name}:{classType}");
                }
                if (property.Value["inputs"] is not JObject inputs)
                {
                    continue;
                }
                foreach (JProperty input in inputs.Properties())
                {
                    if (ComfyDeclaresInput(classType, input.Name) == false)
                    {
                        undeclared.Add($"{property.Name}:{classType}.{input.Name}");
                    }
                    if (!IsLink(input.Value) || workflow[$"{input.Value[0]}"] is not JObject source)
                    {
                        continue;
                    }
                    string expected = ComfyInputType(classType, input.Name);
                    JArray sourceTypes = ComfyOutputTypes($"{source["class_type"]}");
                    int slot = input.Value[1].Value<int>();
                    if (expected is null || sourceTypes is null || slot >= sourceTypes.Count)
                    {
                        continue;
                    }
                    string actual = $"{sourceTypes[slot]}";
                    if (expected != "*" && actual != "*" && !expected.SplitFast(',').Contains(actual) && !actual.SplitFast(',').Contains(expected))
                    {
                        typeErrors.Add($"{property.Name}:{classType}.{input.Name} expects {expected}, gets {actual} from {input.Value[0]}:{slot}:{source["class_type"]}");
                    }
                }
            }
            result["type_errors"] = typeErrors;
            result["undeclared_inputs"] = undeclared;
            result["classes_missing_from_object_info"] = unknownClasses;
        }
        JObject prompts = [];
        foreach ((string text, List<string> labels) in PromptLabels)
        {
            foreach (string label in labels)
            {
                prompts[$"@{label}#{Sha8(text)}"] = text.Length;
            }
        }
        result["prompts"] = prompts;
        JObject media = [];
        foreach ((string nodeId, MediaRecord record) in Media)
        {
            string value = workflow[nodeId]?["inputs"] is JObject mediaInputs ? mediaInputs.Properties().Where(p => p.Name.EndsWith("_base64")).Select(p => $"{p.Value}").FirstOrDefault() : null;
            media[nodeId] = $"@{record.Param}#{(value is null ? "?" : Sha8(value))}{MediaSize(record)}";
        }
        result["media"] = media;
        JObject parameters = [];
        foreach ((string key, ParamReadRecord read) in ParamReads.OrderBy(p => p.Key))
        {
            JToken first = FormatParamValue(read.Type, read.FirstValue);
            JToken last = FormatParamValue(read.Type, read.LastValue);
            string firstText = first?.ToString(Formatting.None) ?? "null";
            string lastText = last?.ToString(Formatting.None) ?? "null";
            parameters[key] = firstText == lastText && read.FirstSource == read.LastSource ? $"{firstText} ({read.FirstSource})" : $"{firstText} ({read.FirstSource}) then {lastText} ({read.LastSource})";
        }
        result["params_read"] = parameters;
        if (Gen.UserInput.SectionParamOverrides.Count > 0)
        {
            JObject sections = [];
            foreach ((int section, T2IParamSet set) in Gen.UserInput.SectionParamOverrides)
            {
                JObject values = [];
                foreach ((string key, object value) in set.ValuesInput)
                {
                    T2IParamTypes.TryGetType(key, out T2IParamType type, Gen.UserInput);
                    values[key] = FormatParamValue(type, value);
                }
                sections[$"{section}"] = values;
            }
            result["section_overrides"] = sections;
        }
        return result;
    }

    /// <summary>Builds the log-only trace detail: per-step tracker state, node reuse, and the raw input values.</summary>
    public JObject BuildDetail()
    {
        JObject states = [];
        foreach (StepRecord step in StepRecords.Where(s => s.State is not null))
        {
            states[$"{step.Index} {step.Name}"] = step.State;
        }
        JArray reuses = [];
        foreach (NodeRecord node in Nodes.Values.Where(n => n.ReusedBy.Count > 0 || n.Overwrites > 0))
        {
            reuses.Add($"{node.Id}:{node.ClassType} reused by [{node.ReusedBy.JoinString(", ")}] overwrites={node.Overwrites}");
        }
        JObject input = [];
        foreach ((string key, object value) in Gen.UserInput.InternalSet.ValuesInput)
        {
            T2IParamTypes.TryGetType(key, out T2IParamType type, Gen.UserInput);
            input[key] = FormatParamValue(type, value);
        }
        return new JObject()
        {
            ["step_states"] = states,
            ["node_reuse"] = reuses,
            ["input"] = input
        };
    }
}
