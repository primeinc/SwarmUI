using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using FreneticUtilities.FreneticExtensions;
using FreneticUtilities.FreneticToolkit;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUI.Builtin_ComfyUIBackend;

/// <summary>Records how a <see cref="WorkflowGenerator"/> built its workflow, without altering the workflow.
/// Covers the steps that ran, the stage and origin of every node, the Swarm type of every node output, every model load and LoRA, every parameter read, prompt and media placeholders, node removals, rewires and in-place edits with the step that made each, and the input chains of every sampler.
/// Exists only for generations where <see cref="Settings.AddDebugData"/> is enabled, or where a caller attaches one explicitly.</summary>
public class WorkflowTrace(WorkflowGenerator gen)
{
    /// <summary>Version of the trace output format.</summary>
    public const int FormatVersion = MetadataHelper.WorkflowTraceVersion;

    /// <summary>ExtraMeta key the trace summary is stored under.</summary>
    public const string MetadataKey = MetadataHelper.WorkflowTraceKey;

    /// <summary>Node classes treated as samplers for chain analysis. An extension with its own sampler node adds the class here when it loads.</summary>
    public static HashSet<string> SamplerClasses = ["KSampler", "KSamplerAdvanced", "SwarmKSampler", "SamplerCustom", "SamplerCustomAdvanced", "SwarmUnsampler"];

    /// <summary>Node input names that carry prompt text. Text in these inputs is never written raw. An extension with its own text encoder node adds the input name here when it loads.
    /// The text of every prompt-view parameter read is also never written raw, under any input name (<see cref="PromptParamTexts"/>).</summary>
    public static HashSet<string> PromptInputNames = ["text", "prompt", "text_g", "text_l", "lyrics", "caption", "match_text", "style", "tags"];

    /// <summary>Maximum number of nodes listed per sampler input chain.</summary>
    public const int MaxChainNodes = 48;

    /// <summary>The generator being traced.</summary>
    public WorkflowGenerator Gen = gen;

    /// <summary>Optional ComfyUI object_info data, used to add Comfy output types and to check links and inputs the way ComfyUI validates them (<see cref="ComfyInputValidation"/>).</summary>
    public JObject ObjectInfo;

    /// <summary>If true, string inputs that could be prompt text (any whitespace, or longer than 64 characters) are hashed whatever their input name, except model file names. Set for workflows that did not come from the generator, which has no prompt labels for them.</summary>
    public bool HashFreeText;

    /// <summary>File extensions of model names, kept readable when <see cref="HashFreeText"/> is set.</summary>
    public static readonly string[] ModelFileExtensions = [".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx"];

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

        /// <summary>IDs of nodes already in the workflow that this step changed in place.</summary>
        public List<string> Edited = [];

        /// <summary>Node IDs that entered the workflow during this step without <see cref="WorkflowGenerator.CreateNode(string, Action{string, JObject}, string, string, string, int)"/>.</summary>
        public List<string> Inserted = [];

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

        /// <summary>Code locations that received this node from the CreateNode dedup instead of a new node, with the stage of each.</summary>
        public List<NodeReuse> ReusedBy = [];

        /// <summary>Name of the step that removed it from the workflow, if removed.</summary>
        public string RemovedBy;
    }

    /// <summary>A CreateNode call that received an existing node instead of a new one.</summary>
    /// <param name="Origin">Calling code location, as member@file:line.</param>
    /// <param name="Stage">Generation stage of the call.</param>
    public record class NodeReuse(string Origin, string Stage)
    {
        /// <summary>Returns the record's JSON form.</summary>
        public JObject ToJson() => new() { ["origin"] = Origin, ["stage"] = Stage };
    }

    /// <summary>A LoRA application.</summary>
    /// <param name="Lora">LoRA name, without its file extension.</param>
    /// <param name="Weight">Model weight.</param>
    /// <param name="TencWeight">Text encoder weight.</param>
    /// <param name="Section">Section confinement ID the LoRA was applied for, or -1 for a dynamic hook.</param>
    /// <param name="Kind">How it is applied: "loader", "hook", or "dynamic-hook".</param>
    /// <param name="Node">The applying node, as "id:Class".</param>
    /// <param name="Stage">Generation stage it was applied in.</param>
    public record class LoraRecord(string Lora, float Weight, float TencWeight, int Section, string Kind, string Node, string Stage)
    {
        /// <summary>Returns the record's JSON form.</summary>
        public JObject ToJson() => new() { ["lora"] = Lora, ["weight"] = Weight, ["tenc_weight"] = TencWeight, ["section"] = Section, ["kind"] = Kind, ["node"] = Node, ["stage"] = Stage };
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
        public List<LoraRecord> Loras = [];

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

        /// <summary>Where the first value came from.</summary>
        public T2IParamInput.ParamReadSource FirstSource;

        /// <summary>Last value returned.</summary>
        public object LastValue;

        /// <summary>Where the last value came from.</summary>
        public T2IParamInput.ParamReadSource LastSource;

        /// <summary>Number of reads.</summary>
        public int Count;

        /// <summary>Distinct values returned, in first-read order, up to <see cref="MaxDistinctParamValues"/>.</summary>
        public List<object> Values = [];
    }

    /// <summary>Most distinct values kept per parameter read record.</summary>
    public const int MaxDistinctParamValues = 8;

    /// <summary>A media input embedded into a node.</summary>
    /// <param name="Param">Source parameter name, eg 'initimage' or 'promptimages.0'.</param>
    /// <param name="Media">The media file.</param>
    /// <param name="Width">Embedded width, if resized.</param>
    /// <param name="Height">Embedded height, if resized.</param>
    public record class MediaRecord(string Param, MediaFile Media, int? Width, int? Height);

    /// <summary>What a media file's header says (<see cref="MediaHeader"/>); each fact is null when the header does not give it.</summary>
    /// <param name="Width">Image width.</param>
    /// <param name="Height">Image height.</param>
    /// <param name="Frames">Frame count of an image with more than one frame.</param>
    /// <param name="DurationSeconds">Audio or video duration, rounded to a hundredth of a second.</param>
    /// <param name="SampleRate">Audio sample rate, in Hz.</param>
    /// <param name="Channels">Audio channel count.</param>
    public record class MediaFacts(int? Width = null, int? Height = null, int? Frames = null, double? DurationSeconds = null, double? SampleRate = null, int? Channels = null)
    {
        /// <summary>Facts of a file whose header says nothing readable.</summary>
        public static readonly MediaFacts None = new();

        /// <summary>Adds each known fact to a JSON object, as "width", "height", "frames", "duration_s", "sample_rate" and "channels".</summary>
        public void AddTo(JObject target)
        {
            if (Width is int width)
            {
                target["width"] = width;
            }
            if (Height is int height)
            {
                target["height"] = height;
            }
            if (Frames is int frames)
            {
                target["frames"] = frames;
            }
            if (DurationSeconds is double duration)
            {
                target["duration_s"] = duration;
            }
            if (SampleRate is double rate)
            {
                target["sample_rate"] = rate;
            }
            if (Channels is int channels)
            {
                target["channels"] = channels;
            }
        }
    }

    /// <summary>A change to a node already in the workflow: an input set, changed or removed, its class changed, or the whole node replaced without CreateNode.</summary>
    /// <param name="Step">Index of the innermost step running when the change was made, or -1.</param>
    /// <param name="NodeId">Workflow node ID.</param>
    /// <param name="ClassType">Node class after the change.</param>
    /// <param name="Field">Input name, "class_type", or "(node)" for a replaced node.</param>
    /// <param name="Before">Value before, or null if absent.</param>
    /// <param name="After">Value after, or null if removed.</param>
    public record class EditRecord(int Step, string NodeId, string ClassType, string Field, JToken Before, JToken After);

    /// <summary>A node as it was at the last <see cref="Checkpoint"/>.</summary>
    /// <param name="Node">The node object in the workflow.</param>
    /// <param name="ClassType">Its class.</param>
    /// <param name="Inputs">A copy of its inputs.</param>
    public record class NodeSnapshot(JObject Node, string ClassType, JObject Inputs);

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
    public List<LoraRecord> Loras = [];

    /// <summary>Parameter reads by "id" or "id@section".</summary>
    public Dictionary<string, ParamReadRecord> ParamReads = [];

    /// <summary>Registered prompt texts, mapped to their labels.</summary>
    public Dictionary<string, List<string>> PromptLabels = [];

    /// <summary>Text values of prompt-view parameters read (<see cref="ParamViewType.PROMPT"/>, such as the negative prompt or the Text2Audio style), mapped to the read key ("id" or "id@section") that first read each.
    /// A node input with one of these values is written as a hash under any input name (<see cref="FormatString"/>).</summary>
    public Dictionary<string, string> PromptParamTexts = [];

    /// <summary>Media inputs by node ID.</summary>
    public Dictionary<string, MediaRecord> Media = [];

    /// <summary>Every <see cref="WGNodeData"/> created, for output typing.</summary>
    public List<WGNodeData> Data = [];

    /// <summary>Arguments of every CreateKSampler result, by node ID.</summary>
    public Dictionary<string, JObject> SamplerCalls = [];

    /// <summary>Segment parts processed, in order.</summary>
    public JArray SegmentParts = [];

    /// <summary>Cache hits inside the generator (conditioning, VAE loaders, model loaders), in order (<see cref="CacheHit"/>).</summary>
    public List<JObject> CacheHits = [];

    /// <summary>Connection rewires, in order (<see cref="Rewired"/>).</summary>
    public List<JObject> Rewires = [];

    /// <summary>Changes to nodes already in the workflow, in order.</summary>
    public List<EditRecord> Edits = [];

    /// <summary>Every node as it was at the last <see cref="Checkpoint"/>, by ID.</summary>
    public Dictionary<string, NodeSnapshot> Snapshots = [];

    /// <summary>Node IDs CreateNode wrote since the last <see cref="Checkpoint"/>.</summary>
    public HashSet<string> CreatedSinceCheckpoint = [];

    /// <summary>Index of the step during which each node entered the workflow without CreateNode, by node ID.</summary>
    public Dictionary<string, int> InsertedBy = [];

    /// <summary>Events that changed the course of generation, such as a step that stopped further steps, each with its "event" name.</summary>
    public List<JObject> Events = [];

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

    /// <summary>Returns the SHA-256 of a token's indented JSON text as UTF-8, the same as hashing <c>token.ToString()</c>, without building the text.</summary>
    public static string HashIndentedJson(JToken token)
    {
        return MetadataHelper.HashWritten(writer =>
        {
            JsonTextWriter json = new(writer) { Formatting = Formatting.Indented };
            token.WriteTo(json);
            json.Flush();
        });
    }

    /// <summary>Formats a workflow path (eg ["4", 0]) as "4:0".</summary>
    public static string PathText(JArray path) => path is null ? null : path.Count == 2 ? $"{path[0]}:{path[1]}" : path.ToString(Formatting.None);

    /// <summary>Parameter reads made after the generator finished, while the backend prepared the workflow text to send (<see cref="OnSendParamRead"/>, recorded by <see cref="RecordSent"/>), by "id" or "id@section".</summary>
    public Dictionary<string, ParamReadRecord> SendParamReads = [];

    /// <summary>Begins tracing a generation.</summary>
    public void Begin()
    {
        Gen.UserInput.ParamReadTracker = OnParamRead;
    }

    /// <summary>Records a parameter read into <see cref="ParamReads"/>, and the text of a prompt-view parameter (<see cref="PromptParamTexts"/>).</summary>
    public void OnParamRead(T2IParamType type, int section, object value, T2IParamInput.ParamReadSource source)
    {
        RecordParamRead(ParamReads, type, section, value, source);
    }

    /// <summary>Records a parameter read made while the backend prepares the workflow text to send, into <see cref="SendParamReads"/>.</summary>
    public void OnSendParamRead(T2IParamType type, int section, object value, T2IParamInput.ParamReadSource source)
    {
        RecordParamRead(SendParamReads, type, section, value, source);
    }

    /// <summary>Records a parameter read into a read table, and the text of a prompt-view parameter (<see cref="PromptParamTexts"/>).</summary>
    public void RecordParamRead(Dictionary<string, ParamReadRecord> reads, T2IParamType type, int section, object value, T2IParamInput.ParamReadSource source)
    {
        string key = section == 0 ? type.ID : $"{type.ID}@{section}";
        if (type.ViewType == ParamViewType.PROMPT && value is string text && text.Length > 0)
        {
            PromptParamTexts.TryAdd(text, key);
        }
        if (!reads.TryGetValue(key, out ParamReadRecord record))
        {
            record = new() { Type = type, Section = section, FirstValue = value, FirstSource = source };
            reads[key] = record;
        }
        record.LastValue = value;
        record.LastSource = source;
        record.Count++;
        if (record.Values.Count < MaxDistinctParamValues)
        {
            foreach (object seen in record.Values)
            {
                if (Equals(seen, value))
                {
                    return;
                }
            }
            record.Values.Add(value);
        }
    }

    /// <summary>Records a <see cref="WGNodeData"/>, read when the trace finishes.</summary>
    public void TrackData(WGNodeData data)
    {
        Data.Add(data);
    }

    /// <summary>Records that a step began.</summary>
    public void BeginStep(WorkflowGenerator.WorkflowGenStep step, bool isModelGen)
    {
        Checkpoint(InnermostStep);
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

    /// <summary>Index of the last top-level step that ended (<see cref="EndStep"/>), or -1. Model-generation steps that ran inside it come after it in <see cref="StepRecords"/>.</summary>
    public int LastTopLevelStep = -1;

    /// <summary>Records that the innermost step ended.</summary>
    public void EndStep()
    {
        Checkpoint(InnermostStep);
        StepRecord record = StepRecords[StepStack[^1]];
        StepStack.RemoveAt(StepStack.Count - 1);
        if (StepStack.Count > 0)
        {
            return;
        }
        LastTopLevelStep = record.Index;
        record.State = new()
        {
            ["model"] = DataJson(Gen.CurrentModel),
            ["textenc"] = DataJson(Gen.CurrentTextEnc),
            ["vae"] = DataJson(Gen.CurrentVae),
            ["media"] = DataJson(Gen.CurrentMedia),
            ["input_image"] = DataJson(Gen.BasicInputImage),
            ["loaded_model"] = Gen.FinalLoadedModel?.Name,
            ["mask"] = PathText(Gen.FinalMask),
            ["mask_crop"] = Gen.MaskShrunkInfo?.BoundsNode is null ? null : new JObject()
            {
                ["bounds"] = Gen.MaskShrunkInfo.BoundsNode,
                ["cropped_mask"] = Gen.MaskShrunkInfo.CroppedMask,
                ["masked_latent"] = Gen.MaskShrunkInfo.MaskedLatent,
                ["scaled_image"] = Gen.MaskShrunkInfo.ScaledImage
            },
            ["differential"] = Gen.IsDifferentialDiffusion
        };
    }

    /// <summary>Index of the innermost running step, or -1.</summary>
    public int InnermostStep => StepStack.Count > 0 ? StepStack[^1] : -1;

    /// <summary>Returns the name of a step by index, or "(outside any step)" for -1.</summary>
    public string StepName(int step) => step >= 0 ? StepRecords[step].Name : "(outside any step)";

    /// <summary>Attributes every change to the workflow since the last checkpoint to the given step: nodes removed, nodes inserted without CreateNode, and inputs or classes of existing nodes changed in place.
    /// Runs at every step boundary, so a change is attributed to the innermost step that was running when it was made.</summary>
    /// <param name="step">Index of the step that was innermost since the last checkpoint, or -1.</param>
    public void Checkpoint(int step)
    {
        JObject workflow = Gen.Workflow;
        StepRecord record = step >= 0 ? StepRecords[step] : null;
        List<string> gone = null;
        foreach (string id in LiveNodes)
        {
            if (!workflow.ContainsKey(id))
            {
                (gone ??= []).Add(id);
            }
        }
        if (gone is not null)
        {
            foreach (string id in gone)
            {
                LiveNodes.Remove(id);
                Snapshots.Remove(id);
                record?.Removed.Add(id);
                if (Nodes.TryGetValue(id, out NodeRecord node))
                {
                    node.RemovedBy = StepName(step);
                }
            }
        }
        foreach (JProperty property in workflow.Properties())
        {
            if (property.Value is not JObject node)
            {
                continue;
            }
            string id = property.Name;
            string classType = node["class_type"]?.ToString() ?? "";
            JObject inputs = node["inputs"] as JObject;
            bool created = CreatedSinceCheckpoint.Contains(id);
            if (!Snapshots.TryGetValue(id, out NodeSnapshot prior))
            {
                if (!created && !Nodes.ContainsKey(id) && InsertedBy.TryAdd(id, step))
                {
                    record?.Inserted.Add(id);
                }
                Snapshots[id] = new(node, classType, inputs?.DeepClone() as JObject);
                continue;
            }
            if (prior.Node != node)
            {
                if (!created)
                {
                    Edits.Add(new(step, id, classType, "(node)", prior.ClassType, classType));
                    if (record is not null && !record.Edited.Contains(id))
                    {
                        record.Edited.Add(id);
                    }
                }
                Snapshots[id] = new(node, classType, inputs?.DeepClone() as JObject);
                continue;
            }
            if (prior.ClassType == classType && JToken.DeepEquals(prior.Inputs, inputs))
            {
                continue;
            }
            if (prior.ClassType != classType)
            {
                Edits.Add(new(step, id, classType, "class_type", prior.ClassType, classType));
            }
            List<string> names = [.. prior.Inputs?.Properties().Select(p => p.Name) ?? []];
            names.AddRange(inputs?.Properties().Select(p => p.Name).Where(n => !names.Contains(n)) ?? []);
            foreach (string name in names)
            {
                JToken before = prior.Inputs?[name], after = inputs?[name];
                if (!JToken.DeepEquals(before, after))
                {
                    Edits.Add(new(step, id, classType, name, before, after?.DeepClone()));
                }
            }
            if (record is not null && !record.Edited.Contains(id))
            {
                record.Edited.Add(id);
            }
            Snapshots[id] = new(node, classType, inputs?.DeepClone() as JObject);
        }
        if (Snapshots.Count > workflow.Count)
        {
            foreach (string id in Snapshots.Keys.Where(k => !workflow.ContainsKey(k)).ToList())
            {
                Snapshots.Remove(id);
            }
        }
        CreatedSinceCheckpoint.Clear();
    }

    /// <summary>Records that the top-level step that just ended stopped all further steps.</summary>
    public void SkippedFurtherSteps()
    {
        Events.Add(new JObject() { ["event"] = "stopped_further_steps", ["step"] = StepName(LastTopLevelStep) });
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
        string stage = CurrentStage;
        Nodes[id] = new()
        {
            Id = id,
            ClassType = classType,
            Step = step,
            Stage = stage,
            Origin = Origin(member, file, line),
            Compat = LoadStack.Count > 0 ? Loads[LoadStack[^1]].Compat : StageCompat.TryGetValue(stage, out string stageCompat) ? stageCompat : Gen.FinalLoadedModel?.ModelClass?.CompatClass?.ID,
            Load = LoadStack.Count > 0 ? LoadStack[^1] : -1,
            Overwrites = overwrites
        };
        LiveNodes.Add(id);
        CreatedSinceCheckpoint.Add(id);
    }

    /// <summary>Records that CreateNode returned an existing identical node.</summary>
    public void NodeReused(string id, string member, string file, int line)
    {
        if (Nodes.TryGetValue(id, out NodeRecord node))
        {
            node.ReusedBy.Add(new(Origin(member, file, line), CurrentStage));
        }
    }

    /// <summary>Formats a code location as member@file:line.</summary>
    public static string Origin(string member, string file, int line)
    {
        return $"{member}@{(file ?? "").Replace('\\', '/').AfterLast('/')}:{line}";
    }

    /// <summary>Records a cache hit inside the generator.</summary>
    /// <param name="kind">What was cached: "conditioning", "vaeloader" or "modelloader".</param>
    /// <param name="result">The cached output returned, as "id:slot".</param>
    /// <param name="model">The model the cached loader loads, for a loader.</param>
    /// <param name="type">The loader type argument, for a model loader.</param>
    public void CacheHit(string kind, string result, string model = null, string type = null)
    {
        JObject hit = new() { ["kind"] = kind };
        if (model is not null)
        {
            hit["model"] = T2IParamTypes.CleanModelName(model);
        }
        if (type is not null)
        {
            hit["type"] = type;
        }
        hit["result"] = result;
        hit["stage"] = CurrentStage;
        CacheHits.Add(hit);
    }

    /// <summary>Compat class of the latest model load in each stage, by stage. A node created outside a load takes its stage's (<see cref="NodeRecord.Compat"/>), else the generator's final loaded model's.</summary>
    public Dictionary<string, string> StageCompat = [];

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
        StageCompat[record.Stage] = record.Compat;
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
        StageCompat[CurrentStage] = model?.ModelClass?.CompatClass?.ID;
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
        CacheHit("modelloader", PathText(modelOut), model?.Name ?? "?", type);
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
        LoraRecord record = new(T2IParamTypes.CleanModelName(lora?.Name ?? "?"), weight, tencWeight, confinement, kind, $"{nodeId}:{classType}", CurrentStage);
        Loras.Add(record);
        if (LoadStack.Count > 0)
        {
            Loads[LoadStack[^1]].Loras.Add(record);
        }
    }

    /// <summary>Registers a prompt text that will reach a text encoder node, under a label. The first positive prompt labeled in a segment's stage is that segment's prompt.</summary>
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
        if (label == $"{CurrentStage}.pos" && SegmentParts.Count > 0 && SegmentParts[^1] is JObject segment && $"{segment["stage"]}" == CurrentStage && segment["prompt"].Type == JTokenType.Null)
        {
            segment["prompt"] = $"@{label}#{MetadataHelper.Sha8(text)}";
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

    /// <summary>Records a segment part being processed. Its "prompt" is filled by <see cref="LabelPrompt"/> with the text its conditioning encodes, and stays null if none is created.</summary>
    public void SegmentPart(PromptRegion.Part part, int index)
    {
        SegmentParts.Add(new JObject()
        {
            ["index"] = index,
            ["stage"] = CurrentStage,
            ["section"] = part.ContextID,
            ["mask_text"] = $"@{CurrentStage}.mask#{MetadataHelper.Sha8(part.DataText)}",
            ["threshold"] = part.Strength,
            ["creativity"] = part.Strength2,
            ["prompt"] = JValue.CreateNull()
        });
    }

    /// <summary>Records the crop made for the segment part being processed: its bounds, cropped mask, masked latent and scaled image nodes, and the grow amount. <see cref="Build"/> reads the threshold, aspect, scale size and paste-back nodes from the graph.</summary>
    public void SegmentCrop(WorkflowGenerator.ImageMaskCropData crop, int grow)
    {
        if (crop is null || SegmentParts.Count == 0 || SegmentParts[^1] is not JObject segment)
        {
            return;
        }
        segment["crop"] = new JObject()
        {
            ["bounds"] = crop.BoundsNode,
            ["cropped_mask"] = crop.CroppedMask,
            ["masked_latent"] = crop.MaskedLatent,
            ["scaled_image"] = crop.ScaledImage,
            ["grow"] = grow
        };
    }

    /// <summary>Records a connection rewire: every input that read one output now reads another.</summary>
    /// <param name="from">The output the inputs read before.</param>
    /// <param name="to">The output they read now.</param>
    /// <param name="count">Number of inputs rewired.</param>
    public void Rewired(JArray from, JArray to, int count)
    {
        Rewires.Add(new JObject() { ["from"] = PathText(from), ["to"] = PathText(to), ["inputs"] = count, ["step"] = StepName(InnermostStep) });
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
            Checkpoint(InnermostStep);
            Logs.Debug($"{LogReference} generation failed, partial trace:\n{LogJson(Build())}\n{LogReference} detail:\n{LogJson(BuildDetail())}");
        }
        catch (Exception traceEx)
        {
            Logs.Debug($"{LogReference} generation failed and the partial trace could not be built: {traceEx.ReadableString()}");
        }
    }

    /// <summary>The record stored in the image metadata, once finished.</summary>
    public JObject Record;

    /// <summary>Key on this trace's server log lines, unique per trace; the image record stores it as "log_key".</summary>
    public readonly string LogKey = NewLogKey();

    /// <summary>Returns a new log key: the first 12 hex digits of a random GUID.</summary>
    public static string NewLogKey() => Guid.NewGuid().ToString("N")[..12];

    /// <summary>Returns the label server log lines carry for a log key.</summary>
    public static string LogLabel(string logKey) => string.IsNullOrEmpty(logKey) ? "[WorkflowTrace]" : $"[WorkflowTrace {logKey}]";

    /// <summary>The label this trace's server log lines carry.</summary>
    public string LogReference => LogLabel(LogKey);

    /// <summary>Returns where a Debug-level log line is kept under the current server log settings (<see cref="Logs.LogWithColor"/>): every one in the in-memory Debug tracker,
    /// on the console only when the log level is Debug or lower, and in the log file only then and with SaveLogToFile on.</summary>
    public static string DebugLogRetention()
    {
        string memory = $"kept in the server's last {Logs.LogTracker.MaxTracked} Debug messages in memory";
        if (Logs.MinimumLevel > Logs.LogLevel.Debug)
        {
            return $"{memory} only (the server log level is {Logs.MinimumLevel}, so it is not printed or saved)";
        }
        return Logs.LogsToSave is null ? $"{memory} and printed to the console, not saved (SaveLogToFile is off)" : $"{memory}, printed to the console and saved to the log file";
    }

    /// <summary>Returns the "full_trace" text of a record: the label of the log line with the full trace, and where that line is kept.</summary>
    public static string FullTraceNote(string logKey) => $"server log line '{LogLabel(logKey)}' at Debug level, {DebugLogRetention()}";

    /// <summary>Finishes the trace after generation: logs the summary plus detail, and stores the image record in the input's ExtraMeta: the summary itself when the image's metadata has no size limit,
    /// otherwise (<see cref="MetadataMayBeLimited"/>) the tier of <see cref="RecordTiers"/> that <see cref="PickTier"/> picks, which <see cref="FitRecordToSave"/> picks again when the image is saved.</summary>
    public void Finish()
    {
        Gen.UserInput.ParamReadTracker = null;
        Checkpoint(InnermostStep);
        Summary = Build();
        Logs.Debug($"{LogReference} request {Gen.UserInput.UserRequestId}:\n{LogJson(Summary)}\n{LogReference} detail:\n{LogJson(BuildDetail())}");
        string format = ImageFormatOf(Gen.UserInput);
        Record = MetadataMayBeLimited(Gen.UserInput, format) ? PickTier(RecordTiers(Summary), Gen.UserInput, format) : Summary;
        Gen.UserInput.ExtraMeta[MetadataKey] = Record;
    }

    /// <summary>JSON writer for trace log lines: indents the top-level entries and the entries of top-level tables and lists, and writes anything deeper on its entry's line.</summary>
    /// <param name="textWriter">The text writer to write to.</param>
    public class LogJsonWriter(TextWriter textWriter) : MetadataHelper.CompactingJsonWriter(textWriter)
    {
        /// <summary>True inside an entry of a top-level table or list.</summary>
        public override bool Compact => OpenContainers > 2;
    }

    /// <summary>Formats a trace record for the server log, in one pass (<see cref="LogJsonWriter"/>): indented JSON with every top-level entry, and every entry of a top-level table or list, on its own "\n"-ended line, and anything deeper on that line.</summary>
    public static string LogJson(JObject record)
    {
        StringWriter text = new(new StringBuilder(4096), CultureInfo.InvariantCulture) { NewLine = "\n" };
        using (LogJsonWriter writer = new(text) { Formatting = Formatting.Indented })
        {
            record.WriteTo(writer);
        }
        return text.ToString();
    }

    /// <summary>Returns true if an image of this input in the given format has a metadata size limit: JPEG keeps metadata in EXIF (<see cref="MetadataHelper.MaxJpegExifChars"/>), and stealth metadata in PNG or WebP pixels holds as much as the image has pixels for (<see cref="MetadataHelper.StealthCapacityBits"/>).
    /// PNG text chunks and WebP's EXIF chunk have no limit.</summary>
    public static bool MetadataMayBeLimited(T2IParamInput input, string format)
    {
        return ImageFile.IsJpegFormat(format) || ImageFile.StealthApplies(format, StealthModeOf(input));
    }

    /// <summary>Returns the stealth metadata mode images of this input are saved with, as Session.ApplyMetadata reads it, or "false" when metadata is not saved.</summary>
    public static string StealthModeOf(T2IParamInput input)
    {
        Settings.User.FileFormatData fileFormat = input.SourceSession?.User?.Settings?.FileFormat ?? Program.ServerSettings?.DefaultUser?.FileFormat;
        return fileFormat is null || !fileFormat.SaveMetadata ? "false" : fileFormat.StealthMetadata ?? "false";
    }

    /// <summary>Records that <see cref="Finish"/> threw: logs the error, detaches the parameter tracker, and stores the error as the image record. The generated workflow is unaffected.</summary>
    public void FinishFailed(Exception ex)
    {
        Gen.UserInput.ParamReadTracker = null;
        Logs.Error($"{LogReference}: the trace could not be finished; the generation is unaffected: {ex.ReadableString()}");
        Gen.UserInput.ExtraMeta[MetadataKey] = new JObject()
        {
            ["version"] = FormatVersion,
            ["log_key"] = LogKey,
            ["trace_error"] = $"{ex.GetType().Name}: {ex.Message}"
        };
    }

    /// <summary>Returns the length of an input's image metadata text without the trace record, as it stands: its parameter metadata and other extra data, built without side effects (no model hashing).</summary>
    public static int OtherMetadataChars(T2IParamInput input)
    {
        JObject extra = input.BuildExtraDataJObject();
        extra.Remove(MetadataKey);
        JObject metadata = new()
        {
            ["sui_image_params"] = input.GenParameterMetadata() ?? [],
            ["sui_extra_data"] = extra
        };
        return T2IParamInput.MetadataToString(metadata).Length;
    }

    /// <summary>Returns the length of the metadata text an image of this input is saved with (<see cref="T2IParamInput.GenFullMetadataObject"/>: parameters, extra data, unused parameter list and model hashes) without the trace record.
    /// Model hashes are read as the save reads them, so a model the generation did not hash yet is hashed here, ahead of the save.</summary>
    public static int SavedMetadataChars(T2IParamInput input)
    {
        JObject metadata = input.GenFullMetadataObject();
        (metadata["sui_extra_data"] as JObject)?.Remove(MetadataKey);
        return T2IParamInput.MetadataToString(metadata).Length;
    }

    /// <summary>Builds the image record tiers, most complete first; <see cref="ForImageMetadata"/> keeps the first that fits the format. Tier 0 is the summary itself (<see cref="Build"/>), unchanged.
    /// Tier 1 reduces every node to its class and provenance (<see cref="NodeJson"/> brief), and the parameter reads to those made for a section. Tier 2 drops the node table, names each node of the samplers' input chains as "id:Class" instead, and gives the steps, removals, rewires, edits, cache hits and node reuse as counts.
    /// Tier 3 keeps the workflow identity, the model and the samplers' nodes and calls. Tier 4 keeps the workflow identity and the sampler count, so it fits whenever the rest of the metadata leaves room for a few hundred characters.
    /// Each tier below 0 says what it omitted and where the full trace is in the server log.</summary>
    /// <param name="summary">The trace summary (<see cref="Build"/>).</param>
    public List<JObject> RecordTiers(JObject summary)
    {
        JObject full = summary;
        JObject reduced = (JObject)full.DeepClone();
        JObject briefNodes = [];
        foreach (JProperty property in Gen.Workflow.Properties())
        {
            if (property.Value is JObject node)
            {
                briefNodes[property.Name] = NodeJson(property.Name, node, true);
            }
        }
        reduced["nodes"] = briefNodes;
        reduced.Remove("params_read");
        JObject sectionReads = [];
        foreach (JProperty read in (summary["params_read"] as JObject ?? []).Properties().Where(p => p.Name.Contains('@')))
        {
            sectionReads[read.Name] = read.Value.DeepClone();
        }
        if (sectionReads.Count > 0)
        {
            reduced["params_read_by_section"] = sectionReads;
        }
        reduced["omitted"] = $"node inputs and output types, and parameter reads outside sections, to fit the image metadata; the full trace is in the {full["full_trace"]}";
        JObject counted = (JObject)reduced.DeepClone();
        counted.Remove("nodes");
        foreach (JObject sampler in (counted["samplers"] as JArray ?? []).OfType<JObject>())
        {
            foreach (JProperty input in (sampler["inputs"] as JObject ?? []).Properties())
            {
                foreach (JValue entry in (input.Value as JArray ?? []).OfType<JValue>().ToList())
                {
                    string id = $"{entry}";
                    if (Gen.Workflow[id] is JObject chainNode)
                    {
                        entry.Value = $"{id}:{chainNode["class_type"]}";
                    }
                }
            }
        }
        JObject counts = [];
        foreach (string key in new[] { "steps", "removed", "rewires", "edits", "cache_hits", "node_reuse" })
        {
            if (counted[key] is JArray list)
            {
                counts[key] = list.Count;
                counted.Remove(key);
            }
        }
        counted["decision_counts"] = counts;
        counted["omitted"] = $"the node table, parameter reads outside sections, and the lists of steps, removals, rewires, edits, cache hits and node reuse, to fit the image metadata; the full trace is in the {full["full_trace"]}";
        JArray brief = [];
        foreach (JObject sampler in ((JArray)full["samplers"]).OfType<JObject>())
        {
            JObject entry = new() { ["node"] = sampler["node"].DeepClone() };
            if (sampler["call"] is JToken call)
            {
                entry["call"] = call.DeepClone();
            }
            brief.Add(entry);
        }
        JObject minimal = new()
        {
            ["version"] = FormatVersion,
            ["log_key"] = full["log_key"]?.DeepClone(),
            ["full_trace"] = full["full_trace"].DeepClone(),
            ["workflow"] = full["workflow"]?.DeepClone(),
            ["model"] = full["model"]?.DeepClone(),
            ["samplers"] = brief,
            ["omitted"] = $"everything but the samplers' nodes and calls, to fit the image metadata; the full trace is in the {full["full_trace"]}"
        };
        JObject notice = MetadataHelper.TraceNotice(full, $"the whole trace, to fit the image metadata; the full trace is in the {full["full_trace"]}");
        return [full, reduced, counted, minimal, notice];
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

    /// <summary>Returns the length a record adds to image metadata text, nested where Swarm stores it (sui_extra_data.debug_trace).</summary>
    public static int RecordMetadataChars(JObject record)
    {
        return T2IParamInput.MetadataToString(new JObject() { ["sui_extra_data"] = new JObject() { [MetadataKey] = record } }).Length;
    }

    /// <summary>Returns the record tier to store in a file's metadata: for JPEG, whose metadata is in one EXIF segment, the first tier that fits <see cref="MetadataHelper.MaxJpegExifChars"/> together with the rest of the metadata, else the last tier;
    /// for every other format (PNG text chunks, WebP's EXIF chunk, video and audio comments), the first.</summary>
    /// <param name="tiers">Record tiers, most complete first (<see cref="RecordTiers"/>).</param>
    /// <param name="format">Image format, as <see cref="T2IParamTypes.ImageFormat"/> values ("PNG", "JPG", "WEBP", ...), or a video or audio file extension.</param>
    /// <param name="otherChars">Length of the rest of the metadata text (<see cref="OtherMetadataChars"/> or <see cref="SavedMetadataChars"/>).</param>
    /// <param name="maxChars">Limit for JPEG, or -1 for <see cref="MetadataHelper.MaxJpegExifChars"/>.</param>
    public static JObject ForImageMetadata(List<JObject> tiers, string format, int otherChars, int maxChars = -1)
    {
        if (maxChars < 0)
        {
            maxChars = MetadataHelper.MaxJpegExifChars;
        }
        if (!ImageFile.IsJpegFormat(format))
        {
            return tiers[0];
        }
        foreach (JObject tier in tiers)
        {
            if (otherChars + RecordMetadataChars(tier) <= maxChars)
            {
                return tier;
            }
        }
        return tiers[^1];
    }

    /// <summary>Annotation on every tier of a stored image record (<see cref="JToken.AddAnnotation"/>; annotations are not serialized): the whole tier list, so the record can be picked again for the metadata the image is saved with.</summary>
    /// <param name="tiers">The record tiers, most complete first.</param>
    public class RecordTierSet(List<JObject> tiers)
    {
        /// <summary>The record tiers, most complete first.</summary>
        public List<JObject> Tiers = tiers;
    }

    /// <summary>1 once <see cref="FitRecordToSave"/> is subscribed to <see cref="T2IEngine.PostGenerateEvent"/>. The first tiered record subscribes it, so a server that never stores one (no JPEG or stealth-metadata saves while tracing) runs no handler, and it runs after the handlers subscribed at startup, whose metadata it then measures.
    /// The JPEG limit leaves a margin for what changes after it runs: post-generate handlers subscribed after it, and the seed of a later image in a batch.</summary>
    public static int FitSubscribed;

    /// <summary>Returns the tier of a record for an image of the given format with the metadata the input has now (<see cref="ForImageMetadata"/>), after annotating every tier with the tier list (<see cref="RecordTierSet"/>) for <see cref="FitRecordToSave"/>, and subscribing that.</summary>
    /// <param name="tiers">Record tiers, most complete first.</param>
    /// <param name="input">The input the record is stored in.</param>
    /// <param name="format">The input's image format (<see cref="ImageFormatOf"/>).</param>
    public static JObject PickTier(List<JObject> tiers, T2IParamInput input, string format)
    {
        RecordTierSet set = new(tiers);
        foreach (JObject tier in tiers)
        {
            tier.AddAnnotation(set);
        }
        if (Interlocked.Exchange(ref FitSubscribed, 1) == 0)
        {
            Action<T2IEngine.PostGenerationEventParams> current, combined;
            do
            {
                current = T2IEngine.PostGenerateEvent;
                combined = current + FitRecordToSave;
            }
            while (Interlocked.CompareExchange(ref T2IEngine.PostGenerateEvent, combined, current) != current);
        }
        return ForImageMetadata(tiers, format, OtherMetadataChars(input));
    }

    /// <summary>Post-generate handler (<see cref="T2IEngine.PostGenerateEvent"/>): picks the record tier again for the file being saved and the metadata it is saved with (<see cref="T2IParamInput.GenFullMetadataObject"/>), which by now includes everything written after the trace finished:
    /// the posted workflow, the backend's debug data with the prompt, the generation times, the unused parameter list and the model hashes.
    /// A video or audio file keeps the first tier (its metadata is a track comment). An image is converted to the input's image format: JPEG gets the first tier that fits one EXIF segment (<see cref="ForImageMetadata"/>),
    /// and with stealth metadata on (PNG or WebP), the first tier from there whose stealth data fits the image's pixels (<see cref="MetadataHelper.StealthCapacityBits"/>; the size comes from the decode <see cref="ImageFile.ConvertTo"/> reuses for the save).
    /// Does nothing for an input without a tiered record. An error is logged, never thrown: the file is saved with the record it had.</summary>
    public static void FitRecordToSave(T2IEngine.PostGenerationEventParams args)
    {
        T2IParamInput input = args.UserInput;
        if (input?.ExtraMeta.GetValueOrDefault(MetadataKey) is not JObject stored || stored.Annotation<RecordTierSet>() is not RecordTierSet set)
        {
            return;
        }
        JObject extra = null;
        try
        {
            bool isImage = args.File is null || args.File.Type.MetaType == MediaMetaType.Image;
            string format = isImage ? ImageFormatOf(input) : args.File.Type.Extension;
            JObject metadata = input.GenFullMetadataObject();
            extra = metadata["sui_extra_data"] as JObject;
            extra?.Remove(MetadataKey);
            JObject pick = ForImageMetadata(set.Tiers, format, T2IParamInput.MetadataToString(metadata).Length);
            string stealthMode = StealthModeOf(input);
            if (isImage && extra is not null && args.File is ImageFile image && ImageFile.StealthApplies(format, stealthMode))
            {
                SixLabors.ImageSharp.Image decoded = image.ToIS;
                long capacity = MetadataHelper.StealthCapacityBits(decoded.Width, decoded.Height, stealthMode);
                for (int i = set.Tiers.IndexOf(pick); i < set.Tiers.Count; i++)
                {
                    pick = set.Tiers[i];
                    extra[MetadataKey] = pick;
                    if (MetadataHelper.StealthPayloadBits(T2IParamInput.MetadataToString(metadata), stealthMode) <= capacity)
                    {
                        break;
                    }
                }
            }
            input.ExtraMeta[MetadataKey] = pick;
        }
        catch (Exception ex)
        {
            Logs.Error($"{LogLabel($"{stored["log_key"]}")}: the record could not be fitted to the saved metadata; the file keeps the record it had: {ex.ReadableString()}");
        }
        finally
        {
            extra?.Remove(MetadataKey);
        }
    }

    /// <summary>The "posted" value of a workflow record until a ComfyUI backend posts it through <see cref="ComfyUIAPIAbstractBackend.AwaitJobLive"/>.</summary>
    public const string NotPosted = "not observed: no post through ComfyUIAPIAbstractBackend.AwaitJobLive";

    /// <summary>Returns the "workflow" objects to update for an input: the summary's, the stored image record's, and that of every tier of the stored record (<see cref="RecordTierSet"/>), each once.</summary>
    public static List<JObject> WorkflowRecords(T2IParamInput input, JObject summary)
    {
        List<JObject> records = [];
        void add(JObject record)
        {
            if (record?["workflow"] is JObject info && !records.Any(r => ReferenceEquals(r, info)))
            {
                records.Add(info);
            }
        }
        add(summary);
        if (input?.ExtraMeta.GetValueOrDefault(MetadataKey) is JObject stored)
        {
            add(stored);
            foreach (JObject tier in stored.Annotation<RecordTierSet>()?.Tiers ?? [])
            {
                add(tier);
            }
        }
        return records;
    }

    /// <summary>Returns the "${...}" placeholders in a workflow text, each once, in order: parsed by <see cref="StringConversionHelper.QuickSimpleTagFiller"/> with the prefix, suffix and subtag setting <see cref="ComfyUIAPIAbstractBackend.CreateWorkflow"/> fills custom and raw workflows with.</summary>
    public static List<string> Placeholders(string text)
    {
        List<string> result = [];
        StringConversionHelper.QuickSimpleTagFiller(text ?? "", "${", "}", tag =>
        {
            string placeholder = $"${{{tag}}}";
            if (!result.Contains(placeholder))
            {
                result.Add(placeholder);
            }
            return null;
        }, false);
        return result;
    }

    /// <summary>Records the workflow text the backend will send to ComfyUI, in the summary and the stored image metadata record: its hash and length, whether anything rewrote the generator's text, which "${...}" placeholders the rewrite filled (such as uploaded image names),
    /// and the parameters the backend read while preparing the text (<see cref="SendParamReads"/>, such as the image size it resized uploaded images to). An error is logged, never thrown: the send is unaffected.</summary>
    /// <param name="workflow">The final workflow JSON text.</param>
    /// <param name="generated">The workflow text the generator returned.</param>
    public void RecordSent(string workflow, string generated)
    {
        try
        {
            string sha = MetadataHelper.HashText(workflow);
            List<JObject> records = WorkflowRecords(Gen.UserInput, Summary);
            if (records.Count == 0)
            {
                return;
            }
            bool rewritten = workflow != generated;
            List<string> filled = rewritten ? [.. Placeholders(generated).Except(Placeholders(workflow))] : [];
            JObject sendReads = SendParamReads.Count > 0 ? FormatParamReads(SendParamReads) : null;
            foreach (JObject info in records)
            {
                info["sent_sha256"] = sha;
                info["sent_length"] = workflow.Length;
                info["rewritten_after_generate"] = rewritten;
                if (rewritten)
                {
                    info["placeholders_filled"] = new JArray(filled.Select(p => (JToken)p));
                }
                if (sendReads is not null)
                {
                    info["params_read"] = sendReads.DeepClone();
                }
                info["posted"] = NotPosted;
            }
            Logs.Debug($"{LogReference}: sent workflow sha256={sha} length={workflow.Length} rewritten_after_generate={rewritten}{(rewritten ? $" placeholders_filled=[{filled.JoinString(", ")}]" : "")}");
        }
        catch (Exception ex)
        {
            Logs.Error($"{LogReference}: the sent workflow could not be recorded; the generation is unaffected: {ex.ReadableString()}");
        }
    }

    /// <summary>Records the workflow a ComfyUI backend posts to /prompt for an input, into its stored trace record and every tier of it. Does nothing for an input without one. An error is logged, never thrown: the post is unaffected.</summary>
    /// <param name="input">The input being generated.</param>
    /// <param name="workflow">The workflow JSON text posted as the "prompt".</param>
    /// <param name="backend">Name of the posting backend type.</param>
    public static void RecordPosted(T2IParamInput input, string workflow, string backend)
    {
        if (input?.ExtraMeta.GetValueOrDefault(MetadataKey) is not JObject stored || stored["workflow"] is not JObject info)
        {
            return;
        }
        string label = LogLabel($"{stored["log_key"]}");
        try
        {
            string sha = MetadataHelper.HashText(workflow);
            bool? matchesSent = info["sent_sha256"] is JValue sent ? $"{sent}" == sha : null;
            foreach (JObject record in WorkflowRecords(input, null))
            {
                record["posted"] = new JObject()
                {
                    ["sha256"] = sha,
                    ["length"] = workflow.Length,
                    ["backend"] = backend,
                    ["matches_sent"] = matchesSent
                };
            }
            Logs.Debug($"{label}: {backend} posted workflow sha256={sha} length={workflow.Length} matches_sent={matchesSent?.ToString() ?? "unknown"}");
        }
        catch (Exception ex)
        {
            Logs.Error($"{label}: the posted workflow could not be recorded; the generation is unaffected: {ex.ReadableString()}");
        }
    }

    /// <summary>Records a workflow that did not come from the generator (custom or raw workflows): logs its description under a new log key, and stores it in the input's ExtraMeta with that key.
    /// When the file's metadata cannot hold it, the stored record drops the node list, then the parameter reads (<see cref="PickTier"/>, then <see cref="FitRecordToSave"/>).
    /// An error is logged and stored as the record, never thrown: the generation is unaffected.</summary>
    /// <param name="input">The input the workflow was filled from.</param>
    /// <param name="source">"custom" or "raw".</param>
    /// <param name="workflow">The filled workflow text.</param>
    /// <param name="reads">The parameter reads that filled it (<see cref="ComfyUIAPIAbstractBackend.CreateWorkflow"/>), recorded as "params_read" with free text hashed, or null.</param>
    public static void RecordExternalWorkflow(T2IParamInput input, string source, string workflow, Dictionary<string, ParamReadRecord> reads = null)
    {
        string logKey = NewLogKey();
        try
        {
            JObject summary = DescribeExternalWorkflow(source, workflow);
            summary["log_key"] = logKey;
            if (reads is not null)
            {
                summary["params_read"] = FormatParamReads(reads, hashFreeText: true);
            }
            Logs.Debug($"{LogLabel(logKey)} request {input.UserRequestId} ({source} workflow):\n{LogJson(summary)}");
            string format = ImageFormatOf(input);
            if (!MetadataMayBeLimited(input, format))
            {
                input.ExtraMeta[MetadataKey] = summary;
                return;
            }
            JObject withoutNodes = (JObject)summary.DeepClone();
            withoutNodes.Remove("nodes");
            withoutNodes["omitted"] = $"the node list, to fit the image metadata; the full description is in the {FullTraceNote(logKey)}";
            List<JObject> tiers = [summary, withoutNodes];
            if (withoutNodes.ContainsKey("params_read"))
            {
                JObject withoutReads = (JObject)withoutNodes.DeepClone();
                withoutReads.Remove("params_read");
                withoutReads["omitted"] = $"the node list and the parameter reads, to fit the image metadata; the full description is in the {FullTraceNote(logKey)}";
                tiers.Add(withoutReads);
            }
            input.ExtraMeta[MetadataKey] = PickTier(tiers, input, format);
        }
        catch (Exception ex)
        {
            Logs.Error($"{LogLabel(logKey)}: the {source} workflow could not be recorded; the generation is unaffected: {ex.ReadableString()}");
            input.ExtraMeta[MetadataKey] = new JObject()
            {
                ["version"] = FormatVersion,
                ["log_key"] = logKey,
                ["trace_error"] = $"{ex.GetType().Name}: {ex.Message}"
            };
        }
    }

    /// <summary>Builds a trace summary for a workflow that did not come from the generator (custom or raw workflows): its identity, and every node with its inputs (<see cref="NodeJson"/>), free text hashed.</summary>
    public static JObject DescribeExternalWorkflow(string source, string workflow)
    {
        JObject result = new()
        {
            ["version"] = FormatVersion,
            ["workflow"] = new JObject()
            {
                ["source"] = source,
                ["sent_sha256"] = MetadataHelper.HashText(workflow),
                ["sent_length"] = workflow?.Length ?? 0,
                ["posted"] = NotPosted
            }
        };
        try
        {
            JObject parsed = workflow.ParseToJson();
            WorkflowTrace reader = new(null) { HashFreeText = true };
            JObject nodes = [];
            foreach (JProperty property in parsed.Properties())
            {
                if (property.Value is JObject node)
                {
                    nodes[property.Name] = reader.NodeJson(property.Name, node, false);
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

    /// <summary>Describes a <see cref="WGNodeData"/>: its output path "id:slot", Swarm data type, and the compat class, size, frames, FPS, alpha and attached audio it knows of.</summary>
    public static JObject DataJson(WGNodeData data)
    {
        if (data is null)
        {
            return null;
        }
        JObject result = new() { ["path"] = PathText(data.Path), ["type"] = data.DataType };
        if (data.Compat is not null)
        {
            result["compat"] = data.Compat.ID;
        }
        if (data.Width is int width)
        {
            result["width"] = width;
        }
        if (data.Height is int height)
        {
            result["height"] = height;
        }
        if (data.Frames is int frames)
        {
            result["frames"] = frames;
        }
        if (data.FPS is not null)
        {
            result["fps"] = data.FPS.DeepClone();
        }
        if (data.MayHaveAlpha)
        {
            result["alpha"] = true;
        }
        if (data.AttachedAudio is not null)
        {
            result["audio"] = PathText(data.AttachedAudio.Path);
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

        /// <summary>Adds the slot's typing to a JSON object: "types" and "compat" as lists (more than one entry means the generator typed the slot inconsistently), and "width", "height", "frames" and "alpha" when known.</summary>
        public void AddTo(JObject target)
        {
            target["types"] = new JArray(Types);
            if (Compat.Count > 0)
            {
                target["compat"] = new JArray(Compat);
            }
            if (Width is int width)
            {
                target["width"] = width;
            }
            if (Height is int height)
            {
                target["height"] = height;
            }
            if (Frames is int frames)
            {
                target["frames"] = frames;
            }
            if (Alpha)
            {
                target["alpha"] = true;
            }
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

    /// <summary>Returns a prompt node's inputs as ComfyUI validates them, from <see cref="ObjectInfo"/> and the node's own inputs (<see cref="ComfyInputValidation.FinalizedInputs"/>), or null if <see cref="ObjectInfo"/> does not list the class.
    /// Throws where ComfyUI fails to expand the node's dynamic inputs.</summary>
    public JObject ComfyInputs(string classType, JObject liveInputs)
    {
        if (ObjectInfo is null || ObjectInfo[classType] is not JObject info)
        {
            return null;
        }
        return ComfyInputValidation.FinalizedInputs(info, liveInputs ?? []);
    }

    /// <summary>Returns ComfyUI's declared type of a prompt node's required or optional input: a type name, or a V1 node's choice list.
    /// Null if <see cref="ObjectInfo"/> does not list the class, the inputs cannot be expanded, or the input is not required or optional among them.</summary>
    public JToken ComfyInputType(string classType, JObject liveInputs, string input)
    {
        JObject comfyInputs;
        try
        {
            comfyInputs = ComfyInputs(classType, liveInputs);
        }
        catch (Exception)
        {
            return null;
        }
        foreach (string group in new[] { "required", "optional" })
        {
            if (comfyInputs?[group] is JObject set && set[input] is JArray declaration && declaration.Count > 0)
            {
                return declaration[0];
            }
        }
        return null;
    }

    /// <summary>Returns true if the value is a link to another node, ie ["id", slot].</summary>
    public static bool IsLink(JToken value)
    {
        return value is JArray arr && arr.Count == 2 && arr[0].Type == JTokenType.String && arr[1].Type == JTokenType.Integer;
    }

    /// <summary>Formats a string node input with placeholders for prompts, media, and long text. Prompt text that is part of a registered prompt, rather than all of one, is labeled "~part" with the labels of the shortest registered prompt that contains it.
    /// The text of a prompt-view parameter read (<see cref="PromptParamTexts"/>) is labeled with that parameter under any input name. Embedded media is "@param#hash", the hash its entry in the record's media table (by node ID) carries.</summary>
    public string FormatString(string nodeId, string key, string value)
    {
        if (key.EndsWith("_base64"))
        {
            if (Media.TryGetValue(nodeId, out MediaRecord media))
            {
                return $"@{media.Param}#{MetadataHelper.Sha8(value)}";
            }
            return $"b64#{MetadataHelper.Sha8(value)}({value.Length})";
        }
        if (PromptLabels.TryGetValue(value, out List<string> labels))
        {
            return $"{labels.Select(l => $"@{l}").JoinString("=")}#{MetadataHelper.Sha8(value)}";
        }
        if (PromptParamTexts.TryGetValue(value, out string promptParam))
        {
            return $"@{promptParam}#{MetadataHelper.Sha8(value)}";
        }
        if (HashFreeText && IsFreeText(value))
        {
            return $"#{MetadataHelper.Sha8(value)}({value.Length})";
        }
        if (PromptInputNames.Contains(key))
        {
            string container = null;
            List<string> containerLabels = [];
            foreach ((string text, List<string> textLabels) in PromptLabels)
            {
                if (value.Length == 0 || !text.Contains(value))
                {
                    continue;
                }
                if (container is null || text.Length < container.Length)
                {
                    container = text;
                    containerLabels = [.. textLabels];
                }
                else if (text.Length == container.Length)
                {
                    containerLabels.AddRange(textLabels.Where(l => !containerLabels.Contains(l)));
                }
            }
            if (container is not null)
            {
                return $"{containerLabels.Select(l => $"@{l}").JoinString("=")}~part#{MetadataHelper.Sha8(value)}";
            }
            return $"#{MetadataHelper.Sha8(value)}({value.Length})";
        }
        if (value.Length > 120)
        {
            return $"#{MetadataHelper.Sha8(value)}({value.Length})";
        }
        return value;
    }

    /// <summary>Returns the size and type facts of a media record: "width" and "height" with "resized" true when the generator resized the image to embed it, else what the file's header says (<see cref="MediaHeader"/>), and the "mime" type.</summary>
    public static JObject MediaFields(MediaRecord media)
    {
        JObject fields = [];
        if (media.Width is int width && media.Height is int height)
        {
            fields["width"] = width;
            fields["height"] = height;
            fields["resized"] = true;
        }
        else
        {
            MediaHeader(media.Media).AddTo(fields);
        }
        fields["mime"] = media.Media?.Type?.MimeType;
        return fields;
    }

    /// <summary>What <see cref="MediaHeader"/> read from each media file's bytes, by array reference, so a file is read once however often a trace names it.</summary>
    public static readonly ConditionalWeakTable<byte[], MediaFacts> MediaHeaders = new();

    /// <summary>Returns what a media file's header says, without decoding it. An image or animation gives its width and height, and its frame count when it has more than one: Swarm's own decode if it already has one (<see cref="ImageFile._CacheISImg"/>), else ImageSharp Identify, which parses the header chunks.
    /// Audio and video give the duration, rounded to a hundredth of a second, and audio adds its sample rate and channel count: ATL, which Swarm reads media metadata with, autodetecting the format the way <see cref="VideoFile.GetMetadata"/> does (ATL estimates a PCM WAV's duration from its data size, within a millisecond).
    /// Neither library reads a video's frame size, so a video has none here. <see cref="MediaFacts.None"/> when nothing is readable; for bytes it cannot parse, ATL also prints the exception to the console, as it does for Swarm's own media metadata reads (ATL Settings.OutputStacktracesToConsole, which Swarm leaves on).</summary>
    public static MediaFacts MediaHeader(MediaFile media)
    {
        if (media?.RawData is not byte[] raw || media.Type is null)
        {
            return MediaFacts.None;
        }
        if (MediaHeaders.TryGetValue(raw, out MediaFacts known))
        {
            return known;
        }
        static MediaFacts frameSize(int width, int height, int frames) => new(width, height, frames > 1 ? frames : null);
        MediaMetaType kind = media.Type.MetaType;
        MediaFacts facts = MediaFacts.None;
        try
        {
            if (kind == MediaMetaType.Image || kind == MediaMetaType.Animation)
            {
                if (media is ImageFile img && img._CacheISImg is SixLabors.ImageSharp.Image decoded)
                {
                    facts = frameSize(decoded.Width, decoded.Height, decoded.Frames.Count);
                }
                else
                {
                    SixLabors.ImageSharp.ImageInfo info = SixLabors.ImageSharp.Image.Identify(raw);
                    facts = frameSize(info.Width, info.Height, info.FrameMetadataCollection.Count);
                }
            }
            else if (kind == MediaMetaType.Video || kind == MediaMetaType.Audio)
            {
                using MemoryStream stream = new(raw, false);
                ATL.Track track = new(stream);
                double? duration = track.DurationMs > 0 ? Math.Round(track.DurationMs / 1000, 2, MidpointRounding.AwayFromZero) : null;
                bool audio = kind == MediaMetaType.Audio;
                double? rate = audio && track.SampleRate > 0 ? track.SampleRate : null;
                int? channels = audio && track.ChannelsArrangement?.NbChannels > 0 ? track.ChannelsArrangement.NbChannels : null;
                facts = new(DurationSeconds: duration, SampleRate: rate, Channels: channels);
            }
        }
        catch (Exception)
        {
            facts = MediaFacts.None;
        }
        MediaHeaders.AddOrUpdate(raw, facts);
        return facts;
    }

    /// <summary>Returns the record name of a read source: "section", "base" or "default".</summary>
    public static string SourceName(T2IParamInput.ParamReadSource source) => source switch
    {
        T2IParamInput.ParamReadSource.Section => "section",
        T2IParamInput.ParamReadSource.Base => "base",
        _ => "default"
    };

    /// <summary>Formats a table of parameter reads, by "id" or "id@section": the first value read and its "source" (<see cref="SourceName"/>); "last" and "last_source" when the last read returned something else;
    /// and with more than two distinct values, the number of "reads" and the distinct "values" in first-read order, with "values_capped" when there may be more than <see cref="MaxDistinctParamValues"/>.</summary>
    /// <param name="reads">Reads by "id" or "id@section".</param>
    /// <param name="hashFreeText">If true, string values that could be prompt text are hashed (<see cref="IsFreeText"/>), for workflows the generator did not build.</param>
    public static JObject FormatParamReads(Dictionary<string, ParamReadRecord> reads, bool hashFreeText = false)
    {
        JObject parameters = [];
        foreach ((string key, ParamReadRecord read) in reads.OrderBy(p => p.Key))
        {
            JToken first = FormatParamValue(read.Type, read.FirstValue, hashFreeText);
            JObject entry = new() { ["value"] = first, ["source"] = SourceName(read.FirstSource) };
            JToken last = FormatParamValue(read.Type, read.LastValue, hashFreeText);
            if (!JToken.DeepEquals(first, last) || read.FirstSource != read.LastSource)
            {
                entry["last"] = last;
                entry["last_source"] = SourceName(read.LastSource);
            }
            if (read.Values.Count > 2)
            {
                entry["reads"] = read.Count;
                entry["values"] = new JArray(read.Values.Select(v => FormatParamValue(read.Type, v, hashFreeText)));
                if (read.Values.Count >= MaxDistinctParamValues)
                {
                    entry["values_capped"] = true;
                }
            }
            parameters[key] = entry;
        }
        return parameters;
    }

    /// <summary>Returns true if a string could be prompt text: longer than 64 characters or containing whitespace, and not a model file name (<see cref="ModelFileExtensions"/>).</summary>
    public static bool IsFreeText(string value)
    {
        return (value.Length > 64 || value.Any(char.IsWhiteSpace)) && !ModelFileExtensions.Any(e => value.EndsWith(e, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>Formats a media file as a parameter value: the "raw_sha8" hash of its bytes, its header facts (<see cref="MediaHeader"/>) and its "mime" type.</summary>
    public static JObject MediaValue(MediaFile media)
    {
        JObject value = new() { ["raw_sha8"] = MetadataHelper.Sha8(media.RawData ?? []) };
        MediaHeader(media).AddTo(value);
        value["mime"] = media.Type?.MimeType;
        return value;
    }

    /// <summary>Formats a parameter value with placeholders for prompts, as "@id#hash(length)", and long text, as "#hash(length)"; a model as its "name" and "class", and media as <see cref="MediaValue"/>.
    /// With <paramref name="hashFreeText"/>, strings that could be prompt text are hashed too (<see cref="IsFreeText"/>).</summary>
    public static JToken FormatParamValue(T2IParamType type, object value, bool hashFreeText = false)
    {
        if (value is null)
        {
            return null;
        }
        if (type?.ViewType == ParamViewType.PROMPT && value is string prompt)
        {
            return $"@{type.ID}#{MetadataHelper.Sha8(prompt)}({prompt.Length})";
        }
        string text(string str) => str.Length > 120 || (hashFreeText && IsFreeText(str)) ? $"#{MetadataHelper.Sha8(str)}({str.Length})" : str;
        return value switch
        {
            string str => text(str),
            bool or int or long or float or double => JToken.FromObject(value),
            T2IModel model => new JObject() { ["name"] = T2IParamTypes.CleanModelName(model.Name), ["class"] = model.ModelClass?.ID },
            MediaFile media => MediaValue(media),
            List<string> list => new JArray(list.Select(s => (JToken)text(s))),
            List<T2IModel> models => new JArray(models.Select(m => (JToken)T2IParamTypes.CleanModelName(m.Name))),
            JToken tok => tok.ToString(Formatting.None),
            System.Collections.IEnumerable items => new JArray(items.Cast<object>().Select(i => i is MediaFile m ? MediaValue(m) : (JToken)$"{i}")),
            _ => $"{value}"
        };
    }

    /// <summary>Formats one node input value that is not a link: a string with prompt and media placeholders (<see cref="FormatString"/>), a number or boolean as it is, and any other value as it is unless its JSON is over 120 characters, then as "#hash(length)".</summary>
    public JToken FormatValue(string nodeId, string name, JToken value)
    {
        if (value.Type == JTokenType.String)
        {
            return FormatString(nodeId, name, value.ToString());
        }
        if (value is JValue)
        {
            return value.DeepClone();
        }
        string raw = value.ToString(Formatting.None);
        return raw.Length > 120 ? $"#{MetadataHelper.Sha8(raw)}({raw.Length})" : value.DeepClone();
    }

    /// <summary>Formats one input value of an edit: absent as null, a link as "id:slot", anything else by <see cref="FormatValue"/>.</summary>
    public JToken FormatEditValue(string nodeId, string name, JToken value)
    {
        if (value is null)
        {
            return null;
        }
        return IsLink(value) ? $"{value[0]}:{value[1]}" : FormatValue(nodeId, name, value);
    }

    /// <summary>Describes an edit: the "node" as "id:Class" and the "step" that made it, with the "input" changed and its "before" and "after" values (null when absent; "class_type" for a class change),
    /// or, for a node replaced outside CreateNode, "replaced" true and the class it "was".</summary>
    public JObject EditJson(EditRecord edit)
    {
        JObject result = new() { ["node"] = $"{edit.NodeId}:{edit.ClassType}" };
        if (edit.Field == "(node)")
        {
            result["replaced"] = true;
            result["was"] = edit.Before?.DeepClone();
        }
        else
        {
            result["input"] = edit.Field;
            result["before"] = FormatEditValue(edit.NodeId, edit.Field, edit.Before);
            result["after"] = FormatEditValue(edit.NodeId, edit.Field, edit.After);
        }
        result["step"] = StepName(edit.Step);
        return result;
    }

    /// <summary>Adds where a node came from to a JSON object: its "stage", the "origin" CreateNode call site as member@file:line (which names the code branch that made it), the "step" that made it and that step's "priority",
    /// the model "load" index when made inside one, the "compat" class of the model loaded when it was made, and how many times CreateNode "reused" it for an identical request or "overwritten" it under the same ID.</summary>
    public void AddProvenance(JObject target, NodeRecord record)
    {
        target["stage"] = record.Stage;
        target["origin"] = record.Origin;
        if (record.Step >= 0 && record.Step < StepRecords.Count)
        {
            StepRecord step = StepRecords[record.Step];
            target["step"] = step.Name;
            target["priority"] = step.Priority;
        }
        if (record.Load >= 0)
        {
            target["load"] = record.Load;
        }
        if (record.Compat is not null)
        {
            target["compat"] = record.Compat;
        }
        if (record.ReusedBy.Count > 0)
        {
            target["reused"] = record.ReusedBy.Count;
        }
        if (record.Overwrites > 0)
        {
            target["overwritten"] = record.Overwrites;
        }
    }

    /// <summary>Describes one node: its "class"; its provenance (<see cref="AddProvenance"/>), or for a node the trace did not see created, "untracked" true and the step it was "inserted_by" when known;
    /// then, unless <paramref name="brief"/>, its non-link "inputs" (<see cref="FormatValue"/>), its "links" as "id:slot", and its typed "outputs" by slot, each with the "comfy" type from object_info and the Swarm typing (<see cref="SlotType.AddTo"/>).</summary>
    /// <param name="id">Node ID.</param>
    /// <param name="node">Node JSON.</param>
    /// <param name="brief">If true, only the class and provenance.</param>
    public JObject NodeJson(string id, JObject node, bool brief)
    {
        string classType = $"{node["class_type"]}";
        JObject result = new() { ["class"] = classType };
        if (Nodes.TryGetValue(id, out NodeRecord record))
        {
            AddProvenance(result, record);
        }
        else if (Gen is not null)
        {
            result["untracked"] = true;
            if (InsertedBy.TryGetValue(id, out int step))
            {
                result["inserted_by"] = StepName(step);
            }
        }
        if (brief)
        {
            return result;
        }
        JObject scalars = [];
        JObject links = [];
        if (node["inputs"] is JObject inputs)
        {
            foreach (JProperty input in inputs.Properties())
            {
                if (IsLink(input.Value))
                {
                    links[input.Name] = $"{input.Value[0]}:{input.Value[1]}";
                }
                else if (input.Value.Type != JTokenType.Null)
                {
                    scalars[input.Name] = FormatValue(id, input.Name, input.Value);
                }
            }
        }
        if (scalars.Count > 0)
        {
            result["inputs"] = scalars;
        }
        if (links.Count > 0)
        {
            result["links"] = links;
        }
        JObject outputs = [];
        JArray comfyTypes = ComfyOutputTypes(classType);
        int slotCount = Math.Max(comfyTypes?.Count ?? 0, TypedSlotCounts.GetValueOrDefault(id, 0));
        for (int slot = 0; slot < slotCount; slot++)
        {
            SlotType slotType = SlotTypes.GetValueOrDefault($"{id}:{slot}");
            JToken comfyType = comfyTypes is not null && slot < comfyTypes.Count ? comfyTypes[slot] : null;
            if (slotType is null && comfyType is null)
            {
                continue;
            }
            JObject output = [];
            if (comfyType is not null)
            {
                output["comfy"] = ComfyInputValidation.TypeText(comfyType);
            }
            slotType?.AddTo(output);
            outputs[$"{slot}"] = output;
        }
        if (outputs.Count > 0)
        {
            result["outputs"] = outputs;
        }
        return result;
    }

    /// <summary>Returns the IDs of the upstream nodes feeding a link, nearest first, at most <see cref="MaxChainNodes"/>. Recurses only through nodes of the given stage; nodes of other stages and other samplers are listed as boundaries without recursing.</summary>
    /// <param name="link">The sampler input's link.</param>
    /// <param name="stage">The sampler's stage.</param>
    /// <param name="workflow">The workflow.</param>
    /// <param name="truncated">Set true if the chain has more nodes than are listed.</param>
    public JArray Chain(JArray link, string stage, JObject workflow, out bool truncated)
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
            result.Add(id);
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
        truncated = queue.Any(id => !seen.Contains(id));
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

    /// <summary>Returns the shortest downstream path from a node to every output node it reaches, each path a list of "id:Class" from the node to the output node.</summary>
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
                List<JToken> path = [];
                for (string cur = id; cur is not null; cur = parent[cur])
                {
                    path.Add($"{cur}:{workflow[cur]?["class_type"]}");
                }
                path.Reverse();
                result.Add(new JArray(path));
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

    /// <summary>Returns the "id:Class" reference of a node in a workflow.</summary>
    public static string NodeRef(JObject workflow, string id) => $"{id}:{workflow[id]?["class_type"]}";

    /// <summary>Returns a node input value read from the graph for a record: a link as "id:slot", any other value as it is, an absent input as null.</summary>
    public static JToken InputJson(JToken value) => value is null ? null : IsLink(value) ? $"{value[0]}:{value[1]}" : value.DeepClone();

    /// <summary>Describes a step that ran: its "name", registration "priority" and "source", what it "created", the model "load" it ran inside for a model-generation step,
    /// and the node IDs it "removed", "edited" in place, and "inserted_without_createnode", when any.</summary>
    /// <param name="step">The step record.</param>
    /// <param name="created">What it created: a node count, or a list of node IDs.</param>
    public static JObject StepJson(StepRecord step, JToken created)
    {
        JObject result = new() { ["name"] = step.Name, ["priority"] = step.Priority, ["source"] = step.Source, ["created"] = created };
        if (step.IsModelGen)
        {
            result["load"] = step.Load;
        }
        if (step.Removed.Count > 0)
        {
            result["removed"] = new JArray(step.Removed);
        }
        if (step.Edited.Count > 0)
        {
            result["edited"] = new JArray(step.Edited);
        }
        if (step.Inserted.Count > 0)
        {
            result["inserted_without_createnode"] = new JArray(step.Inserted);
        }
        return result;
    }

    /// <summary>Builds the trace summary, which is also the complete image record (tier 0 of <see cref="RecordTiers"/>): the log key and where the full trace is logged, the workflow identity, the main model, the steps, model loads with their LoRAs and patches,
    /// every sampler with its call, input chains (node IDs into the node table) and paths to the output nodes, segments, the node table with every node's provenance, inputs and output types, removals, reuse, rewires, edits, cache hits, checks against object_info, prompts, media and parameter reads.</summary>
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
            ["log_key"] = LogKey,
            ["full_trace"] = FullTraceNote(LogKey),
            ["workflow"] = new JObject()
            {
                ["source"] = "generated",
                ["sha256"] = HashIndentedJson(workflow),
                ["nodes"] = workflow.Count,
                ["nodes_created"] = Nodes.Count,
                ["nodes_removed"] = Nodes.Values.Count(n => n.RemovedBy is not null),
                ["features"] = new JArray(Gen.Features.OrderBy(f => f).Select(f => (JToken)f))
            },
            ["model"] = mainModel is null ? null : new JObject()
            {
                ["name"] = T2IParamTypes.CleanModelName(mainModel.Name),
                ["class"] = mainModel.ModelClass?.ID,
                ["compat"] = mainModel.ModelClass?.CompatClass?.ID
            }
        };
        if (Error is not null)
        {
            result["error"] = Error;
        }
        JArray steps = [];
        foreach (StepRecord step in StepRecords)
        {
            steps.Add(StepJson(step, step.Created));
        }
        result["steps"] = steps;
        if (Events.Count > 0)
        {
            result["events"] = new JArray(Events);
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
                ["out"] = new JObject() { ["model"] = load.ModelOut, ["textenc"] = load.ClipOut, ["vae"] = load.VaeOut }
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
                loadObj["loras"] = new JArray(load.Loras.Select(l => l.ToJson()));
            }
            JArray patches = [];
            foreach (StepRecord step in StepRecords.Where(s => s.IsModelGen && s.Load == load.Index))
            {
                List<string> created = [.. Nodes.Values.Where(n => n.Step == step.Index).Select(n => n.Id)];
                if (created.Count == 0 && step.Edited.Count == 0 && step.Removed.Count == 0 && step.Inserted.Count == 0)
                {
                    continue;
                }
                patches.Add(StepJson(step, new JArray(created)));
            }
            if (patches.Count > 0)
            {
                loadObj["patches"] = patches;
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
        result["loras"] = new JArray(Loras.Select(l => l.ToJson()));
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
                ["node"] = $"{property.Name}:{classType}"
            };
            if (SamplerCalls.TryGetValue(property.Name, out JObject args))
            {
                samplerObj["call"] = args.DeepClone();
            }
            JObject chains = [];
            JArray truncated = [];
            if (node["inputs"] is JObject inputs)
            {
                foreach (JProperty input in inputs.Properties())
                {
                    if (IsLink(input.Value))
                    {
                        chains[input.Name] = Chain(input.Value as JArray, stage, workflow, out bool cut);
                        if (cut)
                        {
                            truncated.Add(input.Name);
                        }
                    }
                }
            }
            samplerObj["inputs"] = chains;
            if (truncated.Count > 0)
            {
                samplerObj["truncated_inputs"] = truncated;
            }
            samplerObj["to_outputs"] = Downstream(property.Name, workflow, consumers);
            samplers.Add(samplerObj);
        }
        foreach ((string id, JObject args) in SamplerCalls)
        {
            if (!workflow.ContainsKey(id))
            {
                NodeRecord removedSampler = Nodes.GetValueOrDefault(id);
                samplers.Add(new JObject() { ["node"] = $"{id}:{removedSampler?.ClassType ?? "?"}", ["removed_by"] = removedSampler?.RemovedBy ?? "?", ["call"] = args.DeepClone() });
            }
        }
        result["samplers"] = samplers;
        if (SegmentParts.Count > 0)
        {
            JArray segments = (JArray)SegmentParts.DeepClone();
            foreach (JObject segment in segments.OfType<JObject>())
            {
                if (segment["crop"] is not JObject crop || workflow[$"{crop["bounds"]}"]?["inputs"] is not JObject bounds)
                {
                    continue;
                }
                crop["aspect_x"] = InputJson(bounds["aspect_x"]);
                crop["aspect_y"] = InputJson(bounds["aspect_y"]);
                if (IsLink(bounds["mask"]) && workflow[$"{bounds["mask"][0]}"] is JObject threshold && $"{threshold["class_type"]}" == "SwarmMaskThreshold")
                {
                    crop["threshold_min"] = InputJson(threshold["inputs"]?["min"]);
                    crop["threshold_max"] = InputJson(threshold["inputs"]?["max"]);
                }
                if (workflow[$"{crop["scaled_image"]}"]?["inputs"] is JObject scaled)
                {
                    crop["scaled_width"] = InputJson(scaled["width"]);
                    crop["scaled_height"] = InputJson(scaled["height"]);
                }
                crop["pasted_by"] = new JArray(consumers.GetValueOrDefault($"{crop["bounds"]}", []).Where(id => $"{workflow[id]?["class_type"]}".Contains("CompositeMasked")).Select(id => (JToken)NodeRef(workflow, id)));
            }
            result["segments"] = segments;
        }
        JObject nodeTable = [];
        foreach (JProperty property in workflow.Properties())
        {
            if (property.Value is JObject node)
            {
                nodeTable[property.Name] = NodeJson(property.Name, node, false);
            }
        }
        result["nodes"] = nodeTable;
        JArray removed = [];
        foreach (NodeRecord node in Nodes.Values.Where(n => n.RemovedBy is not null))
        {
            JObject entry = new() { ["node"] = $"{node.Id}:{node.ClassType}" };
            AddProvenance(entry, node);
            entry["removed_by"] = node.RemovedBy;
            removed.Add(entry);
        }
        if (removed.Count > 0)
        {
            result["removed"] = removed;
        }
        JArray reuses = [];
        foreach (NodeRecord node in Nodes.Values.Where(n => n.ReusedBy.Count > 0 || n.Overwrites > 0))
        {
            reuses.Add(new JObject() { ["node"] = $"{node.Id}:{node.ClassType}", ["reused_by"] = new JArray(node.ReusedBy.Select(r => r.ToJson())), ["overwrites"] = node.Overwrites });
        }
        if (reuses.Count > 0)
        {
            result["node_reuse"] = reuses;
        }
        if (Rewires.Count > 0)
        {
            result["rewires"] = new JArray(Rewires);
        }
        if (Edits.Count > 0)
        {
            result["edits"] = new JArray(Edits.Select(EditJson));
        }
        if (CacheHits.Count > 0)
        {
            result["cache_hits"] = new JArray(CacheHits);
        }
        JArray untracked = [];
        foreach (JProperty property in workflow.Properties())
        {
            if (!Nodes.ContainsKey(property.Name))
            {
                JObject entry = new() { ["node"] = $"{property.Name}:{property.Value["class_type"]}" };
                if (InsertedBy.TryGetValue(property.Name, out int step))
                {
                    entry["inserted_by"] = StepName(step);
                }
                untracked.Add(entry);
            }
        }
        if (untracked.Count > 0)
        {
            result["untracked_nodes"] = untracked;
        }
        JArray dangling = [];
        foreach (JProperty property in workflow.Properties())
        {
            if (property.Value["inputs"] is not JObject inputs)
            {
                continue;
            }
            foreach (JProperty input in inputs.Properties())
            {
                if (IsLink(input.Value) && !workflow.ContainsKey($"{input.Value[0]}"))
                {
                    dangling.Add(new JObject() { ["node"] = $"{property.Name}:{property.Value["class_type"]}", ["input"] = input.Name, ["link"] = $"{input.Value[0]}:{input.Value[1]}" });
                }
            }
        }
        if (dangling.Count > 0)
        {
            result["dangling_links"] = dangling;
        }
        if (ObjectInfo is not null)
        {
            JArray typeErrors = [];
            JArray inputErrors = [];
            JArray undeclared = [];
            JArray unknownClasses = [];
            foreach (JProperty property in workflow.Properties())
            {
                string classType = $"{property.Value["class_type"]}";
                string nodeRef = $"{property.Name}:{classType}";
                JObject problem(string input, string text)
                {
                    JObject entry = new() { ["node"] = nodeRef };
                    if (input is not null)
                    {
                        entry["input"] = input;
                    }
                    entry["problem"] = text;
                    return entry;
                }
                JObject inputs = property.Value["inputs"] as JObject ?? [];
                JObject comfyInputs;
                try
                {
                    comfyInputs = ComfyInputs(classType, inputs);
                }
                catch (ComfyInputValidation.DynamicInputError ex)
                {
                    inputErrors.Add(problem(ex.InputName, $"invalid dynamic input: {ex.Message}"));
                    continue;
                }
                catch (Exception ex)
                {
                    inputErrors.Add(problem(null, $"declares dynamic inputs ComfyUI cannot expand: {ex.Message}"));
                    continue;
                }
                if (comfyInputs is null)
                {
                    unknownClasses.Add(nodeRef);
                    continue;
                }
                HashSet<string> checkedInputs = [];
                foreach (string category in new[] { "required", "optional" })
                {
                    if (comfyInputs[category] is not JObject declared)
                    {
                        continue;
                    }
                    foreach (JProperty declaration in declared.Properties())
                    {
                        if (!checkedInputs.Add(declaration.Name))
                        {
                            continue;
                        }
                        if (!inputs.TryGetValue(declaration.Name, out JToken value))
                        {
                            if (category == "required")
                            {
                                inputErrors.Add(problem(declaration.Name, "required and missing"));
                            }
                            continue;
                        }
                        if (value is not JArray link || (IsLink(link) && !workflow.ContainsKey($"{link[0]}")))
                        {
                            continue;
                        }
                        string linkProblem = ComfyInputValidation.CheckLink(link, (declaration.Value as JArray)?.FirstOrDefault(), workflow, ObjectInfo, out bool typeMismatch);
                        if (linkProblem is not null)
                        {
                            (typeMismatch ? typeErrors : inputErrors).Add(problem(declaration.Name, linkProblem));
                        }
                    }
                }
                foreach (JProperty input in inputs.Properties())
                {
                    if (!new[] { "required", "optional", "hidden" }.Any(group => comfyInputs[group] is JObject set && set.ContainsKey(input.Name)))
                    {
                        undeclared.Add(new JObject() { ["node"] = nodeRef, ["input"] = input.Name });
                    }
                }
            }
            result["type_errors"] = typeErrors;
            result["input_errors"] = inputErrors;
            result["undeclared_inputs"] = undeclared;
            result["classes_missing_from_object_info"] = unknownClasses;
        }
        else
        {
            result["object_info"] = "not given: link types, declared inputs and node classes were not checked against ComfyUI";
        }
        JObject prompts = [];
        foreach ((string text, List<string> labels) in PromptLabels)
        {
            foreach (string label in labels)
            {
                prompts[$"@{label}#{MetadataHelper.Sha8(text)}"] = text.Length;
            }
        }
        result["prompts"] = prompts;
        JObject media = [];
        foreach ((string nodeId, MediaRecord record) in Media)
        {
            string value = workflow[nodeId]?["inputs"] is JObject mediaInputs ? mediaInputs.Properties().Where(p => p.Name.EndsWith("_base64")).Select(p => p.Value.ToString()).FirstOrDefault() : null;
            JObject entry = new() { ["param"] = record.Param };
            if (value is not null)
            {
                entry["sha8"] = MetadataHelper.Sha8(value);
            }
            else
            {
                entry["raw_sha8"] = record.Media?.RawData is byte[] raw ? MetadataHelper.Sha8(raw) : null;
            }
            entry.Merge(MediaFields(record));
            media[nodeId] = entry;
        }
        result["media"] = media;
        result["params_read"] = FormatParamReads(ParamReads);
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

    /// <summary>Builds the log-only trace detail: per-step tracker state and every input value, formatted by <see cref="FormatParamValue"/> (prompts, long text and media as hashes).</summary>
    public JObject BuildDetail()
    {
        JObject states = [];
        foreach (StepRecord step in StepRecords.Where(s => s.State is not null))
        {
            states[$"{step.Index} {step.Name}"] = step.State;
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
            ["input"] = input
        };
    }
}
