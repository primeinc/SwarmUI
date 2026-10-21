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
    public const int FormatVersion = 1;

    /// <summary>ExtraMeta key the trace summary is stored under.</summary>
    public const string MetadataKey = T2IMultiStepObjectBuilder.WorkflowTraceMetadataKey;

    /// <summary>Node classes treated as samplers for chain analysis.</summary>
    public static HashSet<string> SamplerClasses = ["KSampler", "KSamplerAdvanced", "SwarmKSampler", "SamplerCustom", "SamplerCustomAdvanced", "SwarmUnsampler"];

    /// <summary>Node input names that carry prompt text. Text in these inputs is never written raw.</summary>
    public static HashSet<string> PromptInputNames = ["text", "prompt", "text_g", "text_l", "lyrics", "caption", "match_text", "style"];

    /// <summary>Maximum number of nodes listed per sampler input chain.</summary>
    public static int MaxChainNodes = 48;

    /// <summary>Most characters of image metadata text (parameters, extra data and the trace record together, as <see cref="T2IParamInput.MetadataToString(JObject)"/> writes it) for formats that keep metadata in EXIF (every format but PNG).
    /// EXIF metadata is a UTF-16 UserComment, two bytes per character; within this limit it fits one 64 KiB JPEG APP1 segment (ImageSharp splits a larger profile across several, which most readers do not join), and the margin covers fields Swarm adds when it saves the image.</summary>
    public static int MaxExifMetadataChars = 30_000;

    /// <summary>The generator being traced.</summary>
    public WorkflowGenerator Gen = gen;

    /// <summary>Optional ComfyUI object_info data, used to add Comfy output types and to check link types.</summary>
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

        /// <summary>Distinct values returned, in first-read order, up to <see cref="MaxDistinctParamValues"/>.</summary>
        public List<object> Values = [];
    }

    /// <summary>Most distinct values kept per parameter read record.</summary>
    public static int MaxDistinctParamValues = 8;

    /// <summary>A media input embedded into a node.</summary>
    /// <param name="Param">Source parameter name, eg 'initimage' or 'promptimages.0'.</param>
    /// <param name="Media">The media file.</param>
    /// <param name="Width">Embedded width, if resized.</param>
    /// <param name="Height">Embedded height, if resized.</param>
    public record class MediaRecord(string Param, MediaFile Media, int? Width, int? Height)
    {
        /// <summary>The formatted size and type, once <see cref="MediaSize(MediaRecord)"/> has computed it.</summary>
        public string SizeText;
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

    /// <summary>Changes to nodes already in the workflow, in order.</summary>
    public List<EditRecord> Edits = [];

    /// <summary>Every node as it was at the last <see cref="Checkpoint"/>, by ID.</summary>
    public Dictionary<string, NodeSnapshot> Snapshots = [];

    /// <summary>Node IDs CreateNode wrote since the last <see cref="Checkpoint"/>.</summary>
    public HashSet<string> CreatedSinceCheckpoint = [];

    /// <summary>Index of the step during which each node entered the workflow without CreateNode, by node ID.</summary>
    public Dictionary<string, int> InsertedBy = [];

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

    /// <summary>Strings and byte arrays at least this long are hashed without copying them and have their hash cached in <see cref="LongHashes"/>.</summary>
    public const int CachedHashMinLength = 4096;

    /// <summary>SHA-256 hex of long strings and byte arrays, by reference, so base64 media, image bytes and workflow texts are hashed once however often the trace names them.
    /// Keyed by reference: strings are immutable, and Swarm replaces a media file's RawData array rather than writing into it.</summary>
    public static readonly ConditionalWeakTable<object, string> LongHashes = new();

    /// <summary>Returns the SHA-256 hex of a string's UTF-8 bytes, the same as <c>Utilities.HashSHA256(text.EncodeUTF8())</c>. Long strings are encoded through a small buffer rather than copied, and their hash is cached.</summary>
    public static string HashText(string text)
    {
        text ??= "";
        if (text.Length < CachedHashMinLength)
        {
            return Utilities.HashSHA256(text.EncodeUTF8());
        }
        return LongHashes.GetValue(text, static t => HashWritten(writer => writer.Write((string)t)));
    }

    /// <summary>Returns the SHA-256 hex of some bytes, cached for long arrays.</summary>
    public static string HashBytes(byte[] data)
    {
        data ??= [];
        if (data.Length < CachedHashMinLength)
        {
            return Utilities.HashSHA256(data);
        }
        return LongHashes.GetValue(data, static d => Utilities.HashSHA256((byte[])d));
    }

    /// <summary>Returns the first 8 hex chars of the SHA-256 of some text.</summary>
    public static string Sha8(string text) => HashText(text)[..8];

    /// <summary>Returns the first 8 hex chars of the SHA-256 of some bytes.</summary>
    public static string Sha8(byte[] data) => HashBytes(data)[..8];

    /// <summary>Returns the SHA-256 hex of the text a writer callback writes, as UTF-8 without a byte order mark (<see cref="StringConversionHelper.UTF8Encoding"/>, as <c>EncodeUTF8</c> uses), encoding through a small buffer instead of building the text or its bytes.</summary>
    public static string HashWritten(Action<TextWriter> write)
    {
        using SHA256 sha = SHA256.Create();
        using (CryptoStream crypto = new(Stream.Null, sha, CryptoStreamMode.Write))
        using (StreamWriter writer = new(crypto, StringConversionHelper.UTF8Encoding, 4096))
        {
            write(writer);
        }
        return Utilities.BytesToHex(sha.Hash);
    }

    /// <summary>Returns the SHA-256 of a token's indented JSON text as UTF-8, the same as hashing <c>token.ToString()</c>, without building the text.</summary>
    public static string HashIndentedJson(JToken token)
    {
        return HashWritten(writer =>
        {
            JsonTextWriter json = new(writer) { Formatting = Formatting.Indented };
            token.WriteTo(json);
            json.Flush();
        });
    }

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
            Compat = LoadStack.Count > 0 ? Loads[LoadStack[^1]].Compat : Gen.FinalLoadedModel?.ModelClass?.CompatClass?.ID,
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
            segment["prompt"] = $"@{label}#{Sha8(text)}";
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
            ["mask_text"] = $"@{CurrentStage}.mask#{Sha8(part.DataText)}",
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
            Checkpoint(InnermostStep);
            Logs.Debug($"[WorkflowTrace] generation failed, partial trace:\n{Build().ToString(Formatting.Indented)}\n[WorkflowTrace] detail:\n{BuildDetail().ToString(Formatting.Indented)}");
        }
        catch (Exception traceEx)
        {
            Logs.Debug($"[WorkflowTrace] generation failed and the partial trace could not be built: {traceEx.ReadableString()}");
        }
    }

    /// <summary>The record stored in the image metadata, once finished.</summary>
    public JObject Record;

    /// <summary>Returns how the server debug log labels the trace of an input.</summary>
    public static string LogReference(T2IParamInput input) => $"[WorkflowTrace] request {input.UserRequestId}";

    /// <summary>Finishes the trace after generation: logs the full summary plus detail, and stores the image record in the input's ExtraMeta: <see cref="RecordFull"/> for PNG, otherwise the tier of <see cref="RecordTiers"/> that <see cref="ForImageMetadata"/> picks.</summary>
    public void Finish()
    {
        Gen.UserInput.ParamReadTracker = null;
        Checkpoint(InnermostStep);
        Summary = Build();
        string logReference = LogReference(Gen.UserInput);
        Logs.Debug($"{logReference}:\n{Summary.ToString(Formatting.Indented)}\n[WorkflowTrace] detail:\n{BuildDetail().ToString(Formatting.Indented)}");
        string format = ImageFormatOf(Gen.UserInput);
        Record = format == "PNG" ? RecordFull(Summary, logReference) : ForImageMetadata(RecordTiers(Summary, logReference), format, OtherMetadataChars(Gen.UserInput));
        Gen.UserInput.ExtraMeta[MetadataKey] = Record;
    }

    /// <summary>Records that <see cref="Finish"/> threw: logs the error, detaches the parameter tracker, and stores the error as the image record. The generated workflow is unaffected.</summary>
    public void FinishFailed(Exception ex)
    {
        Gen.UserInput.ParamReadTracker = null;
        Logs.Error($"{LogReference(Gen.UserInput)}: the trace could not be finished; the generation is unaffected: {ex.ReadableString()}");
        Gen.UserInput.ExtraMeta[MetadataKey] = new JObject()
        {
            ["version"] = FormatVersion,
            ["trace_error"] = $"{ex.GetType().Name}: {ex.Message}"
        };
    }

    /// <summary>Returns the length of an input's image metadata text without the trace record: its parameter metadata and other extra data, built without side effects (no model hashing).</summary>
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

    /// <summary>Returns the node ID a description line or output path entry starts with ("id:Class..."), or null for a truncation marker.</summary>
    public static string LineNodeId(string line) => line is null || line.StartsWith("...") || !line.Contains(':') ? null : line.Before(':');

    /// <summary>Builds the complete image record, which PNG images store and <see cref="RecordTiers"/> starts from: every section of the summary (steps, model loads with their LoRAs and patches, parameter reads, decisions, checks, prompts, media),
    /// with each sampler's input chains given as node IDs into a node table that lists every node of the workflow with its provenance, inputs and output types.</summary>
    public JObject RecordFull(JObject summary, string logReference)
    {
        JObject full = new()
        {
            ["version"] = FormatVersion,
            ["full_trace"] = $"server debug log, '{logReference}'"
        };
        foreach (JProperty property in summary.Properties())
        {
            if (property.Name is not ("version" or "samplers" or "nodes"))
            {
                full[property.Name] = property.Value.DeepClone();
            }
        }
        JArray samplers = [];
        foreach (JObject sampler in (summary["samplers"] as JArray ?? []).OfType<JObject>())
        {
            string node = $"{sampler["node"]}";
            string samplerId = LineNodeId(node);
            string classType = samplerId is not null && Gen.Workflow[samplerId] is JObject samplerNode ? $"{samplerNode["class_type"]}" : null;
            JObject entry = new() { ["node"] = classType is null ? node : $"{samplerId}:{classType}" };
            if (sampler["call"] is JToken call)
            {
                entry["call"] = call.DeepClone();
            }
            if (sampler["inputs"] is JObject inputs)
            {
                JObject ids = [];
                foreach (JProperty input in inputs.Properties())
                {
                    JArray chain = [];
                    foreach (JToken line in input.Value as JArray ?? [])
                    {
                        chain.Add(LineNodeId($"{line}") ?? $"{line}");
                    }
                    ids[input.Name] = chain;
                }
                entry["inputs"] = ids;
            }
            if (sampler["to_outputs"] is JToken outputs)
            {
                entry["to_outputs"] = outputs.DeepClone();
            }
            samplers.Add(entry);
        }
        full["samplers"] = samplers;
        JObject nodes = [];
        foreach (JProperty property in Gen.Workflow.Properties())
        {
            if (property.Value is JObject node)
            {
                nodes[property.Name] = NodeLines.GetValueOrDefault(property.Name) ?? DescribeNode(property.Name, node, null);
            }
        }
        full["nodes"] = nodes;
        return full;
    }

    /// <summary>Builds the image record tiers, most complete first; <see cref="ForImageMetadata"/> keeps the first that fits the format. Tier 0 is <see cref="RecordFull"/>.
    /// Tier 1 reduces every node to its class, stage and provenance, and the parameter reads to those made for a section. Tier 2 drops the node table and gives the steps, removals, rewires, edits, cache hits and node reuse as counts.
    /// Tier 3 keeps the workflow identity, the model and the samplers' nodes and calls. Tier 4 keeps the workflow identity and the sampler count, so it fits whenever the rest of the metadata leaves room for a few hundred characters.
    /// Each tier below 0 says what it omitted and where the full trace is in the server log.</summary>
    public List<JObject> RecordTiers(JObject summary, string logReference)
    {
        JObject full = RecordFull(summary, logReference);
        JObject reduced = (JObject)full.DeepClone();
        JObject briefNodes = [];
        foreach (JProperty property in Gen.Workflow.Properties())
        {
            if (property.Value is not JObject node)
            {
                continue;
            }
            NodeRecord record = Nodes.GetValueOrDefault(property.Name);
            briefNodes[property.Name] = record is null ? $"{property.Name}:{node["class_type"]} <untracked>" : $"{property.Name}:{node["class_type"]}@{record.Stage} <{Provenance(record)}>";
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
            ["full_trace"] = full["full_trace"].DeepClone(),
            ["workflow"] = full["workflow"]?.DeepClone(),
            ["model"] = full["model"]?.DeepClone(),
            ["samplers"] = brief,
            ["omitted"] = $"everything but the samplers' nodes and calls, to fit the image metadata; the full trace is in the {full["full_trace"]}"
        };
        JObject notice = new()
        {
            ["version"] = FormatVersion,
            ["full_trace"] = full["full_trace"].DeepClone(),
            ["workflow"] = full["workflow"]?.DeepClone(),
            ["samplers"] = brief.Count,
            ["omitted"] = $"the whole trace, to fit the image metadata; the full trace is in the {full["full_trace"]}"
        };
        return [full, reduced, counted, minimal, notice];
    }

    /// <summary>Node description lines of the last <see cref="Build"/>, by node ID.</summary>
    public Dictionary<string, string> NodeLines = [];

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

    /// <summary>Returns the record tier to store in an image's metadata: the first for PNG, whose text chunks have no size limit; for formats that keep metadata in EXIF, the first tier that fits <see cref="MaxExifMetadataChars"/> together with the rest of the metadata, else the last tier.</summary>
    /// <param name="tiers">Record tiers, most complete first (<see cref="RecordTiers"/>).</param>
    /// <param name="format">Image format, as <see cref="T2IParamTypes.ImageFormat"/> values ("PNG", "JPG", "WEBP", ...).</param>
    /// <param name="otherChars">Length of the rest of the image metadata text (<see cref="OtherMetadataChars"/>).</param>
    /// <param name="maxChars">Limit for EXIF formats, or -1 for <see cref="MaxExifMetadataChars"/>.</param>
    public static JObject ForImageMetadata(List<JObject> tiers, string format, int otherChars, int maxChars = -1)
    {
        if (maxChars < 0)
        {
            maxChars = MaxExifMetadataChars;
        }
        if (format == "PNG")
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

    /// <summary>The "posted" value of a workflow record until a ComfyUI backend posts it through <see cref="ComfyUIAPIAbstractBackend.AwaitJobLive"/>.</summary>
    public const string NotPosted = "not observed: no post through ComfyUIAPIAbstractBackend.AwaitJobLive";

    /// <summary>Returns the "workflow" objects to update for an input: the summary's, and the stored image metadata record's when the image format stores a notice instead of the summary.</summary>
    public static List<JObject> WorkflowRecords(T2IParamInput input, JObject summary)
    {
        List<JObject> records = [];
        if (summary?["workflow"] is JObject info)
        {
            records.Add(info);
        }
        if (input?.ExtraMeta.GetValueOrDefault(MetadataKey) is JObject stored && stored != summary && stored["workflow"] is JObject storedInfo)
        {
            records.Add(storedInfo);
        }
        return records;
    }

    /// <summary>Returns the "${...}" placeholders in a workflow text, in order of first appearance.</summary>
    public static List<string> Placeholders(string text)
    {
        List<string> result = [];
        int start = text.IndexOf("${", StringComparison.Ordinal);
        while (start >= 0)
        {
            int end = text.IndexOf('}', start + 2);
            if (end < 0)
            {
                break;
            }
            string placeholder = text[start..(end + 1)];
            if (!result.Contains(placeholder))
            {
                result.Add(placeholder);
            }
            start = text.IndexOf("${", end + 1, StringComparison.Ordinal);
        }
        return result;
    }

    /// <summary>Records the workflow text the backend will send to ComfyUI, in the summary and the stored image metadata record: its hash and length, whether anything rewrote the generator's text, and which "${...}" placeholders the rewrite filled (such as uploaded image names).
    /// An error is logged, never thrown: the send is unaffected.</summary>
    /// <param name="workflow">The final workflow JSON text.</param>
    /// <param name="generated">The workflow text the generator returned.</param>
    public void RecordSent(string workflow, string generated)
    {
        try
        {
            string sha = HashText(workflow);
            List<JObject> records = WorkflowRecords(Gen.UserInput, Summary);
            if (records.Count == 0)
            {
                return;
            }
            bool rewritten = workflow != generated;
            List<string> filled = rewritten ? [.. Placeholders(generated).Except(Placeholders(workflow))] : [];
            foreach (JObject info in records)
            {
                info["sent_sha256"] = sha;
                info["sent_length"] = workflow.Length;
                info["rewritten_after_generate"] = rewritten;
                if (rewritten)
                {
                    info["placeholders_filled"] = new JArray(filled.Select(p => (JToken)p));
                }
                info["posted"] = NotPosted;
            }
            Logs.Debug($"{LogReference(Gen.UserInput)}: sent workflow sha256={sha} length={workflow.Length} rewritten_after_generate={rewritten}{(rewritten ? $" placeholders_filled=[{filled.JoinString(", ")}]" : "")}");
        }
        catch (Exception ex)
        {
            Logs.Error($"{LogReference(Gen.UserInput)}: the sent workflow could not be recorded; the generation is unaffected: {ex.ReadableString()}");
        }
    }

    /// <summary>Records the workflow a ComfyUI backend posts to /prompt for an input, into its stored trace record. Does nothing for an input without one. An error is logged, never thrown: the post is unaffected.</summary>
    /// <param name="input">The input being generated.</param>
    /// <param name="workflow">The workflow JSON text posted as the "prompt".</param>
    /// <param name="backend">Name of the posting backend type.</param>
    public static void RecordPosted(T2IParamInput input, string workflow, string backend)
    {
        if (input?.ExtraMeta.GetValueOrDefault(MetadataKey) is not JObject stored || stored["workflow"] is not JObject info)
        {
            return;
        }
        try
        {
            string sha = HashText(workflow);
            bool? matchesSent = info["sent_sha256"] is JValue sent ? $"{sent}" == sha : null;
            info["posted"] = new JObject()
            {
                ["sha256"] = sha,
                ["length"] = workflow.Length,
                ["backend"] = backend,
                ["matches_sent"] = matchesSent
            };
            Logs.Debug($"{LogReference(input)}: {backend} posted workflow sha256={sha} length={workflow.Length} matches_sent={matchesSent?.ToString() ?? "unknown"}");
        }
        catch (Exception ex)
        {
            Logs.Error($"{LogReference(input)}: the posted workflow could not be recorded; the generation is unaffected: {ex.ReadableString()}");
        }
    }

    /// <summary>Records a workflow that did not come from the generator (custom or raw workflows): logs its description, and stores it in the input's ExtraMeta, without the node list when the image format cannot hold it (<see cref="ForImageMetadata"/>).
    /// An error is logged and stored as the record, never thrown: the generation is unaffected.</summary>
    public static void RecordExternalWorkflow(T2IParamInput input, string source, string workflow)
    {
        try
        {
            JObject summary = DescribeExternalWorkflow(source, workflow);
            string logReference = LogReference(input);
            Logs.Debug($"{logReference} ({source} workflow):\n{summary.ToString(Formatting.Indented)}");
            string format = ImageFormatOf(input);
            if (format == "PNG")
            {
                input.ExtraMeta[MetadataKey] = summary;
                return;
            }
            JObject withoutNodes = (JObject)summary.DeepClone();
            withoutNodes.Remove("nodes");
            withoutNodes["omitted"] = $"the node list, to fit the image metadata; the full description is in the server debug log, '{logReference}'";
            input.ExtraMeta[MetadataKey] = ForImageMetadata([summary, withoutNodes], format, OtherMetadataChars(input));
        }
        catch (Exception ex)
        {
            Logs.Error($"{LogReference(input)}: the {source} workflow could not be recorded; the generation is unaffected: {ex.ReadableString()}");
            input.ExtraMeta[MetadataKey] = new JObject()
            {
                ["version"] = FormatVersion,
                ["trace_error"] = $"{ex.GetType().Name}: {ex.Message}"
            };
        }
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
                ["sent_sha256"] = HashText(workflow),
                ["sent_length"] = workflow?.Length ?? 0,
                ["posted"] = NotPosted
            }
        };
        try
        {
            JObject parsed = workflow.ParseToJson();
            WorkflowTrace reader = new(null) { HashFreeText = true };
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

    /// <summary>Formats a string node input with placeholders for prompts, media, and long text. Prompt text that is part of a registered prompt, rather than all of one, is labeled "~part" with the labels of the shortest registered prompt that contains it.</summary>
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
        if (HashFreeText && (value.Length > 64 || value.Any(char.IsWhiteSpace)) && !ModelFileExtensions.Any(e => value.EndsWith(e, StringComparison.OrdinalIgnoreCase)))
        {
            return $"#{Sha8(value)}({value.Length})";
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
                return $"{containerLabels.Select(l => $"@{l}").JoinString("=")}~part#{Sha8(value)}";
            }
            return $"#{Sha8(value)}({value.Length})";
        }
        if (value.Length > 120)
        {
            return $"#{Sha8(value)}({value.Length})";
        }
        return value;
    }

    /// <summary>Returns " WxH mime" for a media record, reading the image size from its header if it was not resized. Computed once per record.</summary>
    public static string MediaSize(MediaRecord media)
    {
        if (media.SizeText is not null)
        {
            return media.SizeText;
        }
        int? width = media.Width, height = media.Height;
        if ((width is null || height is null) && media.Media is ImageFile img && img.Type.MetaType == MediaMetaType.Image && img.RawData is not null)
        {
            try
            {
                SixLabors.ImageSharp.ImageInfo info = SixLabors.ImageSharp.Image.Identify(img.RawData);
                width = info.Width;
                height = info.Height;
            }
            catch (Exception)
            {
                width = null;
                height = null;
            }
        }
        string size = width is null || height is null ? "" : $" {width}x{height}";
        media.SizeText = $"{size} {media.Media?.Type?.MimeType ?? "?"}";
        return media.SizeText;
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

    /// <summary>Formats one node input value: a link as "id:slot", a string with prompt and media placeholders (<see cref="FormatString"/>), anything over 120 characters as a hash, an absent value as "(absent)".</summary>
    public string FormatInput(string nodeId, string name, JToken value)
    {
        if (value is null)
        {
            return "(absent)";
        }
        if (IsLink(value))
        {
            return $"{value[0]}:{value[1]}";
        }
        if (value.Type == JTokenType.String)
        {
            return FormatString(nodeId, name, value.ToString());
        }
        string raw = value.ToString(Formatting.None);
        return raw.Length > 120 ? $"#{Sha8(raw)}({raw.Length})" : raw;
    }

    /// <summary>Describes an edit as "id:Class.input: before -&gt; after by step", or "id:Class replaced outside CreateNode (was Class) by step".</summary>
    public string DescribeEdit(EditRecord edit)
    {
        if (edit.Field == "(node)")
        {
            return $"{edit.NodeId}:{edit.ClassType} replaced outside CreateNode (was {edit.Before}) by {StepName(edit.Step)}";
        }
        return $"{edit.NodeId}:{edit.ClassType}.{edit.Field}: {FormatInput(edit.NodeId, edit.Field, edit.Before)} -> {FormatInput(edit.NodeId, edit.Field, edit.After)} by {StepName(edit.Step)}";
    }

    /// <summary>Describes one node as a single line: "id:Class@stage &lt;origin&gt; [scalars] {links} =&gt; outputs".</summary>
    /// <param name="id">Node ID.</param>
    /// <param name="node">Node JSON.</param>
    /// <param name="relativeStage">If non-null, the stage tag is only shown when the node's stage differs from this, and the origin is omitted.</param>
    public string DescribeNode(string id, JObject node, string relativeStage)
    {
        if (DescribeCache is null)
        {
            return DescribeNodeUncached(id, node, relativeStage);
        }
        string key = $"{id}\n{relativeStage}";
        if (!DescribeCache.TryGetValue(key, out string text))
        {
            text = DescribeNodeUncached(id, node, relativeStage);
            DescribeCache[key] = text;
        }
        return text;
    }

    /// <summary>Node descriptions by ID and relative stage while <see cref="Build"/> runs (the workflow does not change during it), or null.</summary>
    public Dictionary<string, string> DescribeCache;

    /// <summary>Formats where a node came from: "member@file:line by 'step' p{priority}", plus " load#N" inside a model load, " compat=ID" for the model loaded when it was created,
    /// and " reused xN" or " overwritten xN" when CreateNode returned it for an identical request or replaced it. The CreateNode call site names the code branch that made the node.</summary>
    public string Provenance(NodeRecord record)
    {
        string text = record.Origin;
        if (record.Step >= 0 && record.Step < StepRecords.Count)
        {
            StepRecord step = StepRecords[record.Step];
            text += $" by '{step.Name}' p{step.Priority}";
        }
        if (record.Load >= 0)
        {
            text += $" load#{record.Load}";
        }
        if (record.Compat is not null)
        {
            text += $" compat={record.Compat}";
        }
        if (record.ReusedBy.Count > 0)
        {
            text += $" reused x{record.ReusedBy.Count}";
        }
        if (record.Overwrites > 0)
        {
            text += $" overwritten x{record.Overwrites}";
        }
        return text;
    }

    /// <summary>Implements <see cref="DescribeNode"/>.</summary>
    public string DescribeNodeUncached(string id, JObject node, string relativeStage)
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
            sb.Append($" <{Provenance(record)}>");
        }
        else if (record is null && Gen is not null)
        {
            sb.Append(InsertedBy.TryGetValue(id, out int step) ? $" <untracked, inserted by {StepName(step)}>" : " <untracked>");
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
                else if (input.Value.Type != JTokenType.Null)
                {
                    scalars.Add($"{input.Name}={FormatInput(id, input.Name, input.Value)}");
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
        DescribeCache = [];
        try
        {
            return BuildSummary();
        }
        finally
        {
            DescribeCache = null;
        }
    }

    /// <summary>Implements <see cref="Build"/>.</summary>
    public JObject BuildSummary()
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
                ["sha256"] = HashIndentedJson(workflow),
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
            if (step.Edited.Count > 0)
            {
                text += $" edited=[{step.Edited.JoinString(",")}]";
            }
            if (step.Inserted.Count > 0)
            {
                text += $" inserted_without_createnode=[{step.Inserted.JoinString(",")}]";
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
            JArray patches = [];
            foreach (StepRecord step in StepRecords.Where(s => s.IsModelGen && s.Load == load.Index))
            {
                List<string> created = [.. Nodes.Values.Where(n => n.Step == step.Index).Select(n => n.Id)];
                if (created.Count == 0 && step.Edited.Count == 0 && step.Removed.Count == 0 && step.Inserted.Count == 0)
                {
                    continue;
                }
                string patch = $"{step.Priority} {step.Name} ({step.Source}) created=[{created.JoinString(",")}]";
                if (step.Edited.Count > 0)
                {
                    patch += $" edited=[{step.Edited.JoinString(",")}]";
                }
                if (step.Removed.Count > 0)
                {
                    patch += $" removed=[{step.Removed.JoinString(",")}]";
                }
                if (step.Inserted.Count > 0)
                {
                    patch += $" inserted_without_createnode=[{step.Inserted.JoinString(",")}]";
                }
                patches.Add(patch);
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
            JArray segments = (JArray)SegmentParts.DeepClone();
            foreach (JObject segment in segments.OfType<JObject>())
            {
                if (segment["crop"] is not JObject crop || workflow[$"{crop["bounds"]}"]?["inputs"] is not JObject bounds)
                {
                    continue;
                }
                crop["aspect"] = $"{bounds["aspect_x"]}:{bounds["aspect_y"]}";
                if (IsLink(bounds["mask"]) && workflow[$"{bounds["mask"][0]}"] is JObject threshold && $"{threshold["class_type"]}" == "SwarmMaskThreshold")
                {
                    crop["threshold"] = $"{threshold["inputs"]?["min"]}..{threshold["inputs"]?["max"]}";
                }
                if (workflow[$"{crop["scaled_image"]}"]?["inputs"] is JObject scaled)
                {
                    crop["scaled_to"] = $"{scaled["width"]}x{scaled["height"]}";
                }
                crop["pasted_by"] = new JArray(consumers.GetValueOrDefault($"{crop["bounds"]}", []).Where(id => $"{workflow[id]?["class_type"]}".Contains("CompositeMasked")).Select(id => (JToken)$"{id}:{workflow[id]["class_type"]}"));
            }
            result["segments"] = segments;
        }
        JObject nodesByStage = [];
        NodeLines = [];
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
            string line = DescribeNode(property.Name, node, null);
            NodeLines[property.Name] = line;
            stageNodes.Add(line);
        }
        result["nodes"] = nodesByStage;
        JArray removed = [];
        foreach (NodeRecord node in Nodes.Values.Where(n => n.RemovedBy is not null))
        {
            removed.Add($"{node.Id}:{node.ClassType}@{node.Stage} <{Provenance(node)}> removed by {node.RemovedBy}");
        }
        if (removed.Count > 0)
        {
            result["removed"] = removed;
        }
        JArray reuses = [];
        foreach (NodeRecord node in Nodes.Values.Where(n => n.ReusedBy.Count > 0 || n.Overwrites > 0))
        {
            reuses.Add($"{node.Id}:{node.ClassType} reused by [{node.ReusedBy.JoinString(", ")}] overwrites={node.Overwrites}");
        }
        if (reuses.Count > 0)
        {
            result["node_reuse"] = reuses;
        }
        if (Rewires.Count > 0)
        {
            result["rewires"] = new JArray(Rewires.Select(r => (JToken)r));
        }
        if (Edits.Count > 0)
        {
            result["edits"] = new JArray(Edits.Select(e => (JToken)DescribeEdit(e)));
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
                untracked.Add($"{property.Name}:{property.Value["class_type"]}{(InsertedBy.TryGetValue(property.Name, out int step) ? $" inserted by {StepName(step)}" : "")}");
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
                    dangling.Add($"{property.Name}:{property.Value["class_type"]}.{input.Name} links to missing node {input.Value[0]}:{input.Value[1]}");
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
        else
        {
            result["object_info"] = "not given: link types, declared inputs and node classes were not checked against ComfyUI";
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
            string value = workflow[nodeId]?["inputs"] is JObject mediaInputs ? mediaInputs.Properties().Where(p => p.Name.EndsWith("_base64")).Select(p => p.Value.ToString()).FirstOrDefault() : null;
            string hash = value is not null ? Sha8(value) : record.Media?.RawData is byte[] raw ? $"raw:{Sha8(raw)}" : "?";
            media[nodeId] = $"@{record.Param}#{hash}{MediaSize(record)}";
        }
        result["media"] = media;
        JObject parameters = [];
        foreach ((string key, ParamReadRecord read) in ParamReads.OrderBy(p => p.Key))
        {
            JToken first = FormatParamValue(read.Type, read.FirstValue);
            JToken last = FormatParamValue(read.Type, read.LastValue);
            string firstText = first?.ToString(Formatting.None) ?? "null";
            string lastText = last?.ToString(Formatting.None) ?? "null";
            string text = firstText == lastText && read.FirstSource == read.LastSource ? $"{firstText} ({read.FirstSource})" : $"{firstText} ({read.FirstSource}) then {lastText} ({read.LastSource})";
            if (read.Values.Count > 2)
            {
                text += $"; {read.Values.Count}{(read.Values.Count >= MaxDistinctParamValues ? "+" : "")} distinct values over {read.Count} reads: {read.Values.Select(v => FormatParamValue(read.Type, v)?.ToString(Formatting.None) ?? "null").JoinString(" | ")}";
            }
            parameters[key] = text;
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

    /// <summary>Builds the log-only trace detail: per-step tracker state and the raw input values.</summary>
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
