using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using static SwarmUITests.WorkflowGeneratorHarness;

namespace SwarmUITests;

/// <summary>Tests the ComfyUI object_info side of <see cref="WorkflowTrace"/> typing: Comfy output types on every slot, link type checks, undeclared inputs, and output-node detection.</summary>
public class WorkflowTraceTypingTests : SwarmUITest
{
    /// <summary>Prepares the basics.</summary>
    [OneTimeSetUp]
    public static void PreInit()
    {
        Setup();
        Init();
    }

    /// <summary>ComfyUI object_info entries, in the shape of ComfyUI server.py node_info() and comfy_api _io.py get_v1_info(), for every node class in an SDXL checkpoint text-to-image workflow.
    /// Each entry is transcribed from the node's INPUT_TYPES / define_schema, RETURN_TYPES and OUTPUT_NODE; python_module names the source. Choice lists are cut to the values the workflow uses.
    /// Sources: ComfyUI 7ef81780 nodes.py (CheckpointLoaderSimple, EmptyLatentImage, VAEDecode), comfy_extras/nodes_clip_sdxl.py (CLIPTextEncodeSDXL);
    /// SwarmUI ExtraNodes/SwarmComfyCommon/SwarmKSampler.py and SwarmSaveImageWS.py.</summary>
    public const string SdxlObjectInfo = """
        {
            "CheckpointLoaderSimple": {
                "input": { "required": { "ckpt_name": [["test\\ckpt\\stable-diffusion-xl-v1-base.safetensors"], {}] } },
                "output": ["MODEL", "CLIP", "VAE"], "output_name": ["MODEL", "CLIP", "VAE"], "output_node": false, "python_module": "nodes"
            },
            "EmptyLatentImage": {
                "input": { "required": { "width": ["INT", {"default": 1024}], "height": ["INT", {"default": 1024}], "batch_size": ["INT", {"default": 1}] } },
                "output": ["LATENT"], "output_name": ["LATENT"], "output_node": false, "python_module": "nodes"
            },
            "CLIPTextEncodeSDXL": {
                "input": { "required": {
                    "clip": ["CLIP", {}], "width": ["INT", {"default": 1024}], "height": ["INT", {"default": 1024}], "crop_w": ["INT", {"default": 0}], "crop_h": ["INT", {"default": 0}],
                    "target_width": ["INT", {"default": 1024}], "target_height": ["INT", {"default": 1024}], "text_g": ["STRING", {"multiline": true}], "text_l": ["STRING", {"multiline": true}] } },
                "output": ["CONDITIONING"], "output_name": ["CONDITIONING"], "output_node": false, "python_module": "comfy_extras.nodes_clip_sdxl"
            },
            "SwarmKSampler": {
                "input": {
                    "required": {
                        "model": ["MODEL"], "noise_seed": ["INT", {"default": 0}], "steps": ["INT", {"default": 20}], "cfg": ["FLOAT", {"default": 8.0}],
                        "sampler_name": [["euler"]], "scheduler": [["normal"]], "positive": ["CONDITIONING"], "negative": ["CONDITIONING"], "latent_image": ["LATENT"],
                        "start_at_step": ["INT", {"default": 0}], "end_at_step": ["INT", {"default": 10000}], "var_seed": ["INT", {"default": 0}], "var_seed_strength": ["FLOAT", {"default": 0.0}],
                        "sigma_max": ["FLOAT", {"default": -1}], "sigma_min": ["FLOAT", {"default": -1}], "rho": ["FLOAT", {"default": 7.0}], "add_noise": [["enable", "disable"]],
                        "return_with_leftover_noise": [["disable", "enable"]], "previews": [["default", "none", "one", "second", "iterate", "animate"]],
                        "tile_sample": ["BOOLEAN", {"default": false}], "tile_size": ["INT", {"default": 1024}]
                    },
                    "optional": { "model_negative": ["MODEL"] }
                },
                "output": ["LATENT"], "output_name": ["LATENT"], "output_node": false, "python_module": "custom_nodes.SwarmComfyCommon"
            },
            "VAEDecode": {
                "input": { "required": { "samples": ["LATENT", {}], "vae": ["VAE", {}] } },
                "output": ["IMAGE"], "output_name": ["IMAGE"], "output_node": false, "python_module": "nodes"
            },
            "SwarmSaveImageWS": {
                "input": { "required": { "images": ["IMAGE"] }, "optional": { "bit_depth": [["8bit", "16bit", "raw"], {"default": "8bit"}] } },
                "output": [], "output_name": [], "output_node": true, "python_module": "custom_nodes.SwarmComfyCommon"
            }
        }
        """;

    /// <summary>ComfyUI object_info entries, in the shape of comfy_api _io.py get_v1_info(), for the V3 nodes with dynamic inputs that the generator uses for image work: ResizeImageMaskNode (WorkflowGeneratorSteps.cs, ControlNet images to a multiple of 8)
    /// and BatchImagesNode (WorkflowGenerator.cs, video end frames). Transcribed from ComfyUI 7ef81780 comfy_extras/nodes_post_processing.py lines 417-494 and 573-589, keeping what validation reads: types, templates, options, limits and defaults, without tooltips and display text.</summary>
    public const string DynamicObjectInfo = """
        {
            "ResizeImageMaskNode": {
                "input": { "required": {
                    "input": ["COMFY_MATCHTYPE_V3", { "template": { "template_id": "input_type", "allowed_types": "IMAGE,MASK" } }],
                    "resize_type": ["COMFY_DYNAMICCOMBO_V3", { "options": [
                        { "key": "scale dimensions", "inputs": { "required": { "width": ["INT", { "default": 512, "min": 0, "max": 16384, "step": 1 }], "height": ["INT", { "default": 512, "min": 0, "max": 16384, "step": 1 }], "crop": ["COMBO", { "multiselect": false, "options": ["disabled", "center"], "default": "center" }] } } },
                        { "key": "scale by multiplier", "inputs": { "required": { "multiplier": ["FLOAT", { "default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01 }] } } },
                        { "key": "scale longer dimension", "inputs": { "required": { "longer_size": ["INT", { "default": 512, "min": 0, "max": 16384, "step": 1 }] } } },
                        { "key": "scale shorter dimension", "inputs": { "required": { "shorter_size": ["INT", { "default": 512, "min": 0, "max": 16384, "step": 1 }] } } },
                        { "key": "scale width", "inputs": { "required": { "width": ["INT", { "default": 512, "min": 0, "max": 16384, "step": 1 }] } } },
                        { "key": "scale height", "inputs": { "required": { "height": ["INT", { "default": 512, "min": 0, "max": 16384, "step": 1 }] } } },
                        { "key": "scale total pixels", "inputs": { "required": { "megapixels": ["FLOAT", { "default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01 }] } } },
                        { "key": "match size", "inputs": { "required": { "match": ["IMAGE,MASK", {}], "crop": ["COMBO", { "multiselect": false, "options": ["disabled", "center"], "default": "center" }] } } },
                        { "key": "scale to multiple", "inputs": { "required": { "multiple": ["INT", { "default": 8, "min": 1, "max": 16384, "step": 1 }] } } }
                    ] }],
                    "scale_method": ["COMBO", { "multiselect": false, "options": ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], "default": "area" }]
                } },
                "output": ["COMFY_MATCHTYPE_V3"], "output_matchtypes": ["input_type"], "output_name": ["resized"], "output_node": false, "python_module": "comfy_extras.nodes_post_processing"
            },
            "BatchImagesNode": {
                "input": { "required": {
                    "images": ["COMFY_AUTOGROW_V3", { "template": { "input": { "required": { "image": ["IMAGE", {}] } }, "prefix": "image", "min": 1, "max": 50 } }]
                } },
                "output": ["IMAGE"], "output_matchtypes": null, "output_name": ["IMAGE"], "output_node": false, "python_module": "comfy_extras.nodes_post_processing"
            }
        }
        """;

    /// <summary>Prompt inputs for each <see cref="DynamicObjectInfo"/> class as the generator sends them, with links to the SDXL workflow's VAEDecode (8).</summary>
    public static readonly (string Class, string Inputs)[] DynamicSamples =
    [
        ("ResizeImageMaskNode", """{ "input": ["8", 0], "resize_type": "scale to multiple", "resize_type.multiple": 8, "scale_method": "lanczos" }"""),
        ("BatchImagesNode", """{ "images.image0": ["100", 0], "images.image1": ["8", 0] }""")
    ];

    /// <summary>Returns a record's list of node input problems ("type_errors" or "input_errors") as (node, input, problem).</summary>
    public static List<(string Node, string Input, string Problem)> Problems(JToken list) => [.. ((JArray)list).Select(e => ($"{e["node"]}", e["input"]?.ToString(), $"{e["problem"]}"))];

    /// <summary>Returns a record's "undeclared_inputs" as (node, input).</summary>
    public static List<(string Node, string Input)> Undeclared(JObject summary) => [.. ((JArray)summary["undeclared_inputs"]).Select(e => ($"{e["node"]}", $"{e["input"]}"))];

    /// <summary>Builds the SDXL checkpoint text-to-image workflow, traced with the fixture object_info.</summary>
    public static (JObject Workflow, WorkflowTrace Trace) BuildTypedSdxl()
    {
        T2IParamInput input = BuildInput(new() { Model = ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false), Prompt = "zqxbase man on a road" });
        WorkflowGenerator generator = new() { UserInput = input, Features = [.. FullFeatures] };
        generator.Trace = new(generator) { ObjectInfo = JObject.Parse(SdxlObjectInfo) };
        JObject workflow = generator.Generate();
        return (workflow, generator.Trace);
    }

    /// <summary>With object_info, every node class is known, every link is type-checked and valid, every slot shows its Comfy type beside its Swarm type, and the one input ComfyUI does not declare is reported.</summary>
    [Test]
    public void TestObjectInfoTypesTheSdxlGraph()
    {
        (JObject workflow, WorkflowTrace trace) = BuildTypedSdxl();
        JObject summary = trace.Build();
        Assert.That(summary["classes_missing_from_object_info"], Is.Empty, summary["classes_missing_from_object_info"]?.ToString(Formatting.None));
        List<string> untypedLinks = [];
        foreach (JProperty node in workflow.Properties())
        {
            foreach (JProperty input in ((JObject)node.Value["inputs"]).Properties().Where(i => WorkflowTrace.IsLink(i.Value)))
            {
                if (trace.ComfyInputType($"{node.Value["class_type"]}", (JObject)node.Value["inputs"], input.Name) is null)
                {
                    untypedLinks.Add($"{node.Name}:{node.Value["class_type"]}.{input.Name}");
                }
            }
        }
        Assert.That(untypedLinks, Is.Empty, "every link must be type-checked");
        Assert.That(summary["type_errors"], Is.Empty, summary["type_errors"].ToString(Formatting.None));
        // SwarmKSampler.py INPUT_TYPES has no control_after_generate; the generator sends it (WorkflowGenerator.cs CreateKSampler) and ComfyUI ignores it.
        Assert.That(Undeclared(summary), Is.EquivalentTo(new[] { ("10:SwarmKSampler", "control_after_generate") }));
        JObject loader = (JObject)summary["nodes"]["4"];
        Assert.That(($"{loader["class"]}", $"{loader["stage"]}"), Is.EqualTo(("CheckpointLoaderSimple", "base")));
        foreach ((string slot, string comfy, string swarm) in new[] { ("0", "MODEL", "MODEL"), ("1", "CLIP", "TEXTENC"), ("2", "VAE", "VAE") })
        {
            JObject output = (JObject)loader["outputs"][slot];
            Assert.That(($"{output["comfy"]}", output["types"].Select(t => $"{t}"), output["compat"].Select(t => $"{t}")), Is.EqualTo((comfy, new[] { swarm }, new[] { "stable-diffusion-xl-v1" })), $"slot {slot}: {output.ToString(Formatting.None)}");
        }
        Assert.That(trace.IsOutputNode("SwarmSaveImageWS"), Is.True);
        Assert.That(trace.IsOutputNode("VAEDecode"), Is.False);
    }

    /// <summary>A link whose source type does not match the input's declared type is reported with both types and the source.</summary>
    [Test]
    public void TestObjectInfoCatchesMiswiredLink()
    {
        (JObject workflow, WorkflowTrace trace) = BuildTypedSdxl();
        Assert.That(workflow["8"]["inputs"]["vae"].ToString(Formatting.None), Is.EqualTo("""["4",2]"""));
        workflow["8"]["inputs"]["vae"] = new JArray("4", 1);
        JObject summary = trace.Build();
        Assert.That(Problems(summary["type_errors"]), Is.EquivalentTo(new[] { ("8:VAEDecode", "vae", "expects VAE, gets CLIP from 4:1:CheckpointLoaderSimple") }));
    }

    /// <summary>V3 dynamic inputs are declared under the prefixed names the generator sends and are type-checked, as ComfyUI expands them; a match-type output links into an IMAGE input.
    /// Without the V3 marker the expansion does not apply and the same inputs read as undeclared, so the expansion is what declares them.</summary>
    [Test]
    public void TestDynamicInputsAreDeclaredAndTyped()
    {
        (JObject workflow, WorkflowTrace trace) = BuildTypedSdxl();
        foreach (JProperty entry in JObject.Parse(DynamicObjectInfo).Properties())
        {
            trace.ObjectInfo[entry.Name] = entry.Value;
        }
        workflow["100"] = new JObject() { ["class_type"] = DynamicSamples[0].Class, ["inputs"] = JObject.Parse(DynamicSamples[0].Inputs) };
        workflow["101"] = new JObject() { ["class_type"] = DynamicSamples[1].Class, ["inputs"] = JObject.Parse(DynamicSamples[1].Inputs) };
        workflow["102"] = JObject.Parse("""{ "class_type": "SwarmSaveImageWS", "inputs": { "images": ["101", 0] } }""");
        JObject summary = trace.Build();
        Assert.That(summary["type_errors"], Is.Empty, summary["type_errors"].ToString(Formatting.None));
        Assert.That(summary["input_errors"], Is.Empty, summary["input_errors"].ToString(Formatting.None));
        Assert.That(Undeclared(summary), Is.EquivalentTo(new[] { ("10:SwarmKSampler", "control_after_generate") }));
        foreach ((string id, string input) in new[] { ("100", "input"), ("101", "images.image0"), ("101", "images.image1") })
        {
            Assert.That(trace.ComfyInputType($"{workflow[id]["class_type"]}", (JObject)workflow[id]["inputs"], input), Is.Not.Null, $"{id}.{input} is type-checked");
        }
        workflow["101"]["inputs"]["images.image1"] = new JArray("10", 0);
        Assert.That(Problems(trace.Build()["type_errors"]), Is.EqualTo(new[] { ("101:BatchImagesNode", "images.image1", "expects IMAGE, gets LATENT from 10:0:SwarmKSampler") }));
        ((JObject)workflow["101"]["inputs"]).Remove("images.image0");
        Assert.That(Problems(trace.Build()["input_errors"]), Is.EqualTo(new[] { ("101:BatchImagesNode", "images.image0", "required and missing") }));
        workflow["101"]["inputs"] = JObject.Parse(DynamicSamples[1].Inputs);
        foreach (string dynamicClass in new[] { "ResizeImageMaskNode", "BatchImagesNode" })
        {
            ((JObject)trace.ObjectInfo[dynamicClass]).Remove("output_matchtypes");
        }
        Assert.That(Undeclared(trace.Build()), Is.SupersetOf(new[] { ("100:ResizeImageMaskNode", "resize_type.multiple"), ("101:BatchImagesNode", "images.image0"), ("101:BatchImagesNode", "images.image1") }));
    }

    /// <summary>A node class ComfyUI does not have is reported as missing, and its inputs are not reported as undeclared.</summary>
    [Test]
    public void TestObjectInfoReportsMissingClass()
    {
        (JObject workflow, WorkflowTrace trace) = BuildTypedSdxl();
        ((JObject)trace.ObjectInfo).Remove("VAEDecode");
        JObject summary = trace.Build();
        Assert.That(((JArray)summary["classes_missing_from_object_info"]).Select(t => $"{t}"), Is.EquivalentTo(new[] { "8:VAEDecode" }));
        Assert.That(Undeclared(summary).Select(u => u.Node), Has.None.StartsWith("8:"));
    }
}
