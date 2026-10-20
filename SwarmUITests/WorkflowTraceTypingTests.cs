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
                if (trace.ComfyInputType($"{node.Value["class_type"]}", input.Name) is null)
                {
                    untypedLinks.Add($"{node.Name}:{node.Value["class_type"]}.{input.Name}");
                }
            }
        }
        Assert.That(untypedLinks, Is.Empty, "every link must be type-checked");
        Assert.That(summary["type_errors"], Is.Empty, summary["type_errors"].ToString(Formatting.None));
        // SwarmKSampler.py INPUT_TYPES has no control_after_generate; the generator sends it (WorkflowGenerator.cs CreateKSampler) and ComfyUI ignores it.
        Assert.That(((JArray)summary["undeclared_inputs"]).Select(t => $"{t}"), Is.EquivalentTo(new[] { "10:SwarmKSampler.control_after_generate" }));
        string loader = ((JArray)summary["nodes"]["base"]).Select(t => $"{t}").Single(d => d.StartsWith("4:CheckpointLoaderSimple"));
        Assert.That(loader, Does.Contain("0:MODEL=MODEL/stable-diffusion-xl-v1"));
        Assert.That(loader, Does.Contain("1:CLIP=TEXTENC/stable-diffusion-xl-v1"));
        Assert.That(loader, Does.Contain("2:VAE=VAE/stable-diffusion-xl-v1"));
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
        Assert.That(((JArray)summary["type_errors"]).Select(t => $"{t}"), Is.EquivalentTo(new[] { "8:VAEDecode.vae expects VAE, gets CLIP from 4:1:CheckpointLoaderSimple" }));
    }

    /// <summary>A node class ComfyUI does not have is reported as missing, and its inputs are not reported as undeclared.</summary>
    [Test]
    public void TestObjectInfoReportsMissingClass()
    {
        (JObject workflow, WorkflowTrace trace) = BuildTypedSdxl();
        ((JObject)trace.ObjectInfo).Remove("VAEDecode");
        JObject summary = trace.Build();
        Assert.That(((JArray)summary["classes_missing_from_object_info"]).Select(t => $"{t}"), Is.EquivalentTo(new[] { "8:VAEDecode" }));
        Assert.That(((JArray)summary["undeclared_inputs"]).Select(t => $"{t}"), Has.None.StartsWith("8:"));
    }
}
