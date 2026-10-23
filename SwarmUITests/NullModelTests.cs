using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using static SwarmUITests.NullModels;

namespace SwarmUITests;

/// <summary>Tests for <see cref="NullModels"/>: the shape rules reproduce the released Qwen Image 2.1 and Qwen3-VL headers, and the zero-weight files Swarm reads back are real safetensors that Swarm's own class detection identifies.</summary>
public class NullModelTests : SwarmUITest
{
    /// <summary>Prepares the basics.</summary>
    [OneTimeSetUp]
    public static void PreInit()
    {
        Setup();
        WorkflowGeneratorHarness.Init();
    }

    /// <summary>Family folder of the committed Qwen Image 2.1 headers.</summary>
    public const string Family = "qwen-image-2.1";

    /// <summary>The name <see cref="NullModelFiles.DiffusionModelName"/> writes the zero-weight diffusion model under is the name Swarm gives the harness's qwen-image-2.1 model.</summary>
    [Test]
    public void TestLiveDiffusionModelNameIsTheHarnessModel()
    {
        Assert.That(WorkflowGeneratorHarness.ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true).Name, Is.EqualTo(NullModelFiles.DiffusionModelName));
    }

    /// <summary>Returns the keys whose rule-derived shape differs from the real shape.</summary>
    public static List<string> Mismatches(List<Tensor> tensors, Func<Tensor, long[]> rule)
    {
        List<string> result = [];
        foreach (Tensor tensor in tensors)
        {
            long[] derived;
            try
            {
                derived = rule(tensor);
            }
            catch (ArgumentException ex)
            {
                result.Add(ex.Message);
                continue;
            }
            if (!derived.SequenceEqual(tensor.Shape))
            {
                result.Add($"{tensor.Key}: real [{string.Join(",", tensor.Shape)}], rule [{string.Join(",", derived)}]");
            }
        }
        return result;
    }

    /// <summary>Every manifest pins its repository to a commit, and every committed header fixture is the header of that commit's file: its bytes hash to the manifest's header_sha256, the hash of the header bytes fetched from the pinned commit
    /// (SwarmUITests/ComfyLive `just headers` fetches them again and compares). Negative controls: a fixture with one byte changed, and a branch name as revision, are both reported.</summary>
    [Test]
    public void TestHeaderFixturesArePinned()
    {
        Dictionary<string, JObject> manifests = Manifests(Family);
        Assert.That(manifests.Keys, Is.EquivalentTo(new[] { "manifest.json", "tiny-random-manifest.json" }));
        Assert.That(manifests.Values.Sum(m => ((JArray)m["files"]).Count), Is.EqualTo(5), "every committed header fixture is listed");
        foreach ((string name, JObject manifest) in manifests)
        {
            List<string> problems = ManifestProblems(manifest, header => ReadHeaderBytes(Family, header));
            Assert.That(problems, Is.Empty, $"{name}: {string.Join("\n", problems)}");
        }
        JObject manifest21 = manifests["manifest.json"];
        string changed = $"{manifest21["files"][0]["header"]}";
        List<string> tampered = ManifestProblems(manifest21, header =>
        {
            byte[] bytes = ReadHeaderBytes(Family, header);
            if (header == changed)
            {
                bytes[^1] ^= 1;
            }
            return bytes;
        });
        Assert.That(tampered, Has.Count.EqualTo(1), "a one-byte change to a fixture is reported");
        Assert.That(tampered[0], Does.Contain(changed));
        JObject branch = (JObject)manifest21.DeepClone();
        branch["revision"] = "main";
        List<string> branchProblems = ManifestProblems(branch, header => ReadHeaderBytes(Family, header));
        Assert.That(branchProblems, Has.Count.EqualTo(1), "a branch name as revision is reported");
        Assert.That(branchProblems[0], Does.Contain("is not a commit"));
    }

    /// <summary>The committed DiT header is the released file's header (its size matches Hugging Face), and the DiT shape rules, at the released dimensions, reproduce every one of its tensor shapes.</summary>
    [Test]
    public void TestQwen21DiTShapeRulesReproduceRealHeader()
    {
        byte[] headerBytes = ReadHeaderBytes(Family, "qwen_image_2.1_bf16");
        List<Tensor> tensors = Tensors(ReadHeader(Family, "qwen_image_2.1_bf16"));
        Assert.That(8 + headerBytes.Length + tensors.Sum(t => t.Bytes), Is.EqualTo(RealSize(Family, "diffusion_models/qwen_image_2.1_bf16.safetensors")));
        Assert.That(tensors.Count, Is.EqualTo(9 + 32 * 8), "9 top-level tensors and 32 blocks of 8");
        List<string> mismatches = Mismatches(tensors, t => Qwen21Shape(t.Key, Qwen21Dims.Real));
        Assert.That(mismatches, Is.Empty, string.Join("\n", mismatches));
    }

    /// <summary>The DiT shape rules also reproduce an independent Qwen Image 2.1 checkpoint: the diffusers tiny-random transformer (unfused MLP, 2 blocks, F32), at its own dimensions (inner 32, head 16, text 16).</summary>
    [Test]
    public void TestQwen21DiTShapeRulesReproduceTinyRandomHeader()
    {
        byte[] headerBytes = ReadHeaderBytes(Family, "tiny-random-qwen-image-2.1.transformer");
        List<Tensor> tensors = Tensors(ReadHeader(Family, "tiny-random-qwen-image-2.1.transformer"));
        Assert.That(8 + headerBytes.Length + tensors.Sum(t => t.Bytes), Is.EqualTo(RealSize(Family, "transformer/diffusion_pytorch_model.safetensors", "tiny-random-manifest")));
        Qwen21Dims tiny = new(32, 16, 16, 3, 64, 64);
        List<string> mismatches = Mismatches(tensors, t => Qwen21Shape(t.Key, tiny));
        Assert.That(mismatches, Is.Empty, string.Join("\n", mismatches));
    }

    /// <summary>The committed LoRA header is the released turbo LoRA's header, and the LoRA shape rules, at the released dimensions and uncapped rank, reproduce every one of its tensor shapes.</summary>
    [Test]
    public void TestQwen21LoraShapeRulesReproduceRealHeader()
    {
        byte[] headerBytes = ReadHeaderBytes(Family, "qwen_image_2.1_turbo_lora_avg_rank_178_bf16");
        List<Tensor> tensors = Tensors(ReadHeader(Family, "qwen_image_2.1_turbo_lora_avg_rank_178_bf16"));
        Assert.That(8 + headerBytes.Length + tensors.Sum(t => t.Bytes), Is.EqualTo(RealSize(Family, "loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors")));
        List<string> mismatches = Mismatches(tensors, t => Qwen21LoraShape(t.Key, t.Shape, Qwen21Dims.Real, long.MaxValue));
        Assert.That(mismatches, Is.Empty, string.Join("\n", mismatches));
    }

    /// <summary>The committed text encoder header is the released int8 Qwen3-VL-8B's header, every quantization side tensor belongs to a weight, and the text encoder shape rules, at the released dimensions, reproduce the shape of every weight.</summary>
    [Test]
    public void TestQwen3VLShapeRulesReproduceRealHeader()
    {
        byte[] headerBytes = ReadHeaderBytes(Family, "qwen3vl_8b_int8_convrot");
        List<Tensor> tensors = Tensors(ReadHeader(Family, "qwen3vl_8b_int8_convrot"));
        Assert.That(8 + headerBytes.Length + tensors.Sum(t => t.Bytes), Is.EqualTo(RealSize(Family, "text_encoders/qwen3vl_8b_int8_convrot.safetensors")));
        HashSet<string> keys = [.. tensors.Select(t => t.Key)];
        List<string> orphans = [.. tensors.Where(t => IsQuantKey(t.Key) && !keys.Contains($"{t.Key.BeforeLast('.')}.weight")).Select(t => t.Key)];
        Assert.That(orphans, Is.Empty, string.Join("\n", orphans));
        Assert.That(keys.Count(k => k.EndsWith(".input_layernorm.weight")), Is.EqualTo(36), "36 language layers");
        Assert.That(keys.Count(k => k.StartsWith("model.visual.blocks.") && k.EndsWith(".norm1.weight")), Is.EqualTo(27), "27 vision blocks");
        Assert.That(keys.Count(k => k.StartsWith("model.visual.deepstack_merger_list.") && k.EndsWith(".norm.weight")), Is.EqualTo(3), "3 DeepStack mergers");
        List<string> mismatches = Mismatches([.. tensors.Where(t => !IsQuantKey(t.Key))], t => Qwen3VLShape(t.Key, Qwen3VLDims.Real8B));
        Assert.That(mismatches, Is.Empty, string.Join("\n", mismatches));
    }

    /// <summary>The VAE shape rules keep the released VAE's header at its own dimensions, and a written null VAE carries the dimensions ComfyUI derives from it (comfy/sd.py: "dim" from encoder.conv1.weight, "dec_dim" from decoder.head.0.gamma), with the RGBA head and 64 latent channels kept.</summary>
    [Test]
    public void TestQwen21VaeShapeRules()
    {
        List<Tensor> tensors = Tensors(ReadHeader(Family, "qwen_image_2.1_vae_bf16"));
        Assert.That(8 + ReadHeaderBytes(Family, "qwen_image_2.1_vae_bf16").Length + tensors.Sum(t => t.Bytes), Is.EqualTo(RealSize(Family, "vae/qwen_image_2.1_vae_bf16.safetensors")), "the header fixture is the released file's header");
        List<string> mismatches = Mismatches(tensors, t => Qwen21VaeShape(t.Key, t.Shape, Qwen21VaeDims.Real));
        Assert.That(mismatches, Is.Empty, string.Join("\n", mismatches));
        string path = Path.Combine(Path.GetTempPath(), $"swarmui-null-vae-{Guid.NewGuid():N}.safetensors").Replace('\\', '/');
        try
        {
            long size = WriteQwen21Vae(path, Qwen21VaeDims.Null);
            TestContext.Out.WriteLine($"null VAE {size} bytes");
            Dictionary<string, long[]> shapes = Tensors(T2IModel.GetMetadataHeaderFrom(path)).ToDictionary(t => t.Key, t => t.Shape);
            Assert.That(shapes.Count, Is.EqualTo(tensors.Count));
            Assert.That(shapes["encoder.conv1.weight"][0], Is.EqualTo(Qwen21VaeDims.Null.EncoderDim));
            Assert.That(shapes["decoder.head.0.gamma"][0], Is.EqualTo(Qwen21VaeDims.Null.DecoderDim));
            Assert.That(shapes["decoder.head.2.weight"], Is.EqualTo(new long[] { 4, Qwen21VaeDims.Null.DecoderDim, 1, 3, 3 }));
            Assert.That(shapes["decoder.conv1.weight"], Is.EqualTo(new long[] { 8 * Qwen21VaeDims.Null.DecoderDim, 64, 1, 3, 3 }));
            Assert.That(shapes["encoder.head.2.weight"], Is.EqualTo(new long[] { 128, 8 * Qwen21VaeDims.Null.EncoderDim, 1, 3, 3 }));
            Assert.That(size, Is.LessThan(1_000_000));
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>Reads a written file back with Swarm's own header reader, checks its layout, and returns the class Swarm's own detection assigns it.</summary>
    public static string SwarmClassOf(string type, string folder, string name)
    {
        string path = $"{folder}/{name}";
        JObject header = T2IModel.GetMetadataHeaderFrom(path);
        Assert.That(Tensors(header), Is.Not.Empty, path);
        Assert.That(new FileInfo(path).Length, Is.EqualTo(8 + BitConverter.ToInt64(File.ReadAllBytes(path).AsSpan(0, 8)) + Tensors(header).Sum(t => t.Bytes)), path);
        T2IModel model = new(Program.T2IModelSets[type], folder, path, name);
        return T2IModelClassSorter.IdentifyClassFor(model, header, type)?.ID;
    }

    /// <summary>The micro DiT, LoRA and VAE (every real key, small dimensions) are real safetensors files that Swarm classifies as Qwen Image 2.1, and the null text encoder is a valid safetensors file, with no tensor at its released size.</summary>
    [Test]
    public void TestQwen21NullModelsAreDetectedBySwarm()
    {
        string root = Path.Combine(Path.GetTempPath(), $"swarmui-null-models-{Guid.NewGuid():N}").Replace('\\', '/');
        try
        {
            long ditSize = WriteQwen21DiT($"{root}/diffusion_models/qwen_image_2.1_bf16.safetensors", Qwen21Dims.Micro);
            WriteQwen21UnfusedDiT($"{root}/diffusion_models/qwen_image_2.1_unfused.safetensors", Qwen21Dims.Micro);
            Assert.That(SwarmClassOf("Stable-Diffusion", $"{root}/diffusion_models", "qwen_image_2.1_unfused.safetensors"), Is.EqualTo("qwen-image-2.1"), "the unfused-MLP layout is Qwen Image 2.1 too");
            long loraSize = WriteQwen21Lora($"{root}/loras/qwen_image_2.1_turbo_lora.safetensors", Qwen21Dims.Micro, 4);
            long vaeSize = WriteQwen21Vae($"{root}/vae/qwen_image_2.1_vae_bf16.safetensors", Qwen21VaeDims.Null);
            long teSize = WriteQwen3VLTextEncoder($"{root}/text_encoders/qwen3vl_8b_int8_convrot.safetensors", Qwen3VLDims.Null);
            TestContext.Out.WriteLine($"micro DiT {ditSize} bytes, micro LoRA {loraSize} bytes, null VAE {vaeSize} bytes, null text encoder {teSize} bytes");
            Assert.That(ditSize, Is.LessThan(20_000_000));
            Assert.That(loraSize, Is.LessThan(5_000_000));
            Assert.That(teSize, Is.LessThan(20_000_000));
            JObject teHeader = T2IModel.GetMetadataHeaderFrom($"{root}/text_encoders/qwen3vl_8b_int8_convrot.safetensors");
            Dictionary<string, Tensor> realTe = Tensors(ReadHeader(Family, "qwen3vl_8b_int8_convrot")).ToDictionary(t => t.Key);
            List<Tensor> nullTe = Tensors(teHeader);
            Assert.That(nullTe.Count, Is.EqualTo(realTe.Values.Count(t => !IsQuantKey(t.Key))), "every weight of the released encoder, no quantization side tensors");
            List<string> fullSize = [.. nullTe.Where(t => realTe[t.Key].Shape.Aggregate(1L, (a, b) => a * b) == t.Shape.Aggregate(1L, (a, b) => a * b) && t.Shape.Length > 1).Select(t => t.Key)];
            Assert.That(fullSize, Is.Empty, "no matrix keeps its released size: " + string.Join(", ", fullSize));
            Assert.That(SwarmClassOf("Stable-Diffusion", $"{root}/diffusion_models", "qwen_image_2.1_bf16.safetensors"), Is.EqualTo("qwen-image-2.1"));
            Assert.That(SwarmClassOf("LoRA", $"{root}/loras", "qwen_image_2.1_turbo_lora.safetensors"), Is.EqualTo("qwen-image-2.1/lora"));
            Assert.That(SwarmClassOf("VAE", $"{root}/vae", "qwen_image_2.1_vae_bf16.safetensors"), Is.EqualTo("qwen-image-2.1/vae"));
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, true);
            }
        }
    }
}
