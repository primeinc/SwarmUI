using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using static SwarmUITests.NullModels;

namespace SwarmUITests;

/// <summary>Tests for <see cref="NullModels"/>: the shape rules reproduce the released Qwen Image 2.1 headers, and the zero-weight files Swarm reads back are real safetensors that Swarm's own class detection identifies.</summary>
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

    /// <summary>The micro DiT and LoRA (every real key, one 128-wide head) and the full-size zero VAE are real safetensors files that Swarm classifies as Qwen Image 2.1.</summary>
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
            WriteVerbatim($"{root}/vae/qwen_image_2.1_vae_bf16.safetensors", ReadHeaderBytes(Family, "qwen_image_2.1_vae_bf16"), RealSize(Family, "vae/qwen_image_2.1_vae_bf16.safetensors"));
            TestContext.Out.WriteLine($"micro DiT {ditSize} bytes, micro LoRA {loraSize} bytes");
            Assert.That(ditSize, Is.LessThan(20_000_000));
            Assert.That(loraSize, Is.LessThan(5_000_000));
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
