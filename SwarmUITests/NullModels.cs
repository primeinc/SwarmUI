using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>Builds zero-weight safetensors files from real model headers: real tensor keys, dtypes and layout, every weight 0.
/// The real loaders (Swarm's header reader and class detection, ComfyUI's model detection, construction and execution) then run with no real weights.</summary>
public static class NullModels
{
    /// <summary>Bytes per element of each safetensors dtype.</summary>
    public static readonly Dictionary<string, long> DtypeBytes = new() { ["BOOL"] = 1, ["U8"] = 1, ["I8"] = 1, ["F8_E4M3"] = 1, ["F8_E5M2"] = 1, ["I16"] = 2, ["U16"] = 2, ["F16"] = 2, ["BF16"] = 2, ["I32"] = 4, ["U32"] = 4, ["F32"] = 4, ["I64"] = 8, ["U64"] = 8, ["F64"] = 8 };

    /// <summary>One tensor entry of a safetensors header.</summary>
    /// <param name="Key">Tensor name.</param>
    /// <param name="Dtype">Safetensors dtype name.</param>
    /// <param name="Shape">Tensor shape; empty for a scalar.</param>
    public record class Tensor(string Key, string Dtype, long[] Shape)
    {
        /// <summary>Payload size in bytes.</summary>
        public long Bytes => Shape.Aggregate(1L, (a, b) => a * b) * DtypeBytes[Dtype];
    }

    /// <summary>Returns the folder holding the committed real headers for a model family.</summary>
    public static string FixtureDir(string family) => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "null-models", family);

    /// <summary>Reads a committed real header (the exact header bytes of the real file).</summary>
    public static byte[] ReadHeaderBytes(string family, string name) => File.ReadAllBytes(Path.Combine(FixtureDir(family), $"{name}.header.json"));

    /// <summary>Reads a committed real header as JSON.</summary>
    public static JObject ReadHeader(string family, string name) => Encoding.UTF8.GetString(ReadHeaderBytes(family, name)).ParseToJson();

    /// <summary>Returns the tensors of a header in file order (by data offset), checking the payload is contiguous and each tensor is sized by its shape and dtype.</summary>
    public static List<Tensor> Tensors(JObject header)
    {
        List<(long Start, long End, Tensor Tensor)> entries = [];
        foreach (JProperty property in header.Properties().Where(p => p.Name != "__metadata__"))
        {
            JArray offsets = (JArray)property.Value["data_offsets"];
            Tensor tensor = new(property.Name, $"{property.Value["dtype"]}", [.. ((JArray)property.Value["shape"]).Select(v => v.Value<long>())]);
            entries.Add((offsets[0].Value<long>(), offsets[1].Value<long>(), tensor));
        }
        entries.Sort((a, b) => a.Start.CompareTo(b.Start));
        long expected = 0;
        foreach ((long start, long end, Tensor tensor) in entries)
        {
            if (start != expected || end - start != tensor.Bytes)
            {
                throw new InvalidDataException($"Tensor '{tensor.Key}' at [{start}, {end}) does not continue the payload at {expected} with {tensor.Bytes} bytes");
            }
            expected = end;
        }
        return [.. entries.Select(e => e.Tensor)];
    }

    /// <summary>Writes a safetensors file holding the given tensors, all zero, laid out contiguously in the given order. The header is padded to 8 bytes with spaces. Returns the file size.</summary>
    public static long Write(string path, JObject metadata, IReadOnlyList<Tensor> tensors)
    {
        JObject header = [];
        if (metadata is not null)
        {
            header["__metadata__"] = metadata.DeepClone();
        }
        long offset = 0;
        foreach (Tensor tensor in tensors)
        {
            header[tensor.Key] = new JObject()
            {
                ["dtype"] = tensor.Dtype,
                ["shape"] = JArray.FromObject(tensor.Shape),
                ["data_offsets"] = JArray.FromObject(new[] { offset, offset + tensor.Bytes })
            };
            offset += tensor.Bytes;
        }
        byte[] headerBytes = Encoding.UTF8.GetBytes(header.ToString(Formatting.None));
        int pad = (8 - (headerBytes.Length % 8)) % 8;
        byte[] padded = [.. headerBytes, .. Enumerable.Repeat((byte)' ', pad)];
        return WriteFile(path, padded, 8 + padded.Length + offset);
    }

    /// <summary>Writes a zero-weight copy of a real file: the real header bytes unchanged, then a zero payload up to the real file size.</summary>
    public static long WriteVerbatim(string path, byte[] headerBytes, long realSize)
    {
        long payload = Tensors(Encoding.UTF8.GetString(headerBytes).ParseToJson()).Sum(t => t.Bytes);
        if (8 + headerBytes.Length + payload != realSize)
        {
            throw new InvalidDataException($"Header describes {8 + headerBytes.Length + payload} bytes, but the real file is {realSize}");
        }
        return WriteFile(path, headerBytes, realSize);
    }

    /// <summary>Writes the 8-byte little-endian header length and the header, then extends the file with zeros to its full size without writing them.</summary>
    public static long WriteFile(string path, byte[] headerBytes, long size)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        using FileStream file = new(path, FileMode.Create, FileAccess.Write);
        file.Write(BitConverter.GetBytes((ulong)headerBytes.Length));
        file.Write(headerBytes);
        file.SetLength(size);
        return size;
    }

    /// <summary>Dimensions of a Qwen Image 2.1 DiT, as ComfyUI's comfy/model_detection.py derives them from tensor shapes.</summary>
    /// <param name="Inner">Model width, from img_in.weight shape[0]; heads = Inner / HeadDim.</param>
    /// <param name="HeadDim">Attention head width, from norm_q.weight shape[0]. Fixed at 128 by the rope axes (16, 56, 56) in comfy/ldm/qwen_image21/model.py.</param>
    /// <param name="Context">Text encoder width, from txt_in.text_norm.weight shape[0]. Must match the Qwen3-VL-8B text encoder (4096).</param>
    /// <param name="MlpRatio">MLP hidden width over Inner, from img_mlp.gate_up.weight.</param>
    /// <param name="InChannels">Latent channels in, from img_in.weight shape[1].</param>
    /// <param name="OutChannels">Latent channels out, from proj_out.weight shape[0].</param>
    public record class Qwen21Dims(long Inner, long HeadDim, long Context, long MlpRatio, long InChannels, long OutChannels)
    {
        /// <summary>The released model.</summary>
        public static readonly Qwen21Dims Real = new(4096, 128, 4096, 3, 64, 64);

        /// <summary>The smallest model ComfyUI builds and runs: one 128-wide head, real text encoder width, real latent channels.</summary>
        public static readonly Qwen21Dims Micro = new(128, 128, 4096, 3, 64, 64);
    }

    /// <summary>Returns the shape of a Qwen Image 2.1 DiT weight, following the module constructors in ComfyUI comfy/ldm/qwen_image21/model.py.</summary>
    public static long[] Qwen21Shape(string key, Qwen21Dims d)
    {
        string local = key.StartsWith("transformer_blocks.") ? key.After('.').After('.') : key;
        long mlp = d.Inner * d.MlpRatio;
        return local switch
        {
            "img_in.weight" => [d.Inner, d.InChannels],
            "txt_in.text_norm.weight" => [d.Context],
            "txt_in.in_layer.weight" => [d.Inner, d.Context],
            "txt_in.out_layer.weight" => [d.Inner, d.Inner],
            "time_text_embed.timestep_embedder.linear_1.weight" => [d.Inner, 256],
            "time_text_embed.timestep_embedder.linear_2.weight" => [d.Inner, d.Inner],
            "modulation.1.weight" => [4 * d.Inner, d.Inner],
            "norm_out.linear.weight" => [d.Inner, d.Inner],
            "proj_out.weight" => [d.OutChannels, d.Inner],
            "attn.to_q.weight" or "attn.to_k.weight" or "attn.to_v.weight" or "attn.to_out.0.weight" => [d.Inner, d.Inner],
            "attn.norm_q.weight" or "attn.norm_k.weight" => [d.HeadDim],
            "img_mlp.gate_up.weight" => [2 * mlp, d.Inner],
            "img_mlp.out.weight" => [d.Inner, mlp],
            _ => throw new ArgumentException($"No Qwen Image 2.1 DiT shape rule for '{key}'")
        };
    }

    /// <summary>Returns the shape of a Qwen Image 2.1 LoRA tensor ("diffusion_model.{module}" + ".lora_down.weight", ".lora_up.weight", ".alpha" or ".diff") for a DiT of the given dimensions, capping the rank.</summary>
    public static long[] Qwen21LoraShape(string key, long[] realShape, Qwen21Dims d, long maxRank)
    {
        if (!key.StartsWith("diffusion_model."))
        {
            throw new ArgumentException($"No Qwen Image 2.1 LoRA shape rule for '{key}'");
        }
        string module = key["diffusion_model.".Length..];
        if (module.EndsWith(".alpha"))
        {
            return realShape;
        }
        if (module.EndsWith(".diff"))
        {
            return Qwen21Shape($"{module.BeforeLast('.')}.weight", d);
        }
        if (module.EndsWith(".lora_down.weight"))
        {
            long[] weight = Qwen21Shape($"{module[..^".lora_down.weight".Length]}.weight", d);
            return [Math.Min(realShape[0], maxRank), weight[1]];
        }
        if (module.EndsWith(".lora_up.weight"))
        {
            long[] weight = Qwen21Shape($"{module[..^".lora_up.weight".Length]}.weight", d);
            return [weight[0], Math.Min(realShape[1], maxRank)];
        }
        throw new ArgumentException($"No Qwen Image 2.1 LoRA shape rule for '{key}'");
    }

    /// <summary>Writes a zero-weight Qwen Image 2.1 DiT with every real key, at the given dimensions.</summary>
    public static long WriteQwen21DiT(string path, Qwen21Dims d)
    {
        JObject header = ReadHeader("qwen-image-2.1", "qwen_image_2.1_bf16");
        return Write(path, header["__metadata__"] as JObject, [.. Tensors(header).Select(t => t with { Shape = Qwen21Shape(t.Key, d) })]);
    }

    /// <summary>Writes a zero-weight Qwen Image 2.1 LoRA with every key of the released turbo LoRA, shaped for a DiT of the given dimensions.</summary>
    public static long WriteQwen21Lora(string path, Qwen21Dims d, long maxRank)
    {
        JObject header = ReadHeader("qwen-image-2.1", "qwen_image_2.1_turbo_lora_avg_rank_178_bf16");
        return Write(path, header["__metadata__"] as JObject, [.. Tensors(header).Select(t => t with { Shape = Qwen21LoraShape(t.Key, t.Shape, d, maxRank) })]);
    }

    /// <summary>Returns the real size of a model file listed in a family's manifest.</summary>
    public static long RealSize(string family, string repoPath)
    {
        JObject manifest = File.ReadAllText(Path.Combine(FixtureDir(family), "manifest.json")).ParseToJson();
        return ((JArray)manifest["files"]).Single(f => $"{f["path"]}" == repoPath)["size"].Value<long>();
    }
}
