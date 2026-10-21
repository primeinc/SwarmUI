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

    /// <summary>Writes the 8-byte little-endian header length and the header, then extends the file with zeros to its full size without writing them.
    /// An existing file of that size and header is kept as it is: a running ComfyUI may hold it memory-mapped, and Windows refuses to rewrite a mapped file.</summary>
    public static long WriteFile(string path, byte[] headerBytes, long size)
    {
        if (File.Exists(path) && new FileInfo(path).Length == size)
        {
            using FileStream existing = new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            byte[] start = new byte[8 + headerBytes.Length];
            existing.ReadExactly(start);
            if (start.AsSpan(0, 8).SequenceEqual(BitConverter.GetBytes((ulong)headerBytes.Length)) && start.AsSpan(8).SequenceEqual(headerBytes))
            {
                return size;
            }
        }
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
    /// <param name="Context">Text encoder width, from txt_in.text_norm.weight shape[0]. Must match the text encoder's hidden size.</param>
    /// <param name="MlpRatio">MLP hidden width over Inner, from img_mlp.gate_up.weight.</param>
    /// <param name="InChannels">Latent channels in, from img_in.weight shape[1].</param>
    /// <param name="OutChannels">Latent channels out, from proj_out.weight shape[0].</param>
    public record class Qwen21Dims(long Inner, long HeadDim, long Context, long MlpRatio, long InChannels, long OutChannels)
    {
        /// <summary>The released model.</summary>
        public static readonly Qwen21Dims Real = new(4096, 128, 4096, 3, 64, 64);

        /// <summary>The smallest model ComfyUI builds and runs: one 128-wide head, the <see cref="Qwen3VLDims.Null"/> text encoder width, real latent channels.</summary>
        public static readonly Qwen21Dims Micro = new(128, 128, Qwen3VLDims.Null.Hidden, 3, 64, 64);
    }

    /// <summary>Dimensions of a Qwen3-VL text encoder: comfy/text_encoders/llama.py Qwen3VL_8BConfig for the language model, comfy/text_encoders/qwen3vl.py QWEN3VL_VISION for the vision tower.</summary>
    /// <param name="Vocab">Token vocabulary size. The tokenizer emits ids up to the real vocabulary, so it stays real.</param>
    /// <param name="Hidden">Language model width.</param>
    /// <param name="Intermediate">MLP width.</param>
    /// <param name="Heads">Attention heads.</param>
    /// <param name="KvHeads">Key/value heads.</param>
    /// <param name="HeadDim">Head width. A class constant (128) in llama.py, not a config field.</param>
    /// <param name="VisionHidden">Vision tower width. QWEN3VL_VISION_COMMON fixes 16 heads, so it is a multiple of 16.</param>
    /// <param name="VisionIntermediate">Vision MLP width.</param>
    public record class Qwen3VLDims(long Vocab, long Hidden, long Intermediate, long Heads, long KvHeads, long HeadDim, long VisionHidden, long VisionIntermediate)
    {
        /// <summary>The released Qwen3-VL-8B.</summary>
        public static readonly Qwen3VLDims Real8B = new(151936, 4096, 12288, 32, 8, 128, 1152, 4304);

        /// <summary>The smallest text encoder that runs: real vocabulary and head width, everything else small.</summary>
        public static readonly Qwen3VLDims Null = new(151936, 16, 16, 1, 1, 128, 64, 64);
    }

    /// <summary>Returns the part of a key after "{prefix}{index}.", or null if the key does not start with the prefix.</summary>
    public static string AfterIndex(string key, string prefix) => key.StartsWith(prefix) ? key[prefix.Length..].After('.') : null;

    /// <summary>Returns true if a key is a quantization side tensor (ComfyUI comfy/ops.py: "comfy_quant" layer config, "weight_scale").</summary>
    public static bool IsQuantKey(string key) => key.EndsWith(".comfy_quant") || key.EndsWith(".weight_scale");

    /// <summary>Returns the shape of a Qwen3-VL text encoder weight, following the module constructors in ComfyUI comfy/text_encoders/llama.py (Llama2_), qwen3vl.py and qwen35.py (vision tower).</summary>
    public static long[] Qwen3VLShape(string key, Qwen3VLDims d)
    {
        long q = d.Heads * d.HeadDim, kv = d.KvHeads * d.HeadDim, vh = d.VisionHidden, merge = 4 * d.VisionHidden;
        if (AfterIndex(key, "model.layers.") is string layer)
        {
            return layer switch
            {
                "input_layernorm.weight" or "post_attention_layernorm.weight" => [d.Hidden],
                "self_attn.q_proj.weight" => [q, d.Hidden],
                "self_attn.k_proj.weight" or "self_attn.v_proj.weight" => [kv, d.Hidden],
                "self_attn.o_proj.weight" => [d.Hidden, q],
                "self_attn.q_norm.weight" or "self_attn.k_norm.weight" => [d.HeadDim],
                "mlp.gate_proj.weight" or "mlp.up_proj.weight" => [d.Intermediate, d.Hidden],
                "mlp.down_proj.weight" => [d.Hidden, d.Intermediate],
                _ => throw new ArgumentException($"No Qwen3-VL text encoder shape rule for '{key}'")
            };
        }
        if (AfterIndex(key, "model.visual.blocks.") is string block)
        {
            return block switch
            {
                "norm1.weight" or "norm1.bias" or "norm2.weight" or "norm2.bias" or "attn.proj.bias" or "mlp.linear_fc2.bias" => [vh],
                "attn.qkv.weight" => [3 * vh, vh],
                "attn.qkv.bias" => [3 * vh],
                "attn.proj.weight" => [vh, vh],
                "mlp.linear_fc1.weight" => [d.VisionIntermediate, vh],
                "mlp.linear_fc1.bias" => [d.VisionIntermediate],
                "mlp.linear_fc2.weight" => [vh, d.VisionIntermediate],
                _ => throw new ArgumentException($"No Qwen3-VL text encoder shape rule for '{key}'")
            };
        }
        string merger = AfterIndex(key, "model.visual.deepstack_merger_list.") ?? (key.StartsWith("model.visual.merger.") ? key["model.visual.merger.".Length..] : null);
        if (merger is not null)
        {
            bool deepstack = key.StartsWith("model.visual.deepstack_merger_list.");
            return merger switch
            {
                "norm.weight" or "norm.bias" => [deepstack ? merge : vh],
                "linear_fc1.weight" => [merge, merge],
                "linear_fc1.bias" => [merge],
                "linear_fc2.weight" => [d.Hidden, merge],
                "linear_fc2.bias" => [d.Hidden],
                _ => throw new ArgumentException($"No Qwen3-VL text encoder shape rule for '{key}'")
            };
        }
        return key switch
        {
            "model.embed_tokens.weight" or "lm_head.weight" => [d.Vocab, d.Hidden],
            "model.norm.weight" => [d.Hidden],
            "model.visual.patch_embed.proj.weight" => [vh, 3, 2, 16, 16],
            "model.visual.patch_embed.proj.bias" => [vh],
            "model.visual.pos_embed.weight" => [2304, vh],
            _ => throw new ArgumentException($"No Qwen3-VL text encoder shape rule for '{key}'")
        };
    }

    /// <summary>Writes a zero-weight Qwen3-VL-8B text encoder with every weight key of the released int8 file, unquantized (BF16), at the given dimensions.
    /// ComfyUI builds this encoder from a fixed config, so it only runs at non-real dimensions with the SwarmUITests/ComfyLive null-model hook.</summary>
    public static long WriteQwen3VLTextEncoder(string path, Qwen3VLDims d)
    {
        JObject header = ReadHeader("qwen-image-2.1", "qwen3vl_8b_int8_convrot");
        return Write(path, header["__metadata__"] as JObject, [.. Tensors(header).Where(t => !IsQuantKey(t.Key)).Select(t => t with { Dtype = "BF16", Shape = Qwen3VLShape(t.Key, d) })]);
    }

    /// <summary>Dimensions of a Qwen Image 2.1 VAE, as ComfyUI comfy/sd.py derives them for comfy/ldm/wan/vae2_2.py WanVAE: "dim" from encoder.conv1.weight shape[0], "dec_dim" from decoder.head.0.gamma shape[0].</summary>
    public record class Qwen21VaeDims(long EncoderDim, long DecoderDim)
    {
        /// <summary>The released VAE.</summary>
        public static readonly Qwen21VaeDims Real = new(96, 144);

        /// <summary>The smallest VAE.</summary>
        public static readonly Qwen21VaeDims Null = new(4, 4);
    }

    /// <summary>Returns the shape of a Qwen Image 2.1 VAE weight at the given dimensions. Every channel axis of the encoder is a multiple of "dim" and of the decoder a multiple of "dec_dim" (dim_mult [1, 2, 4, 8, 8], times 2 for time_conv and 3 for to_qkv);
    /// the other axes are kernel sizes, image channels (4) and latent channels (64, 128 for mean and log-variance), and none of them is a multiple of 96 or 144.</summary>
    public static long[] Qwen21VaeShape(string key, long[] realShape, Qwen21VaeDims d)
    {
        (long unit, long to) = key.StartsWith("encoder.") ? (Qwen21VaeDims.Real.EncoderDim, d.EncoderDim) : key.StartsWith("decoder.") ? (Qwen21VaeDims.Real.DecoderDim, d.DecoderDim) : (1, 1);
        return [.. realShape.Select(axis => axis % unit == 0 ? axis / unit * to : axis)];
    }

    /// <summary>Writes a zero-weight Qwen Image 2.1 VAE with every key of the released VAE, at the given dimensions.</summary>
    public static long WriteQwen21Vae(string path, Qwen21VaeDims d)
    {
        JObject header = ReadHeader("qwen-image-2.1", "qwen_image_2.1_vae_bf16");
        return Write(path, header["__metadata__"] as JObject, [.. Tensors(header).Select(t => t with { Shape = Qwen21VaeShape(t.Key, t.Shape, d) })]);
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
            "img_mlp.proj.weight" or "img_mlp.gate_layer.weight" => [mlp, d.Inner],
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

    /// <summary>Writes a zero-weight Qwen Image 2.1 DiT with the unfused MLP layout (img_mlp.proj and gate_layer) of the diffusers tiny-random checkpoint's keys, at the given dimensions.</summary>
    public static long WriteQwen21UnfusedDiT(string path, Qwen21Dims d)
    {
        JObject header = ReadHeader("qwen-image-2.1", "tiny-random-qwen-image-2.1.transformer");
        return Write(path, header["__metadata__"] as JObject, [.. Tensors(header).Select(t => t with { Dtype = "BF16", Shape = Qwen21Shape(t.Key, d) })]);
    }

    /// <summary>Returns the real size of a model file listed in a family's manifest.</summary>
    public static long RealSize(string family, string repoPath, string manifestName = "manifest")
    {
        JObject manifest = File.ReadAllText(Path.Combine(FixtureDir(family), $"{manifestName}.json")).ParseToJson();
        return ((JArray)manifest["files"]).Single(f => $"{f["path"]}" == repoPath)["size"].Value<long>();
    }
}
