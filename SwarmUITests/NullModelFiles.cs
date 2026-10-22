using System;
using System.IO;
using NUnit.Framework;
using static SwarmUITests.NullModels;

namespace SwarmUITests;

/// <summary>The zero-weight Qwen Image 2.1 file set a live run reads: the micro DiT, a LoRA, the null Qwen3-VL text encoder and the null VAE. `just models` in SwarmUITests/ComfyLive writes it with <see cref="WriteLiveSet"/>.</summary>
[Category("NullModelFiles")]
public class NullModelFiles
{
    /// <summary>Name SwarmUI lists the zero-weight diffusion model under: the qwen-image-2.1 model in diffusion_models format that <see cref="WorkflowGeneratorHarness"/> registers.</summary>
    public const string DiffusionModelName = "test/dm/qwen-image-2.1.safetensors";

    /// <summary>Name of the zero-weight LoRA, without its extension.</summary>
    public const string LoraName = "qwen21-null";

    /// <summary>Writes the file set into the model folders ComfyUI and SwarmUI read under a model root.</summary>
    public static void WriteQwen21LiveSet(string root, string diffusionModelName)
    {
        WriteQwen21DiT($"{root}/diffusion_models/{diffusionModelName}", Qwen21Dims.Micro);
        WriteQwen21Lora($"{root}/loras/{LoraName}.safetensors", Qwen21Dims.Micro, 4);
        WriteQwen3VLTextEncoder($"{root}/text_encoders/qwen3vl_8b_int8_convrot.safetensors", Qwen3VLDims.Null);
        WriteQwen21Vae($"{root}/vae/QwenImage/qwen_image_2.1_vae_bf16.safetensors", Qwen21VaeDims.Null);
    }

    /// <summary>Writes the file set into the model root named by SWARMUI_TEST_COMFY_MODELS; skipped without it.</summary>
    [Test]
    public void WriteLiveSet()
    {
        string root = Environment.GetEnvironmentVariable("SWARMUI_TEST_COMFY_MODELS")?.Replace('\\', '/');
        if (string.IsNullOrWhiteSpace(root))
        {
            Assert.Ignore("Set SWARMUI_TEST_COMFY_MODELS, or run 'just models' in SwarmUITests/ComfyLive.");
        }
        WriteQwen21LiveSet(root, DiffusionModelName);
        Assert.That(File.Exists($"{root}/diffusion_models/{DiffusionModelName}"), $"the diffusion model is written under {root}");
    }
}
