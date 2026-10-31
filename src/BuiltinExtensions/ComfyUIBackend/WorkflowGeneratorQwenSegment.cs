using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUI.Builtin_ComfyUIBackend;

public partial class WorkflowGenerator
{
    /// <summary>
    /// Wires a segment crop into the exact ComfyUI TextEncodeQwenImage21 node.
    /// Upstream owns Qwen vision tokens, reference VAE latents, and the starting
    /// edit latent. No Qwen-specific encoding or sampling is reimplemented here.
    /// </summary>
    public (JArray Positive, JArray Negative, JArray Latent) CreateQwen21SegmentConditioning(
        string prompt, string negativePrompt, WGNodeData clip, WGNodeData vae,
        JArray croppedImage, JArray croppedMask, int sectionId)
    {
        if (croppedImage is null || croppedMask is null || clip is null || vae is null)
        {
            throw new SwarmUserErrorException("Qwen segment requires a source crop, text encoder and VAE.");
        }
        string mode = UserInput.Get(T2IParamTypes.SegmentQwen21Sampling, "Masked Img2Img", sectionId: sectionId);
        if (mode != "Reference Edit" && mode != "Masked Img2Img")
        {
            throw new SwarmUserErrorException($"Unsupported Qwen segment sampling mode '{mode}'.");
        }

        // The native node has no support for scheduled CLIP LoRA hooks or tiled
        // reference VAE encoding. Reject those inputs instead of silently
        // dropping their effects or creating yet another custom Qwen node.
        if (UserInput.DynamicLoraIndices.Count > 0)
        {
            throw new SwarmUserErrorException("Native TextEncodeQwenImage21 cannot apply scheduled LoRA hooks. Static segment LoRAs are supported.");
        }
        if (UserInput.TryGet(T2IParamTypes.VAETileSize, out _) ||
            UserInput.TryGet(T2IParamTypes.VAETemporalTileSize, out _))
        {
            throw new SwarmUserErrorException("Native TextEncodeQwenImage21 does not support VAE tiling for reference images.");
        }

        // Swarm weight tags are parsed by SwarmTextEncodeAdvanced, not by the
        // native Qwen encoder. Reuse Swarm\'s existing prompt-text flattener
        // so Qwen sees the actual subject and LoRA trigger, not literal markup.
        JObject inputs = new()
        {
            ["clip"] = clip.Path,
            ["vae"] = vae.Path,
            ["prompt"] = T2IPromptHandling.ProcessPromptLikeForLength(prompt ?? "").Trim(),
            ["negative_prompt"] = T2IPromptHandling.ProcessPromptLikeForLength(negativePrompt ?? "").Trim(),
            ["resolution"] = 0
        };
        bool useOriginal = UserInput.Get(T2IParamTypes.SegmentQwenUseOriginalReference, true, sectionId: sectionId);
        int slot = 1;
        if (useOriginal)
        {
            // For InitEdit, this *replaces* the full-frame Prompt Images[0].
            // Feeding both the crop and the full frame duplicates the subject
            // that is supposed to be replaced.
            inputs["images.image_1"] = croppedImage;
            slot++;
        }
        if (UserInput.TryGet(T2IParamTypes.PromptImages, out List<Image> references)
            && UserInput.Get(ComfyUIBackendExtension.TextEncodedImage, "auto") != "none")
        {
            for (int i = useOriginal ? 1 : 0; i < references.Count; i++)
            {
                if (slot > 16)
                {
                    throw new SwarmUserErrorException("TextEncodeQwenImage21 accepts at most 16 reference images.");
                }
                JArray image = GetPromptImage(true, true, i);
                if (image is not null)
                {
                    inputs[$"images.image_{slot}"] = image;
                    slot++;
                }
            }
        }

        string native = CreateNode("TextEncodeQwenImage21", inputs);
        JArray positive = NodePath(native, 0);
        JArray negative = NodePath(native, 1);
        if (UserInput.TryGet(T2IParamTypes.ConditioningMultiplier, out double positiveMultiplier)
            && positiveMultiplier != 1)
        {
            positive = NodePath(CreateNode("ConditioningMultiply", new JObject()
            {
                ["conditioning"] = positive,
                ["multiplier"] = positiveMultiplier
            }), 0);
        }
        if (UserInput.TryGet(T2IParamTypes.NegativeConditioningMultiplier, out double negativeMultiplier)
            && negativeMultiplier != 1)
        {
            negative = NodePath(CreateNode("ConditioningMultiply", new JObject()
            {
                ["conditioning"] = negative,
                ["multiplier"] = negativeMultiplier
            }), 0);
        }
        if (string.IsNullOrWhiteSpace(negativePrompt) && ShouldZeroNegative())
        {
            negative = NodePath(CreateNode("ConditioningZeroOut", new JObject()
            {
                ["conditioning"] = negative
            }), 0);
        }
        // These are distinct upstream workflows, not aliases:
        // Reference Edit uses ComfyUI TextEncodeQwenImage21's own edit latent
        // (the path used by ordinary Qwen editing). Masked Img2Img uses
        // LanPaint's masked VAE latent, as in its official inpaint example.
        // No handwritten inpaint encoder or step planner is required.
        if (mode == "Reference Edit")
        {
            return (positive, negative, NodePath(native, 2));
        }
        string maskedEncode = CreateNode("LanPaint_ImageEncode", new JObject()
        {
            ["image"] = croppedImage,
            ["vae"] = vae.Path,
            ["mask"] = croppedMask
        });
        return (positive, negative, NodePath(maskedEncode, 0));
    }
}
