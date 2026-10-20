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
    /// Translate the already-processed Swarm prompt syntax into the plain
    /// text expected by native TextEncodeQwenImage21. Never generate an edit
    /// instruction, insert an image token, or reinterpret the author's words.
    /// PromptRegion already removed the segment directive and the request
    /// parser already resolved Swarm wildcards and dynamic prompt tags.
    /// </summary>
    public static string FlattenQwen21SegmentPrompt(string prompt)
    {
        string raw = prompt ?? "";
        // ProcessPromptLikeForLength is Swarm's existing tag flattener,
        // but its estimators would replace scheduled alternatives with the
        // LONGEST variant (and silently discard embedded tokens). Those
        // operations are estimates, not Qwen prompt semantics. Never pick a
        // different instruction behind the user's back.
        // Use Swarm's own registered tag names, not a hand-maintained list.
        // A tag boundary matters: an ordinary Qwen token such as <image1>
        // or free text such as <alternative> must not be misclassified.
        foreach (string registeredTag in T2IPromptHandling.PromptTagLengthEstimators.Keys)
        {
            if (registeredTag.Equals("weight", StringComparison.OrdinalIgnoreCase))
            {
                continue; // This one may be flattened without changing its words.
            }
            string marker = "<" + registeredTag;
            int scan = 0;
            while ((scan = raw.IndexOf(marker, scan, StringComparison.OrdinalIgnoreCase)) >= 0)
            {
                int boundary = scan + marker.Length;
                if (boundary < raw.Length && raw[boundary] is ':' or '[' or '>')
                {
                    throw new SwarmUserErrorException(
                        $"Qwen segment has unsupported residual Swarm tag '<{registeredTag}>'. " +
                        "This tag cannot be flattened without changing the authored prompt; " +
                        "resolve it before native Qwen conditioning.");
                }
                scan = boundary;
            }
        }
        string text = T2IPromptHandling.ProcessPromptLikeForLength(raw);
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new SwarmUserErrorException("Qwen segment prompt has no text after Swarm markup processing.");
        }
        return text;
    }

    /// <summary>
    /// Skip PromptImages[0] only when that image was also the crop source.
    /// An independent InitImage source does not consume PromptImages[0].
    /// </summary>
    public static int FirstQwen21AdditionalReferenceIndex(bool useOriginal, bool hasInitImage)
    {
        return useOriginal && !hasInitImage ? 1 : 0;
    }

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

        // The native Qwen encoder has no Swarm weight-markup parser. Flatten
        // Swarm tags only; pass the user's actual segment and negative text
        // without generating replacement sentences or image references.
        bool useOriginal = UserInput.Get(T2IParamTypes.SegmentQwenUseOriginalReference, true, sectionId: sectionId);
        JObject inputs = new()
        {
            ["clip"] = clip.Path,
            ["vae"] = vae.Path,
            ["prompt"] = FlattenQwen21SegmentPrompt(prompt),
            ["negative_prompt"] = T2IPromptHandling.ProcessPromptLikeForLength(negativePrompt ?? ""),
            ["resolution"] = 0
        };
        int slot = 1;
        if (useOriginal)
        {
            // The crop is the native source reference. An additional Prompt
            // Image is skipped only if it was itself the crop's original source.
            inputs["images.image_1"] = croppedImage;
            slot++;
        }
        if (UserInput.TryGet(T2IParamTypes.PromptImages, out List<Image> references)
            && UserInput.Get(ComfyUIBackendExtension.TextEncodedImage, "auto") != "none")
        {
            bool hasInitImage = UserInput.TryGet(T2IParamTypes.InitImage, out Image initSource)
                && initSource is not null;
            for (int i = FirstQwen21AdditionalReferenceIndex(useOriginal, hasInitImage);
                i < references.Count; i++)
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
