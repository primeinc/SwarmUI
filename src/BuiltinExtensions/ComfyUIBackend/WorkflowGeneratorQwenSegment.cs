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
    /// With no Init Image, PromptImages[0] is the selected segment source,
    /// regardless of whether its additional vision reference is enabled.
    /// Never add that same source back as an independent extra reference when
    /// SegmentQwenUseOriginalReference is disabled. An independent Init Image
    /// does not consume PromptImages[0]. The useOriginal argument is retained
    /// for callers of this public helper.
    /// </summary>
    public static int FirstQwen21AdditionalReferenceIndex(bool useOriginal, bool hasInitImage)
    {
        return hasInitImage ? 0 : 1;
    }

    /// <summary>
    /// Native TextEncodeQwenImage21(resolution=0) rounds every independent
    /// reference to multiples of 32 before BOTH vision and VAE encoding.
    /// Match Python's round-to-even; no common-size ImageBatch is involved.
    /// </summary>
    public static (int Width, int Height) NativeQwen21ReferenceDimensions(int width, int height)
    {
        return (Math.Max(32, (int)Math.Round(width / 32.0) * 32),
            Math.Max(32, (int)Math.Round(height / 32.0) * 32));
    }

    /// <summary>
    /// One native Qwen reference image, shared by normal image editing and
    /// segments. Text vision and ReferenceLatent VAE MUST see the same
    /// original pixels at independent, 32-pixel-aligned geometry.
    /// </summary>
    public JArray GetNativeQwen21PromptReference(int index)
    {
        if (!UserInput.TryGet(T2IParamTypes.PromptImages, out List<Image> references)
            || index < 0 || index >= references.Count)
        {
            return null;
        }
        JArray image = GetPromptImage(false, false, index);
        if (image is null)
        {
            return null;
        }
        (int rawWidth, int rawHeight) = references[index].GetResolution();
        (int width, int height) = NativeQwen21ReferenceDimensions(rawWidth, rawHeight);
        if (rawWidth != width || rawHeight != height)
        {
            string resize = CreateNode("ImageScale", new JObject()
            {
                ["image"] = image,
                ["width"] = width,
                ["height"] = height,
                ["crop"] = "disabled",
                ["upscale_method"] = "lanczos"
            });
            image = NodePath(resize, 0);
        }
        return image;
    }

    /// <summary>
    /// Qwen 2.1 regions use the SAME Swarm prompt engine as ordinary Qwen
    /// generation. Do not flatten weights, substitute scheduled text, discard
    /// dynamic LoRA hooks, or manufacture an edit instruction.
    ///
    /// Native Comfy nodes own image VAE encoding, reference latents and the
    /// optional empty edit latent. No Qwen tokenizer or sampler is reimplemented.
    /// </summary>
    public (JArray Positive, JArray Negative, JArray Latent) CreateQwen21SegmentConditioning(
        string prompt, string negativePrompt, WGNodeData clip, WGNodeData vae,
        JArray croppedImage, JArray croppedMask, int sectionId, int steps = 20)
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
        if (steps <= 0)
        {
            throw new SwarmUserErrorException("Qwen segment requires at least one sampling step.");
        }
        if (UserInput.TryGet(T2IParamTypes.VAETileSize, out _) ||
            UserInput.TryGet(T2IParamTypes.VAETemporalTileSize, out _))
        {
            throw new SwarmUserErrorException("Qwen segment reference VAE encoding does not support tiling.");
        }

        bool useOriginal = UserInput.Get(T2IParamTypes.SegmentQwenUseOriginalReference, true, sectionId: sectionId);
        List<JArray> referenceImages = [];
        if (useOriginal)
        {
            referenceImages.Add(croppedImage);
        }
        if (UserInput.TryGet(T2IParamTypes.PromptImages, out List<Image> references)
            && UserInput.Get(ComfyUIBackendExtension.TextEncodedImage, "auto") != "none")
        {
            bool hasInitImage = UserInput.TryGet(T2IParamTypes.InitImage, out Image initSource)
                && initSource is not null;
            for (int i = FirstQwen21AdditionalReferenceIndex(useOriginal, hasInitImage);
                i < references.Count; i++)
            {
                // The same native reference is used by text vision and VAE
                // without generic prompt-image resampling or ImageBatch.
                JArray image = GetNativeQwen21PromptReference(i);
                if (image is not null)
                {
                    referenceImages.Add(image);
                    if (referenceImages.Count > 16)
                    {
                        throw new SwarmUserErrorException("Qwen Image 2.1 accepts at most 16 reference images.");
                    }
                }
            }
        }

        // Follow ordinary SwarmUI's image-batch and prompt-tag handling.
        // This uses the actual cropped source image as vision reference #1,
        // not the generated base frame. No reference is silently duplicated.
        // The legacy images IMAGE batch is correct for one reference. Two
        // differently sized images must use separate Swarm encoder inputs:
        // Comfy ImageBatch resizes the second image to the first geometry.
        JArray textImages = referenceImages.Count == 1 ? referenceImages[0] : null;
        string cropSize = CreateNode("GetImageSize", new JObject()
        {
            ["image"] = croppedImage
        });
        JArray width = NodePath(cropSize, 0), height = NodePath(cropSize, 1);
        JArray dynamicHooks = CreateDynamicLoraHooks();
        JArray encodeSwarmText(string text)
        {
            JObject textInputs = new()
            {
                ["clip"] = clip.Path,
                ["lora_hooks"] = dynamicHooks,
                ["steps"] = steps,
                ["prompt"] = text ?? "",
                // Upstream TextEncodeQwenImage21 passes prevent_empty_text=True
                // to BOTH prompt tokenizers. Preserve native Qwen conditioning
                // for an empty negative without manufacturing prompt text.
                ["qwen21_prevent_empty_text"] = true,
                ["width"] = width,
                ["height"] = height,
                ["target_width"] = width,
                ["target_height"] = height,
                ["images"] = textImages
            };
            if (referenceImages.Count > 1)
            {
                for (int i = 0; i < referenceImages.Count; i++)
                {
                    textInputs[$"image_{i + 1}"] = referenceImages[i];
                }
            }
            string node = CreateNode("SwarmTextEncodeAdvanced", textInputs);
            return NodePath(node, 0);
        }
        JArray positive = encodeSwarmText(prompt);
        JArray negative = encodeSwarmText(negativePrompt);

        // Comfy's ReferenceLatent attaches the same native metadata that
        // TextEncodeQwenImage21 attaches, while SwarmTextEncodeAdvanced keeps
        // the user's original scheduled/weighted/LoRA-tagged text intact.
        string cropEncode = CreateNode("VAEEncode", new JObject()
        {
            ["pixels"] = croppedImage,
            ["vae"] = vae.Path
        });
        JArray cropLatent = NodePath(cropEncode, 0);
        for (int i = 0; i < referenceImages.Count; i++)
        {
            JArray referenceLatent = i == 0 && useOriginal ? cropLatent :
                NodePath(CreateNode("VAEEncode", new JObject()
                {
                    ["pixels"] = referenceImages[i],
                    ["vae"] = vae.Path
                }), 0);
            positive = NodePath(CreateNode("ReferenceLatent", new JObject()
            {
                ["conditioning"] = positive,
                ["latent"] = referenceLatent
            }), 0);
            negative = NodePath(CreateNode("ReferenceLatent", new JObject()
            {
                ["conditioning"] = negative,
                ["latent"] = referenceLatent
            }), 0);
        }

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
        if (mode == "Reference Edit")
        {
            // The first source VAE latent has the correct Qwen channel count,
            // alpha support and exact crop geometry. Native LatentMultiply(0)
            // produces the empty latent needed for full-schedule reference edit.
            string empty = CreateNode("LatentMultiply", new JObject()
            {
                ["samples"] = cropLatent,
                ["multiplier"] = 0.0
            });
            return (positive, negative, NodePath(empty, 0));
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
