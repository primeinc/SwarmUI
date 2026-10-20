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
    public static string FlattenQwen21SegmentPrompt(string prompt, bool allowEmpty = false)
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
                // The request parser rejects nonnumeric weight metadata but leaves
                // the original tag in place. The length estimator would nevertheless
                // strip that malformed tag, silently changing the author's text.
                // Validate EVERY residual weight occurrence before using it.
                const string weightMarker = "<weight";
                int weightScan = 0;
                while ((weightScan = raw.IndexOf(weightMarker, weightScan, StringComparison.OrdinalIgnoreCase)) >= 0)
                {
                    int boundary = weightScan + weightMarker.Length;
                    if (boundary >= raw.Length || raw[boundary] is not (':' or '[' or '>'))
                    {
                        weightScan = boundary; // eg literal <weighted> is not a Swarm weight tag
                        continue;
                    }
                    int close = raw[boundary] == '[' ? raw.IndexOf(']', boundary + 1) : -1;
                    bool isValid = raw[weightScan..boundary] == weightMarker
                        && close > boundary + 1
                        && close + 1 < raw.Length
                        && raw[close + 1] == ':'
                        && double.TryParse(raw[(boundary + 1)..close].Trim(), out double weight)
                        && double.IsFinite(weight);
                    // A numeric prefix alone is not a complete Swarm tag.
                    // Match nested angle brackets before the length estimator
                    // can touch the input. A missing outer '>' must fail rather
                    // than reaching the native Qwen encoder unchanged.
                    if (isValid)
                    {
                        int depth = 1;
                        bool complete = false;
                        for (int pos = close + 2; pos < raw.Length; pos++)
                        {
                            if (raw[pos] == '<')
                            {
                                depth++;
                            }
                            else if (raw[pos] == '>' && --depth == 0)
                            {
                                complete = true;
                                break;
                            }
                        }
                        isValid = complete;
                    }
                    if (!isValid)
                    {
                        throw new SwarmUserErrorException(
                            "Qwen segment has a malformed residual Swarm weight tag. " +
                            "Only numeric <weight[number]:text> tags can be flattened; " +
                            "the authored prompt has not been reinterpreted.");
                    }
                    weightScan = close + 2;
                }
                continue; // Valid numeric weights can be flattened without changing the words.
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
        if (!allowEmpty && string.IsNullOrWhiteSpace(text))
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
                JArray image = GetPromptImage(true, true, i);
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
        JArray textImages = null;
        foreach (JArray image in referenceImages)
        {
            textImages = textImages is null ? image : NodePath(CreateNode("ImageBatch", new JObject()
            {
                ["image1"] = textImages,
                ["image2"] = image
            }), 0);
        }
        string cropSize = CreateNode("GetImageSize", new JObject()
        {
            ["image"] = croppedImage
        });
        JArray width = NodePath(cropSize, 0), height = NodePath(cropSize, 1);
        JArray dynamicHooks = CreateDynamicLoraHooks();
        JArray encodeSwarmText(string text)
        {
            string node = CreateNode("SwarmTextEncodeAdvanced", new JObject()
            {
                ["clip"] = clip.Path,
                ["lora_hooks"] = dynamicHooks,
                ["steps"] = steps,
                ["prompt"] = text ?? "",
                ["width"] = width,
                ["height"] = height,
                ["target_width"] = width,
                ["target_height"] = height,
                ["images"] = textImages
            });
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
