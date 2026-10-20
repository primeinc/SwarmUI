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
    /// Qwen Image 2.1 segments reuse Swarm's ordinary text-conditioning path for LoRAs,
    /// or ComfyUI's native TextEncodeQwenImage21 for genuine image references.
    /// Only the segment crop / sampling schedule belongs to this helper.
    /// </summary>
    public (JArray Positive, JArray Negative, JArray Latent, int StartStep) CreateQwen21SegmentConditioning(
        string prompt, string negativePrompt, WGNodeData clip, WGNodeData vae,
        ImageMaskCropData crop, int steps, double creativity, int sectionId)
    {
        if (crop?.ScaledImage is null || vae is null || clip is null)
        {
            throw new SwarmUserErrorException("Qwen Image 2.1 segment is missing its scaled crop, text encoder, or VAE.");
        }

        string mode = UserInput.Get(T2IParamTypes.SegmentQwen21Sampling, "Reference Edit", sectionId: sectionId);
        if (mode != "Reference Edit" && mode != "Masked Img2Img")
        {
            throw new SwarmUserErrorException($"Unknown Qwen Image 2.1 segment sampling mode '{mode}'.");
        }

        // A confined model LoRA must be encoded with the same Swarm/CLIP route that
        // already works in ordinary generation. A source-image reference can overwhelm
        // an identity edit, so only text-only segments add the crop automatically.
        bool confinedLora = HasLorasForConfinement(sectionId);
        bool dynamicLora = UserInput.DynamicLoraIndices.Count > 0;
        bool originalReference = UserInput.Get(T2IParamTypes.SegmentQwenUseOriginalReference, true, sectionId: sectionId);
        List<JArray> references = [];
        if (originalReference && !confinedLora && !dynamicLora)
        {
            references.Add(NodePath(crop.ScaledImage, 0));
        }
        if (UserInput.TryGet(T2IParamTypes.PromptImages, out List<Image> images)
            && UserInput.Get(ComfyUIBackendExtension.TextEncodedImage, "auto") != "none")
        {
            if (dynamicLora && images.Count > 0)
            {
                throw new SwarmUserErrorException("Native Qwen Image 2.1 reference conditioning cannot apply Swarm's scheduled dynamic LoRA hooks. Disable Prompt Images for that segment or use a static LoRA.");
            }
            if (references.Count + images.Count > 16)
            {
                throw new SwarmUserErrorException($"Qwen Image 2.1 supports at most sixteen reference images; got {references.Count + images.Count}.");
            }
            for (int i = 0; i < images.Count; i++)
            {
                JArray reference = GetPromptImage(true, true, i);
                if (reference is not null)
                {
                    references.Add(reference);
                }
            }
        }

        JArray positive;
        JArray negative;
        if (references.Count == 0)
        {
            // Exactly the ordinary Qwen LoRA/dynamic-hook/prompt-tag conditioner.
            positive = CreateConditioning(prompt ?? "", clip.Path, FinalLoadedModel, true, steps: steps);
            negative = CreateConditioning(negativePrompt ?? "", clip.Path, FinalLoadedModel, false, steps: steps);
        }
        else
        {
            if (UserInput.TryGet(T2IParamTypes.VAETileSize, out _) || UserInput.TryGet(T2IParamTypes.VAETemporalTileSize, out _))
            {
                throw new SwarmUserErrorException("ComfyUI's native TextEncodeQwenImage21 node does not expose tiled reference VAE encoding. Disable VAE tiling for referenced Qwen segments.");
            }
            if ((prompt ?? "").Contains('<') || (negativePrompt ?? "").Contains('<'))
            {
                throw new SwarmUserErrorException("ComfyUI's native TextEncodeQwenImage21 does not parse Swarm's advanced prompt tags with image references. Use a plain referenced segment prompt, or turn off Original Reference.");
            }

            JObject native = new()
            {
                ["clip"] = clip.Path,
                ["vae"] = vae.Path,
                ["prompt"] = prompt ?? "",
                ["negative_prompt"] = negativePrompt ?? "",
                ["resolution"] = 0 // Native Qwen preserves the actual crop/reference geometry.
            };
            for (int i = 0; i < references.Count; i++)
            {
                native[$"images.image_{i + 1}"] = references[i];
            }
            string encoder = CreateNode("TextEncodeQwenImage21", native);
            positive = NodePath(encoder, 0);
            negative = NodePath(encoder, 1);
            // The ordinary conditioner already applies these modifiers in the
            // no-reference case. Apply the same existing Comfy nodes here.
            if (UserInput.TryGet(T2IParamTypes.ConditioningMultiplier, out double posMult) && posMult != 1)
            {
                positive = NodePath(CreateNode("ConditioningMultiply", new JObject()
                {
                    ["conditioning"] = positive,
                    ["multiplier"] = posMult
                }), 0);
            }
            if (UserInput.TryGet(T2IParamTypes.NegativeConditioningMultiplier, out double negMult) && negMult != 1)
            {
                negative = NodePath(CreateNode("ConditioningMultiply", new JObject()
                {
                    ["conditioning"] = negative,
                    ["multiplier"] = negMult
                }), 0);
            }
            if (string.IsNullOrWhiteSpace(negativePrompt) && ShouldZeroNegative())
            {
                negative = NodePath(CreateNode("ConditioningZeroOut", new JObject()
                {
                    ["conditioning"] = negative
                }), 0);
            }
        }

        JArray latent = mode == "Reference Edit"
            ? NodePath(CreateVAEEncode(vae.Path, NodePath(crop.ScaledImage, 0)), 0)
            : NodePath(crop.MaskedLatent, 0);
        int startStep = (int)Math.Round(steps * (1 - creativity));
        Logs.Info($"[QWEN21-SEG] conditioning section={sectionId} mode='{mode}' source={(references.Count == 0 ? "SwarmTextEncodeAdvanced" : "TextEncodeQwenImage21")} references={references.Count} scoped_lora={confinedLora} start={startStep}");
        return (positive, negative, latent, startStep);
    }
}
