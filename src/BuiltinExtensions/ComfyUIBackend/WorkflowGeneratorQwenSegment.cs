using System;
using Newtonsoft.Json.Linq;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUI.Builtin_ComfyUIBackend;

public partial class WorkflowGenerator
{
    /// <summary>
    /// Builds Qwen Image 2.1 segment conditioning from the actual crop.
    /// Reference Edit follows the same edit-stage sampler contract used by the working Base2Edit
    /// implementation: native-style Qwen conditioning plus a plain VAE encoding of the crop,
    /// with edit strength expressed through start_at_step.
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

        bool useOriginalReference = UserInput.Get(T2IParamTypes.SegmentQwenUseOriginalReference, true, sectionId: sectionId);
        JObject encoderInputs = new()
        {
            ["clip"] = clip.Path,
            ["vae"] = vae.Path,
            ["image"] = NodePath(crop.ScaledImage, 0),
            ["prompt"] = prompt ?? "",
            ["negative_prompt"] = negativePrompt ?? "",
            ["use_original_reference"] = useOriginalReference,
            ["steps"] = steps,
            ["lora_hooks"] = CreateDynamicLoraHooks()
        };
        // Match WGNodeData.EncodeToLatent's VAE tiling contract for reference images.
        // The source crop and each optional reference use the same encoding mode.
        if (UserInput.TryGet(T2IParamTypes.VAETileSize, out _) || UserInput.TryGet(T2IParamTypes.VAETemporalTileSize, out _))
        {
            encoderInputs["reference_tile_size"] = UserInput.Get(T2IParamTypes.VAETileSize, 256);
            encoderInputs["reference_tile_overlap"] = UserInput.Get(T2IParamTypes.VAETileOverlap, 64);
            encoderInputs["reference_temporal_tile_size"] = UserInput.Get(T2IParamTypes.VAETemporalTileSize, 32);
            encoderInputs["reference_temporal_tile_overlap"] = UserInput.Get(T2IParamTypes.VAETemporalTileOverlap, 4);
        }
        // Qwen supports sixteen individual references; the segment crop consumes one
        // when enabled. Keep every optional prompt image in its own native vision slot.
        if (UserInput.TryGet(T2IParamTypes.PromptImages, out System.Collections.Generic.List<Image> referenceImages)
            && UserInput.Get(ComfyUIBackendExtension.TextEncodedImage, "auto") != "none")
        {
            int maxPromptImages = useOriginalReference ? 15 : 16;
            if (referenceImages.Count > maxPromptImages)
            {
                throw new SwarmUserErrorException($"Qwen Image 2.1 segment accepts at most {maxPromptImages} Prompt Images with the current Original Reference setting; got {referenceImages.Count}.");
            }
            for (int i = 0; i < referenceImages.Count; i++)
            {
                JArray referenceImage = GetPromptImage(true, true, i);
                if (referenceImage is not null)
                {
                    encoderInputs[$"image_{i + 1}"] = referenceImage;
                }
            }
        }
        string encoder = CreateNode("SwarmQwenImage21SegmentEncode", encoderInputs);

        int startStep = (int)Math.Round(steps * (1 - creativity));
        JArray latent;
        if (mode == "Reference Edit")
        {
            // Base2Edit's working Qwen 2.1 edit path samples from the current image latent.
            // Do not use the encoder's fresh/noise latent here, and do not attach the segment
            // mask to this latent; the existing pixel-space composite confines the final paste.
            string encodedCrop = CreateVAEEncode(vae.Path, NodePath(crop.ScaledImage, 0));
            latent = NodePath(encodedCrop, 0);
        }
        else
        {
            latent = NodePath(crop.MaskedLatent, 0);
        }

        JArray positive = NodePath(encoder, 0);
        JArray negative = NodePath(encoder, 1);
        // Match the standard CreateConditioningDirect / CreateConditioning post-processing.
        if (UserInput.TryGet(T2IParamTypes.ConditioningMultiplier, out double positiveMultiplier) && positiveMultiplier != 1)
        {
            positive = NodePath(CreateNode("ConditioningMultiply", new JObject()
            {
                ["conditioning"] = positive,
                ["multiplier"] = positiveMultiplier
            }), 0);
        }
        if (UserInput.TryGet(T2IParamTypes.NegativeConditioningMultiplier, out double negativeMultiplier) && negativeMultiplier != 1)
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

        Logs.Info($"[QWEN21-SEG] build section={sectionId} mode='{mode}' steps={steps} start={startStep} creativity={creativity:0.###} encoder={encoder} original_reference={useOriginalReference} latent={(mode == "Reference Edit" ? "plain-crop-vae" : "masked-crop-vae")}");
        return (positive, negative, latent, startStep);
    }
}
