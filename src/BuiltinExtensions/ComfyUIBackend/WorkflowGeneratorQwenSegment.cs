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
    /// Reference Edit follows upstream ComfyUI's native Qwen Image 2.1 contract:
    /// condition on the crop as a vision/reference image and sample from the encoder's
    /// empty Qwen latent. A VAE-encoded crop anchors the original subject too strongly
    /// for identity-changing edits. Masked Img2Img retains its original latent route.
    /// </summary>
    public (JArray Positive, JArray Negative, JArray Latent, int StartStep, int SamplingSteps) CreateQwen21SegmentConditioning(
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
        // In an InitEdit request the first Prompt Image is the original whole
        // image, which is already represented here by the selected crop.
        // Passing both would condition the model on the same original person
        // twice, and strongly resist a requested identity replacement.
        // Preserve additional Prompt Images, or every explicitly supplied image
        // when the dedicated Original Reference option is disabled.
        if (UserInput.TryGet(T2IParamTypes.PromptImages, out System.Collections.Generic.List<Image> referenceImages)
            && UserInput.Get(ComfyUIBackendExtension.TextEncodedImage, "auto") != "none")
        {
            int firstExtra = useOriginalReference ? 1 : 0;
            int maxExtras = useOriginalReference ? 15 : 16;
            int extraCount = Math.Max(0, referenceImages.Count - firstExtra);
            if (extraCount > maxExtras)
            {
                throw new SwarmUserErrorException($"Qwen Image 2.1 segment accepts at most {maxExtras} additional Prompt Images with Original Reference {(useOriginalReference ? "enabled" : "disabled")}; got {extraCount}.");
            }
            for (int i = firstExtra; i < referenceImages.Count; i++)
            {
                JArray referenceImage = GetPromptImage(true, true, i);
                if (referenceImage is not null)
                {
                    encoderInputs[$"image_{i - firstExtra + 1}"] = referenceImage;
                }
            }
        }
        string encoder = CreateNode("SwarmQwenImage21SegmentEncode", encoderInputs);

        // ComfyUI KSampler.set_steps(steps, denoise) expands the sigma schedule to
        // int(steps / denoise), then keeps its final (steps + 1) sigmas.
        // SwarmKSampler uses start_at_step instead: expand the schedule here as
        // well, otherwise a 4-step turbo LoRA at creativity 0.8 runs only 3 steps.
        int samplingSteps = steps;
        int startStep;
        if (mode == "Reference Edit" && creativity > 0 && creativity < 1)
        {
            (samplingSteps, startStep) = PlanQwen21ReferenceEditSteps(steps, creativity);
        }
        else
        {
            startStep = (int)Math.Round(steps * (1 - creativity));
        }
        JArray latent;
        if (mode == "Reference Edit")
        {
            // ComfyUI TextEncodeQwenImage21's third output is an empty latent sized to the
            // first reference. The working non-segmented Qwen edit likewise generates
            // from an empty latent, using the source image solely as reference conditioning.
            // Re-encoding the girl's crop as the starting latent reconstructs her instead
            // of letting the edit replace her. Pixel masking belongs to recomposition.
            latent = NodePath(encoder, 2);
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

        Logs.Info($"[QWEN21-SEG] build section={sectionId} mode='{mode}' configured_steps={steps} scheduled_steps={samplingSteps} start={startStep} executed_steps={samplingSteps - startStep} creativity={creativity:0.###} encoder={encoder} original_reference={useOriginalReference} latent={(mode == "Reference Edit" ? "native-empty-qwen" : "masked-crop-vae")}");
        return (positive, negative, latent, startStep, samplingSteps);
    }

    /// <summary>Use upstream ComfyUI's partial-denoise sigma schedule while keeping
    /// exactly the user-configured count of model evaluations. This controls the
    /// Reference Edit schedule only; Masked Img2Img retains its existing behavior.</summary>
    public static (int SamplingSteps, int StartStep) PlanQwen21ReferenceEditSteps(int steps, double creativity)
    {
        if (steps < 1 || steps > 10000 || !double.IsFinite(creativity) || creativity <= 0 || creativity > 1)
        {
            throw new SwarmUserErrorException($"Invalid Qwen reference-edit step count ({steps}) or creativity ({creativity}).");
        }
        double scheduleLength = steps / creativity;
        if (!double.IsFinite(scheduleLength) || scheduleLength > 10000)
        {
            throw new SwarmUserErrorException($"Qwen Reference Edit requires {scheduleLength:0} scheduled steps for creativity {creativity:0.###} and {steps} sampling iterations, but the sampler supports at most 10000.");
        }
        int samplingSteps = Math.Max(steps, (int)scheduleLength);
        return (samplingSteps, samplingSteps - steps);
    }
}
