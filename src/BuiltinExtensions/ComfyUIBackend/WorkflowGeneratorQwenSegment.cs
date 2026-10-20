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
    /// Reuses the same Qwen 2.1 text/image encoding and reference-latent plumbing
    /// as a working non-segmented edit. Only the image source (the masked crop)
    /// and the number of effective denoising passes are segment-specific.
    /// </summary>
    public (JArray Positive, JArray Negative, JArray Latent, int SamplerSteps, int StartStep) CreateQwen21SegmentConditioning(
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

        // The first Prompt Image is the original full image being edited. When
        // using the source crop, substitute it into that slot instead of adding
        // the full-image reference AGAIN, which conflicts with a local replace.
        bool originalReference = UserInput.Get(T2IParamTypes.SegmentQwenUseOriginalReference, true, sectionId: sectionId);
        List<JArray> references = [];
        if (originalReference)
        {
            references.Add(NodePath(crop.ScaledImage, 0));
        }
        if (UserInput.TryGet(T2IParamTypes.PromptImages, out List<Image> images)
            && UserInput.Get(ComfyUIBackendExtension.TextEncodedImage, "auto") != "none")
        {
            int firstExtra = originalReference ? 1 : 0;
            if (references.Count + images.Count - firstExtra > 16)
            {
                throw new SwarmUserErrorException("Qwen Image 2.1 supports at most sixteen reference images.");
            }
            for (int i = firstExtra; i < images.Count; i++)
            {
                JArray image = GetPromptImage(true, true, i);
                if (image is not null)
                {
                    references.Add(image);
                }
            }
        }

        // SwarmTextEncodeAdvanced is the same encoder used outside segments, and
        // already supports Qwen vision tokens, prompt tags and dynamic LoRA hooks.
        // ImageBatch is the same multi-image combiner the ordinary Qwen path uses.
        JArray visionImage = null;
        foreach (JArray reference in references)
        {
            if (visionImage is null)
            {
                visionImage = reference;
            }
            else
            {
                string batch = CreateNode("ImageBatch", new JObject()
                {
                    ["image1"] = visionImage,
                    ["image2"] = reference
                });
                visionImage = NodePath(batch, 0);
            }
        }
        JArray positive = CreateConditioning(prompt ?? "", clip.Path, FinalLoadedModel, true,
            attachImages: visionImage, steps: steps);
        JArray negative = CreateConditioning(negativePrompt ?? "", clip.Path, FinalLoadedModel, false,
            attachImages: visionImage, steps: steps);
        if (references.Count > 0)
        {
            // Attach precisely the same VAE reference latents as normal Qwen
            // sampling. Qwen consumes them on BOTH positive and negative paths.
            (positive, negative) = WithSegmentSamplingVae(vae, () =>
            {
                (JArray p, JArray n, _, _) = BuildInputImageHandling(
                    references, positive, negative, NodePath(crop.MaskedLatent, 0));
                return (p, n);
            });
        }

        JArray latent = mode == "Reference Edit"
            ? NodePath(CreateVAEEncode(vae.Path, NodePath(crop.ScaledImage, 0)), 0)
            : NodePath(crop.MaskedLatent, 0);
        // KSampler's 'steps' counts the entire schedule, while start_at_step
        // skips the prefix. To execute four requested passes at 0.8 denoise,
        // construct a 5-step schedule and skip one, exactly as Comfy's
        // KSampler denoise setting does. A 4-step Turbo must run four passes.
        if (steps <= 0 || creativity <= 0 || creativity > 1)
        {
            throw new SwarmUserErrorException("Qwen segment requires positive steps and creativity in (0, 1].");
        }
        double neededSchedule = Math.Ceiling(steps / creativity - 1e-9);
        if (neededSchedule > 10000)
        {
            throw new SwarmUserErrorException($"Qwen segment creativity {creativity:0.####} requires {neededSchedule} scheduler steps to execute {steps} effective passes; ComfyUI supports at most 10000.");
        }
        int samplerSteps = Math.Max(steps, (int)neededSchedule);
        int startStep = samplerSteps - steps;
        Logs.Info($"[QWEN21-SEG] conditioning section={sectionId} mode='{mode}' encoder=SwarmTextEncodeAdvanced references={references.Count} effective_steps={steps} schedule_steps={samplerSteps} start_at_step={startStep}");
        return (positive, negative, latent, samplerSteps, startStep);
    }
}
