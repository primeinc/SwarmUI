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
        string encoder = CreateNode("SwarmQwenImage21SegmentEncode", new JObject()
        {
            ["clip"] = clip.Path,
            ["vae"] = vae.Path,
            ["image"] = NodePath(crop.ScaledImage, 0),
            ["prompt"] = prompt ?? "",
            ["negative_prompt"] = negativePrompt ?? "",
            ["use_original_reference"] = useOriginalReference
        });

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
