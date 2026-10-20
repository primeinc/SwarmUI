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

        string encoder = CreateNode("SwarmQwenImage21SegmentEncode", new JObject()
        {
            ["clip"] = clip.Path,
            ["vae"] = vae.Path,
            ["image"] = NodePath(crop.ScaledImage, 0),
            ["prompt"] = prompt ?? "",
            ["negative_prompt"] = negativePrompt ?? ""
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

        Logs.Info($"[QWEN21-SEG] build section={sectionId} mode='{mode}' steps={steps} start={startStep} creativity={creativity:0.###} encoder={encoder} latent={(mode == "Reference Edit" ? "plain-crop-vae" : "masked-crop-vae")}");
        return (NodePath(encoder, 0), NodePath(encoder, 1), latent, startStep);
    }
    /// <summary>
    /// Builds the segment path used when the section contains a static Qwen LoRA.
    /// This deliberately cuts Qwen's reference-image conditioning so the adapter is not
    /// fighting an instruction to preserve the source subject. The existing crop latent
    /// remains the img2img anchor and the pixel-space segment mask still confines the paste.
    /// </summary>
    public (JArray Positive, JArray Negative, JArray Latent, int StartStep) CreateQwen21LoraImg2ImgConditioning(
        string prompt, string negativePrompt, WGNodeData clip, WGNodeData vae,
        ImageMaskCropData crop, int steps, double creativity, int sectionId, bool maskedLatent = false)
    {
        if (crop?.ScaledImage is null || vae is null || clip is null)
        {
            throw new SwarmUserErrorException("Qwen Image 2.1 LoRA segment is missing its scaled crop, text encoder, or VAE.");
        }

        int width = UserInput.GetImageWidth();
        int height = UserInput.GetImageHeight();

        string encodeText(string text)
        {
            return CreateNode("SwarmTextEncodeAdvanced", new JObject()
            {
                ["clip"] = clip.Path,
                ["lora_hooks"] = CreateDynamicLoraHooks(),
                ["steps"] = steps,
                ["prompt"] = text ?? "",
                ["width"] = width,
                ["height"] = height,
                ["target_width"] = width,
                ["target_height"] = height
                // Deliberately no "images" input: match Qwen 2.1 T2I conditioning.
            });
        }

        string positive = encodeText(prompt);
        string negative = encodeText(negativePrompt);
        JArray latent;
        if (maskedLatent)
        {
            latent = NodePath(crop.MaskedLatent, 0);
        }
        else
        {
            string encodedCrop = CreateVAEEncode(vae.Path, NodePath(crop.ScaledImage, 0));
            latent = NodePath(encodedCrop, 0);
        }
        int startStep = (int)Math.Round(steps * (1 - creativity));

        Logs.Info($"[QWEN21-SEG] build section={sectionId} mode='LoRA Img2Img' steps={steps} start={startStep} creativity={creativity:0.###} conditioning=text-only latent={(maskedLatent ? "masked-crop-vae" : "plain-crop-vae")} reference=cut");
        return (NodePath(positive, 0), NodePath(negative, 0), latent, startStep);
    }


}
