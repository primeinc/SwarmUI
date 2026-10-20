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
    /// Reference Edit follows the same edit-stage sampling shape used by the working Base2Edit Qwen path:
    /// native-style Qwen conditioning plus a plain VAE encoding of the current crop, with edit strength
    /// expressed through start_at_step. It deliberately does not sample from a fresh latent.
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

        JArray latent;
        int startStep = (int)Math.Round(steps * (1 - creativity));
        if (mode == "Reference Edit")
        {
            // Match the proven Base2Edit Qwen 2.1 sampler contract: current image latent as the edit anchor.
            // Do not attach the segment mask to the latent here; the existing pixel-space composite confines
            // the regenerated crop to the selected segment after decode.
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
}
