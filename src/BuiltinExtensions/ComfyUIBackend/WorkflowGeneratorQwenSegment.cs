using System;
using Newtonsoft.Json.Linq;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUI.Builtin_ComfyUIBackend;

public partial class WorkflowGenerator
{
    /// <summary>
    /// Builds Qwen Image 2.1 segment conditioning from the actual crop using a Swarm-owned
    /// compatibility node that mirrors the upstream edit contract and emits a fresh latent.
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

        JArray latent = mode == "Reference Edit" ? NodePath(encoder, 2) : NodePath(crop.MaskedLatent, 0);
        int startStep = mode == "Reference Edit" ? 0 : (int)Math.Round(steps * (1 - creativity));

        Logs.Info($"[QWEN21-SEG] build section={sectionId} mode='{mode}' steps={steps} start={startStep} creativity={creativity:0.###} encoder={encoder}");
        return (NodePath(encoder, 0), NodePath(encoder, 1), latent, startStep);
    }
}
