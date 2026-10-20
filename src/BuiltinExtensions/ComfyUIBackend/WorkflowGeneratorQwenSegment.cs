using System;
using Newtonsoft.Json.Linq;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmUI.Builtin_ComfyUIBackend;

public partial class WorkflowGenerator
{
    /// <summary>
    /// Builds Qwen Image 2.1 segment conditioning using Swarm's bundled multimodal text encoder.
    /// This intentionally avoids requiring a newer upstream ComfyUI TextEncodeQwenImage21 node.
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

        JArray cropImage = NodePath(crop.ScaledImage, 0);
        JArray textImages = cropImage;
        for (int i = 0; i < 15; i++)
        {
            JArray image = GetPromptImage(true, true, i);
            if (image is null)
            {
                break;
            }
            string batched = CreateNode("ImageBatch", new JObject()
            {
                ["image1"] = textImages,
                ["image2"] = image
            });
            textImages = NodePath(batched, 0);
        }

        int width = UserInput.GetImageWidth();
        int height = UserInput.GetImageHeight();
        JObject encoderInputs(string text) => new()
        {
            ["clip"] = clip.Path,
            ["lora_hooks"] = CreateDynamicLoraHooks(),
            ["steps"] = steps,
            ["prompt"] = text ?? "",
            ["width"] = width,
            ["height"] = height,
            ["target_width"] = width,
            ["target_height"] = height,
            ["images"] = textImages
        };

        string positiveEncoder = CreateNode("SwarmTextEncodeAdvanced", encoderInputs(prompt));
        string negativeEncoder = CreateNode("SwarmTextEncodeAdvanced", encoderInputs(negativePrompt));
        JArray positive = NodePath(positiveEncoder, 0);
        JArray negative = NodePath(negativeEncoder, 0);

        void addReferenceLatent(JArray image)
        {
            string encoded = CreateVAEEncode(vae.Path, image);
            string positiveRef = CreateNode("ReferenceLatent", new JObject()
            {
                ["conditioning"] = positive,
                ["latent"] = NodePath(encoded, 0)
            });
            positive = NodePath(positiveRef, 0);
            string negativeRef = CreateNode("ReferenceLatent", new JObject()
            {
                ["conditioning"] = negative,
                ["latent"] = NodePath(encoded, 0)
            });
            negative = NodePath(negativeRef, 0);
        }

        // The crop must be visible to both Qwen's vision tokenizer and its reference-latent path.
        addReferenceLatent(cropImage);
        for (int i = 0; i < 15; i++)
        {
            JArray image = GetPromptImage(true, false, i);
            if (image is null)
            {
                break;
            }
            addReferenceLatent(image);
        }

        // Both modes use the existing correctly-sized masked crop latent.
        // Reference Edit starts from the full schedule; Masked Img2Img preserves partial-denoise creativity.
        JArray latent = NodePath(crop.MaskedLatent, 0);
        int startStep = mode == "Reference Edit" ? 0 : (int)Math.Round(steps * (1 - creativity));
        return (positive, negative, latent, startStep);
    }
}
