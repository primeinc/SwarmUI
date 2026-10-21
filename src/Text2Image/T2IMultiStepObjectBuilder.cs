using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SwarmUI.Accounts;
using SwarmUI.Core;
using SwarmUI.Utils;
using ISImage = SixLabors.ImageSharp.Image;
using Image = SwarmUI.Utils.Image;
using SixLabors.ImageSharp.PixelFormats;
using System;
using SwarmUI.Media;

namespace SwarmUI.Text2Image;

/// <summary>This special utility class generates prompts that have multi-step object generation included.
/// Called by <see cref="T2IEngine"/>, should not be called directly.</summary>
public class T2IMultiStepObjectBuilder
{
    /// <summary>ExtraMeta key for the debug record of how an object prompt's image was composed. Set when <see cref="Settings.AddDebugData"/> is enabled.</summary>
    public const string TraceMetadataKey = "debug_trace_objects";

    /// <summary>ExtraMeta key a backend's workflow trace is stored under (WorkflowTrace.MetadataKey in the ComfyUI backend).</summary>
    public const string WorkflowTraceMetadataKey = "debug_trace";

    /// <summary>Returns "#" plus the first 8 hex chars of the SHA-256 of some text, so prompt text is never written raw into debug records.</summary>
    public static string TraceHash(string text) => $"#{Utilities.HashSHA256((text ?? "").EncodeUTF8())[..8]}";

    /// <summary>Describes one object pass: the object's region, the crop of the base image that is inpainted, the inpaint size, and the mask inside it. The crop box is also where the result is pasted back, resized to the crop size.</summary>
    public static JObject TraceObjectPass(PromptRegion.Part part, int pixelX, int pixelY, int pixelWidth, int pixelHeight, int extraX, int extraY, int extraWidth, int extraHeight, int fixedWidth, int fixedHeight, double scaleWidth, double scaleHeight)
    {
        return new JObject()
        {
            ["pass"] = $"object{part.ContextID}",
            ["prompt"] = $"@object{part.ContextID}{TraceHash(part.Prompt)}",
            ["region"] = $"{part.X},{part.Y} {part.Width}x{part.Height}",
            ["object_px"] = $"{pixelX},{pixelY} {pixelWidth}x{pixelHeight}",
            ["crop_px"] = $"{extraX},{extraY} {extraWidth}x{extraHeight}",
            ["inpaint_size"] = $"{fixedWidth}x{fixedHeight}",
            ["mask_px"] = $"{(int)((pixelX - extraX) * scaleWidth)},{(int)((pixelY - extraY) * scaleHeight)} {(int)(pixelWidth * scaleWidth)}x{(int)(pixelHeight * scaleHeight)}",
            ["creativity"] = part.Strength2
        };
    }

    /// <summary>Adds the outcome of a finished pass to its debug record: the seed and model used, the pass's own workflow trace identity and log key if its backend recorded one, and the result image's size and hash.</summary>
    public static void TracePassResult(JObject passTrace, T2IParamInput passInput, MediaFile result)
    {
        passTrace["seed"] = passInput.InternalSet.ValuesInput.GetValueOrDefault(T2IParamTypes.Seed.Type.ID) is long seed ? seed : null;
        passTrace["model"] = passInput.InternalSet.ValuesInput.GetValueOrDefault(T2IParamTypes.Model.Type.ID) is T2IModel model ? T2IParamTypes.CleanModelName(model.Name) : null;
        JObject passRecord = passInput.ExtraMeta.GetValueOrDefault(WorkflowTraceMetadataKey) as JObject;
        passTrace["workflow"] = passRecord?["workflow"]?.DeepClone();
        passTrace["log_key"] = passRecord?["log_key"]?.DeepClone();
        if (result is ImageFile img)
        {
            try
            {
                ImageInfo info = ISImage.Identify(img.RawData);
                passTrace["result"] = $"{info.Width}x{info.Height} #{Utilities.HashSHA256(img.RawData)[..8]} {img.Type.MimeType}";
            }
            catch (Exception ex)
            {
                passTrace["result"] = $"unreadable image ({ex.GetType().Name}) #{Utilities.HashSHA256(img.RawData ?? [])[..8]} {img.Type?.MimeType}";
            }
        }
        else
        {
            passTrace["result"] = result is null ? "no output" : $"not an image ({result.Type?.MimeType})";
        }
    }

    public static async Task<Image> CreateFullImage(string prompt, T2IParamInput user_input, string batchId, Session.GenClaim claim, Action<JObject> output, Action<string> setError, bool isWS, float backendTimeoutMin)
    {
        int obj = 0;
        async Task<MediaFile> createImageDirect(T2IParamInput user_input)
        {
            MediaFile result = null;
            await T2IEngine.CreateImageTask(user_input, batchId + (obj++), claim, output, setError, isWS, backendTimeoutMin, (img, meta) => { result = img.File; }, false);
            return result;
        }
        if (string.IsNullOrWhiteSpace(prompt) || !prompt.Contains("<object:"))
        {
            return null;
        }
        PromptRegion regions = new(prompt);
        PromptRegion.Part[] objects = [.. regions.Parts.Where(p => p.Type == PromptRegion.PartType.Object)];
        if (objects.IsEmpty())
        {
            return null;
        }
        T2IParamInput outerInput = user_input;
        user_input = user_input.Clone();
        JArray tracePasses = null;
        if (Program.ServerSettings.AddDebugData)
        {
            tracePasses = [];
            outerInput.ExtraMeta[TraceMetadataKey] = new JObject()
            {
                ["version"] = 1,
                ["source"] = "multistep-objects",
                ["note"] = "the base image is generated, then each object's crop is inpainted by its own generation and pasted back by Swarm; each pass's own workflow trace is in the server log under its log_key",
                ["passes"] = tracePasses
            };
        }
        if (user_input.TryGet(T2IParamTypes.AltResolutionHeightMult, out _))
        {
            user_input.Set(T2IParamTypes.Height, user_input.GetImageHeight());
            user_input.Remove(T2IParamTypes.AspectRatio);
            user_input.Remove(T2IParamTypes.AltResolutionHeightMult);
        }
        user_input.Remove(T2IParamTypes.RefinerModel);
        user_input.Remove(T2IParamTypes.RefinerUpscale);
        user_input.Remove(T2IParamTypes.RefinerMethod);
        //user_input.Set(T2IParamTypes.EndStepsEarly, 0.2);
        user_input.Set(T2IParamTypes.Seed, user_input.Get(T2IParamTypes.Seed) + 1);
        claim.Extend(1 + objects.Length);
        T2IParamInput basicInput = user_input.Clone();
        foreach (T2IParamTypes.ControlNetParamHolder controlnet in T2IParamTypes.Controlnets)
        {
            basicInput.Remove(controlnet.Model);
            basicInput.Remove(controlnet.Strength);
        }
        MediaFile file = await createImageDirect(basicInput);
        if (tracePasses is not null)
        {
            JObject baseTrace = new() { ["pass"] = "base", ["removed_params"] = "refinermodel, refinerupscale, refinermethod, controlnet models and strengths" };
            TracePassResult(baseTrace, basicInput, file);
            tracePasses.Add(baseTrace);
        }
        if (file is not ImageFile img)
        {
            return null;
        }
        //user_input.Set(T2IParamTypes.EndStepsEarly, 0.6); // TODO: Configurable
        ISImage liveImg = img.ToIS.Clone(_ => { });
        float overBound = 0.1f;
        foreach (PromptRegion.Part part in objects)
        {
            if (claim.ShouldCancel)
            {
                return null;
            }
            user_input.Set(T2IParamTypes.Seed, user_input.Get(T2IParamTypes.Seed) + 1);
            T2IParamInput objInput = user_input.Clone();
            int mpTarget = liveImg.Width * liveImg.Height;
            if (objInput.TryGet(T2IParamTypes.RegionalObjectInpaintingModel, out T2IModel model))
            {
                objInput.Set(T2IParamTypes.Model, model);
                if (model.StandardWidth > 0)
                {
                    mpTarget = model.StandardWidth * model.StandardHeight;
                }
            }
            objInput.Set(T2IParamTypes.Prompt, part.Prompt);
            int pixelX = (int)(part.X * liveImg.Width);
            int pixelY = (int)(part.Y * liveImg.Height);
            int pixelWidth = (int)(part.Width * liveImg.Width);
            int pixelHeight = (int)(part.Height * liveImg.Height);
            int extraX = Math.Max((int)((part.X - overBound * 0.5) * liveImg.Width), 0);
            int extraY = Math.Max((int)((part.Y - overBound * 0.5) * liveImg.Height), 0);
            int extraWidth = Math.Min((int)((part.Width + overBound) * liveImg.Width), liveImg.Width - extraX);
            int extraHeight = Math.Min((int)((part.Height + overBound) * liveImg.Height), liveImg.Height - extraY);
            (int fixedWidth, int fixedHeight) = Utilities.ResToModelFit(extraWidth, extraHeight, mpTarget);
            double scaleWidth = fixedWidth / (double)extraWidth;
            double scaleHeight = fixedHeight / (double)extraHeight;
            using ISImage subImage = liveImg.Clone(i => i.Crop(new Rectangle(extraX, extraY, extraWidth, extraHeight)).Resize(fixedWidth, fixedHeight));
            using ISImage maskImage = ISImage.LoadPixelData<Rgb24>(Enumerable.Repeat(Color.Black.ToPixel<Rgb24>(), 1).ToArray(), 1, 1);
            using ISImage maskInnerImage = ISImage.LoadPixelData<Rgb24>(Enumerable.Repeat(Color.White.ToPixel<Rgb24>(), 1).ToArray(), 1, 1);
            maskInnerImage.Mutate(i => i.Resize((int)(pixelWidth * scaleWidth), (int)(pixelHeight * scaleHeight)));
            maskImage.Mutate(i => i.Resize(fixedWidth, fixedHeight).DrawImage(maskInnerImage, new Point((int)((pixelX - extraX) * scaleWidth), (int)((pixelY - extraY) * scaleHeight)), 1));
            objInput.Set(T2IParamTypes.MaskImage, new(maskImage));
            objInput.Set(T2IParamTypes.Prompt, part.Prompt);
            objInput.Set(T2IParamTypes.InitImage, new Image(subImage));
            objInput.Set(T2IParamTypes.InitImageCreativity, part.Strength2);
            objInput.Set(T2IParamTypes.Width, fixedWidth);
            objInput.Set(T2IParamTypes.Height, fixedHeight);
            MediaFile objFile = await createImageDirect(objInput);
            if (tracePasses is not null)
            {
                JObject passTrace = TraceObjectPass(part, pixelX, pixelY, pixelWidth, pixelHeight, extraX, extraY, extraWidth, extraHeight, fixedWidth, fixedHeight, scaleWidth, scaleHeight);
                TracePassResult(passTrace, objInput, objFile);
                tracePasses.Add(passTrace);
            }
            if (objFile is not ImageFile objImg)
            {
                return null;
            }
            ISImage objISImg = objImg.ToIS;
            objISImg = objISImg.Clone(i => i.Resize(extraWidth, extraHeight));
            liveImg.Mutate(i => i.DrawImage(objISImg, new Point(extraX, extraY), 1));
            if (user_input.Get(T2IParamTypes.OutputIntermediateImages, false))
            {
                output(new JObject() { ["image"] = "data:image/png;base64," + new Image(liveImg).AsBase64, ["batch_index"] = $"{batchId}{obj++}", ["request_id"] = $"{user_input.UserRequestId}", ["metadata"] = null });
            }
        }
        Image composite = new(liveImg);
        if (tracePasses is not null && outerInput.ExtraMeta.GetValueOrDefault(TraceMetadataKey) is JObject objectsTrace)
        {
            objectsTrace["composite"] = $"{liveImg.Width}x{liveImg.Height} #{Utilities.HashSHA256(composite.RawData)[..8]} {composite.Type.MimeType}";
        }
        return composite;
    }
}
