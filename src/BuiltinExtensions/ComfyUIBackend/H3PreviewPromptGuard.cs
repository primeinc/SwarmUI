using System;
using System.Buffers.Binary;
using System.Text.Json;

namespace SwarmUI.Builtin_ComfyUIBackend;

/// <summary>
/// Checks the prompt identity embedded in ComfyUI binary event-4 preview metadata.
///
/// SwarmKSampler attaches prompt_id to every custom preview and SwarmInternalUtil
/// maps its 9999123 message to Comfy event 4. The websocket is reusable: a
/// previous sampler's late async frame must never be attributed to the current
/// request. Legacy/native preview events that do not identify a prompt remain
/// accepted rather than breaking unrelated generation paths.
/// </summary>
public static class H3PreviewPromptGuard
{
    public static bool Matches(byte[]? message, string? expectedPromptId)
    {
        if (message is null || message.Length < 8)
        {
            return false;
        }
        int eventType = BinaryPrimitives.ReadInt32BigEndian(message.AsSpan(0, 4));
        if (eventType != 4)
        {
            return true;
        }
        int metadataLength = BinaryPrimitives.ReadInt32BigEndian(message.AsSpan(4, 4));
        if (metadataLength < 0 || metadataLength > message.Length - 8)
        {
            return false;
        }
        try
        {
            using JsonDocument document = JsonDocument.Parse(message.AsMemory(8, metadataLength));
            JsonElement metadata = document.RootElement;
            if (metadata.ValueKind != JsonValueKind.Object)
            {
                return false;
            }
            if (!metadata.TryGetProperty("prompt_id", out JsonElement promptId) ||
                promptId.ValueKind == JsonValueKind.Null)
            {
                // Native/legacy metadata does not always include a prompt ID.
                return true;
            }
            return promptId.ValueKind == JsonValueKind.String &&
                string.Equals(promptId.GetString(), expectedPromptId, StringComparison.Ordinal);
        }
        catch (JsonException)
        {
            return false;
        }
    }
}
