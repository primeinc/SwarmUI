using System;
using System.Buffers.Binary;
using System.Text;
using SwarmUI.Builtin_ComfyUIBackend;

internal static class Program
{
    private static int _checks;

    private static void Check(bool actual, bool expected, string caseName)
    {
        if (actual != expected)
        {
            throw new Exception($"FAIL {caseName}: expected {expected} but got {actual}");
        }
        _checks++;
    }

    private static byte[] PreviewMessage(string metadata, int eventId = 4, byte[]? pixels = null)
    {
        byte[] json = Encoding.UTF8.GetBytes(metadata);
        byte[] message = new byte[8 + json.Length + (pixels?.Length ?? 0)];
        BinaryPrimitives.WriteInt32BigEndian(message.AsSpan(0, 4), eventId);
        BinaryPrimitives.WriteInt32BigEndian(message.AsSpan(4, 4), json.Length);
        json.CopyTo(message, 8);
        pixels?.CopyTo(message, 8 + json.Length);
        return message;
    }

    private static int Main()
    {
        // These are the actual ComfyUI binary event-4 metadata bytes created
        // by SwarmKSampler.py and its 9999123 -> 4 remapping shim.
        byte[] current = PreviewMessage("{\"prompt_id\":\"current\",\"node_id\":\"17\",\"mime_type\":\"image/jpeg\",\"id\":0}", pixels: [255, 216, 255]);
        byte[] stale = PreviewMessage("{\"prompt_id\":\"previous\",\"node_id\":\"17\",\"mime_type\":\"image/jpeg\",\"id\":0}", pixels: [255, 216, 255]);
        Check(H3PreviewPromptGuard.Matches(current, "current"), true, "current job preview");
        Check(H3PreviewPromptGuard.Matches(stale, "current"), false, "stale previous-job preview");
        Check(H3PreviewPromptGuard.Matches(stale, "previous"), true, "previous job still valid for itself");
        Check(H3PreviewPromptGuard.Matches(PreviewMessage("{\"mime_type\":\"image/jpeg\"}"), "current"), true, "legacy preview without identity is accepted");
        Check(H3PreviewPromptGuard.Matches(PreviewMessage("{\"prompt_id\":null}"), "current"), true, "legacy null prompt identity is accepted");
        Check(H3PreviewPromptGuard.Matches(PreviewMessage("{\"prompt_id\":\"current-2\"}"), "current"), false, "exact ID match not prefix");
        Check(H3PreviewPromptGuard.Matches(PreviewMessage("{\"prompt_id\":17}"), "17"), false, "malformed numeric identity rejected");
        Check(H3PreviewPromptGuard.Matches(PreviewMessage("{unparseable"), "current"), false, "invalid JSON rejected");

        byte[] invalidLength = (byte[])current.Clone();
        BinaryPrimitives.WriteInt32BigEndian(invalidLength.AsSpan(4, 4), int.MaxValue);
        Check(H3PreviewPromptGuard.Matches(invalidLength, "current"), false, "invalid metadata length rejected");
        Check(H3PreviewPromptGuard.Matches([0, 0, 0, 4, 0, 0, 0], "current"), false, "truncated preview rejected");

        // Native ComfyUI previews have no job identity. Do not break them.
        Check(H3PreviewPromptGuard.Matches(PreviewMessage("not-json", eventId: 1), "current"), true, "legacy binary preview untouched");

        // Negative control: old behavior forwarded every binary frame after
        // isMe=true, including a stale frame. The new guard must reject it.
        bool oldBehaviorForwardsStale = true;
        Check(H3PreviewPromptGuard.Matches(stale, "current") != oldBehaviorForwardsStale, true,
              "new guard detects cross-generation leak");

        Console.WriteLine($"H3_PREVIEW_PROMPT_GUARD_OK {_checks} cases");
        return 0;
    }
}
