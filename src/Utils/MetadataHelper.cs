using FreneticUtilities.FreneticExtensions;
using FreneticUtilities.FreneticToolkit;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Text2Image;
using System.IO;
using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;

namespace SwarmUI.Utils;

/// <summary>Helper for metadata handling, especially "stealth metadata".</summary>
public static class MetadataHelper
{
    // The stealth info implementation of reforge was used as a reference for this file.

    /// <summary>ExtraMeta key a backend's workflow trace record is stored under (the ComfyUI backend's WorkflowTrace writes it; image metadata copies and object passes read it).</summary>
    public const string WorkflowTraceKey = "debug_trace";

    /// <summary>Version of the workflow trace record format.</summary>
    public const int WorkflowTraceVersion = 1;

    /// <summary>Returns the record stored under <see cref="WorkflowTraceKey"/> before a backend generates, which a backend that traces replaces with its record: it says the named backend recorded no workflow trace.</summary>
    /// <param name="backend">The backend, as "ID (type name)".</param>
    public static JObject NoWorkflowTraceNotice(string backend)
    {
        return new JObject()
        {
            ["version"] = WorkflowTraceVersion,
            ["notice"] = $"no workflow trace: backend {backend} recorded none. Only this server's ComfyUI backends record how they built the workflow; a remote Swarm backend's server keeps its own trace, if it records one, in its own log."
        };
    }

    /// <summary>Strings and byte arrays at least this long are hashed without copying them and have their hash cached in <see cref="LongHashes"/>.</summary>
    public const int CachedHashMinLength = 4096;

    /// <summary>SHA-256 hex of long strings and byte arrays, by reference, so base64 media, image bytes and workflow texts are hashed once however often a trace names them.
    /// Keyed by reference: strings are immutable, and Swarm replaces a media file's RawData array rather than writing into it.</summary>
    public static readonly ConditionalWeakTable<object, string> LongHashes = new();

    /// <summary>Returns the SHA-256 hex of a string's UTF-8 bytes, the same as <c>Utilities.HashSHA256(text.EncodeUTF8())</c>. Long strings are encoded through a small buffer rather than copied, and their hash is cached.</summary>
    public static string HashText(string text)
    {
        text ??= "";
        if (text.Length < CachedHashMinLength)
        {
            return Utilities.HashSHA256(text.EncodeUTF8());
        }
        return LongHashes.GetValue(text, static t => HashWritten(writer => writer.Write((string)t)));
    }

    /// <summary>Returns the SHA-256 hex of some bytes, cached for long arrays.</summary>
    public static string HashBytes(byte[] data)
    {
        data ??= [];
        if (data.Length < CachedHashMinLength)
        {
            return Utilities.HashSHA256(data);
        }
        return LongHashes.GetValue(data, static d => Utilities.HashSHA256((byte[])d));
    }

    /// <summary>Returns the first 8 hex chars of the SHA-256 of some text, the short hash trace records name text by instead of writing it.</summary>
    public static string Sha8(string text) => HashText(text)[..8];

    /// <summary>Returns the first 8 hex chars of the SHA-256 of some bytes.</summary>
    public static string Sha8(byte[] data) => HashBytes(data)[..8];

    /// <summary>Returns the SHA-256 hex of the text a writer callback writes, as UTF-8 without a byte order mark (<see cref="StringConversionHelper.UTF8Encoding"/>, as <c>EncodeUTF8</c> uses), encoding through a small buffer instead of building the text or its bytes.</summary>
    public static string HashWritten(Action<TextWriter> write)
    {
        using SHA256 sha = SHA256.Create();
        using (CryptoStream crypto = new(Stream.Null, sha, CryptoStreamMode.Write))
        using (StreamWriter writer = new(crypto, StringConversionHelper.UTF8Encoding, 4096))
        {
            write(writer);
        }
        return Utilities.BytesToHex(sha.Hash);
    }

    /// <summary>Most characters of metadata text (as <see cref="T2IParamInput.MetadataToString(JObject)"/> writes it) for a JPEG image, whose metadata is a UTF-16 EXIF UserComment, two bytes per character.
    /// Within this limit the EXIF profile fits one JPEG APP1 segment; ImageSharp splits a larger one across several, and the ExifReader the web UI reads metadata with takes only the last Exif segment's data.</summary>
    public const int MaxJpegExifChars = 30_000;

    /// <summary>Returns the notice form of a workflow trace record (the ExtraMeta value under <see cref="WorkflowTraceKey"/>): its version, log key, full trace pointer, workflow identity and sampler count, with what was omitted and why.</summary>
    /// <param name="record">A trace record, of any tier.</param>
    /// <param name="omitted">What the notice omits and why.</param>
    public static JObject TraceNotice(JObject record, string omitted)
    {
        JObject notice = [];
        foreach (string key in new[] { "version", "log_key", "full_trace", "workflow" })
        {
            if (record[key] is JToken value)
            {
                notice[key] = value.DeepClone();
            }
        }
        if (record["samplers"] is JArray samplers)
        {
            notice["samplers"] = samplers.Count;
        }
        else if (record["samplers"] is JToken samplerCount)
        {
            notice["samplers"] = samplerCount.DeepClone();
        }
        notice["omitted"] = omitted;
        return notice;
    }

    /// <summary>Returns metadata text for a JPEG copy of an image's metadata (<see cref="Media.ImageFile.ToMetadataJpg"/>: history thumbnails, model and workflow preview images): text within <see cref="MaxJpegExifChars"/> as it is,
    /// otherwise with a workflow trace record (sui_extra_data.debug_trace) replaced by its notice (<see cref="TraceNotice"/>). Text without a record, or that is not JSON, is returned as it is.</summary>
    public static string FitTextToJpegExif(string metadataText)
    {
        string key = WorkflowTraceKey;
        if (metadataText is null || metadataText.Length <= MaxJpegExifChars || !metadataText.Contains($"\"{key}\""))
        {
            return metadataText;
        }
        JObject metadata;
        try
        {
            using JsonTextReader reader = new(new StringReader(metadataText)) { DateParseHandling = DateParseHandling.None };
            metadata = JObject.Load(reader);
        }
        catch (JsonException)
        {
            return metadataText;
        }
        if (metadata["sui_extra_data"]?[key] is not JObject record)
        {
            return metadataText;
        }
        metadata["sui_extra_data"][key] = TraceNotice(record, $"the trace record of the image this JPEG copy was made from, to fit one EXIF segment; the full trace is in the {record["full_trace"]}");
        return T2IParamInput.MetadataToString(metadata);
    }

    /// <summary>Returns the number of bits <see cref="EncodeStealthMetadata"/> embeds for a metadata text: the signature, the 32-bit length and the gzip-compressed text.</summary>
    /// <param name="metadata">The metadata text.</param>
    /// <param name="mode">The encoding mode: "Alpha" or "RGB".</param>
    public static long StealthPayloadBits(string metadata, string mode)
    {
        return Encoding.UTF8.GetByteCount(StealthSignature(mode.ToLowerInvariant(), true)) * 8L + 32 + Compress(Encoding.UTF8.GetBytes(metadata)).Length * 8L;
    }

    /// <summary>Returns the number of bits <see cref="EncodeStealthMetadata"/> can embed in an image of the given size: one per pixel in "Alpha" mode, three in "RGB" mode. Bits past this are not embedded.</summary>
    public static long StealthCapacityBits(int width, int height, string mode)
    {
        return (long)width * height * (mode.ToLowerInvariant() == "rgb" ? 3 : 1);
    }

    /// <summary>Returns the signature that starts stealth metadata in the given mode ("alpha" or "rgb", lowercase).</summary>
    private static string StealthSignature(string mode, bool compressed) => $"stealth_{(mode == "alpha" ? "png" : "rgb")}{(compressed ? "comp" : "info")}";

    /// <summary>Returns the gzip-compressed form of the given bytes, as stealth metadata stores them.</summary>
    private static byte[] Compress(byte[] data)
    {
        using MemoryStream ms = new();
        using (GZipStream gzip = new(ms, CompressionLevel.Optimal, true))
        {
            gzip.Write(data, 0, data.Length);
        }
        return ms.ToArray();
    }

    /// <summary>Encodes the given metadata string into the LSBs of the image's pixels.</summary>
    /// <param name="image">The image to modify. Must be in Rgba32 format.</param>
    /// <param name="metadata">The metadata string to embed.</param>
    /// <param name="mode">The encoding mode: "Alpha" or "RGB".</param>
    /// <param name="format">The target image format (e.g., "webp", "png").</param>
    public static void EncodeStealthMetadata(Image<Rgba32> image, string metadata, string mode, string format)
    {
        string actualMode = mode.ToLowerInvariant();
        // stupid hack to stop the alpha layer from getting deleted
        if (actualMode == "alpha" && format.StartsWith("WEBP"))
        {
            PrepareImageForWebPAlpha(image);
        }

        string binaryData = PrepareData(metadata, actualMode, true);
        if (actualMode == "alpha")
        {
            EmbedAlpha(image, binaryData);
        }
        else if (actualMode == "rgb")
        {
            EmbedRgb(image, binaryData);
        }
    }

    /// <summary>
    /// Pre-processes the image to prevent the WebP encoder from discarding the alpha channel.
    /// It does this by creating a semi-transparent vertical line on the right edge of the image.
    /// </summary>
    public static void PrepareImageForWebPAlpha(Image<Rgba32> image)
    {
        for (int y = 0; y < image.Height; y++)
        {
            Rgba32 pixel = image[image.Width - 1, y];
            pixel.A = 254;
            image[image.Width - 1, y] = pixel;
        }
    }

    private static string PrepareData(string metadata, string mode, bool compressed)
    {
        string signature = StealthSignature(mode, compressed);
        byte[] signatureBytes = Encoding.UTF8.GetBytes(signature);
        string binarySignature = BytesToBinaryString(signatureBytes);
        byte[] paramBytes = Encoding.UTF8.GetBytes(metadata);
        if (compressed)
        {
            paramBytes = Compress(paramBytes);
        }
        string binaryParam = BytesToBinaryString(paramBytes);
        string binaryParamLen = Convert.ToString(binaryParam.Length, 2).PadLeft(32, '0');
        return binarySignature + binaryParamLen + binaryParam;
    }

    private static string BytesToBinaryString(byte[] bytes)
    {
        StringBuilder sb = new(bytes.Length * 8);
        foreach (byte b in bytes)
        {
            sb.Append(Convert.ToString(b, 2).PadLeft(8, '0'));
        }
        return sb.ToString();
    }

    private static void EmbedAlpha(Image<Rgba32> image, string binaryData)
    {
        int dataIndex = 0;
        for (int x = 0; x < image.Width; x++)
        {
            for (int y = 0; y < image.Height; y++)
            {
                if (dataIndex >= binaryData.Length)
                {
                    return;
                }
                Rgba32 pixel = image[x, y];
                int bit = binaryData[dataIndex] - '0';
                pixel.A = (byte)((pixel.A & 0xFE) | bit);
                image[x, y] = pixel;
                dataIndex++;
            }
        }
    }

    private static void EmbedRgb(Image<Rgba32> image, string binaryData)
    {
        int dataIndex = 0;
        for (int x = 0; x < image.Width; x++)
        {
            for (int y = 0; y < image.Height; y++)
            {
                if (dataIndex >= binaryData.Length)
                {
                    return;
                }
                Rgba32 pixel = image[x, y];
                pixel.R = (byte)((pixel.R & 0xFE) | (binaryData[dataIndex] - '0'));
                if (dataIndex + 1 < binaryData.Length)
                {
                    pixel.G = (byte)((pixel.G & 0xFE) | (binaryData[dataIndex + 1] - '0'));
                }
                if (dataIndex + 2 < binaryData.Length)
                {
                    pixel.B = (byte)((pixel.B & 0xFE) | (binaryData[dataIndex + 2] - '0'));
                }
                image[x, y] = pixel;
                dataIndex += 3;
            }
        }
    }
}
