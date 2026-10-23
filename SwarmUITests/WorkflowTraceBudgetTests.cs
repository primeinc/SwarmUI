using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Linq;
using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SixLabors.ImageSharp.Memory;
using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SwarmUITests.Identity;
using static SwarmUITests.WorkflowGeneratorHarness;

namespace SwarmUITests;

/// <summary>Cost budgets for <see cref="WorkflowTrace"/>, measured per thread as allocation and as the minimum time of several runs. A traced build's extra allocation and time stay bounded,
/// the trace's own work does not grow with the size of the media, a checkpoint of an unchanged graph is small, and tracing returns every shared pool array it rents (<see cref="PoolBalance"/>), so subtracting pool refills hides no allocation of its own.
/// The media-growth check measures the trace's own Finish, where the trace reads media values. A case over the time bound is measured again before it fails (<see cref="TimeRechecks"/>).</summary>
public class WorkflowTraceBudgetTests : SwarmUITest
{
    /// <summary>Prepares the basics.</summary>
    [OneTimeSetUp]
    public static void PreInit()
    {
        Setup();
        Init();
    }

    /// <summary>Most a traced build may allocate beyond the untraced build of the same request, plus one read of each of its images' bytes:
    /// the trace sizes an image the generator did not resize with ImageSharp's Image.Identify, which reads every PNG chunk's data into a buffer, image data included, and checks the CRC of IHDR, PLTE, IDAT and fdAT
    /// (ImageSharp 3.1.12 PngDecoderCore.TryReadChunk, ReadChunkData and ValidateChunk; PngChunk.IsCritical under the default PngCrcChunkHandling.IgnoreNonCritical).
    /// With ImageSharp's default allocator that buffer is pooled and returned; this test allocates every buffer (<see cref="SimpleGcMemoryAllocator"/>), so it counts.</summary>
    public const long MaxTracedExtraBytes = 1536 * 1024;

    /// <summary>Most a first <see cref="WorkflowTrace.Finish"/> (hash cache empty) may allocate more with large media than with small: far less than one copy of the large media's base64 text (about 1.6 MB here).
    /// Finish is where the trace reads media values (hashes and sizes); the generation hooks never read input values.</summary>
    public const long MaxMediaGrowthBytes = 64 * 1024;

    /// <summary>Most one checkpoint of an unchanged graph may allocate.</summary>
    public const long MaxCheckpointBytes = 1536;

    /// <summary>Most <see cref="WorkflowTrace.Build"/> may allocate.</summary>
    public const long MaxBuildBytes = 1024 * 1024;

    /// <summary>Most a traced build may take, as a multiple of the untraced build's time, plus <see cref="TimeSlackMs"/>. Traced builds measure 1.1x to 1.9x, the most with large media.</summary>
    public const double MaxTracedTimeRatio = 2;

    /// <summary>Fixed allowance added to the time bound, for timer and scheduling noise on short builds.</summary>
    public const double TimeSlackMs = 10;

    /// <summary>Times a case that is over the time bound is measured again (minimum of 5 alternating runs each) before it fails: load from other processes can slow one measurement, and a real slowdown repeats.</summary>
    public const int TimeRechecks = 2;

    /// <summary>Shared-pool refills subtracted from every measurement, while <see cref="TestTraceCostStaysWithinBudget"/> runs.</summary>
    public static GeneratorIdentity.PoolRefills Refills;

    /// <summary>Tracks, on one thread, the shared char and byte pool arrays rented and not yet returned: ArrayPoolEventSource BufferRented (event 1: buffer ID, size, pool ID, bucket ID) and BufferReturned (event 3: buffer ID, size, pool ID), Verbose events whose buffer ID is the array's hash code (.NET 8 SharedArrayPool.Rent and Return).
    /// A pool array that is rented and returned costs a refill at most, which the measurement subtracts as pool state (<see cref="GeneratorIdentity.PoolRefills"/>); one that is never returned is the code's own allocation, which the subtraction would hide, so traced builds must leave none that untraced builds do not.
    /// The listener allocates for every event on the measured thread, so it never runs during an allocation measurement.</summary>
    public sealed class PoolBalance : EventListener
    {
        /// <summary>Managed thread whose rents and returns are tracked: the one that creates the listener.</summary>
        public readonly int ThreadId = Environment.CurrentManagedThreadId;

        /// <summary>Bytes of each array rented on <see cref="ThreadId"/> and not yet returned, by buffer ID.</summary>
        public readonly Dictionary<int, long> Outstanding = [];

        /// <inheritdoc/>
        protected override void OnEventSourceCreated(EventSource source)
        {
            if (source.Name == "System.Buffers.ArrayPoolEventSource")
            {
                EnableEvents(source, EventLevel.Verbose);
            }
        }

        /// <inheritdoc/>
        protected override void OnEventWritten(EventWrittenEventArgs e)
        {
            if (Environment.CurrentManagedThreadId != ThreadId)
            {
                return;
            }
            if (e.EventId == 1)
            {
                int pool = (int)e.Payload[2];
                long size = (int)e.Payload[1];
                if (pool == GeneratorIdentity.PoolRefills.CharPoolId || pool == GeneratorIdentity.PoolRefills.BytePoolId)
                {
                    Outstanding[(int)e.Payload[0]] = pool == GeneratorIdentity.PoolRefills.CharPoolId ? size * 2 : size;
                }
            }
            else if (e.EventId == 3)
            {
                Outstanding.Remove((int)e.Payload[0]);
            }
        }

        /// <summary>Runs an action on this thread and returns the bytes of shared pool arrays it rented and did not return.</summary>
        public static long Unreturned(Action action)
        {
            using PoolBalance balance = new();
            action();
            return balance.Outstanding.Values.Sum();
        }
    }

    /// <summary>Cost of one action: elapsed ticks and bytes allocated on this thread, less <see cref="Refills"/>.</summary>
    /// <param name="Ticks">Elapsed <see cref="Stopwatch"/> ticks.</param>
    /// <param name="Bytes">Bytes allocated on the measuring thread, less shared-pool refills.</param>
    public record class Cost(long Ticks, long Bytes)
    {
        /// <summary>Elapsed milliseconds.</summary>
        public double Ms => Ticks * 1000.0 / Stopwatch.Frequency;

        /// <summary>Formats as "ms, KiB".</summary>
        public override string ToString() => $"{Ms:0.000} ms, {Bytes / 1024.0:0.0} KiB";
    }

    /// <summary>Runs an action several times on this thread, returning the minimum time and the minimum allocation.</summary>
    public static Cost Measure(Action action, int repeats = 3)
    {
        long ticks = long.MaxValue, bytes = long.MaxValue;
        for (int i = 0; i < repeats; i++)
        {
            long allocated = GC.GetAllocatedBytesForCurrentThread() - (Refills?.Bytes ?? 0);
            long start = Stopwatch.GetTimestamp();
            action();
            ticks = Math.Min(ticks, Stopwatch.GetTimestamp() - start);
            bytes = Math.Min(bytes, GC.GetAllocatedBytesForCurrentThread() - (Refills?.Bytes ?? 0) - allocated);
        }
        return new(ticks, bytes);
    }

    /// <summary>Returns a PNG of random pixels. Noise does not compress, so the PNG's size, and its base64 text in the workflow, scale with its area.</summary>
    public static Image NoiseImage(int size, int seed)
    {
        byte[] pixels = new byte[size * size * 3];
        new Random(seed).NextBytes(pixels);
        using SixLabors.ImageSharp.Image<Rgb24> image = SixLabors.ImageSharp.Image.LoadPixelData<Rgb24>(pixels, size, size);
        return new Image(image);
    }

    /// <summary>Costs of one request, untraced and traced, and of the trace's phases after the traced build.</summary>
    /// <param name="Label">Case label.</param>
    /// <param name="MediaBytes">Total size of the request's images.</param>
    /// <param name="Untraced">Untraced build.</param>
    /// <param name="Traced">Traced build, including Finish.</param>
    /// <param name="Checkpoint">One checkpoint of the finished, unchanged graph.</param>
    /// <param name="Phases">Costs of the trace's phases, by name.</param>
    /// <param name="Make">Makes the case's request.</param>
    /// <param name="UntracedUnreturned">Bytes of shared pool arrays an untraced build rents and does not return (<see cref="PoolBalance"/>).</param>
    /// <param name="TracedUnreturned">The same for a traced build.</param>
    /// <param name="PhasesUnreturned">The same for one run of every phase of the trace.</param>
    public record class CaseCost(string Label, long MediaBytes, Cost Untraced, Cost Traced, Cost Checkpoint, Dictionary<string, Cost> Phases, Func<Request> Make, long UntracedUnreturned, long TracedUnreturned, long PhasesUnreturned)
    {
        /// <summary>Bytes the traced build allocates beyond the untraced one.</summary>
        public long ExtraBytes => Traced.Bytes - Untraced.Bytes;
    }

    /// <summary>Returns an untraced and a traced build of a request, the traced one storing its generator in <paramref name="last"/>.</summary>
    public static (Action Untraced, Action Traced) Builds(Func<Request> make, Action<WorkflowGenerator> last = null)
    {
        void untracedBuild()
        {
            Request request = make();
            new WorkflowGenerator() { UserInput = BuildInput(request), Features = [.. request.Features] }.Generate();
        }
        void tracedBuild()
        {
            Request request = make();
            WorkflowGenerator generator = new() { UserInput = BuildInput(request), Features = [.. request.Features] };
            generator.Trace = new(generator);
            generator.Generate();
            last?.Invoke(generator);
        }
        return (untracedBuild, tracedBuild);
    }

    /// <summary>Runs untraced and traced builds alternately, returning the minimum cost of each.</summary>
    public static (Cost Untraced, Cost Traced) MeasureBuilds(Action untracedBuild, Action tracedBuild, int repeats)
    {
        Cost untraced = new(long.MaxValue, long.MaxValue), traced = new(long.MaxValue, long.MaxValue);
        for (int i = 0; i < repeats; i++)
        {
            Cost u = Measure(untracedBuild, 1), t = Measure(tracedBuild, 1);
            untraced = new(Math.Min(untraced.Ticks, u.Ticks), Math.Min(untraced.Bytes, u.Bytes));
            traced = new(Math.Min(traced.Ticks, t.Ticks), Math.Min(traced.Bytes, t.Bytes));
        }
        return (untraced, traced);
    }

    /// <summary>Builds a request untraced and traced, then measures the trace's phases on the traced build, and last, with no measurement running, the shared pool arrays each leaves unreturned.
    /// Each build runs once unmeasured first, then untraced and traced runs alternate, so caches and pools warmed by one do not favor the other.</summary>
    public static CaseCost MeasureCase(string label, Func<Request> make, long mediaBytes, int repeats = 5)
    {
        WorkflowGenerator last = null;
        (Action untracedBuild, Action tracedBuild) = Builds(make, g => last = g);
        untracedBuild();
        tracedBuild();
        (Cost untraced, Cost traced) = MeasureBuilds(untracedBuild, tracedBuild, repeats);
        WorkflowTrace trace = last.Trace;
        T2IParamInput input = last.UserInput;
        const int checkpoints = 50;
        Cost checkpointRun = Measure(() =>
        {
            for (int i = 0; i < checkpoints; i++)
            {
                trace.Checkpoint(-1);
            }
        });
        JObject summary = trace.Build();
        List<JObject> tiers = trace.RecordTiers(summary);
        Dictionary<string, Action> phaseActions = new()
        {
            ["Build"] = () => trace.Build(),
            ["BuildDetail"] = () => trace.BuildDetail(),
            ["workflow hash"] = () => WorkflowTrace.HashIndentedJson(last.Workflow),
            ["log text"] = () => _ = WorkflowTrace.LogJson(summary) + WorkflowTrace.LogJson(trace.BuildDetail()),
            ["RecordTiers"] = () => trace.RecordTiers(summary),
            ["OtherMetadataChars"] = () => WorkflowTrace.OtherMetadataChars(input),
            ["JPEG refit at save"] = () => WorkflowTrace.ForImageMetadata(tiers, "JPG", WorkflowTrace.SavedMetadataChars(input)),
            ["Finish"] = trace.Finish,
            ["Finish, hash cache cleared"] = () =>
            {
                MetadataHelper.LongHashes.Clear();
                trace.Finish();
            }
        };
        Dictionary<string, Cost> phases = phaseActions.ToDictionary(p => p.Key, p => Measure(p.Value));
        long untracedUnreturned = PoolBalance.Unreturned(untracedBuild), tracedUnreturned = PoolBalance.Unreturned(tracedBuild);
        long phasesUnreturned = PoolBalance.Unreturned(() =>
        {
            foreach (Action phase in phaseActions.Values)
            {
                phase();
            }
        });
        return new(label, mediaBytes, untraced, traced, new(checkpointRun.Ticks / checkpoints, checkpointRun.Bytes / checkpoints), phases, make, untracedUnreturned, tracedUnreturned, phasesUnreturned);
    }

    /// <summary>The budgets, for a segment request with an init image and a prompt image across three model families, with small and with large media.
    /// ImageSharp's default allocator pools image buffers, which made a build's allocation differ by about 2 MiB between processes; measurement uses <see cref="SimpleGcMemoryAllocator"/>, which allocates every buffer,
    /// and subtracts the shared array pools' refills (<see cref="GeneratorIdentity.PoolRefills"/>), so the counts repeat.</summary>
    [Test]
    public void TestTraceCostStaysWithinBudget()
    {
        MemoryAllocator priorAllocator = SixLabors.ImageSharp.Configuration.Default.MemoryAllocator;
        SixLabors.ImageSharp.Configuration.Default.MemoryAllocator = new SimpleGcMemoryAllocator();
        using GeneratorIdentity.PoolRefills refills = new();
        Refills = refills;
        try
        {
            MeasureAndCheckBudgets();
        }
        finally
        {
            Refills = null;
            SixLabors.ImageSharp.Configuration.Default.MemoryAllocator = priorAllocator;
        }
    }

    /// <summary>Implements <see cref="TestTraceCostStaysWithinBudget"/>.</summary>
    public static void MeasureAndCheckBudgets()
    {
        Lora("test_alien_lora");
        string prompt = $"{WorkflowTraceTests.PromptMarkerBase} <segment:man> {WorkflowTraceTests.PromptMarkerSegment} <lora:test_alien_lora>";
        (T2IModel Model, string Label)[] models =
        [
            (ModelFor(T2IModelClassSorter.ModelClasses["stable-diffusion-xl-v1-base"], false), "sdxl"),
            (ModelFor(T2IModelClassSorter.ModelClasses["qwen-image-2.1"], true), "qwen-image-2.1"),
            (ModelFor(T2IModelClassSorter.ModelClasses["flux.1-dev/kontext"], false), "flux-kontext")
        ];
        (string Name, Image Init, Image Prompt)[] media =
        [
            ("small", NoiseImage(64, 1), NoiseImage(48, 2)),
            ("large", NoiseImage(512, 3), NoiseImage(384, 4))
        ];
        Stopwatch timer = Stopwatch.StartNew();
        List<CaseCost> costs = [];
        foreach ((T2IModel model, string modelLabel) in models)
        {
            foreach ((string name, Image init, Image promptImage) in media)
            {
                List<Image> promptImages = [promptImage];
                Request make() => new() { Model = model, Prompt = prompt, Setters = { i => i.Set(T2IParamTypes.Width, 512), i => i.Set(T2IParamTypes.Height, 512), i => i.Set(T2IParamTypes.InitImage, init), i => i.Set(T2IParamTypes.InitImageCreativity, 0.6), i => i.Set(T2IParamTypes.PromptImages, promptImages) } };
                costs.Add(MeasureCase($"{modelLabel} {name}", make, init.RawData.Length + promptImage.RawData.Length));
            }
        }
        timer.Stop();
        TestContext.Out.WriteLine($"{costs.Count} cases measured in {timer.ElapsedMilliseconds} ms (after one warm-up, min of 5 alternating runs each, this thread only, {(Refills?.Bytes ?? 0) / 1024} KiB of shared-pool refills subtracted):");
        foreach (CaseCost cost in costs)
        {
            TestContext.Out.WriteLine($"  {cost.Label} (media {cost.MediaBytes / 1024} KiB): untraced {cost.Untraced}; traced {cost.Traced}; extra {cost.ExtraBytes / 1024.0:0.0} KiB; checkpoint {cost.Checkpoint}; pool arrays left unreturned: untraced {cost.UntracedUnreturned} B, traced {cost.TracedUnreturned} B, phases {cost.PhasesUnreturned} B");
            TestContext.Out.WriteLine($"      {cost.Phases.Select(p => $"{p.Key} {p.Value}").JoinString("; ")}");
        }
        List<string> failures = [];
        foreach (CaseCost cost in costs)
        {
            long allowedExtra = MaxTracedExtraBytes + cost.MediaBytes;
            if (cost.ExtraBytes > allowedExtra)
            {
                failures.Add($"{cost.Label}: tracing allocates {cost.ExtraBytes / 1024} KiB beyond the untraced build, over {MaxTracedExtraBytes / 1024} KiB plus one read of its {cost.MediaBytes / 1024} KiB of images");
            }
            if (cost.TracedUnreturned > cost.UntracedUnreturned || cost.PhasesUnreturned > 0)
            {
                failures.Add($"{cost.Label}: tracing rents shared pool arrays it never returns ({cost.TracedUnreturned} B unreturned by a traced build against {cost.UntracedUnreturned} B untraced, {cost.PhasesUnreturned} B by the trace's phases), an allocation the refill subtraction hides");
            }
            if (cost.Checkpoint.Bytes > MaxCheckpointBytes)
            {
                failures.Add($"{cost.Label}: a checkpoint of an unchanged graph allocates {cost.Checkpoint.Bytes} bytes, over {MaxCheckpointBytes}");
            }
            if (cost.Phases["Build"].Bytes > MaxBuildBytes)
            {
                failures.Add($"{cost.Label}: Build allocates {cost.Phases["Build"].Bytes / 1024} KiB, over {MaxBuildBytes / 1024} KiB");
            }
            (double untracedMs, double tracedMs) = (cost.Untraced.Ms, cost.Traced.Ms);
            for (int attempt = 1; tracedMs > untracedMs * MaxTracedTimeRatio + TimeSlackMs && attempt <= TimeRechecks; attempt++)
            {
                (Action untracedBuild, Action tracedBuild) = Builds(cost.Make);
                (Cost untraced, Cost traced) = MeasureBuilds(untracedBuild, tracedBuild, 5);
                TestContext.Out.WriteLine($"  {cost.Label}: time recheck {attempt}: untraced {untraced.Ms:0.0} ms, traced {traced.Ms:0.0} ms (was {tracedMs:0.0} against {untracedMs:0.0})");
                (untracedMs, tracedMs) = (untraced.Ms, traced.Ms);
            }
            if (tracedMs > untracedMs * MaxTracedTimeRatio + TimeSlackMs)
            {
                failures.Add($"{cost.Label}: the traced build takes {tracedMs:0.0} ms against {untracedMs:0.0} ms untraced, over {MaxTracedTimeRatio}x + {TimeSlackMs} ms, in the first measurement and {TimeRechecks} rechecks");
            }
        }
        Func<Request> leakCase = costs[0].Make;
        List<WorkflowGenerator.WorkflowGenStep> steps = [.. WorkflowGenerator.Steps];
        try
        {
            WorkflowGenerator.AddStep(g =>
            {
                if (g.Trace is not null)
                {
                    ArrayPool<char>.Shared.Rent(4096);
                }
            }, 10.5, "budget test: leak a pool array when traced");
            (Action untracedBuild, Action tracedBuild) = Builds(leakCase);
            untracedBuild();
            tracedBuild();
            long untracedLeak = PoolBalance.Unreturned(untracedBuild), tracedLeak = PoolBalance.Unreturned(tracedBuild);
            TestContext.Out.WriteLine($"Negative control, a step that rents a 4096-char array and keeps it when traced: unreturned untraced {untracedLeak} B, traced {tracedLeak} B");
            if (tracedLeak < untracedLeak + 8192)
            {
                failures.Add($"negative control: a traced build that keeps a rented 4096-char array shows {tracedLeak} B unreturned against {untracedLeak} B untraced; the pool balance check cannot see a leak");
            }
        }
        finally
        {
            WorkflowGenerator.Steps = steps;
        }
        foreach ((T2IModel _, string modelLabel) in models)
        {
            CaseCost small = costs.Single(c => c.Label == $"{modelLabel} small"), large = costs.Single(c => c.Label == $"{modelLabel} large");
            long growth = large.Phases["Finish, hash cache cleared"].Bytes - small.Phases["Finish, hash cache cleared"].Bytes;
            if (growth > MaxMediaGrowthBytes)
            {
                failures.Add($"{modelLabel}: a first Finish allocates {growth / 1024} KiB more with {large.MediaBytes / 1024} KiB of media than with {small.MediaBytes / 1024} KiB, over {MaxMediaGrowthBytes / 1024} KiB: something copies the media");
            }
        }
        Assert.That(failures, Is.Empty, string.Join("\n", failures));
    }
}
