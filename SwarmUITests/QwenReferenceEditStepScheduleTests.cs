using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;

namespace SwarmUITests;

/// <summary>
/// Checks the actual Qwen Reference Edit sampling schedule arithmetic.
/// This is NOT a diffusion-inference or LoRA-fidelity test.
/// </summary>
public class QwenReferenceEditStepScheduleTests
{
    [TestCase(4, 1.0, 4, 0)]
    [TestCase(4, 0.8, 5, 1)]
    [TestCase(4, 0.5, 8, 4)]
    [TestCase(20, 0.8, 25, 5)]
    [TestCase(4, 0.1, 40, 36)]
    public void ReferenceEditRetainsEveryRequestedIteration(int steps, double creativity, int expectedScheduleSteps, int expectedStart)
    {
        (int scheduled, int start) = WorkflowGenerator.PlanQwen21ReferenceEditSteps(steps, creativity);
        Assert.That(scheduled, Is.EqualTo(expectedScheduleSteps));
        Assert.That(start, Is.EqualTo(expectedStart));
        Assert.That(scheduled - start, Is.EqualTo(steps),
            "Never truncate a 4-step Turbo LoRA to three evaluations when creativity is below 1.");
    }

    [TestCase(4, 0.0)]
    [TestCase(4, -0.1)]
    [TestCase(4, 1.1)]
    [TestCase(4, double.NaN)]
    [TestCase(0, 0.8)]
    public void InvalidReferenceEditScheduleIsRejected(int steps, double creativity)
    {
        Assert.Throws<SwarmUI.Utils.SwarmUserErrorException>(
            () => WorkflowGenerator.PlanQwen21ReferenceEditSteps(steps, creativity));
    }
}
