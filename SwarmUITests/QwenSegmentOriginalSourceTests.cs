using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Utils;

namespace SwarmUITests;

/// <summary>Exercises the same source selector used by the production graph.</summary>
public class QwenSegmentOriginalSourceTests
{
    [Test]
    public void SegmentOverrideUsesConfiguredModelNotEarlierBaseModel()
    {
        string qwen = T2IModelClassSorter.CompatQwenImage21.ID;
        Assert.That(WorkflowGeneratorSteps.UsesQwen21SegmentModel(
            baseIsQwen21: false, hasSegmentOverride: true, segmentModelCompat: qwen), Is.True);
        Assert.That(WorkflowGeneratorSteps.UsesQwen21SegmentModel(
            baseIsQwen21: true, hasSegmentOverride: true, segmentModelCompat: "stable-diffusion-xl"), Is.False);
        Assert.That(WorkflowGeneratorSteps.UsesQwen21SegmentModel(
            baseIsQwen21: true, hasSegmentOverride: false, segmentModelCompat: null), Is.True);
        Assert.That(WorkflowGeneratorSteps.UsesQwen21SegmentModel(
            baseIsQwen21: false, hasSegmentOverride: false, segmentModelCompat: qwen), Is.False);
    }

    [Test]
    public void ReplicatedSourceBatchUsesDistinctDeterministicSegmentSeeds()
    {
        long seed = 20261020;
        Assert.That(WorkflowGeneratorSteps.Qwen21BatchSegmentSeed(seed, 0), Is.EqualTo(seed));
        Assert.That(WorkflowGeneratorSteps.Qwen21BatchSegmentSeed(seed, 1), Is.EqualTo(seed + 1));
        Assert.That(WorkflowGeneratorSteps.Qwen21BatchSegmentSeed(seed, 2), Is.EqualTo(seed + 2));
        Assert.That(WorkflowGeneratorSteps.Qwen21BatchSegmentSeed(long.MaxValue, 1), Is.EqualTo(0));
    }

    [TestCase("<segment:girl,0.8,0.1> Edit the image", true, false, true, true)]
    [TestCase("sunrise, a girl on the street <segment:girl,0.75,0.15> <weight[1.2]:HannahLynn>", true, false, true, true)]
    [TestCase("sunrise, a girl on the street <segment:girl,0.75,0.15> HannahLynn", true, false, false, false)]
    [TestCase("<segment:girl,0.8,0.1> Edit the image", true, false, false, true)]
    [TestCase("sunrise, a girl <segment:girl,0.8,0.1> HannahLynn", false, false, true, false)]
    [TestCase("sunrise, a girl <segment:girl,0.8,0.1> HannahLynn", true, true, true, false)]
    [TestCase("sunrise, a girl on the street", true, false, true, false)]
    public void OriginalImageSelectionMustNotDependOnWhetherGlobalPromptIsEmpty(
        string text, bool isQwen21, bool isSingleImage, bool hasExplicitSource, bool expectOriginal)
    {
        PromptRegion prompt = new(text);
        Assert.That(WorkflowGeneratorSteps.ShouldQwenSegmentUseOriginalSource(
            prompt, isQwen21, isSingleImage, hasExplicitSource), Is.EqualTo(expectOriginal));
    }
}
