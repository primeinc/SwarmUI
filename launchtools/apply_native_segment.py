"""One-shot, revision-checked source patch. Removed after the CI proof commit."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
paths = {
 'generator': ('src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs', 'dca3b2f2f17852e584d5c054ef478b3fc8781857'),
 'steps': ('src/BuiltinExtensions/ComfyUIBackend/WorkflowGeneratorSteps.cs', '36a3bc2518241cfad441aac124f592a7eb63ebcb'),
 'params': ('src/Text2Image/T2IParamTypes.cs', 'b58b0eed54f7f3aee7584d990a699779d2c3090a'),
 'extension': ('src/BuiltinExtensions/ComfyUIBackend/ComfyUIBackendExtension.cs', '0b0d970e8f5ab7a11fd5b8d6b01ed22d7a3ef0e3'),
}
sources = {}
for key, (name, expected) in paths.items():
    actual = subprocess.check_output(['git', 'hash-object', name], cwd=root, text=True).strip()
    if actual != expected:
        raise RuntimeError(f'Refusing to patch changed source {name}: {actual} != {expected}')
    sources[key] = (root / name).read_text(encoding='utf-8')

def replace(key, old, new):
    count = sources[key].count(old)
    if count != 1:
        raise RuntimeError(f'{key}: expected exactly one anchor, found {count}: {old[:120]}')
    sources[key] = sources[key].replace(old, new, 1)

replace('generator', 'int sectionId = 0, WGNodeData negativeModel = null)',
    'int sectionId = 0, WGNodeData negativeModel = null, bool referenceConditioningHandled = false)')
replace('generator', '''        JArray imgNeg = null;
        if (UserInput.TryGet(T2IParamTypes.PromptImages, out List<Image> images) && images.Count > 0)
        {
            (pos, neg, latent, imgNeg) = BuildInputImageHandling(LoadPromptImagesForMainRef(images), pos, neg, latent);
        }
        else
        {
            if (IsKontext() || (IsQwenImage() && IsQwenImageEdit()) || ShouldUseQwen21AutomaticReference())''',
'''        JArray imgNeg = null;
        // Native paired encoders have already attached matching vision/reference conditioning.
        // Do not append global or inferred crop references for a second time.
        if (!referenceConditioningHandled && UserInput.TryGet(T2IParamTypes.PromptImages, out List<Image> images) && images.Count > 0)
        {
            (pos, neg, latent, imgNeg) = BuildInputImageHandling(LoadPromptImagesForMainRef(images), pos, neg, latent);
        }
        else if (!referenceConditioningHandled)
        {
            if (IsKontext() || (IsQwenImage() && IsQwenImageEdit()) || ShouldUseQwen21AutomaticReference())''')
replace('steps', '''                    PromptRegion.Part part = parts[i];
                    string[] segmentSections = part.DataText.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);''',
'''                    PromptRegion.Part part = parts[i];
                    if (g.IsQwenImage21() && part.Strength2 <= 0)
                    {
                        continue;
                    }
                    string[] segmentSections = part.DataText.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);''')
replace('steps', '''                    JArray prompt = g.CreateConditioning(part.Prompt, clip.Path, t2iModel, true, steps: steps);
                    string neg = negativeParts.FirstOrDefault(p => p.DataText == part.DataText)?.Prompt ?? negativeRegion.GlobalPrompt;
                    JArray negPrompt = g.CreateConditioning(neg, clip.Path, t2iModel, false, steps: steps);
                    int startStep = (int)Math.Round(steps * (1 - part.Strength2));''',
'''                    string neg = negativeParts.FirstOrDefault(p => p.DataText == part.DataText)?.Prompt ?? negativeRegion.GlobalPrompt;
                    int startStep = (int)Math.Round(steps * (1 - part.Strength2));
                    bool nativeQwenSegment = g.IsQwenImage21();
                    JArray prompt, negPrompt;
                    JArray segmentLatent = NodePath(g.MaskShrunkInfo.MaskedLatent, 0);
                    if (nativeQwenSegment)
                    {
                        (prompt, negPrompt, segmentLatent, startStep) = g.CreateQwen21SegmentConditioning(
                            part.Prompt, neg, clip, vae, g.MaskShrunkInfo, steps, part.Strength2, part.ContextID);
                    }
                    else
                    {
                        prompt = g.CreateConditioning(part.Prompt, clip.Path, t2iModel, true, steps: steps);
                        negPrompt = g.CreateConditioning(neg, clip.Path, t2iModel, false, steps: steps);
                    }''')
replace('steps', '''string sampler = g.WithSegmentSamplingVae(vae, () => g.CreateKSampler(model.Path, prompt, negPrompt, [g.MaskShrunkInfo.MaskedLatent, 0], cfg, steps, startStep, 10000, seed, false, true, sectionId: part.ContextID));''',
'''string sampler = g.WithSegmentSamplingVae(vae, () => g.CreateKSampler(model.Path, prompt, negPrompt, segmentLatent, cfg, steps, startStep, 10000, seed, false, true, sectionId: part.ContextID, referenceConditioningHandled: nativeQwenSegment));''')
match = re.search(r'public (?:static )?class T2IParamTypes\s*\n\{', sources['params'])
if match is None:
    raise RuntimeError('T2IParamTypes class declaration not found')
replace('params', match.group(), match.group() + '''
    /// <summary>Qwen 2.1 segment sampling mode: native reference editing or partial-denoise masked img2img.</summary>
    public static T2IRegisteredParam<string> SegmentQwen21Sampling;
''')
anchor = '        // ================================================ Advanced Sampling ================================================'
replace('params', anchor, '''        SegmentQwen21Sampling = Register<string>(new("Segment Qwen 2.1 Sampling", "Reference Edit regenerates each selected crop using native Qwen image conditioning and the full configured step schedule, then composites through the segment mask. Segment creativity zero skips the edit; other creativity values do not shorten this mode. Select Masked Img2Img to retain partial-denoise Segment Creativity and latent masking. Applies only to Qwen Image 2.1.",
            "Reference Edit", IgnoreIf: "Reference Edit", GetValues: _ => ["Reference Edit", "Masked Img2Img"], IsAdvanced: true, Group: GroupSegmentOverrides, OrderPriority: 5.6
            ));
''' + anchor)
replace('extension', '        ["SwarmKSampler"] = "variation_seed",', '        ["SwarmKSampler"] = "variation_seed",\n        ["TextEncodeQwenImage21"] = "qwen21_native_conditioning",')
for key, (name, _) in paths.items():
    (root / name).write_text(sources[key], encoding='utf-8', newline='\n')
helper = root / 'launchtools/qwen-native-helper.cs.txt'
(root / 'src/BuiltinExtensions/ComfyUIBackend/WorkflowGeneratorQwenSegment.cs').write_text('using System;\n' + helper.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
helper.unlink()
print('Native segment patch applied to exact pinned source files.')
