"""Image-space LoRA confinement metrics with real Pillow pixels, no inference required.

The live acceptance harness obtains a binary segmentation mask from the
submitted Comfy graph. These helpers measure ON-vs-OFF differences inside
the subject and strictly outside the mask's expanded blend margin.
Source: ComfyUI MaskToImage / upstream InpaintStitchImproved masked blend.
"""
from PIL import ImageChops, ImageFilter, ImageStat


def normalized_lora_path(name):
    # Comfy model catalogs use actual filenames; Linux paths are case-sensitive.
    # Different-case adapters in one folder are NOT interchangeable.
    normalized = str(name).replace("\\", "/").strip("/")
    suffix = ".safetensors"
    return normalized[:-len(suffix)] if normalized.lower().endswith(suffix) else normalized


def binary_mask(image):
    return image.convert("L").point(lambda n: 255 if n >= 128 else 0)


def region_rms(first, second, mask, margin=48):
    if first.size != second.size or mask.size != first.size:
        raise AssertionError("Segment image and real detector mask dimensions must agree")
    if margin < 0:
        raise ValueError("Blend exclusion margin cannot be negative")
    subject = binary_mask(mask)
    if subject.getbbox() is None:
        raise AssertionError("Segment detector supplied an empty subject mask")
    kernel = margin * 2 + 1
    outer = ImageChops.invert(subject.filter(ImageFilter.MaxFilter(kernel)))
    if outer.getbbox() is None:
        raise AssertionError("Segment mask covers the canvas including blend margin")
    diff = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    inside = ImageStat.Stat(diff, mask=subject).rms
    outside = ImageStat.Stat(diff, mask=outer).rms
    return (
        (sum(x*x for x in inside) / len(inside)) ** 0.5,
        (sum(x*x for x in outside) / len(outside)) ** 0.5,
    )


def assert_same_detector_mask(*masks):
    if not masks or any(mask is None for mask in masks):
        raise AssertionError("Missing actual segment detector mask for LoRA comparison")
    first = binary_mask(masks[0])
    if any(mask.size != masks[0].size or
           ImageChops.difference(first, binary_mask(mask)).getbbox()
           for mask in masks[1:]):
        raise AssertionError("OFF/ON/OFF used different detector masks; LoRA comparison invalid")
    return first
