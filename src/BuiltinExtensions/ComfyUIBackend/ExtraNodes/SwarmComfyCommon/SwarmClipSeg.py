import torch
from PIL import Image
import numpy as np
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import folder_paths
import os, requests, json, tempfile, threading

_CLIPSEG_FILES = (
    "config.json", "merges.txt", "model.safetensors",
    "preprocessor_config.json", "special_tokens_map.json",
    "tokenizer_config.json", "vocab.json",
)
_CLIPSEG_DOWNLOAD_LOCK = threading.Lock()


def _clipseg_file_valid(path, filename):
    """Check on-disk completeness, including safetensors size/header metadata."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
        if filename.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return isinstance(json.load(f), dict)
        if filename.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(path, framework="pt") as data:
                return bool(data.keys())
        return True
    except (OSError, ValueError, UnicodeError, Exception):
        # Any unreadable or invalid cached artifact must be repaired, never
        # treated as successfully downloaded merely because its path exists.
        return False


def get_path():
    if "clipseg" in folder_paths.folder_names_and_paths:
        paths = folder_paths.folder_names_and_paths["clipseg"]
        return paths[0][0]
    else:
        # Jank backup path if you're not running properly in Swarm
        path = os.path.dirname(os.path.realpath(__file__)) + "/models"
        return path


# Keep the model artifacts in safetensors, never Python pickle. An interrupted
# download must not poison subsequent detections with a preexisting directory
# or a partially published file.
def download_model(path, urlbase):
    os.makedirs(path, exist_ok=True)
    with _CLIPSEG_DOWNLOAD_LOCK:
        for filename in _CLIPSEG_FILES:
            filepath = os.path.join(path, filename)
            if _clipseg_file_valid(filepath, filename):
                continue
            fd, pending = tempfile.mkstemp(prefix=f".{filename}.", suffix=".partial", dir=path)
            try:
                with os.fdopen(fd, "wb") as out:
                    print(f"[SwarmClipSeg] Downloading/repairing '{filename}'...")
                    response = requests.get(
                        f"{urlbase}{filename}", stream=True, timeout=(10, 120),
                        headers={"Accept-Encoding": "identity"})
                    try:
                        response.raise_for_status()
                        expected_size = response.headers.get("Content-Length")
                        expected_size = int(expected_size) if expected_size else None
                        written = 0
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                out.write(chunk)
                                written += len(chunk)
                        if written == 0:
                            raise IOError(f"CLIPSeg '{filename}' download was empty")
                        if expected_size is not None and written != expected_size:
                            raise IOError(
                                f"CLIPSeg '{filename}' was truncated: "
                                f"expected {expected_size} bytes, got {written}")
                    finally:
                        response.close()
                if not _clipseg_file_valid(pending, filename):
                    raise IOError(f"CLIPSeg '{filename}' has invalid model/JSON content")
                os.replace(pending, filepath)
            finally:
                if os.path.exists(pending):
                    os.unlink(pending)

class SwarmClipSeg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "match_text": ("STRING", {"multiline": True, "tooltip": "A short description (a few words) to describe something within the image to find and mask."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.01, "round": False, "tooltip": "Threshold to apply to the mask, higher values will make the mask more strict. Without sufficient thresholding, CLIPSeg may include random stray content around the edges."}),
            }
        }

    CATEGORY = "SwarmUI/masks"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "seg"
    DESCRIPTION = "Segment an image using CLIPSeg, creating a mask of what part of an image appears to match the given text."

    def seg(self, images, match_text, threshold):
        if images.ndim != 4 or images.shape[0] < 1:
            raise ValueError("SwarmClipSeg expects a nonempty BHWC image batch")
        path = get_path() + "/clipseg-rd64-refined-fp16-safetensors/"
        download_model(path, "https://huggingface.co/mcmonkey/clipseg-rd64-refined-fp16/resolve/main/")
        processor = CLIPSegProcessor.from_pretrained(path)
        model = CLIPSegForImageSegmentation.from_pretrained(path)

        # The Comfy MASK batch must have the same length and order as IMAGE.
        # The former implementation inspected only images[0] and returned
        # a single mask even when the caller supplied multiple images.
        # Process one image at a time with one loaded detector so each frame
        # is thresholded and normalized independently, not across the batch.
        masks = []
        for frame in images:
            array = 255.0 * frame[..., :3].cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            with torch.no_grad():
                logits = model(**processor(
                    text=match_text, images=img, return_tensors="pt", padding=True))[0]
            mask = torch.nn.functional.threshold(logits.sigmoid(), threshold, 0)
            mask -= mask.min()
            maximum = mask.max()
            if maximum > 0:
                mask /= maximum
            while mask.ndim < 4:
                mask = mask.unsqueeze(0)
            mask = torch.nn.functional.interpolate(
                mask, size=(images.shape[1], images.shape[2]), mode="bilinear")
            if mask.shape[0] != 1 or mask.shape[1] != 1:
                raise ValueError("SwarmClipSeg detector returned multiple masks for one image")
            masks.append(mask[:, 0])
        return (torch.cat(masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "SwarmClipSeg": SwarmClipSeg,
}
