"""Image preprocessing, one small function per operation so each can be shown and explained.

Pipeline used for the baseline (full frame):
    load RGB -> resize to 224 x 308 (keeps the 4:3 shape, nothing cropped or padded)
             -> scale to [0,1] -> subtract ImageNet mean, divide by ImageNet std

Pipeline used for the MegaDetector mitigation (animal crop):
    load RGB -> cut the detector box (+10% context margin) -> pad to a square with mean grey
             -> resize to 224 x 224 -> same normalisation

Augmentations (train-time only, defined here so they can be inspected; the frozen-backbone
baseline embeds clean images, see NOTES.md Phase 3):
    horizontal flip, random-resized-crop, colour jitter, random greyscale.
"""
import numpy as np
import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Per-channel statistics of the ImageNet training set. DINOv2 / ResNet were trained on inputs
# normalised with exactly these numbers, so we must feed them the same scale.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FULL_H, FULL_W = 224, 308      # 308 = 22 patches of 14 px; 224 = 16 patches. 308/224 = 1.375 ~ 4:3
CROP_SIZE = 224
CROP_MARGIN = 0.10             # extra context around a detector box, as a fraction of box size


def load_rgb(path) -> Image.Image:
    """Open as 3-channel RGB. Night/IR frames are stored with R=G=B, so they simply become a grey RGB image."""
    return Image.open(path).convert("RGB")


# ---- geometry ---------------------------------------------------------------------------
def resize_full(img: Image.Image) -> Image.Image:
    """Chosen: resize the WHOLE frame to 224x308. Frames are ~4:3 so distortion is below 3%."""
    return img.resize((FULL_W, FULL_H), Image.BICUBIC)


def center_crop_square(img: Image.Image, size: int = 224) -> Image.Image:
    """Rejected alternative A: shrink the short side then cut a central square (loses ~27% of width)."""
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    return TF.center_crop(img, [size, size])


def letterbox_pad_square(img: Image.Image, size: int = 224, fill=(124, 116, 104)) -> Image.Image:
    """Rejected alternative B for full frames (used for crops): scale long side to `size`, pad the rest with mean colour."""
    w, h = img.size
    s = size / max(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return canvas


def crop_box(img: Image.Image, bbox_norm, margin: float = CROP_MARGIN) -> Image.Image:
    """Cut a MegaDetector box. bbox_norm = [x, y, w, h] as fractions of the image (MegaDetector format)."""
    W, H = img.size
    x, y, w, h = bbox_norm
    x0, y0, x1, y1 = x * W, y * H, (x + w) * W, (y + h) * H
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    box = (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx)), min(H, int(y1 + my)))
    return img.crop(box)


def resize_crop(img: Image.Image) -> Image.Image:
    return letterbox_pad_square(img, CROP_SIZE)


# ---- augmentations ----------------------------------------------------------------------
def aug_hflip(img):
    return ImageOps.mirror(img)


def aug_random_resized_crop(img, seed=0, scale=(0.6, 1.0)):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    t = T.RandomResizedCrop((FULL_H, FULL_W), scale=scale, ratio=(1.25, 1.5))
    return t(img)


def aug_color_jitter(img, seed=0):
    torch.manual_seed(seed)
    return T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)(img)


def aug_grayscale(img):
    return TF.rgb_to_grayscale(img, num_output_channels=3)


# ---- tensor conversion ------------------------------------------------------------------
def to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    """PIL uint8 (H,W,3, 0..255) -> float tensor (3,H,W): x/255, then (x - mean) / std per channel."""
    x = TF.to_tensor(img)                       # divides by 255 -> [0, 1]
    return TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)


def denormalize(x: torch.Tensor) -> np.ndarray:
    """Inverse of the above, for plotting only."""
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * s + m).clamp(0, 1).permute(1, 2, 0).numpy()
