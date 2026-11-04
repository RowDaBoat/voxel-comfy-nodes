import os

import numpy
import torch
from PIL import Image, ImageOps

from .naming import naturally_sorted


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def image_paths_in(folder):
    if not os.path.isdir(folder):
        raise ValueError(f"no folder at {folder}")
    names = [name for name in os.listdir(folder) if name.lower().endswith(IMAGE_EXTENSIONS)]
    return [os.path.join(folder, name) for name in naturally_sorted(names)]


def channels_of(picture):
    return numpy.asarray(picture, dtype=numpy.float32) / 255.0


def image_and_mask(path):
    """An IMAGE `(1,H,W,3)` and a MASK `(1,H,W)` the way ComfyUI's Load Image makes them:
    the mask is one minus alpha, empty when the picture has no alpha."""
    with Image.open(path) as opened:
        picture = ImageOps.exif_transpose(opened)
        image = torch.from_numpy(channels_of(picture.convert("RGB"))).unsqueeze(0)
        if "A" in picture.getbands():
            alpha = torch.from_numpy(channels_of(picture.getchannel("A")))
            mask = 1.0 - alpha
        else:
            mask = torch.zeros(image.shape[1:3])
    return image, mask.unsqueeze(0)
