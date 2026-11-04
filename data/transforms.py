"""Image preprocessing and the mapping between original and resized coordinates.

Outside this file, every point is (x, y) in ORIGINAL image pixels, origin top-left.
Inside the model, images are resized to a square of cfg["image_size"] pixels (Lang behaviour),
so x and y are scaled separately.
"""

import numpy as np
import torch
from PIL import Image


def preprocess_image(img, cfg, flip=False):
    """PIL image -> normalised tensor [1, 3, S, S]. Same steps as the Lang pipeline."""
    size = cfg["image_size"]
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(cfg["image_mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(cfg["image_std"], dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def to_resized(xy, orig_wh, image_size):
    """Points [K, 2] in original pixels -> resized-square pixels. Accepts numpy or torch."""
    xy = torch.as_tensor(xy, dtype=torch.float32).clone().reshape(-1, 2)
    xy[:, 0] *= image_size / orig_wh[0]
    xy[:, 1] *= image_size / orig_wh[1]
    return xy


def to_original(xy_resized, orig_wh, image_size):
    """Points [K, 2] in resized-square pixels -> original pixels (torch tensor)."""
    xy = torch.as_tensor(xy_resized, dtype=torch.float32).clone().reshape(-1, 2)
    xy[:, 0] *= orig_wh[0] / image_size
    xy[:, 1] *= orig_wh[1] / image_size
    return xy


def flip_points(xy, width):
    """Mirror x coordinates for an image of the given width (original or resized pixels)."""
    xy = np.array(xy, dtype=np.float32, copy=True).reshape(-1, 2)
    xy[:, 0] = width - xy[:, 0]
    return xy


def flip_sample(sample):
    """Mirror both images of a sample and all its points (training augmentation)."""
    sample = dict(sample)
    src_w, trg_w = sample["src_img"].size[0], sample["trg_img"].size[0]
    sample["src_img"] = sample["src_img"].transpose(Image.FLIP_LEFT_RIGHT)
    sample["trg_img"] = sample["trg_img"].transpose(Image.FLIP_LEFT_RIGHT)
    sample["src_xy"] = flip_points(sample["src_xy"], src_w)[0]
    if sample.get("trg_xys") is not None:
        sample["trg_xys"] = flip_points(sample["trg_xys"], trg_w)
    # Cached features belong to the unflipped images.
    sample["src_image"] = f"{sample['src_image']}#flip"
    sample["trg_image"] = f"{sample['trg_image']}#flip"
    return sample


def clip_point(xy, wh):
    """Keep a point inside the image: x in [0, w], y in [0, h]."""
    return np.clip(np.asarray(xy, dtype=np.float64), [0.0, 0.0], [float(wh[0]), float(wh[1])])
