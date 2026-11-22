"""Image preprocessing and the mapping between original and resized coordinates.

Outside this file, every point is (x, y) in ORIGINAL image pixels, origin top-left.
Inside the model, images are resized to a square of cfg["image_size"] pixels:
- letterbox=False (default/Lang): stretch W and H independently to S x S.
- letterbox=True: scale longest side to S preserving aspect ratio, and center-pad with
  ImageNet mean (zero after normalisation) to S x S.
"""

import numpy as np
import torch
from PIL import Image

IMAGENET_PAD_RGB = (124, 116, 104)  # round([0.485, 0.456, 0.406] * 255) -> ~0.0 after norm


def letterbox_params(orig_wh, image_size):
    """Return (new_w, new_h, pad_x, pad_y) for aspect-preserving resize + center-pad to image_size."""
    w, h = float(orig_wh[0]), float(orig_wh[1])
    scale = float(image_size) / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    pad_x = (int(image_size) - new_w) // 2
    pad_y = (int(image_size) - new_h) // 2
    return new_w, new_h, pad_x, pad_y


def preprocess_image(img, cfg, flip=False):
    """PIL image -> normalised tensor [1, 3, S, S]."""
    size = int(cfg["image_size"])
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if cfg.get("letterbox"):
        new_w, new_h, pad_x, pad_y = letterbox_params(img.size, size)
        resized = img.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new("RGB", (size, size), IMAGENET_PAD_RGB)
        canvas.paste(resized, (pad_x, pad_y))
    else:
        canvas = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(cfg["image_mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(cfg["image_std"], dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def to_resized(xy, orig_wh, image_size, letterbox=False):
    """Points [K, 2] in original pixels -> resized-square pixels. Accepts numpy or torch."""
    xy = torch.as_tensor(xy, dtype=torch.float32).clone().reshape(-1, 2)
    if letterbox:
        new_w, new_h, pad_x, pad_y = letterbox_params(orig_wh, image_size)
        xy[:, 0] = xy[:, 0] * (new_w / orig_wh[0]) + pad_x
        xy[:, 1] = xy[:, 1] * (new_h / orig_wh[1]) + pad_y
    else:
        xy[:, 0] *= image_size / orig_wh[0]
        xy[:, 1] *= image_size / orig_wh[1]
    return xy


def to_original(xy_resized, orig_wh, image_size, letterbox=False):
    """Points [K, 2] in resized-square pixels -> original pixels (torch tensor)."""
    xy = torch.as_tensor(xy_resized, dtype=torch.float32).clone().reshape(-1, 2)
    if letterbox:
        new_w, new_h, pad_x, pad_y = letterbox_params(orig_wh, image_size)
        xy[:, 0] = (xy[:, 0] - pad_x) * (orig_wh[0] / new_w)
        xy[:, 1] = (xy[:, 1] - pad_y) * (orig_wh[1] / new_h)
    else:
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
