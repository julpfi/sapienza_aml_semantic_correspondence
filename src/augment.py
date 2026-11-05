"""Training augmentation of one sample (docs/PRD_finetuning.md sections 3.3 and 5.5).

Source and target are different photos, so each is augmented separately: with probability
`aug_image_prob` an image goes through random crop, rotation, colour jitter, grayscale and blur;
otherwise it is left as it is. Points are moved with the same transform as their image.
Edge rule: after a geometric transform every kept point must be at least `aug_edge_margin` pixels
(in resized space) away from the border and from the padding a rotation adds. A failing source
point, or a target image that loses all accepted targets, rejects the geometric transform of that
image (tried `aug_max_tries` times, then the image keeps its original geometry). Targets that fail
are dropped. Randomness comes from the `random` module, whose state last.pt saves for --resume.
"""

import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from data.transforms import flip_sample


def _margins(wh, cfg):
    """aug_edge_margin (resized pixels) in original pixels, per axis."""
    return np.array([cfg["aug_edge_margin"] * wh[0] / cfg["image_size"],
                     cfg["aug_edge_margin"] * wh[1] / cfg["image_size"]])


def _inside(xy, wh, margin):
    """[K] bool: points at least `margin` [2] away from every border of a wh image."""
    return ((xy >= margin) & (xy <= np.asarray(wh, dtype=np.float64) - margin)).all(axis=1)


def _random_crop(img, xy, cfg):
    w, h = img.size
    side = random.uniform(*cfg["aug_crop_scale"])
    cw, ch = max(1, round(w * side)), max(1, round(h * side))
    x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
    return img.crop((x0, y0, x0 + cw, y0 + ch)), xy - np.array([x0, y0])


def _rotate(img, xy, angle_deg, margin):
    """Rotate about the centre (counter-clockwise, PIL convention). Also returns the points that are
    far enough from the padding: their distance to the original border is kept by the rotation."""
    w, h = img.size
    cx, cy, t = w / 2, h / 2, math.radians(angle_deg)
    dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
    new = np.stack([cx + dx * math.cos(t) + dy * math.sin(t), cy - dx * math.sin(t) + dy * math.cos(t)], 1)
    away_from_padding = _inside(xy, (w, h), margin)
    fill = tuple(int(v) for v in np.asarray(img).reshape(-1, 3).mean(0))
    return img.rotate(angle_deg, resample=Image.BICUBIC, center=(cx, cy), fillcolor=fill), new, away_from_padding


def _geometric(img, xy, cfg):
    """One random crop/rotation of an image and its points [K, 2]. Returns (img, xy, keep [K])."""
    keep = np.ones(len(xy), dtype=bool)
    if cfg["aug_crop_prob"] and random.random() < cfg["aug_crop_prob"]:
        img, xy = _random_crop(img, xy, cfg)
    if cfg["aug_rotation_deg"] and random.random() < cfg["aug_rotation_prob"]:
        angle = random.uniform(-cfg["aug_rotation_deg"], cfg["aug_rotation_deg"])
        img, xy, keep = _rotate(img, xy, angle, _margins(img.size, cfg))
    return img, xy, keep & _inside(xy, img.size, _margins(img.size, cfg))


def _photometric(img, cfg):
    if cfg["aug_color_jitter"] and random.random() < cfg["aug_color_prob"]:
        j = cfg["aug_color_jitter"]
        for enhance in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            img = enhance(img).enhance(random.uniform(1 - j, 1 + j))
        if cfg["aug_hue_jitter"]:
            shift = random.uniform(-cfg["aug_hue_jitter"], cfg["aug_hue_jitter"])
            hsv = np.asarray(img.convert("HSV")).copy()
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + round(shift * 255)) % 256
            img = Image.fromarray(hsv, "HSV").convert("RGB")
    if cfg["aug_grayscale_prob"] and random.random() < cfg["aug_grayscale_prob"]:
        img = ImageOps.grayscale(img).convert("RGB")
    if cfg["aug_blur_prob"] and random.random() < cfg["aug_blur_prob"]:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))
    return img


def _augment_image(img, xy, cfg, need_all):
    """Geometric transform with the edge rule, then colour. need_all: every point must be kept."""
    if random.random() >= cfg["aug_image_prob"]:
        return img, xy.astype(np.float32)
    for _ in range(cfg["aug_max_tries"]):
        new_img, new_xy, keep = _geometric(img, xy, cfg)
        if keep.all() if need_all else keep.any():
            img, xy = new_img, new_xy[keep]
            break  # otherwise the image keeps its original geometry
    return _photometric(img, cfg), xy.astype(np.float32)


def augment_sample(sample, cfg):
    """Return an augmented copy of a training sample (pair flip first, then each image separately)."""
    if cfg["flip_augment"] and random.random() < cfg["flip_augment_prob"]:
        sample = flip_sample(sample)
    sample = dict(sample)
    sample["src_img"], src = _augment_image(sample["src_img"], np.asarray(sample["src_xy"], np.float64)[None],
                                            cfg, need_all=True)
    sample["src_xy"] = src[0]
    sample["trg_img"], sample["trg_xys"] = _augment_image(sample["trg_img"],
                                                          np.asarray(sample["trg_xys"], np.float64), cfg,
                                                          need_all=False)
    sample["src_image"] = f"{sample['src_image']}#aug"  # never reuse cached features of an augmented image
    sample["trg_image"] = f"{sample['trg_image']}#aug"
    return sample


def augmentation_enabled(cfg):
    """True when any augmentation besides the pair flip can change an image."""
    if not cfg["aug_image_prob"]:
        return False
    return bool(cfg["aug_crop_prob"] or (cfg["aug_rotation_deg"] and cfg["aug_rotation_prob"])
                or (cfg["aug_color_jitter"] and cfg["aug_color_prob"]) or cfg["aug_grayscale_prob"]
                or cfg["aug_blur_prob"])
