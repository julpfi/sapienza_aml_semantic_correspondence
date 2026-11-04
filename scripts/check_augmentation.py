"""Draw augmented training pairs with their points, and check the edge rule on many samples.

    python scripts/check_augmentation.py                  # -> outputs/augmentation_check.png
    python scripts/check_augmentation.py --n 16 --checks 500

Look at the picture: every green target and red source point must sit on the same object part as
before augmentation, and never on the grey padding of a rotation (docs/PRD_finetuning.md section 3.3).
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.datasets import build_dataset  # noqa: E402
from src.augment import _margins, augment_sample  # noqa: E402
from src.config import get_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8, help="pairs to draw")
    parser.add_argument("--checks", type=int, default=500, help="augmentations checked for the edge rule")
    args = parser.parse_args()
    cfg = get_config()
    random.seed(cfg["seed"])
    dataset = build_dataset(cfg, "train")

    violations, dropped, near_edge_in_data = 0, 0, 0
    for i in range(args.checks):
        s = dataset[i % len(dataset)]
        a = augment_sample(s, cfg)
        for orig_img, orig_xy, img, xy in ((s["src_img"], s["src_xy"][None], a["src_img"], a["src_xy"][None]),
                                           (s["trg_img"], s["trg_xys"], a["trg_img"], a["trg_xys"])):
            w = orig_img.size[0]
            unmoved = np.concatenate([orig_xy, np.stack([w - orig_xy[:, 0], orig_xy[:, 1]], 1)])  # or flipped
            m = _margins(img.size, cfg)
            for point in xy:
                if ((point >= m - 1e-6) & (point <= np.asarray(img.size) - m + 1e-6)).all():
                    continue
                if np.abs(unmoved - point).max(axis=1).min() < 1e-3:
                    near_edge_in_data += 1  # already near the border in the data; geometry was not changed
                else:
                    violations += 1
        dropped += len(s["trg_xys"]) - len(a["trg_xys"])
    print(f"edge rule: {violations} points moved next to the border in {args.checks} augmented samples "
          f"(must be 0); {near_edge_in_data} points were already near the border in the data; "
          f"{dropped} accepted targets dropped")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(args.n, 4, figsize=(16, 3.6 * args.n), squeeze=False)
    for row in range(args.n):
        s = dataset[random.randrange(len(dataset))]
        a = augment_sample(s, cfg)
        for col, (img, xy, color, title) in enumerate([
                (s["src_img"], s["src_xy"][None], "red", f"{s['query_id']} source"),
                (s["trg_img"], s["trg_xys"], "lime", "target"),
                (a["src_img"], a["src_xy"][None], "red", "augmented source"),
                (a["trg_img"], a["trg_xys"], "lime", "augmented target")]):
            axes[row, col].imshow(img)
            axes[row, col].scatter(xy[:, 0], xy[:, 1], c=color, s=50, edgecolors="black")
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    fig.tight_layout()
    out = Path(cfg["output_dir"]) / "augmentation_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=60)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
