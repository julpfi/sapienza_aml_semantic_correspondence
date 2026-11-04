"""Analyse validation results: error breakdown and a visual check of a few predictions."""

from pathlib import Path


def error_breakdown(per_query, alpha):
    """Where the errors are: near misses (alpha..2*alpha) and far misses (> 0.5)."""
    dist = per_query["dist"]
    return {
        "median_dist": float(dist.median()),
        "near_miss_pct": 100 * float(((dist >= alpha) & (dist < 2 * alpha)).mean()),
        "far_miss_pct": 100 * float((dist > 0.5).mean()),
    }


def plot_predictions(cfg, model, dataset, out_png, n=4):
    """Draw source query, accepted targets and prediction for the first n queries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, len(dataset))
    if n == 0:
        return None
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n), squeeze=False)
    for i in range(n):
        s = dataset[i]
        pred_xy, _ = model.predict(s, cfg)
        axes[i, 0].imshow(s["src_img"])
        axes[i, 0].scatter([s["src_xy"][0]], [s["src_xy"][1]], c="red", s=60)
        axes[i, 0].set_title(f"{s['query_id']}: source query")
        axes[i, 1].imshow(s["trg_img"])
        if s["trg_xys"] is not None:
            axes[i, 1].scatter(s["trg_xys"][:, 0], s["trg_xys"][:, 1], c="lime", s=60, label="accepted")
        axes[i, 1].scatter([pred_xy[0]], [pred_xy[1]], c="red", marker="x", s=100, label="prediction")
        axes[i, 1].legend(loc="lower right")
        axes[i, 1].set_title("target")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    return out_png
