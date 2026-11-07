"""Plot of the worst validation predictions (with the CLS similarity of each pair) and the evaluation report on W&B."""

import numpy as np

# Evaluation summary keys kept out of W&B. The error-type *_pct values are also left out: the
# error-type bar plot shows them.
NOT_LOGGED = {"score", "sec_per_query", "n_queries", "n_pairs"}


def _thumbnail(img, max_px):
    """Image downscaled so its longest side is max_px, and the factor applied to its points."""
    factor = min(1.0, max_px / max(img.size))
    if factor < 1.0:
        img = img.resize((round(img.size[0] * factor), round(img.size[1] * factor)))
    return np.asarray(img), factor


def plot_worst_predictions(cfg, model, dataset, frame):
    """The n_visual_checks largest errors (d/L), one row each with four panels:
    source + query | source CLS map | target + accepted targets and prediction | target CLS map.

    CLS map: cosine similarity between the image's DINOv2 CLS token and each patch token (current
    weights), drawn on the patch grid and stretched to the image, so the points line up. The row
    title also gives CLS cos(src, trg), the global similarity of the pair. Predictions come from
    `frame` (PCKMeter.frame()). Returns the figure, or None when nothing can be drawn.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = {qid: i for i, qid in enumerate(dataset.df[cfg["csv_columns"]["query_id"]].astype(str))}
    rows = frame.sort_values("dist", ascending=False)
    rows = rows[rows["query_id"].isin(index)].head(cfg["n_visual_checks"])
    if rows.empty:
        return None
    model.eval()
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.3 * len(rows)), dpi=80, squeeze=False)
    for ax_row, row in zip(axes, rows.itertuples()):
        s = dataset[index[row.query_id]]
        src_cls, src_map = model.cls_embedding(s["src_img"])
        trg_cls, trg_map = model.cls_embedding(s["trg_img"])
        src, f_src = _thumbnail(s["src_img"], cfg["visual_thumb_px"])
        trg, f_trg = _thumbnail(s["trg_img"], cfg["visual_thumb_px"])
        query = np.asarray(s["src_xy"], dtype=np.float64) * f_src
        pred = np.array([row.pred_x, row.pred_y]) * f_trg
        accepted = np.asarray(s["trg_xys"], dtype=np.float64).reshape(-1, 2) * f_trg
        nearest = accepted[np.linalg.norm(accepted - pred, axis=1).argmin()]

        for ax, img, cls_map in ((ax_row[0:2], src, src_map), (ax_row[2:4], trg, trg_map)):
            ax[0].imshow(img)
            h, w = img.shape[:2]
            ax[1].imshow(cls_map.float().cpu().numpy(), cmap="viridis", interpolation="nearest",
                         extent=(0, w, h, 0))
        for ax in ax_row[0:2]:
            ax.scatter([query[0]], [query[1]], c="red", s=50, edgecolors="white")
        for ax in ax_row[2:4]:
            ax.plot([pred[0], nearest[0]], [pred[1], nearest[1]], "w--", lw=1)
            ax.scatter(accepted[:, 0], accepted[:, 1], c="lime", s=50, edgecolors="black", label="accepted")
            ax.scatter([pred[0]], [pred[1]], c="red", marker="x", s=80, label="prediction")
        ax_row[2].legend(loc="lower right", fontsize=7)

        category = "" if row.category == "all" else f" | {row.category}"
        ax_row[0].set_title(f"{row.query_id}{category}: source", fontsize=9)
        ax_row[1].set_title(f"source CLS map | CLS cos(src, trg) = {float(src_cls @ trg_cls):.2f}", fontsize=9)
        ax_row[2].set_title(f"target: d/L = {row.dist:.2f} | {row.error_type}", fontsize=9)
        ax_row[3].set_title("target CLS map", fontsize=9)
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    return fig


def log_evaluation(wandb_run, summary, meter, prefix="val", step=None, figure=None, sweep=None):
    """Log one evaluation to W&B: scalars, group table, error types, worst predictions, sweep table."""
    if wandb_run is None:
        return
    import matplotlib.pyplot as plt
    import wandb

    data = {f"{prefix}/{k}": v for k, v in summary.items()
            if isinstance(v, (int, float)) and k not in NOT_LOGGED and not k.endswith("_pct")}
    data[f"{prefix}/per_group"] = wandb.Table(dataframe=meter.per_group().fillna(0))
    errors = meter.error_types().reset_index(names="error_type")
    data[f"{prefix}/error_types"] = wandb.plot.bar(wandb.Table(dataframe=errors), "error_type", "n_queries",
                                                    title="Error types")
    data[f"{prefix}/dist_hist"] = wandb.Histogram(np.clip(meter.frame()["dist"].to_numpy(), 0, 1))
    if figure is not None:
        figure.canvas.draw()  # JPEG: photos are ~10x smaller than the default PNG
        pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        data[f"{prefix}/worst_predictions"] = wandb.Image(pixels, file_type="jpg")
        plt.close(figure)
    if sweep is not None and len(sweep):
        data[f"{prefix}/sweep"] = wandb.Table(dataframe=sweep)
    wandb_run.log(data, step=step)
