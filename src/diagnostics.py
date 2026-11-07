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
    """The n_visual_checks largest errors (d/L): source query, accepted targets and prediction.

    Each row title also shows the cosine similarity of the two images' DINOv2 CLS tokens (global
    similarity of the pair, current weights). Predictions come from `frame` (PCKMeter.frame()).
    Returns the figure, or None when nothing can be drawn.
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
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 3.6 * len(rows)), dpi=80, squeeze=False)
    for ax_row, row in zip(axes, rows.itertuples()):
        s = dataset[index[row.query_id]]
        cls_cos = float(model.cls_embedding(s["src_img"]) @ model.cls_embedding(s["trg_img"]))
        src, f_src = _thumbnail(s["src_img"], cfg["visual_thumb_px"])
        trg, f_trg = _thumbnail(s["trg_img"], cfg["visual_thumb_px"])
        pred = np.array([row.pred_x, row.pred_y]) * f_trg
        accepted = np.asarray(s["trg_xys"], dtype=np.float64).reshape(-1, 2) * f_trg
        nearest = accepted[np.linalg.norm(accepted - pred, axis=1).argmin()]

        ax_row[0].imshow(src)
        ax_row[0].scatter([s["src_xy"][0] * f_src], [s["src_xy"][1] * f_src], c="red", s=50)
        category = "" if row.category == "all" else f" | {row.category}"
        ax_row[0].set_title(f"{row.query_id}{category}\nCLS cos(src, trg) = {cls_cos:.2f}", fontsize=9)
        ax_row[1].imshow(trg)
        ax_row[1].plot([pred[0], nearest[0]], [pred[1], nearest[1]], "w--", lw=1)
        ax_row[1].scatter(accepted[:, 0], accepted[:, 1], c="lime", s=50, label="accepted")
        ax_row[1].scatter([pred[0]], [pred[1]], c="red", marker="x", s=80, label="prediction")
        ax_row[1].legend(loc="lower right", fontsize=7)
        ax_row[1].set_title(f"d/L = {row.dist:.2f} | {row.error_type}", fontsize=9)
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
