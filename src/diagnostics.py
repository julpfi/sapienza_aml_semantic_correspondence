"""Visual check of the worst validation predictions and the evaluation report on W&B."""

from pathlib import Path

import numpy as np

# Evaluation summary keys kept out of the W&B charts (docs/REVIEW_finetuning.md 2.2). The error-type
# *_pct values are also left out: the error-type bar plot shows them.
SUMMARY_ONLY = ("n_queries", "n_pairs")  # constant: run summary only
NOT_LOGGED = {"score", "sec_per_query", "loss/small", "loss/medium", "loss/large"}


def worst_queries(frame, n):
    """query_ids of the n largest errors (d/L)."""
    return frame.sort_values("dist", ascending=False)["query_id"].head(n).tolist()


def plot_predictions(cfg, model, dataset, query_ids, out_png=None):
    """Draw source query, accepted targets and prediction for the given queries. Returns the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = {qid: i for i, qid in enumerate(dataset.df[cfg["csv_columns"]["query_id"]].astype(str))}
    query_ids = [q for q in query_ids if q in index]
    if not query_ids:
        return None
    fig, axes = plt.subplots(len(query_ids), 2, figsize=(10, 4 * len(query_ids)), squeeze=False)
    for row, qid in enumerate(query_ids):
        s = dataset[index[qid]]
        pred_xy, _ = model.predict(s, cfg)
        dist = np.linalg.norm(s["trg_xys"] - pred_xy, axis=1).min() / s["scale"] if s["trg_xys"] is not None else None
        axes[row, 0].imshow(s["src_img"])
        axes[row, 0].scatter([s["src_xy"][0]], [s["src_xy"][1]], c="red", s=60)
        axes[row, 0].set_title(f"{qid}: source query")
        axes[row, 1].imshow(s["trg_img"])
        if s["trg_xys"] is not None:
            axes[row, 1].scatter(s["trg_xys"][:, 0], s["trg_xys"][:, 1], c="lime", s=60, label="accepted")
        axes[row, 1].scatter([pred_xy[0]], [pred_xy[1]], c="red", marker="x", s=100, label="prediction")
        axes[row, 1].legend(loc="lower right")
        axes[row, 1].set_title("target" + (f"  d/L = {dist:.2f}" if dist is not None else ""))
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=80)
    return fig


def log_evaluation(wandb_run, summary, meter, prefix="val", step=None, figure=None, sweep=None):
    """Log one evaluation to W&B: scalars, group table, error types, worst predictions, sweep table."""
    if wandb_run is None:
        return
    import wandb

    numbers = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
    wandb_run.summary.update({f"{prefix}/{k}": numbers.pop(k) for k in SUMMARY_ONLY if k in numbers})
    data = {f"{prefix}/{k}": v for k, v in numbers.items() if k not in NOT_LOGGED and not k.endswith("_pct")}
    data[f"{prefix}/per_group"] = wandb.Table(dataframe=meter.per_group().fillna(0))
    errors = meter.error_types().reset_index(names="error_type")
    data[f"{prefix}/error_types"] = wandb.plot.bar(wandb.Table(dataframe=errors), "error_type", "n_queries",
                                                    title="Error types")
    data[f"{prefix}/dist_hist"] = wandb.Histogram(np.clip(meter.frame()["dist"].to_numpy(), 0, 1))
    if figure is not None:
        data[f"{prefix}/worst_predictions"] = wandb.Image(figure)
    if sweep is not None and len(sweep):
        data[f"{prefix}/sweep"] = wandb.Table(dataframe=sweep)
    wandb_run.log(data, step=step)
