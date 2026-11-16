"""Paired comparison of validation results: the decision tool for every experiment.

One run's 95% interval is about +-4.5 points on 453 queries, far too wide for a 1-3 point effect.
Here both sides are scored on the SAME queries and the validation PAIRS are bootstrapped
(queries of a pair are correlated), so the interval is for the difference A - B.

    python scripts/compare_runs.py a_s0,a_s1 b_s0,b_s1
    python scripts/compare_runs.py a_s0,a_s1 b_s0,b_s1 --resamples 5000
    python scripts/compare_runs.py frozen_dino concat_sd                # fusion and training keys are not checked
    python scripts/compare_runs.py --self-test

Each argument is a comma-separated list of run ids (the seeds of one setting). Every run needs
outputs/<run_id>/per_query.csv, written by `python run.py --mode evaluate --run <run_id>`
(training does not write it). Metrics are averaged over the seeds of a side first, per query.

Sign convention: the difference is A - B in percentage points. For PCK, P(A better) is the share of
resamples with A - B > 0. For error rates (lower is better) it is the share with A - B < 0.
Resamples that tie count one half, so two identical runs give exactly 0 and P = 0.5.
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_config  # noqa: E402
from src.utils.run import load_run_config  # noqa: E402

# Settings that must be equal in all runs unless listed in --allow.
MATCHING_KEYS = ["soft_window", "soft_temperature", "flip_test", "image_size", "matcher",
                 "feat_upsample", "pck_alpha", "pck_strict"]
ERROR_TYPES = ["wrong_keypoint", "near_miss", "other_miss", "far_miss"]


class CompareError(Exception):
    pass


def load_side(run_ids, output_dir):
    """Return [(run_id, cfg, per_query frame sorted by query_id)] for one side."""
    runs = []
    for run_id in run_ids:
        folder = Path(output_dir) / run_id
        if not folder.is_dir():
            raise CompareError(f"Run folder not found: {folder}")
        csv_path = folder / "per_query.csv"
        if not csv_path.is_file():
            raise CompareError(
                f"{csv_path} is missing. Training does not write per_query.csv (it validates with save=False); "
                f"run `python run.py --mode evaluate --run {run_id}` first.")
        cfg = load_run_config(run_id, output_dir)
        frame = pd.read_csv(csv_path).sort_values("query_id").reset_index(drop=True)
        if frame["query_id"].duplicated().any():
            raise CompareError(f"{csv_path} has duplicated query_id values.")
        runs.append((run_id, cfg, frame))
    return runs


def check_runs(side_a, side_b, allow):
    """Raise CompareError if query ids differ or matching settings differ. Returns the settings table."""
    everything = side_a + side_b
    ids = everything[0][2]["query_id"].tolist()
    for run_id, _, frame in everything[1:]:
        if frame["query_id"].tolist() != ids:
            raise CompareError(f"Run {run_id} has a different set of query_id than {everything[0][0]}.")
    unknown = set(allow) - set(MATCHING_KEYS)
    if unknown:
        raise CompareError(f"--allow keys not in the checked list {MATCHING_KEYS}: {sorted(unknown)}")
    table = {}
    for key in MATCHING_KEYS:
        values = [(run_id, cfg.get(key)) for run_id, cfg, _ in everything]
        groups = {"A": {json.dumps(cfg.get(key)) for _, cfg, _ in side_a},
                  "B": {json.dumps(cfg.get(key)) for _, cfg, _ in side_b}}
        for name, found in groups.items():
            if len(found) > 1:  # seeds of one setting must agree even for allowed keys
                raise CompareError(f"Runs of side {name} differ in `{key}`: {sorted(found)}")
        if groups["A"] != groups["B"] and key not in allow:
            raise CompareError(f"`{key}` differs between A and B ({values}). "
                               f"Pass --allow {key} if that is what is being tested.")
        table[key] = {"A": json.loads(groups["A"].pop()), "B": json.loads(groups["B"].pop()),
                      "allowed_to_differ": key in allow}
    return table


def side_metrics(side, alpha):
    """Per-query values (mean over seeds) for every metric: {name: (numerator, denominator, lower_is_better)}.

    A query contributes numerator[q] to the metric and denominator[q] (1, or 0 when it is outside the group).
    """
    frames = [frame for _, _, frame in side]
    n = len(frames[0])

    def mean_of(fn):
        return np.mean([np.asarray(fn(f), dtype=float) for f in frames], axis=0)

    ones = np.ones(n)
    metrics = {}
    for a in (0.05, alpha, 0.15):
        column = f"pck@{a}"
        label = f"PCK@{a:.2f}"
        metrics[label] = (mean_of(lambda f: f[column]), ones, False)
    main = f"pck@{alpha}"
    for group in ("single", "multi"):
        # the group is a property of the query, identical in every seed
        in_group = (frames[0]["target_group"] == group).to_numpy(dtype=float)
        metrics[f"PCK@{alpha:.2f} {group}"] = (mean_of(lambda f: f[main]) * in_group, in_group, False)
    for kind in ERROR_TYPES:
        metrics[f"{kind} rate"] = (mean_of(lambda f: f["error_type"] == kind), ones, True)
    return metrics


def paired_bootstrap(a_metrics, b_metrics, pair_ids, resamples, seed):
    pairs, inverse = np.unique(pair_ids, return_inverse=True)
    n_pairs = len(pairs)

    def per_pair(values):
        return np.bincount(inverse, weights=values, minlength=n_pairs)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_pairs, size=(resamples, n_pairs))
    results = {}
    for name, (a_num, a_den, lower_better) in a_metrics.items():
        b_num, b_den, _ = b_metrics[name]
        stats = {}
        for side, num, den in (("A", a_num, a_den), ("B", b_num, b_den)):
            stats[side] = (per_pair(num), per_pair(den))
        with np.errstate(invalid="ignore", divide="ignore"):
            full = {s: 100 * stats[s][0].sum() / stats[s][1].sum() for s in stats}
            boot = {s: 100 * stats[s][0][draws].sum(axis=1) / stats[s][1][draws].sum(axis=1) for s in stats}
        diff = boot["A"] - boot["B"]
        diff = diff[np.isfinite(diff)]
        better = (diff < 0) if lower_better else (diff > 0)
        ties = diff == 0
        results[name] = {
            "A_mean": float(full["A"]), "B_mean": float(full["B"]), "diff": float(full["A"] - full["B"]),
            "ci_low": float(np.percentile(diff, 2.5)), "ci_high": float(np.percentile(diff, 97.5)),
            "p_a_better": float((better.sum() + 0.5 * ties.sum()) / len(diff)),
            "lower_is_better": lower_better,
        }
    return results


def compare(a_ids, b_ids, output_dir, resamples=2000, allow=(), write_dir=None):
    side_a = load_side(a_ids, output_dir)
    side_b = load_side(b_ids, output_dir)
    settings = check_runs(side_a, side_b, set(allow))
    alpha = side_a[0][1]["pck_alpha"]
    seed = int(side_a[0][1]["seed"])
    frame = side_a[0][2]
    a_metrics, b_metrics = side_metrics(side_a, alpha), side_metrics(side_b, alpha)
    results = paired_bootstrap(a_metrics, b_metrics, frame["pair_id"].to_numpy(), resamples, seed)
    report = {
        "A": a_ids, "B": b_ids, "n_queries": len(frame), "n_pairs": int(frame["pair_id"].nunique()),
        "resamples": resamples, "bootstrap_seed": seed, "pck_alpha": alpha,
        "sign": "difference = A - B in points; P(A better) uses > 0 for PCK and < 0 for error rates; ties count 0.5",
        "settings_checked": settings, "metrics": results,
    }
    if write_dir is not None:
        folder = Path(write_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "compare.json").write_text(json.dumps(report, indent=2) + "\n")
        report["folder"] = str(folder)
    return report


def print_report(report):
    print(f"A: {', '.join(report['A'])}")
    print(f"B: {', '.join(report['B'])}")
    print(f"{report['n_queries']} queries, {report['n_pairs']} pairs, {report['resamples']} pair resamples "
          f"(seed {report['bootstrap_seed']})")
    checked = ", ".join(f"{k}={v['A']}" + (f"/{v['B']}" if v["A"] != v["B"] else "")
                        for k, v in report["settings_checked"].items())
    print(f"Settings checked: {checked}")
    print("Difference = A - B in points. P(A better): share of resamples with A - B > 0 for PCK, < 0 for "
          "error rates (lower is better); ties count 0.5.\n")
    header = f"{'metric':<24}{'A':>8}{'B':>8}{'A-B':>8}   {'95% interval':<18}{'P(A better)':>12}"
    print(header)
    print("-" * len(header))
    for name, r in report["metrics"].items():
        interval = f"[{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]"
        print(f"{name:<24}{r['A_mean']:>8.1f}{r['B_mean']:>8.1f}{r['diff']:>+8.1f}   {interval:<18}"
              f"{r['p_a_better']:>12.3f}")


def make_fake_run(root, run_id, frame, **settings):
    folder = Path(root) / run_id
    folder.mkdir(parents=True)
    config = {"seed": 0, "soft_window": 9, "soft_temperature": 0.04, "flip_test": False, "image_size": 644,
              "pck_alpha": 0.1, "pck_strict": True, **settings}
    (folder / "config.json").write_text(json.dumps(config))
    frame.to_csv(folder / "per_query.csv", index=False)


def fake_frame(rng, n_pairs=60, per_pair=5):
    n = n_pairs * per_pair
    dist = rng.uniform(0.0, 0.4, size=n)
    frame = pd.DataFrame({
        "query_id": [f"q{i:04d}" for i in range(n)],
        "pair_id": np.repeat([f"p{i:03d}" for i in range(n_pairs)], per_pair),
        "dist": dist,
        "target_group": np.where(np.arange(n) % 3 == 0, "multi", "single"),
    })
    for alpha in (0.05, 0.1, 0.15):
        frame[f"pck@{alpha}"] = (dist < alpha).astype(float)
    frame["error_type"] = np.where(dist < 0.1, "correct", "near_miss")
    return frame


def self_test():
    rng = np.random.default_rng(0)
    failures = []

    def check(name, ok):
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
        if not ok:
            failures.append(name)

    def refused(**kwargs):
        try:
            compare(**kwargs)
        except CompareError:
            return True
        return False

    with tempfile.TemporaryDirectory() as tmp:
        base = fake_frame(rng)
        broken = base.copy()
        # 12 correct queries, one in each of 12 pairs, become wrong: A - B = 12 / 300 = 4.0 points
        correct = np.flatnonzero(base["pck@0.1"] == 1.0)
        pairs = base["pair_id"].to_numpy()
        chosen, seen = [], set()
        for i in correct:
            if pairs[i] not in seen:
                seen.add(pairs[i])
                chosen.append(i)
            if len(chosen) == 12:
                break
        for column in ("pck@0.05", "pck@0.1", "pck@0.15"):
            broken.loc[chosen, column] = 0.0
        broken.loc[chosen, "error_type"] = "wrong_keypoint"
        make_fake_run(tmp, "a", base)
        make_fake_run(tmp, "b", broken)
        make_fake_run(tmp, "a_copy", base)
        make_fake_run(tmp, "c_ids", base.assign(query_id=base["query_id"].str.replace("q", "z", n=1, regex=False)))
        make_fake_run(tmp, "d_window", base, soft_window=5)
        make_fake_run(tmp, "e_size", base, image_size=756)

        report = compare(["a"], ["b"], tmp, resamples=1000, write_dir=Path(tmp) / "out")
        main = report["metrics"]["PCK@0.10"]
        check("difference equals the known 4.0 points", abs(main["diff"] - 4.0) < 1e-9)
        check("interval contains the known difference", main["ci_low"] <= 4.0 <= main["ci_high"])
        check("A better with P > 0.9", main["p_a_better"] > 0.9)
        wrong = report["metrics"]["wrong_keypoint rate"]
        check("wrong_keypoint rate is lower for A by 4.0 and P(A better) > 0.9",
              abs(wrong["diff"] + 4.0) < 1e-9 and wrong["p_a_better"] > 0.9)
        check("compare.json written", (Path(tmp) / "out" / "compare.json").is_file())

        same = compare(["a"], ["a_copy"], tmp, resamples=500)
        check("identical runs give exactly 0 and P = 0.5",
              all(r["diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0 and r["p_a_better"] == 0.5
                  for r in same["metrics"].values()))

        seeds = compare(["a", "a_copy"], ["b", "a"], tmp, resamples=200)
        check("two seeds per side are averaged (4.0 / 2 = 2.0 points)",
              abs(seeds["metrics"]["PCK@0.10"]["diff"] - 2.0) < 1e-9)

        check("mismatched query ids are refused", refused(a_ids=["a"], b_ids=["c_ids"], output_dir=tmp))
        check("mismatched soft_window is refused", refused(a_ids=["a"], b_ids=["d_window"], output_dir=tmp))
        check("mismatched image_size is refused", refused(a_ids=["a"], b_ids=["e_size"], output_dir=tmp))
        try:
            allowed = compare(["a"], ["e_size"], tmp, resamples=100, allow=["image_size"])
            check("--allow image_size lets the comparison run", allowed["settings_checked"]["image_size"]["allowed_to_differ"])
        except CompareError:
            check("--allow image_size lets the comparison run", False)
        check("missing per_query.csv is refused with the evaluate command",
              _message_has(lambda: compare(["a"], ["nope"], tmp), "nope"))
    print("\nAll checks passed." if not failures else f"\n{len(failures)} check(s) FAILED: {failures}")
    return 1 if failures else 0


def _message_has(fn, text):
    try:
        fn()
    except CompareError as err:
        return text in str(err)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a_runs", nargs="?", help="comma-separated run ids of setting A (seeds)")
    parser.add_argument("b_runs", nargs="?", help="comma-separated run ids of setting B (seeds)")
    parser.add_argument("--resamples", type=int, default=2000, help="bootstrap resamples of the pairs")
    parser.add_argument("--allow", default="",
                        help=f"comma-separated settings allowed to differ between A and B, from {MATCHING_KEYS}")
    parser.add_argument("--self-test", action="store_true", help="check the script on fake runs and exit")
    args = parser.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.a_runs or not args.b_runs:
        parser.error("give A_RUNS and B_RUNS (or --self-test)")

    cfg = get_config()
    def split(text):
        return [item.strip() for item in text.split(",") if item.strip()]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    folder = Path(cfg["output_dir"]) / f"{stamp}_compare"
    try:
        report = compare(split(args.a_runs), split(args.b_runs), cfg["output_dir"], args.resamples,
                         split(args.allow), write_dir=folder)
    except CompareError as err:
        print(f"Refused: {err}", file=sys.stderr)
        sys.exit(1)
    print_report(report)
    print(f"\nWritten to {folder}")


if __name__ == "__main__":
    main()
