"""Command-line entry point.

    python run.py eval      --set data_root=/path/SPair-71k max_pairs=300
    python run.py sweep     --set max_pairs=300            # cheap matching options, features cached
    python run.py featsweep --set max_pairs=300            # layer / img_size / backbone
    python run.py decide    --run outputs/stage1_dinov2_b14 --target 70
    python run.py finetune  --set run_name=stage2_ft2 ft_unfreeze_blocks=2
    python run.py predict   --set ft_checkpoint=outputs/stage2_ft2/best.pth --out submission.csv

`--config file.json` loads a saved Config; `--set k=v` overrides any field.
"""
import argparse
import ast

from soco.config import Config


def parse_sets(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "sweep", "featsweep", "decide", "finetune", "predict"])
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--run")
    ap.add_argument("--baseline")
    ap.add_argument("--target", type=float, default=70.0)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--sample_submission", default="")
    a = ap.parse_args()
    cfg = Config.load(a.config) if a.config else Config()
    cfg = cfg.replace(**parse_sets(a.set))

    if a.cmd == "eval":
        from soco.evaluate import evaluate
        evaluate(cfg)
    elif a.cmd == "sweep":
        from soco.evaluate import sweep_matching
        sweep_matching(cfg, {"matcher": ["argmax", "soft_window"], "feat_upsample": [1, 2],
                             "flip_test": [False, True], "soft_temperature": [0.02, 0.04]})
    elif a.cmd == "featsweep":
        from soco.evaluate import sweep_features
        sweep_features(cfg, {"layer": [-1, -2, -4], "img_size": [448, 518, 644]})
    elif a.cmd == "decide":
        from soco.decide import recommend
        print(recommend(a.run, a.target, baseline_dir=a.baseline, stage=a.stage))
    elif a.cmd == "finetune":
        from soco.finetune import finetune
        finetune(cfg)
    elif a.cmd == "predict":
        from soco.submit import predict_split
        predict_split(cfg, "test", a.out, a.sample_submission)


if __name__ == "__main__":
    main()
