# Decision log

Newest on top. Template:

```
## D-XXX: <title> (YYYY-MM-DD)
Decision: ...
Rejected: <alternative> — <why>, one line each
Files: ...
Rule change: yes/no (which rule)
```

Note: `.claude/AGENTS-SETUP.md` and `.claude/repo_cleanup/SKILL.md` cite D-002 and D-003, but this
file did not exist before D-004, so those entries were never written down.

## D-005: One configuration, no overrides; finish removing the tests; simpler run folders (2026-11-04)
Decision (user request: "remove the test sections, delete the possibility of having experiments, delete unused empty folders, config in the output folder"):
- `tests/` and `docs/tests.md` are actually deleted now (D-004 decided it but they were still there), plus `.pytest_cache/`.
- `--set` is gone from `run.py` and `scripts/download_weights.py`: the only configuration is `BASE` in `src/config.py`.
  `parse_overrides`/`check_keys` removed. Overrides D-004's "Remove `--set` too" rejection.
- Decision gate (`recommend`, `decide_target`, `decide_sanity_floor`) removed: it suggested the next experiment.
- Modes are `train | evaluate | submit`. `predict` merged into `submit` (which already wrote `predictions.csv`).
  `--split` (evaluate always uses val) and `--checkpoint` (use `--run`) removed.
- Run folder: `config.json` + results only. Training writes per-epoch validation to `history.json` (updated every
  epoch) instead of `val_ep<N>/` folders.
- `splits_dir` is now a runtime path (`/kaggle/working/splits` on Kaggle, `SC_SPLITS_DIR` to override), so the Kaggle
  notebook no longer needs `--set`.
- Stray files removed: `__pycache__/`, `*:Zone.Identifier` in `.claude/`. No empty folders were left (`splits/` is kept:
  the split file is written there). `.env.example` added (the docs referred to it).
Rejected:
- Remove W&B and training (head/finetune) too — user chose to remove the overrides only.
Files: run.py, src/config.py, src/train.py, src/evaluate.py, src/predict.py, src/diagnostics.py, scripts/download_weights.py,
colab/run.ipynb, notebooks/kaggle_run.ipynb, environment.yml, README.md, RUN.md, docs/run_explanation.md,
docs/REORGANIZATION.md, data/README.md, .claude/agents/*.md, .claude/AGENTS-SETUP.md, .claude/repo_cleanup/SKILL.md.
Rule change: yes. A variant is an edit of `BASE` plus a new run; there is no experiment log or override flag.
`--run`/`--resume` still reopen old run folders (stale keys in their `config.json` are ignored).

## D-004: Remove the test suite, named experiments and sweeps; one flat run folder (2026-11-04)
Decision (user request):
- Delete `tests/`, `docs/tests.md` and `pytest` from `environment.yml`. Overrides D-003 ("tests are kept by default").
- One configuration: `BASE` in `src/config.py`. `EXPERIMENTS`, `cfg["exp"]` and `--exp` are gone; a variant is
  `--set key=value` (e.g. `--set head=linear train_mode=finetune`), recorded in the run's `config.json`.
- Lang sweeps removed: `--sweep`, `sweep_matching`, `sweep_features`, `sweep_matching_grid`, `sweep_features_grid`.
- Run folder: `<output_dir>/<UTC time>_<mode>/`, `config.json` always written first; evaluation files go directly in
  the run folder (no `eval_<split>/`), `checkpoints/` is created only when training, per-epoch validation in `val_ep<N>/`.
Rejected:
- Keep sweeps — user chose the simplest procedure.
- Remove `--set` too — editing `BASE` for every variant is slower and `config.json` already records overrides.
- Keep `<stamp>_<exp>_<mode>` naming — no `exp` left to put in it.
Files: run.py, src/config.py, src/evaluate.py, src/utils/run.py, src/train.py, src/diagnostics.py,
scripts/download_weights.py, colab/run.ipynb, notebooks/kaggle_run.ipynb, environment.yml, README.md, RUN.md,
docs/run_explanation.md, docs/REORGANIZATION.md, .claude/agents/{ceo,code-reviewer,code-explorer}.md, .claude/AGENTS-SETUP.md.
Rule change: yes. "Done" no longer means "fast/slow tests pass"; it means the code compiles and
`python run.py --mode submit --dry-run` runs (plus val PCK for method changes). New variants are `--set` overrides,
not `EXPERIMENTS` entries. Old run folders still reopen with `--run`/`--resume` (stale `exp`/sweep keys in their
`config.json` are ignored).
