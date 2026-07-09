---
name: lrp-tune
description: Run Optuna hyperparameter tuning for the LightGBM gradient-boosting models in this repo (language-reading-predictors). Use when asked to tune, re-tune, or refresh hyperparameters for one GB model, a family, or all of them, and to review/promote the tuned params. Covers scripts/tune_model.py and scripts/tune_models_batch.py, output locations, the review step, and promotion into the model modules.
---

# Tune GB hyperparameters (Optuna)

Step 0 of the workflow: retune the LightGBM hyperparameters, then **review** before promoting. Tuning never mutates the registry — promotion is a manual, reviewable edit.

## Prerequisites

- `conda activate dse-language-reading-predictors` (verify with `dse-check-env environment.yml`).
- All 50 GB models (`lrp-rli-gbg-001…022` gain, `lrp-rli-gbl-001…028` level) use `LGBMPipeline` (target transform `none`) + the MAE objective, so one uniform policy applies.

## The reviewed policy (#169)

Tune with MAE scoring and the MAE objective, `GroupKFold` by `subject_id` at each model's own `cv_splits`, an inner `GroupShuffleSplit` early-stopping slice (the outer val fold is never shown to early stopping), seed 47:

```bash
python scripts/tune_model.py <model_id> --n-trials 150 --scoring mae --lgbm-objective mae --seed 47
```

The mean best iteration across folds becomes the tuned `n_estimators`. Keep this policy uniform unless you have a specific reason to change it (record the reason in a `notes/` note).

## Single model vs batch

- **One model:** the command above. Writes `best_params.json` to `output/tuning/<model_id>/`.
- **All / a family (preferred for full retunes):** `scripts/tune_models_batch.py` — resumable, auditable, runs each model in its own subprocess **sequentially** (each Optuna trial already saturates all cores via LightGBM `n_jobs=-1`; concurrency oversubscribes and is slower).

```bash
python scripts/tune_models_batch.py --dry-run                    # list planned actions
python scripts/tune_models_batch.py --family all                 # gain|level|core|exploratory|all
python scripts/tune_models_batch.py --models lrp-rli-gbg-012 lrp-rli-gbl-012
python scripts/tune_models_batch.py --force                      # re-tune models already complete
```

A model whose `best_params.json` matches the requested policy is skipped unless `--force`. The batch continues past failures and lists them at the end.

## Outputs

- `output/tuning/<model_id>/best_params.json` — tuned params + CV metrics.
- `output/tuning/retune169_manifest.json` — per-model command, git commit, wall-clock, status, headline CV metric (rewritten after every model, so a killed run resumes cleanly).
- `output/tuning/_logs/<model_id>.log` — per-model tuning log.
- Build a review table (`output/tuning/review_*.csv`): old-vs-new CV MAE ± fold-std, `n_estimators`, boundary/pathology flags, verdict.

## Review before promoting

Small grouped folds mean TPE usually re-lands in the same region — **the value is provenance and parsimony, not predictive gain.** Judge each tune:

- **Within fold-noise?** Accept if |ΔMAE| < ~0.25 × fold-std (fold std ≈ 2–7 MAE units dwarfs the mean differences). Typical verdict is `accept-neutral`.
- **Pathology scan:** flag `n_estimators` ceiling hits (early stopping should engage well below `--max-n-estimators`); the `reg_alpha`/`reg_lambda` 1e-3 log-floor is a benign search-floor artefact (models prefer near-zero explicit L1/L2), not a per-model problem. A collapse to `n_estimators ≈ 3` (e.g. `lrp-rli-gbg-017` nonword-repetition gain) is a near-noise **outcome**, not a tuning defect — flag, do not rerun.

## Promotion (manual)

Only after review: copy the tuned values into `_LGBM_MAE_PARAMS` in each `models/lrp_rli_gbg_*.py` / `models/lrp_rli_gbl_*.py` module. **Preserve each module's existing key schema** (some carry `random_state`, some don't) — edit values only, to keep the diff to parameter values. Remove any stale `retune-pending`/borrowed prose. Guard retired borrowing with `tests/test_borrowed_params.py`.

## After promoting

Validate, then hand off to the GB reporting fit (see the `lrp-fit-gb` skill):

```bash
pytest tests/test_models.py tests/test_borrowed_params.py
python scripts/fit_model.py all --config dev        # smoke test
ruff check src/ && npm run format:check && npm run spellcheck
```

Record the retune (policy, wall-clock, verdicts, exceptions) in a dated `notes/` note.
