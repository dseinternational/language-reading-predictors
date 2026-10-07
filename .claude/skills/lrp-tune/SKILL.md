---
name: lrp-tune
description: Run Optuna hyperparameter tuning for the LightGBM gradient-boosting models in this repo (language-reading-predictors). Use when asked to tune, re-tune, or refresh hyperparameters for one GB model, a family, or all of them, and to review/promote the tuned params. Covers scripts/tune_model.py and scripts/tune_models_batch.py, output locations, the review step, and promotion into the model modules.
---

> [!NOTE]
> Clarity and currency edits by a LLM-based AI tool (Codex/GPT-6).

# Tune GB hyperparameters (Optuna)

Tune the LightGBM parameters, compare the results and review them before copying values into the registry. Tuning does not edit model declarations.

## Prerequisites

- `uv sync --locked`, then either `source .venv/bin/activate` or prefix each command with `uv run`.
- The registered gain and level models use `LGBMPipeline` (target transform `none`) + the Huber objective, so one uniform policy applies.

## The reviewed policy (Huber, 2026-09-22)

Tune with RMSE scoring and the Huber objective. Use `GroupKFold` by `subject_id`, each model's `cv_splits` and seed 47. Early stopping uses an inner `GroupShuffleSplit`; it does not use the outer validation fold. The Huber threshold (`alpha`) is derived per model by `--alpha-rule robust-mad`: 1.345 × 1.4826 × the median absolute deviation (MAD) from the tuned target's median, falling back to 1.345 × the mean absolute deviation from the median when the MAD is zero (`models/objective.py`). It is recorded in `best_params.json` under `alpha_derivation` and promoted with the other parameters:

```bash
uv run python scripts/tune_model.py <model_id> --n-trials 150 --scoring rmse --lgbm-objective huber --alpha-rule robust-mad --seed 47
```

These are the script defaults. The policy replaced the #169 MAE policy after the [objective-sensitivity check](../../../notes/202609221700-gb-objective-sensitivity.md). The mean best iteration across folds becomes the tuned `n_estimators`. Keep this policy uniform unless you have a specific reason to change it (record the reason in a `notes/` note).

## Single model or batch

- **One model:** the command above. Writes `best_params.json` to `output/tuning/<model_id>/`.
- **All models or a family.** `scripts/tune_models_batch.py` runs each model in a separate process, in sequence, and records enough information to resume. Each trial uses LightGBM `n_jobs=-1`, so concurrent tuning can compete for the same cores.

```bash
uv run python scripts/tune_models_batch.py --dry-run                    # list planned actions
uv run python scripts/tune_models_batch.py --family all                 # gain|level|core|exploratory|all
uv run python scripts/tune_models_batch.py --models lrp-rli-gbg-012 lrp-rli-gbl-012
uv run python scripts/tune_models_batch.py --force                      # re-tune models already complete
```

A model whose `best_params.json` matches the requested policy is skipped unless `--force`. The batch continues past failures and lists them at the end.

## Outputs

- `output/tuning/<model_id>/best_params.json` records the tuned parameters and cross-validation metrics.
- `output/tuning/retune169_manifest.json` records each model's command, Git commit, elapsed time, status and headline metric. It is rewritten after each model.
- `output/tuning/_logs/<model_id>.log` records the model's tuning log.
- Build a review table (`output/tuning/review_*.csv`): old and new cross-validation RMSE and variation across folds, `n_estimators`, boundary/pathology flags, verdict.

## Review before promoting

Compare the old and new settings using the same grouped evaluation and record mean error, variation across folds and model complexity. Fold-to-fold variation is descriptive; it is not a confidence interval or a formal acceptance threshold. Hyperparameter selection on these folds makes the comparison internal to the tuning process.

Check whether the search reached the iteration limit or a parameter boundary. Explain any extreme setting before applying it. A model with very few trees may have little predictive signal; changing the search until it finds more trees does not establish better prediction.

## Promotion (manual)

Only after review: copy the tuned values (including `alpha`) into `_LGBM_HUBER_PARAMS` in each `models/lrp_rli_gbg_*.py` / `models/lrp_rli_gbl_*.py` module. **Preserve each module's existing key schema** (some carry `random_state`, some don't). Edit values only. Remove any stale `retune-pending`/borrowed prose. Guard retired borrowing with `tests/test_borrowed_params.py`.

## After promoting

Validate, then hand off to the GB reporting fit (see the `lrp-fit-gb` skill):

```bash
uv run pytest tests/test_models.py tests/test_borrowed_params.py
uv run python scripts/fit_model.py all --config dev        # smoke test
uv run ruff check src/ && npm run format:check && npm run spellcheck
```

Record the retune (policy, wall-clock, verdicts, exceptions) in a dated `notes/` note.
