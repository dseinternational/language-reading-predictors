---
name: lrp-fit-gb
description: Fit, check and render LightGBM models for gains and levels. Use the project methods for predictive interpretation and the runbook for publication.
---

> [!NOTE]
> Clarity and currency edits by a LLM-based AI tool (Codex/GPT-6).
>
> Report-index step added by a LLM-based AI tool (Claude Code/Opus 5).

# Fit gradient-boosting models

LightGBM models rank predictors of gains and concurrent levels. They describe prediction and association. Read [METHODS.md](../../../METHODS.md) for the interpretation rules and [the catalogue](../../../docs/models/README.md) for the model list.

## Fit and render

From the repository root:

```bash
uv sync --locked
uv run python scripts/fit_model.py lrp-rli-gbg-012 --config dev
uv run python scripts/fit_model.py lrp-rli-gbg-012 --config reporting --render
uv run python scripts/fit_model.py all --config reporting --render
```

`all` skips registered variants unless passed `--include-variants`. Development fits omit the extended ranking and SHAP-interaction analyses. Use `reporting` for the full outputs, and inspect the final failure count and process exit status.

Redirect output with `--output-dir` or `DSE_LRP_OUTPUT_DIR`; the command option takes precedence. Results normally go to `output/models/<model_id>/`. The fit records source, data and environment identities and writes `fit_complete=true` only after the analysis and report preparation finish. `metrics.json` alone is not a completion guarantee.

Use `scripts/run_refit_sweep.py --help` for resumable batches. Estimate run time from a comparable recent run on the same host. A full reporting sweep can take much longer than a development fit.

After a sweep, run `uv run python scripts/build_gb_index.py` (with the same `--output-dir`, if any) to write `output/models/index.html`. It links every rendered report, grouped as in the catalogue, with each model's held-out R² and leading predictors. It flags fits that lack a completion record or a clean source commit. It reads stored artefacts only and refits nothing.

## Read the outputs

Read `config.json`, `metrics.json`, `predictor_ranking.csv`, `cluster_ranking.csv` and the SHAP beeswarm together.

- `cv_pooled_r2` uses held-out predictions pooled across child-grouped folds. Hyperparameters were selected on the same grouped folds, so this is internal performance after tuning, not independent validation.
- Permutation importance measures the change in held-out RMSE after predictor values are shuffled between children at the same assessment waves and within identical observed-wave schedules. It measures importance conditional on that schedule. Predictors that cannot change under these shuffles, including `time`, are not assessable by this design. Check the recorded design version and donor support before interpreting saved rankings. The models use a Huber objective and RMSE scoring. Huber predictions need not be conditional means for skewed or floored outcomes; RMSE evaluates the fitted predictions.
- SHAP values show the direction of a predictor's contribution to the fitted prediction. Importance alone has no direction.
- The permutation, SHAP bar and SHAP scatter figures show the ten leading predictors; the CSV tables keep every predictor. Three extra waterfalls explain the observations nearest the 25th, 50th and 75th percentiles of the outcome (`shap_waterfall_observations.csv` names them).
- Same-skill predictors can restate a concurrent level outcome. Gain models include baseline scores, so negative baseline associations can reflect score limits or regression to the mean.

Check these features in the actual fit. Do not copy an earlier run's R², rankings or directions into a new summary. Predictive importance does not identify what would happen if someone changed the predictor. The Bayesian models also need a suitable design and assumptions for causal interpretation.

## Publish and record

The built-in `--upload` option publishes to the public research container and requires working credentials on the current host. A private archive uses a separate destination. Use the [publication runbook](../../../docs/runbooks/full-statistical-model-refit.md) for destination and provenance checks within the user's authorised scope. GB fits have no posterior trace to upload.

Record configuration, fitted and failed models, validation metrics, interpretation limits and output or publication location in a dated note. Label AI-authored prose and run the repository's required checks before a commit or pull request.
