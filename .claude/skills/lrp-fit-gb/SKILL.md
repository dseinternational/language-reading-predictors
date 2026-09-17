---
name: lrp-fit-gb
description: Fit, check and render LightGBM models for gains and levels. Use the project methods for predictive interpretation and the runbook for publication.
---

> [!NOTE]
> Clarity and currency edits by a LLM-based AI tool (Codex/GPT-6).

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

## Read the outputs

Read `config.json`, `metrics.json`, `predictor_ranking.csv`, `cluster_ranking.csv` and the SHAP beeswarm together.

- `cv_pooled_r2` uses held-out predictions pooled across child-grouped folds. Hyperparameters were selected on the same grouped folds, so this is internal performance after tuning, not independent validation.
- Permutation importance measures the change in held-out RMSE after predictor values are shuffled between children as whole blocks. Its scale differs from the MAE tuning objective.
- SHAP values show the direction of a predictor's contribution to the fitted prediction. Importance alone has no direction.
- Same-skill predictors can restate a concurrent level outcome. Gain models include baseline scores, so negative baseline associations can reflect score limits or regression to the mean.

Check these features in the actual fit. Do not copy an earlier run's R², rankings or directions into a new summary. Predictive importance does not identify what would happen if someone changed the predictor. The Bayesian models also need a suitable design and assumptions for causal interpretation.

## Publish and record

The built-in `--upload` option publishes to the public research container and requires working credentials on the current host. A private archive uses a separate destination. Use the [publication runbook](../../../docs/runbooks/full-statistical-model-refit.md) for destination and provenance checks within the user's authorised scope. GB fits have no posterior trace to upload.

Record configuration, fitted and failed models, validation metrics, interpretation limits and output or publication location in a dated note. Label AI-authored prose and run the repository's required checks before a commit or pull request.
