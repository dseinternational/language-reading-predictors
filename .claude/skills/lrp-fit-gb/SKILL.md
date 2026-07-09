---
name: lrp-fit-gb
description: Fit the LightGBM gradient-boosting models in this repo (language-reading-predictors) — Step 1 exploratory analysis. Use when asked to fit/refit GB models, render their reports, upload artefacts, or summarise GB findings. Covers scripts/fit_model.py (dev/test/reporting, all, variants), output locations, both blob-upload paths, and how to read the SHAP + permutation ranking.
---

# Fit gradient-boosting models

Step 1 of the two-step methodology: exploratory LightGBM + permutation importance + SHAP to learn which predictors matter per outcome. There are **50 final GB models** (22 gain `lrp-rli-gbg-001…022`, 28 level `lrp-rli-gbl-001…028`), 0 variants. The cluster-first predictor ranking is the deliverable.

## Prerequisites

`conda activate dse-language-reading-predictors`. Apply tuned params first if retuning (see `lrp-tune`).

## Fit

```bash
python scripts/fit_model.py <model_id>                       # dev config (fast, default)
python scripts/fit_model.py <model_id> --config test         # moderate
python scripts/fit_model.py <model_id> --config reporting     # full (production)
python scripts/fit_model.py all --config reporting --render   # all 50 final models + Quarto reports
python scripts/fit_model.py all --include-variants --config dev
```

- `all` **skips variants** unless `--include-variants` (the 50 finals have none). It continues past failures and prints a `Summary (N fitted, M failed)` table at the end.
- **`dev` skips** the SHAP-interaction analysis and cluster-first ranking; **`reporting` runs them** (plus a Quarto render per model with `--render`). Always use `--config reporting --render` for a publishable run.
- Model ids accept canonical (`lrp-rli-gbg-012`) or legacy (`LRPGBG12`) forms.
- Redirect heavy output to scratch with `DSE_LRP_OUTPUT_DIR=/mnt/scratch/lrp` or `--output-dir` (takes precedence). The resolved root prints at start and is recorded in `config.json`.

Reporting fit of all 50 is ~minutes (LightGBM is fast). Run it backgrounded with a shell timer + a completion marker; check the log doesn't fail on model #1, then wait for completion.

## Outputs — `output/models/<model_id>/`

`config.json`, `metrics.json`, `cv_scores.csv`, `predictor_ranking.csv`, `cluster_ranking.csv`, `cluster_table.csv`, `permutation_importance.*`, many `shap_*.png/svg`, `index.html`/`index.qmd`.

## Upload

Two distinct paths — pick per intent. Traces don't exist for GB (`--include-traces` is a no-op).

**A. Public research site (`--upload`, the built-in flag).** Publishes to the public `dseresearch` container at `$DSERESEARCH_BLOB_CONTAINER_URL` → `projects/language-reading-predictors/output/<run_id>/<model_id>/…`, anonymously readable (open internet). It authenticates with `DefaultAzureCredential`. **Gotcha:** on this VM the managed identity (`$AZURE_CLIENT_ID`) has **no write role** on that account (403 `AuthorizationPermissionMismatch`), and `DefaultAzureCredential` prefers the MI over an `az login`. To publish you must either (a) grant the runner's identity **Storage Blob Data Contributor** on the `dseresearch` account, or (b) run the upload under a principal that has the role with the MI env cleared (see `lrp-fit-statistical` for the `AzureCliCredential` wrapper pattern). Public publishing is outward-facing and hard to reverse — **confirm scope with the user first**; the data is preliminary.

```bash
python scripts/fit_model.py all --config reporting --render --upload
```

**B. Private durable archive (azcopy + managed identity — works today).** The VM's MI _does_ have write on the private `$DSE_RESEARCH_BLOB_ENDPOINT` account, `outputs` container. Mirror the local layout under `language-reading-predictors/output/models/`:

```bash
export AZCOPY_AUTO_LOGIN_TYPE=MSI AZCOPY_MSI_CLIENT_ID="$AZURE_CLIENT_ID"
azcopy copy "output/models" \
  "$DSE_RESEARCH_BLOB_ENDPOINT/outputs/language-reading-predictors/output/" \
  --recursive
```

Verify `Final Job Status: Completed` and `Failed: 0`. `output/` is git-ignored, so nothing but notes is committed.

## Summarise findings

Read direction + uncertainty, never a bare ranking.

- **Headline metric:** `cv_pooled_r2` from `metrics.json` (not per-fold R²: check `cv_r2_per_fold_reliable` — small grouped folds make per-fold R² unreliable).
- **Ranking:** read `predictor_ranking.csv` / `cluster_ranking.csv` **with** the SHAP beeswarm (`shap_summary.png`) — permutation importance and SHAP disagree, so state the **direction**.
- **Expected split:** level (`gbl`) models are autoregressive → high pooled R² (median ~0.6, up to ~0.98 for near-tautological composites); gain (`gbg`) models are near-noise → low R² (~0.02–0.27, regression-to-the-mean). Rankings are dominated by each outcome's own baseline + `time`/`age`, with `trog`, `aptinfo`, `spphon`/`blending`, `yarclet` recurring as cross-predictors. Gain models carry little predictable signal beyond baseline — the **Step 2 Bayesian models do the causal work** (see `lrp-fit-statistical`).

Record the run (config, N fitted/failed, R² spread, blob location) in a dated `notes/` note. Pre-commit: `ruff check src/`, `npm run format:check`, `npm run spellcheck`.
