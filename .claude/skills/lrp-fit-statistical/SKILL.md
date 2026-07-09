---
name: lrp-fit-statistical
description: Fit the Bayesian (PyMC) statistical models in this repo (language-reading-predictors) — Step 2 interactions/causal estimation. Use when asked to fit/refit statistical models, render reports, run the cross-model comparison, upload/publish artefacts, or summarise Bayesian findings. Covers scripts/fit_statistical_model.py (dev/test/reporting, all), the convergence gate, both blob-upload paths incl. the public-publish credential wrapper, and the per-family result CSVs.
---

# Fit statistical (Bayesian) models

Step 2 of the methodology: PyMC models for interactions and DAG-supported causal estimation, with interpretable estimands and quantified uncertainty. **89 models across 16 families** (itt, gain_factors, aligned, did, level_factors, mechanism, dose_response, joint, growth, horseshoe, mediation, lcsm, mediation_multi, corr_factor, adjusted, historical_growth). Only `τ` (and DiD `δ`, gain-factor on-intervention marginal) is causal; everything else is an adjusted association.

## Prerequisites

`conda activate dse-language-reading-predictors`. Each family has a factory in `statistical_models/factories.py` and a pipeline in `pipeline.py`. Models are auto-discovered — a new `lrp_rli_*` / `lrp_rlm_*` module with a top-level `fit()` registers itself.

## Fit

```bash
python scripts/fit_statistical_model.py <model_id> --config dev            # fast smoke
python scripts/fit_statistical_model.py <model_id> --config reporting --render
python scripts/fit_statistical_model.py all --config reporting --render    # all 89 + Quarto reports
python scripts/fit_statistical_model.py <model_id> --config reporting --target-accept 0.97
```

- Sampling presets: `dev` = 500 draws × 500 tune × 2 chains; `test` = 2000×2000×4; **`reporting` = 6000 draws × 6000 tune × 6 chains, target_accept 0.95**.
- `all` fits every registered model sequentially (nutpie fast — a typical ITT reporting fit is ~40 s; growth/mediation/HSGP/LCSM are slower). Full reporting sweep of 89 + renders ≈ ~1.5–2 h on 16 cores. Run backgrounded with a shell timer + completion marker.
- Model ids accept canonical (`lrp-rli-itt-007`) or legacy (`lrpitt07`) forms.
- Failures and render failures are collected; the script exits non-zero if any fit OR render failed OR the upload step raised — **check the exit code, not just the "N fitted, 0 failed" line** (an upload 403 makes it exit 1 even when all fits/renders succeeded).
- `--target-accept` raises the NUTS acceptance to chase away divergences on hard geometries.

## Cross-model comparison

After the full fit:

```bash
python scripts/compare_statistical_models.py --config reporting
```

Writes to `output/statistical_models/comparison/`: `itt_vs_joint_tau.csv` (single-outcome τ vs joint τ_k consistency), `tau_forest.png`, `mechanism_forest.{png,csv}`, and nested PSIS-LOO tables (mechanism / phonics-route / age-moderation / dose / did-dose).

## Outputs — `output/statistical_models/models/<model_id>-<config>/`

`config.json`, `diagnostics_summary.json` (the pass/fail gate), `trace.nc` (large — excluded from upload by default), `priors_table.csv`, diagnostic plots, `index.html`/`index.qmd`, and family-specific result CSVs (below).

## Convergence gate — check BEFORE interpreting

`diagnostics_summary.json` has `passed` and `checks` = {rhat, ess, divergences, bfmi}. Thresholds: **r̂ ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.30, and zero divergences.** The `divergences` check fails on _any_ divergence, so a fit can be flagged while r̂ = 1.00 and ESS is huge. Triage:

- **Divergence-only flags at ≤ ~0.5 % of total draws** (GP/HSGP mechanism surfaces, dose slopes) are within the METHODS ≤ 1 % guidance — usable, note them.
- **A genuine concern** looks like `mm-001` (corr_factor): ~1 % divergences **plus** sub-threshold BFMI — a latent-factor funnel. Hold its structural coefficients pending a non-centred reparameterisation or higher `--target-accept`; its correlations are fine.

## Upload

Traces (`.nc`) are excluded by default (`--include-traces` to include; they're ~16 GB across 89 fits).

**A. Public research site (`--upload`).** Same mechanism and same credential gotcha as GB (see `lrp-fit-gb`): targets the public `dseresearch` container (`$DSERESEARCH_BLOB_CONTAINER_URL`) → `projects/language-reading-predictors/output/<run_id>/<model>-<config>/…`, anonymously readable. The VM managed identity has **no write role** there, and `DefaultAzureCredential` prefers the MI, so the built-in flag fails with 403. **Public + preliminary — confirm scope with the user first.** The reliable way to publish today (Frank's `az login` has the role) is a small wrapper that reuses the same helper with an explicit `AzureCliCredential`, over the already-fitted dirs (no re-fit):

```python
import glob, os, uuid
from azure.identity import AzureCliCredential
from dse_research_utils.storage.azure import upload_directory_to_blob_storage
credential, run_id = AzureCliCredential(), str(uuid.uuid7())   # one run_id for the whole batch
for d in sorted(glob.glob("output/statistical_models/models/*-reporting")):
    res = upload_directory_to_blob_storage(d, os.path.basename(d),
              project="language-reading-predictors", include_traces=False,
              run_id=run_id, credential=credential)
    print(res.report_url)
```

Run it with `unset AZURE_CLIENT_ID` in the process env (so nothing re-selects the MI). Verifies public: `curl -s -o /dev/null -w "%{http_code} %{content_type}\n" <report_url>` → `200 text/html`. To grant the built-in `--upload` first-class: give the runner identity **Storage Blob Data Contributor** on the `dseresearch` account, then `--upload` works unchanged.

**B. Private durable archive (azcopy + managed identity — works today).** Writes to `$DSE_RESEARCH_BLOB_ENDPOINT` (private) `outputs` container. Exclude traces to match the default:

```bash
export AZCOPY_AUTO_LOGIN_TYPE=MSI AZCOPY_MSI_CLIENT_ID="$AZURE_CLIENT_ID"
azcopy copy "output/statistical_models/models" \
  "$DSE_RESEARCH_BLOB_ENDPOINT/outputs/language-reading-predictors/output/statistical_models/" \
  --recursive --exclude-pattern="*.nc"
azcopy copy "output/statistical_models/comparison" \
  "$DSE_RESEARCH_BLOB_ENDPOINT/outputs/language-reading-predictors/output/statistical_models/" --recursive
```

Confirm `Final Job Status: Completed`, `Failed: 0`. **Always mask account/host and truncate `$AZURE_CLIENT_ID` in any displayed output.**

## Summarise findings

Report the posterior — mean + 95 % credible interval + tail probability, no p-values. Positive τ = intervention helps. The headline estimand per family:

| Family                      | File                                                       | Causal term / what to read                                                                                                                                                                                            |
| --------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| itt                         | `tau_summary.csv`                                          | `tau_prob_median` + `tau_prob_lo/hi` + `prob_tau_pos` (risk-difference scale). Floored P/N: primary estimand is the off-floor risk difference (`offfloor_movers.csv`, `rope_summary.csv`), the graded τ is secondary. |
| joint                       | `tau_summary.csv`, `tau_contrast_matrix.csv`               | τ_k per outcome; generalisation contrasts (taught vs not-taught).                                                                                                                                                     |
| did                         | `did_summary.csv`                                          | `delta_*` = within-person DiD causal effect (`beta_period_*` for the dose variants).                                                                                                                                  |
| gain_factors                | `treatment_marginal.csv`                                   | `trt_prob_*` = on-intervention risk difference (the **only** causal term; every covariate is an adjusted association).                                                                                                |
| level_factors               | `factor_summary.csv`                                       | only `b_grp_time[1]` (t2 contrast) is a clean randomised term; later timepoints are post-crossover associations.                                                                                                      |
| mechanism                   | `mechanism_curve.csv` / comparison `mechanism_forest.csv`  | marginal slope — **association**, never "X drives Y".                                                                                                                                                                 |
| mediation / mediation_multi | `mediation_summary.csv`                                    | `total`/`NDE`/`NIE`/`proportion_mediated` (g-formula); NIE per mediator for the two-mediator model.                                                                                                                   |
| dose_response               | `dose_slope_summary.csv`                                   | dose slope (dose is a partial collider → sensitivity view).                                                                                                                                                           |
| aligned                     | `factor_summary.csv`, `cohort_marginal.csv`                | cohort contrast is **not** randomised — flag as association.                                                                                                                                                          |
| lcsm                        | `coupling_summary.csv`                                     | cross-lagged couplings (associations).                                                                                                                                                                                |
| growth                      | `growth_association_summary.csv`                           | between-child gamma associations.                                                                                                                                                                                     |
| horseshoe                   | `predictor_ranking.csv`                                    | `p_abs_gt_delta` selection — cross-check of the GB ranking.                                                                                                                                                           |
| corr_factor                 | `factor_correlation_summary.csv`, `structural_summary.csv` | domain correlations (robust); structural leg cautious (see gate).                                                                                                                                                     |
| adjusted                    | `predictor_associations.csv`                               | adjusted vs bivariate between-child associations.                                                                                                                                                                     |

**The coherent story to expect** (reading/phonics intervention): credible positive ITT on letter-sound knowledge (L), phoneme blending (B), word reading (W), taught expressive vocab (TE); null on broad standardised vocabulary (R/E). Robust to ability/SES adjustment and replicated by DiD + gain-factor ANCOVA. The word-reading gain is **mediated by letter-sound knowledge** (NIE via L credible, via E ≈ 0). Record the run (config, N fitted/failed, gate pass count + divergence caveats, key τ, blob/publish location) in a dated `notes/` note with the AI-authorship label. Pre-commit: `ruff check src/`, `npm run format:check`, `npm run spellcheck`.
