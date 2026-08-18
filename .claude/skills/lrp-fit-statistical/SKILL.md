---
name: lrp-fit-statistical
description: Fit the Bayesian (PyMC) statistical models in this repo (language-reading-predictors) — Step 2 interactions/causal estimation. Use when asked to fit/refit statistical models, render reports, run the cross-model comparison, upload/publish artefacts, or summarise Bayesian findings. Covers scripts/fit_statistical_model.py (dev/test/reporting, all), the convergence gate, both blob-upload paths incl. the public-publish credential wrapper, and the per-family result CSVs.
---

> [!NOTE]
> Available-case modified ITT terminology updated by a LLM-based AI tool (Codex/GPT-5).
>
> Divergent-transition gate guidance, credible-interval standard, model/family counts and sweep figures corrected against `METHODS.md` and the registry by a LLM-based AI tool (Claude Code/Opus 5).

# Fit statistical (Bayesian) models

Step 2 of the methodology: PyMC models for interactions and DAG-supported causal estimation, with interpretable estimands and quantified uncertainty. **220 registered models across 22 families** (adjusted, aligned, block_exposure, concurrent, corr_factor, did, dose_response, gain_factors, growth, historical_growth, historical_joint, horseshoe, itt, joint, joint_mechanism, lcsm, level_factors, long_corr_factor, mechanism, mediation, mediation_multi, survival). Counts drift as models are added — `definitions.KINDS` and `discover_models()` are authoritative, and `docs/models/README.md` is the catalogue. The available-case modified ITT estimate `τ` (and DiD `δ`, gain-factor on-intervention marginal) has a causal reading only under its stated analysis-set assumptions; everything else is an adjusted association.

## Prerequisites

`conda activate dse-language-reading-predictors`. Each family has a factory in `statistical_models/factories.py` and a pipeline in `pipeline.py`. Models are auto-discovered — a new `lrp_rli_*` / `lrp_rlm_*` module with a top-level `fit()` registers itself.

## Fit

```bash
python scripts/fit_statistical_model.py <model_id> --config dev            # fast smoke
python scripts/fit_statistical_model.py <model_id> --config reporting --render
python scripts/fit_statistical_model.py all --config reporting --render    # all 220 + Quarto reports
python scripts/fit_statistical_model.py <model_id> --config reporting --target-accept 0.97
```

- Sampling presets: `dev` = 500 draws × 500 tune × 2 chains; `test` = 2000×2000×4; **`reporting` = 6000 draws × 6000 tune × 6 chains, target_accept 0.95**.
- `all` fits every registered model sequentially (nutpie fast — a typical ITT reporting fit is ~40 s; growth/mediation/HSGP/LCSM are far slower — the mediation g-formula fits run 25–45 min each). Full reporting sweep of 220 + renders ≈ **6–7 h on 16 cores** (measured 2026-08-17). **`all --render` batches every render until after all 220 fits finish**, so an interrupted sweep leaves fitted-but-un-rendered dirs. For a full sweep prefer a resumable per-model driver that renders each model right after its own fit and skips dirs already holding both `release_decision.json` and `index.html` — a monolithic background sweep has been reaped mid-run before.
- Model ids accept canonical (`lrp-rli-itt-007`) or legacy (`lrpitt07`) forms.
- Failures and render failures are collected; the script exits non-zero if any fit OR render failed OR the upload step raised — **check the exit code, not just the "N fitted, 0 failed" line** (an upload 403 makes it exit 1 even when all fits/renders succeeded).
- `--target-accept` raises the NUTS acceptance to chase away divergences on hard geometries. Precedence is CLI override > model-spec default > preset, so prefer this flag over patching the sampler — `config.json` then records the value actually used. It is the wrong tool for an r̂/ESS failure with zero divergences (see the gate section).

## Cross-model comparison

After the full fit:

```bash
python scripts/compare_statistical_models.py --config reporting
```

Writes to `output/statistical_models/comparison/`: `itt_vs_joint_tau.csv` (single-outcome τ vs joint τ_k consistency), `tau_forest.png`, `mechanism_forest.{png,csv}`, and nested PSIS-LOO tables (mechanism / phonics-route / age-moderation / dose / did-dose).

## Outputs — `output/statistical_models/models/<model_id>-<config>/`

`config.json`, `diagnostics_summary.json` (the pass/fail gate), `trace.nc` (large — excluded from upload by default), `priors_table.csv`, diagnostic plots, `index.html`/`index.qmd`, and family-specific result CSVs (below).

## Convergence gate — check BEFORE interpreting

`diagnostics_summary.json` has `passed` and `checks` = {rhat, ess, divergences, bfmi}. Thresholds: **r̂ ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.30, and zero divergences.** The `divergences` check fails on _any_ divergence, so a fit can be flagged while r̂ = 1.00 and ESS is huge.

**Divergences fail closed — there is no percentage threshold.** The "Divergent-transition qualification" policy in `METHODS.md` supersedes the earlier ≤ 1 % guidance: a divergence count or percentage _alone_ can never establish safety, so zero remains the only automatic clean pass. A reporting-tier fit with a genuinely small absolute number of divergences may exceptionally be labelled **QUALIFIED, NOT PASSED**, but only after a trace- and estimand-specific review, and it is never converted to `diagnostics_summary.passed = true`. Qualification requires _all_ of: divergences as the only failed check; a documented geometry-remediation attempt; an independent-seed reporting run; named headline summaries stable within Monte Carlo uncertainty; and a diagnostic run that deliberately maps the problematic geometry to those estimands. Any causal or model-of-record treatment effect, mediation decomposition, floor/survival estimand, nonlinear knee or shape, dose-heterogeneity slope, horseshoe ranking, covariance or latent-structure quantity is **zero-divergence-only** and cannot be qualified at all. Qualified fits are excluded from model comparisons unless their predictive geometry is separately reviewed, and qualification never propagates to sensitivity or leave-one-out refits. **Until a trace-bound verifier is implemented and a fit-specific review is approved, every divergent fit fails closed.** Full policy: `notes/202608021625-divergence-qualification-policy.md`.

Triage a failed gate by _which_ check failed — the remedy differs:

- **Any divergences** — withheld, however few. Remediate with a per-model `--target-accept` raise, then re-check. Do **not** blanket-raise across models: some specs declare a higher value in-module (`mm-001`/`002`/`101`, `mech-093`/`094`/`095` set 0.999), so a uniform 0.99 would _lower_ acceptance for those and can return a false zero-divergence pass. Read `config.json` → `sampling.target_accept` for what the fit actually used before choosing.
- **r̂ / ESS shortfall with zero divergences** — a mixing problem, not a geometry problem: raise draws and chains, not `target_accept` (there are no divergences for it to chase). Typical of weakly-identified hierarchical variance terms — e.g. a child random intercept competing with the Beta-Binomial `kappa` for the same overdispersion on a low-denominator outcome. More draws will clear the gate but will _not_ narrow a genuinely wide posterior; report it as "the computation is now trustworthy", not "the estimate improved".
- **A genuine structural concern** looks like `mm-001` (corr_factor): divergences **plus** sub-threshold BFMI — a latent-factor funnel. Hold its structural coefficients pending a non-centred reparameterisation; its correlations are fine.

**Never re-derive the diagnostics by hand** — in pipeline code, in a one-off script, or when spot-checking a trace. Call `statistical_models.sampling_quality.sampling_quality(trace, var_names=…)`, which returns all four signals unrounded and correctly coerced. `az.summary()` rounds to `rcParams["stats.round_to"]` (`"2g"`) unless passed `round_to="none"` — the **string**; `round_to=None` and `"auto"` both fall through to the rounded default, and omitting it returns a _string_-dtype frame. Rounding erases exactly the digits the gate turns on: **every r̂ from 1.011 to 1.049 rounds to `1.0`**, silently turning an r̂ ≤ 1.01 test into r̂ < 1.05. ESS must be the **minimum of `ess_bulk` and `ess_tail`**, not `ess_bulk` alone. This has gone wrong three times (dseinternational/research#65; again in #440's exact-LOO-refit gate; and in a prototype reporting four fits as `1.0000` when they were 1.0011–1.0022).

## Upload

Traces (`.nc`) are excluded by default (`--include-traces` to include; a full reporting sweep is **~45 GB of `trace.nc` across 220 fits** — measured 2026-08-17, and the dominant term in the output root's ~46 GB).

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

Follow the reporting standard in `METHODS.md` ("Interpret" and "Reporting results") and the evidence-ladder policy from issue #179 (`notes/202606261304-evidence-strength-and-rope-reporting.md`):

- **Report the posterior**: the **median** (transformation-invariant across logit/probability scales; preferred over the mean) + an inner **50 %** and outer **89 %** equal-tailed credible interval + the tail probability (e.g. `P(τ>0) = 0.97`). No p-values. **89 % is the house standard, not 95 %** — 95 % is an arbitrary NHST convention and its 2.5 / 97.5 % limits are the least MCMC-stable quantiles, so the deliberately non-round 89 % is used instead (Kruschke 2021 BARG; `notes/202607172359-credible-interval-standard.md`). The summary CSVs and `az.summary` output carry `eti89_lb` / `eti89_ub` accordingly — never re-derive a 95 % band. Read direction from the tail probability directly, not from whether a band excludes zero. Give both the logit scale and the probability/items scale at sample-mean baseline — the items translation ("≈ +3 of 32 letter sounds") is the most approachable form.
- **Evidence labels** are the fixed claim-oriented ladder — **inconclusive / suggestive / moderate / strong / very strong** at P ≥ 0.75 / 0.91 / 0.97 / 0.99 (round odds 3:1 / 10:1 / 30:1 / 100:1). Rules: a label qualifies the evidence for a **named claim** ("strong evidence the intervention helps"), is **oriented to the favoured direction** (a clearly negative effect is evidence of harm, not "inconclusive"), is reported **after** the probability, and **never describes effect size** — do not write "credible", "null", "leans positive", or fuse size words into labels ("credible, large"). The result CSVs already carry computed labels (`direction_label`, `favoured_direction_label`, `benefit_label` in `tau_summary.csv` / `rope_summary.csv` / `factor_summary.csv` / `did_summary.csv`) — use them rather than re-deriving.
- **Direction and magnitude are separate claims.** `pd`/`prob_*_pos` → the direction label; `prob_benefit_ge_delta` (against the pre-specified per-outcome minimally-important difference δ) → the magnitude (`benefit_label`); `prob_in_rope` quantifies a "probably negligible" reading. Report both when they diverge (e.g. very strong direction, suggestive that the benefit clears δ). A flat result is _inconclusive_ — quantified by ROPE mass — never "null" or "no effect".
- **Causal vs association**: only τ, the DiD δ, and the gain-factor on-intervention marginal are causal. Covariate coefficients are **Table-2-fallacy territory** — they describe _who progresses_, never levers; say so. Covariate sets are DAG-pre-specified, so a skill absent from a model was excluded by the diagram, not found unimportant.
- **Small-sample honesty**: point estimates that clear a threshold are on average magnitude-inflated (winner's curse) — lead with the interval, not the point.

The headline estimand per family:

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

**The coherent story to expect** (reading/phonics intervention): strong-to-very-strong evidence in the available-case modified ITT estimates for benefits on letter-sound knowledge (L), phoneme blending (B), word reading (W) and taught expressive vocab (TE); inconclusive-and-probably-negligible (high ROPE mass) on broad standardised vocabulary (R/E). Robust to ability/SES adjustment and replicated by DiD + gain-factor ANCOVA. The word-reading gain is **mediated by letter-sound knowledge** (very strong evidence for the NIE via L; the path via E is inconclusive and ≈ 0). Record the run (config, N fitted/failed, gate pass count + divergence caveats, key τ, blob/publish location) in a dated `notes/` note with the AI-authorship label. Pre-commit: `ruff check src/`, `npm run format:check`, `npm run spellcheck`.
