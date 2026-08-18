---
name: lrp-fit-statistical
description: Fit the Bayesian (PyMC) statistical models in this repo (language-reading-predictors) — Step 2 interactions/causal estimation. Use when asked to fit/refit statistical models, render reports, run the cross-model comparison, upload/publish artefacts, or summarise Bayesian findings. Covers scripts/fit_statistical_model.py (dev/test/reporting, all), the convergence gate, both blob-upload paths incl. the public-publish credential wrapper, and the per-family result CSVs.
---

> [!NOTE]
> Available-case modified ITT terminology updated by a LLM-based AI tool (Codex/GPT-5).
>
> Divergent-transition gate guidance, credible-interval standard, causal-term guidance, architecture, model/family counts and sweep figures corrected against `METHODS.md`, the registry and stored artefacts by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Fit statistical (Bayesian) models

Step 2 of the methodology: PyMC models for interactions and DAG-supported causal estimation, with interpretable estimands and quantified uncertainty. **220 registered models across 22 families** (adjusted, aligned, block_exposure, concurrent, corr_factor, did, dose_response, gain_factors, growth, historical_growth, historical_joint, horseshoe, itt, joint, joint_mechanism, lcsm, level_factors, long_corr_factor, mechanism, mediation, mediation_multi, survival). Counts drift as models are added — `definitions.KINDS` and `discover_models()` are authoritative, and `docs/models/README.md` is the catalogue. The available-case modified ITT `τ` (including joint-model `τ_k`), DiD `tau_t2`, the period-1-standardised marginal in **interaction-free primary** gain-factor fits, level-factor `b_grp_time[1]` at mean ability, and fitted LCSM window-1 assigned-arm change contrasts have causal readings only under their stated design and analysis-set assumptions. Gain-factor moderation variants are partly post-crossover-informed and never causal. Other DiD and LCSM window summaries, including `arm_gap_t3` and `delta_crossover`, are post-crossover associations; covariate, dose, mechanism and cross-process coupling coefficients are adjusted associations.

## Prerequisites

`conda activate dse-language-reading-predictors`. Shared model-construction helpers remain in `statistical_models/factories.py`; family orchestration lives in one module per kind under `statistical_models/pipelines/`. The aggregate `pipeline.py` facade has been retired. Models are auto-discovered — a new `lrp_rli_*` / `lrp_rlm_*` module with a top-level `fit()` registers itself.

## Fit

```bash
python scripts/fit_statistical_model.py <model_id> --config dev            # fast smoke
python scripts/fit_statistical_model.py <model_id> --config reporting --render
python scripts/fit_statistical_model.py all --config reporting --render    # all 220 + Quarto reports
python scripts/fit_statistical_model.py <model_id> --config reporting --target-accept 0.97
```

- Sampling presets: `dev` = 500 draws × 500 tune × 2 chains; `test` = 2000×2000×4; **`reporting` = 6000 draws × 6000 tune × 6 chains, with target_accept 0.95 as the preset default**. Model-specific defaults and a command-line override take precedence; read the stored value from `config.json`.
- `all` fits every registered model sequentially (nutpie fast — a typical ITT reporting fit is ~40 s; growth/mediation/HSGP/LCSM are far slower — the mediation g-formula fits run 25–45 min each). The August 2026 full reporting sweep of 220 models plus renders took **10.65 h on 16 cores**, of which 10.16 h was per-model fitting (sampling plus prior predictive, LOO, posterior predictive, diagnostics and figures) and about half an hour was rendering. **`all --render` batches every render until after all 220 fits finish**, so an interrupted sweep leaves fitted-but-unrendered dirs. Use the checked-in command unless a durable resumable driver is added under `scripts/`. A resumable driver must preserve its source and digest, validate registry completeness, and compare source commit/dirty state, data and environment hashes, config and sampling settings, `artifact_manifest.json`, and the current `release_decision.json` before skipping a directory. The presence of only `release_decision.json` and `index.html` is not a safe resume criterion.
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

`config.json`, `environment-lock.json`, `diagnostics_summary.json`, `artifact_manifest.json`, `release_decision.json`, `key_findings.json`, `trace.nc` (large — excluded from upload by default), `priors_table.csv`, diagnostic plots, `index.html`/`index.qmd`, family-specific result CSVs (below), and `subfit_provenance.csv` where required.

## Ordered release decision — check BEFORE interpreting

Convergence is necessary but not sufficient. Before reading a scientific result, require `release_decision.json` to have `publishable: true`, then confirm that `key_findings.json` is released and that `environment-lock.json`, `artifact_manifest.json`, and any applicable `subfit_provenance.csv` agree with the fit. The ordered decision checks inputs, computation, required artefacts and robustness; a model can pass convergence and still be withheld. The August 2026 run has 220/220 convergence passes but only 214/220 publishable fits.

### Convergence gate

`diagnostics_summary.json` has `passed` and `checks` = {rhat, ess, divergences, bfmi}. Thresholds: **r̂ ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.30, and zero divergences.** The `divergences` check fails on _any_ divergence, so a fit can be flagged while r̂ = 1.00 and ESS is huge.

**Divergences fail closed — there is no percentage threshold.** The "Divergent-transition qualification" policy in `METHODS.md` supersedes the earlier ≤ 1 % guidance: a divergence count or percentage _alone_ can never establish safety, so zero remains the only automatic clean pass. A reporting-tier fit with a genuinely small absolute number of divergences may exceptionally be labelled **QUALIFIED, NOT PASSED**, but only after a trace- and estimand-specific review, and it is never converted to `diagnostics_summary.passed = true`. Qualification requires _all_ of: divergences as the only failed check; a documented geometry-remediation attempt; an independent-seed reporting run; named headline summaries stable within Monte Carlo uncertainty; and a diagnostic run that deliberately maps the problematic geometry to those estimands. Any causal or model-of-record treatment effect, mediation decomposition, floor/survival estimand, nonlinear knee or shape, dose-heterogeneity slope, horseshoe ranking, covariance or latent-structure quantity is **zero-divergence-only** and cannot be qualified at all. Qualified fits are excluded from model comparisons unless their predictive geometry is separately reviewed, and qualification never propagates to sensitivity or leave-one-out refits. **Until a trace-bound verifier is implemented and a fit-specific review is approved, every divergent fit fails closed.** Full policy: `notes/202608021625-divergence-qualification-policy.md`.

Triage a failed gate by _which_ check failed — the remedy differs:

- **Any divergences** — withheld, however few. Remediate with a per-model `--target-accept` raise, then re-check. Do **not** blanket-raise across models: some specs declare a higher value in-module (`lrp-rli-mm-001`/`002`/`101`/`102` and `lrp-rli-mech-093`/`094`/`095` set 0.999), so a uniform 0.99 would _lower_ acceptance for those and can return a false zero-divergence pass. Read `config.json` → `sampling.target_accept` for what the fit actually used before choosing.
- **r̂ / ESS shortfall with zero divergences** — a mixing problem, not a geometry problem: raise draws and chains, not `target_accept` (there are no divergences for it to chase). Typical of weakly-identified hierarchical variance terms — e.g. a child random intercept competing with the Beta-Binomial `kappa` for the same overdispersion on a low-denominator outcome. More draws will clear the gate but will _not_ narrow a genuinely wide posterior; report it as "the computation is now trustworthy", not "the estimate improved".
- **A genuine structural concern** can present as divergences together with sub-threshold BFMI, suggesting a funnel or another poorly explored geometry; diagnose the affected trace and estimands rather than assigning that diagnosis from model family alone. Do not use `lrp-rli-mm-001` as a current example: after the bare-`LKJCorr` correction its stored reporting fit has zero divergences, maximum r̂ 1.0012, minimum effective sample size 10,509.7 and minimum BFMI 0.966, and it is publishable. Do not conflate it with the separate historical model `lrp-rlm-mm-001`, which has different diagnostics and is withheld at the inputs stage for unresolved denominator provenance.

**Never re-derive the diagnostics by hand** — in pipeline code, in a one-off script, or when spot-checking a trace. Call `statistical_models.sampling_quality.sampling_quality(trace, var_names=…)`, which returns all four signals unrounded and correctly coerced. `az.summary()` rounds to `rcParams["stats.round_to"]` (`"2g"`) unless passed `round_to="none"` — the **string**; `round_to=None` and `"auto"` both fall through to the rounded default, and omitting it returns a _string_-dtype frame. Rounding erases exactly the digits the gate turns on: **every r̂ from 1.011 to 1.049 rounds to `1.0`**, silently turning an r̂ ≤ 1.01 test into r̂ < 1.05. ESS must be the **minimum of `ess_bulk` and `ess_tail`**, not `ess_bulk` alone. This has gone wrong three times (dseinternational/research#65; again in #440's exact-LOO-refit gate; and in a prototype reporting four fits as `1.0000` when they were 1.0011–1.0022).

## Upload

Traces (`.nc`) are excluded by default (`--include-traces` to include; the current August 2026 reporting artefacts contain **51.0 GB decimal / 47.5 GiB of `trace.nc` across 220 fits**, the dominant part of a models root of about 58 GB decimal).

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

- **Report the posterior**: the **median** (transformation-invariant across logit/probability scales; preferred over the mean) + an inner **50 %** and outer **89 %** equal-tailed credible interval + the tail probability (e.g. `P(τ>0) = 0.97`). No p-values. **89 % is the house standard, not 95 %** — 95 % is an arbitrary NHST convention and its 2.5 / 97.5 % limits are the least MCMC-stable quantiles, so the deliberately non-round 89 % is used instead (Kruschke 2021 BARG; `notes/202607172359-credible-interval-standard.md`). The summary CSVs and `az.summary` output carry `eti89_lb` / `eti89_ub` accordingly — never re-derive a 95 % band. Read direction from the tail probability directly, not from whether a band excludes zero. Follow the stored estimand and its reported scale: ITT items effects are average marginals over the fitted rows' observed covariate profiles, not evaluations at one sample-mean profile, and other families may report percentage-point, latent-change, hazard-ratio or association scales. Use the fit's own translated artefact rather than imposing one universal conversion.
- **Evidence labels** are the fixed claim-oriented ladder — **inconclusive / suggestive / moderate / strong / very strong** at P ≥ 0.75 / 0.91 / 0.97 / 0.99 (round odds 3:1 / 10:1 / 30:1 / 100:1). Rules: a label qualifies the evidence for a **named claim** ("strong evidence the intervention helps"), is **oriented to the favoured direction** (a clearly negative effect is evidence of harm, not "inconclusive"), is reported **after** the probability, and **never describes effect size** — do not write "credible", "null", "leans positive", or fuse size words into labels ("credible, large"). The result CSVs already carry computed labels (`direction_label`, `favoured_direction_label`, `benefit_label` in `tau_summary.csv` / `rope_summary.csv` / `factor_summary.csv` / `did_summary.csv`) — use them rather than re-deriving.
- **Direction and magnitude are separate claims.** `pd`/`prob_*_pos` → the direction label; `prob_benefit_ge_delta` (against the per-outcome minimally-important difference δ) → the magnitude (`benefit_label`); `prob_in_rope` quantifies support for a practically small effect. Check the fit's provenance before calling δ pre-specified: some thresholds were agreed only after initial results review and are explicitly post-hoc. Report direction and magnitude when they diverge (e.g. very strong direction, suggestive that the benefit clears δ). A flat result is _inconclusive_ — quantified by ROPE mass — never "null" or "no effect".
- **Causal vs association**: the available-case modified ITT `τ` (and joint `τ_k`), DiD `tau_t2`, the period-1-standardised marginal from interaction-free primary gain-factor fits, level-factor `b_grp_time[1]` at mean ability, and fitted LCSM window-1 assigned-arm change contrasts are the randomisation-anchored terms, subject to each model's analysis-set and specification assumptions. Gain-factor moderation variants, later arm gaps, dose slopes, mechanism slopes, LCSM couplings and covariate coefficients are associations. Covariate coefficients are **Table-2-fallacy territory** — they describe _who progresses_, never levers; say so. Covariate sets are DAG-pre-specified, so a skill absent from a model was excluded by the diagram, not found unimportant.
- **Small-sample honesty**: point estimates that clear a threshold are on average magnitude-inflated (winner's curse) — lead with the interval, not the point.

The headline estimand per family:

| Family                      | File(s)                                                            | What to read                                                                                                                                                                        |
| --------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| itt                         | `tau_summary.csv`, `rope_summary.csv`                              | Outcome-specific available-case modified ITT average marginal; P/N headlines are post-hoc baseline-floor subgroup risk differences.                                                 |
| joint                       | `tau_summary.csv`, `tau_difference.csv`, `tau_contrast_matrix.csv` | Outcome-specific `τ_k`; 015/115 are generalisation contrasts and 016 is a modality contrast. Current factorised fits omit within-child residual covariance from contrast intervals. |
| did                         | `did_summary.csv`                                                  | `tau_t2` is randomisation-anchored; `arm_gap_t1` is balance, while `arm_gap_t3`, `delta_crossover` and dose slopes are associations.                                                |
| gain_factors                | `treatment_marginal.csv`                                           | Period-1-standardised marginal is causal only in interaction-free primaries; moderation-variant marginals are partly post-crossover-informed associations.                          |
| level_factors               | `factor_summary.csv`, `rope_summary.csv`                           | `b_grp_time[1]` at mean ability is randomisation-anchored; later waves and the time-invariant group-by-ability term are associations.                                               |
| mechanism                   | `mechanism_curve.csv`, comparison `mechanism_forest.csv`           | Baseline-conditional exposure association, never "X drives Y"; distinguish score exposures from raw-covariate variants.                                                             |
| joint_mechanism             | `joint_mechanism_slopes.csv`                                       | Within-model difference between two adjusted mechanism slopes; association, with measurement and residual-confounding limits.                                                       |
| mediation / mediation_multi | `mediation_summary.csv`                                            | Model-based `total`/`NDE`/`NIE` decomposition; contemporaneous timing and treatment-induced dose confounding prevent a causal mediation reading.                                    |
| dose_response               | `dose_slope_summary.csv`                                           | Observational dose slope; dose is a partial collider, so treat as sensitivity evidence.                                                                                             |
| aligned                     | `factor_summary.csv`, `cohort_marginal.csv`                        | Per-protocol cohort association; off-floor outcomes are percentage-point risk differences, not items.                                                                               |
| lcsm                        | `coupling_summary.csv`, `itt_window1_contrast.csv`                 | Couplings are associations; fitted first-window assigned-arm change contrasts inherit randomisation, while later windows do not.                                                    |
| growth                      | `growth_association_summary.csv`                                   | Between-child growth associations; check mandatory observation-influence stability where configured.                                                                                |
| historical_growth           | `posterior_growth_summary.csv`                                     | Descriptive within-group growth and secondary group contrasts; no causal terms and affected measures remain input-gated.                                                            |
| historical_joint            | `measure_correlation_summary.csv`, `within_scale_summary.csv`      | Stable-level and within-child correlations; non-resolution is lack of identification, not a null.                                                                                   |
| horseshoe                   | `predictor_ranking.csv`                                            | Shrinkage ranking cross-check of the GB analysis; not a causal ordering.                                                                                                            |
| corr_factor                 | `factor_correlation_summary.csv`, `structural_summary.csv`         | Cross-sectional domain correlations and structural associations; check fit-specific prior sensitivity and input provenance.                                                         |
| long_corr_factor            | `factor_correlation_by_wave.csv`, `trait_state_summary.csv`        | Longitudinal latent correlations and trait/state decomposition; no cross-lagged causal coupling.                                                                                    |
| adjusted                    | `predictor_associations.csv`                                       | Mutually adjusted and bivariate associations; inspect whether the model is single-span or repeated-transition rather than calling every result between-child.                       |
| survival                    | `survival_summary.csv`                                             | Pooled prognostic hazard association under complementary log-log; a hazard ratio is not a probability ratio or a causal effect.                                                     |
| block_exposure              | `block_exposure_summary.csv`                                       | Post-crossover block-exposure associations; direction labels do not establish the parallel-trajectories assumption.                                                                 |
| concurrent                  | `concurrent_associations.csv`, `concurrent_marginals.csv`          | Same-wave adjusted associations and marginals; no temporal or causal ordering.                                                                                                      |

**The current pattern to check, not assume** (reading/phonics intervention): the available-case modified ITT estimates give strong-to-very-strong directional evidence of benefits on letter-sound knowledge (L), phoneme blending (B), word reading (W) and taught expressive vocabulary (TE). Broad standardised receptive and expressive vocabulary (R/E) remain imprecise: their intervals permit material effects and their ROPE probabilities are not high enough to establish practical absence. Ability and SES adjustments are sensitivity checks; DiD, interaction-free gain-factor primaries, level-factor fits and LCSM first-window contrasts reuse the same randomised t2 information under different specifications and are not independent replications. Mediation fits may estimate a positive indirect component through letter-sound knowledge, but contemporaneous mediator/outcome timing and treatment-induced session-dose confounding prevent a causal mediation reading; report the confounding calibration and t3 sensitivity. Before summarising, require `release_decision.publishable`, not only a passed convergence gate. Record the run (config, N fitted/failed, convergence count, publishable/withheld count and reasons, key estimands, source/data/environment provenance, and blob/publish location) in a dated `notes/` note with the AI-authorship label. Pre-commit: `ruff check src/`, `npm run format:check`, `npm run spellcheck`.
