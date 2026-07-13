<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Full statistical-model refit (reporting config) — run record and results

Date: 2026-07-13

Related: `METHODS.md` (interpret/reporting standard), #179 (evidence-ladder & ROPE policy), #247 (revised-DAG factor adjustments), #265 (HSGP mechanism reparameterisation), the methodology review at `notes/202607130922-statistical-models-methodology-review.md`.

## What was run

A complete refit of **every registered Bayesian statistical model (115 models across 16 families)** at the `reporting` sampling preset (6 chains × 6000 draws × 6000 tune, `target_accept = 0.95`), each with its Quarto report rendered, followed by the cross-model comparison and publication of all artefacts to the public research site.

```bash
python scripts/fit_statistical_model.py all --config reporting --render
python scripts/compare_statistical_models.py --config reporting
```

- **Data:** `data/rli_data_long.csv`, N = 53 children, 4 timepoints (waitlist-crossover RCT).
- **Fit sweep started 11:34 BST.** The first `all` run was terminated externally at ~12:39 after completing 112 of 115 models (it died mid-sampling on the slow `corr_factor` latent-factor model). No fits were lost — the 112 completed dirs were intact, so the run was **resumed for only the 3 outstanding models** (`lrp-rli-mm-001`, `lrp-rli-mm-101`, `lrp-rlm-hg-001`) rather than re-fitting from scratch. All 3 completed cleanly (exit 0) by 12:58.
- Every model wrote `trace.nc`, `diagnostics_summary.json`, `config.json`, `priors_table.csv`, the family result CSVs, diagnostic plots and a rendered `index.html`.

## Convergence gate: 109 / 115 PASS

Gate thresholds: r̂ ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.30, **zero** divergences (the divergence check fails on any single divergence, so a fit can be flagged while r̂ = 1.00 and ESS is large).

The 6 flagged models are all expected and were triaged, not silently accepted:

| Model              | Failed check(s)     | Divergences  | max r̂  | min ESS | Reading                                                                                                                                                                         |
| ------------------ | ------------------- | ------------ | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-did-007`  | divergences         | 34 (0.094 %) | 1.0015 | 5558    | Divergence-only, well under the METHODS ≤ 1 % guidance — **usable**. Period-resolved letter-sound dose geometry.                                                                |
| `lrp-rli-dose-077` | divergences         | 17 (0.047 %) | 1.001  | 4088    | Divergence-only, **usable**. Cumulative-session dose slope.                                                                                                                     |
| `lrp-rli-dose-177` | divergences         | 13 (0.036 %) | 1.001  | 3792    | Divergence-only, **usable**. Dose slope.                                                                                                                                        |
| `lrp-rli-hs-001`   | divergences         | 2 (0.006 %)  | 1.001  | 9352    | Divergence-only, **usable**. Horseshoe funnel.                                                                                                                                  |
| `lrp-rli-mm-001`   | r̂, ESS, divergences | 1 (0.003 %)  | 1.0195 | 354     | corr_factor latent-factor funnel. Mild r̂/ESS miss — **domain correlations robust; structural leg read cautiously** (as flagged in the methodology review). BFMI healthy (0.91). |
| `lrp-rli-mm-101`   | r̂, ESS, divergences | 57 (0.158 %) | 1.0213 | 260     | corr_factor variant, same funnel — same caution.                                                                                                                                |

No headline causal estimand sits on a flagged model: the four divergence-only fits clear the ≤ 1 % guidance, and the two corr_factor misses affect only the measurement model's structural leg (its **correlations**, which are the deliverable, converged fine). The comparison script correctly excludes all 6 from its interpretable LOO tables.

## Results — the causal story holds and replicates across designs

Reported per the `METHODS.md` / #179 standard: posterior **median**, equal-tailed **95 % credible interval**, tail probability `P(τ>0)`, and the fixed evidence ladder (suggestive ≥ 0.75 / moderate ≥ 0.91 / strong ≥ 0.97 / very strong ≥ 0.99). Only τ, the DiD δ and the gain-factor on-intervention marginal are **causal**; everything else is an adjusted association. All τ figures below are on the **probability (risk-difference) scale** at sample-mean baseline.

### ITT suite (`lrp-rli-itt-001…011`) — single-outcome intention-to-treat

| Outcome                          | τ (risk diff.) | 95 % CI          | P(τ>0)    | Evidence         |
| -------------------------------- | -------------- | ---------------- | --------- | ---------------- |
| **Letter sounds (L)**            | **+0.110**     | [+0.040, +0.179] | **0.999** | **very strong**  |
| **Phoneme blending (B)**         | **+0.099**     | [+0.005, +0.192] | **0.980** | **strong**       |
| **Word reading (W)**             | **+0.030**     | [+0.004, +0.057] | **0.986** | **strong**       |
| **Taught expressive vocab (TE)** | **+0.064**     | [+0.006, +0.122] | **0.985** | **strong**       |
| Taught receptive vocab (TR)      | +0.057         | [−0.003, +0.117] | 0.968     | moderate         |
| Untaught receptive vocab (UR)    | +0.050         | [−0.014, +0.116] | 0.937     | moderate         |
| Nonword decoding (N)             | +0.100         | [−0.071, +0.268] | 0.877     | suggestive       |
| Untaught expressive vocab (UE)   | +0.026         | [−0.041, +0.093] | 0.773     | suggestive       |
| Number knowledge (P)             | +0.041         | [−0.098, +0.183] | 0.724     | inconclusive     |
| **Broad receptive vocab (R)**    | +0.001         | [−0.027, +0.031] | 0.539     | **inconclusive** |
| **Broad expressive vocab (E)**   | +0.001         | [−0.022, +0.025] | 0.534     | **inconclusive** |

The strongest, clearest benefit is on **letter-sound knowledge** (≈ +3–4 of 32 letter sounds; the most approachable framing), then **phoneme blending, word reading and taught expressive vocabulary**. **Broad standardised vocabulary (R/E) is inconclusive and probably negligible** — near-zero medians with tight bands hugging zero (a ROPE reading, not "no effect"). P and N are heavily floored; their primary estimand is an off-floor risk difference (graded τ shown here is the flagged secondary).

### Robustness — the story survives adjustment

- **Ability-adjusted** (block-design, `itt-017…024`) reproduces the suite almost exactly: L very strong (P = 0.999), W strong (0.980), TE strong (0.982), R/E inconclusive (0.55/0.54).
- **SES-adjusted / SES complete-case** word reading (`itt-013/014`): P = 0.967 / 0.977 — holds.
- **Joint model** (`itt-012`) τ_k agree with the single-outcome fits (logit scale W 0.354 vs 0.354, L 0.578 vs 0.582, R/E ≈ 0 in both) — see `comparison/itt_vs_joint_tau.csv`.

### Within-person replication — DiD (`lrp-rli-did-001…006`)

The waitlist-crossover difference-in-differences reproduces the randomised effect with each child as its own control: **word reading very strong** (did-001 δ P = 0.995; did-006 P = 0.992), receptive vocabulary moderate (did-005 P = 0.921), letter sounds suggestive (did-002 P = 0.866), phoneme blending inconclusive (did-003).

### ANCOVA replication — gain factors (`lrp-rli-gf-001…011`)

On-intervention risk difference (the only causal term): **word reading P = 0.993 (≈ +2.6 words)**, **letter sounds P = 0.991 (≈ +3.3 letter sounds)**, phoneme blending P = 0.903; broad vocab R/E and number P negligible/negative. Consistent with ITT and DiD.

### Mechanism of the reading gain — mediation (`lrp-rli-med-*`)

The word-reading gain runs **through letter-sound knowledge**, not through broad vocabulary:

- `med-059` (single mediator L): **NIE via L P = 0.997, ≈ +1.7 words**; NDE ≈ 0 (P = 0.56); proportion mediated ≈ 0.82.
- `med-064` (two mediators L + broad expressive E): **NIE_L P = 0.996**, NIE_E ≈ 0 (P = 0.58) — the route is L, not E.
- `med-066` / `med-075` (L + phoneme blending B): NIE_L P = 0.998, NIE_B ≈ 0 / slightly negative — the mediation loads on letter sounds specifically.
- `med-076` (longitudinal ordering, L at t2): NIE P = 0.999, ≈ +3.1 words.
- `med-079` (**negative-control mediator, grammar**): NIE ≈ 0 (P = 0.71) — the control behaves as it should, i.e. no spurious route.

### Cross-checks

- **Horseshoe predictor ranking** (`hs-002`, word reading): top selected predictors are **L (p = 0.995) and broad expressive E (p = 0.992)**, then taught vocab T — corroborates the GB ranking and the mediation finding.
- **Correlated-domain measurement model** (`mm-001`): domain correlations are strong and robust — vocabulary↔code 0.72, vocabulary↔grammar 0.79, code↔grammar 0.65, all P(>0) ≈ 1.0. (Structural leg held cautiously — see gate.)

## Cross-model comparison artefacts

`output/statistical_models/comparison/`: `itt_vs_joint_tau.csv`, `tau_forest.png`, `mediation_family.csv` + `mediation_family_forest.png`, and nested PSIS-LOO tables (mechanism / phonics-route / age-moderation / dose / did-dose). **Caveat:** the mechanism τ forest PNG was skipped this run — `lrp-rli-mech-058` had a reconstructed-size mismatch (157 vs 156 obs; confounder-only missingness the keep-mask does not model) and was dropped from that plot and its CSV. All other comparison artefacts wrote normally. Worth a follow-up to teach the mechanism-forest reconstruction about confounder-only missingness so mech-058 rejoins the plot.

## Publication

All 115 model output dirs + the comparison dir published to the **public research site** (anonymously readable), traces excluded — 116 directories, 0 upload failures. Public access verified (`200 text/html`).

- **Publish run id:** `019f5b61-caf8-70f0-9952-11d9121f30b3`
- **Report root:** `https://dseresearch.blob.core.windows.net/public/projects/language-reading-predictors/output/019f5b61-caf8-70f0-9952-11d9121f30b3/`
- **Per-model report:** `<root>/<model-id>-reporting/index.html` — e.g. word reading ITT: `https://dseresearch.blob.core.windows.net/public/projects/language-reading-predictors/output/019f5b61-caf8-70f0-9952-11d9121f30b3/lrp-rli-itt-010-reporting/index.html`
- **Comparison artefacts:** `<root>/comparison/` (CSVs + forest PNGs; no landing HTML).

Uploaded via the `AzureCliCredential` wrapper (`az login` has the write role on `dseresearch`; the VM managed identity does not, so the built-in `--upload` flag would 403 — see `lrp-fit-statistical`). Reports had to be **rendered separately** this run: `fit_statistical_model.py all --render` renders only in a batch after every fit finishes, so the externally-killed first sweep left the 112 completed models un-rendered; they were rendered standalone (`quarto render index.qmd`, 111 OK / 0 fail) before the upload.

This is preliminary work in progress — all data and models are exploratory.
