# Full statistical-model refit and report render (2026-07-19)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

## Goal

Refit **every** registered Bayesian (PyMC) statistical model at the production `reporting` sampling configuration and render each model's Quarto report. This is a from-scratch sweep — `output/statistical_models/` was empty at the start.

## Scope

- **176 models** across all families (enumerated from `statistical_models.registry.discover_models()` — the skill catalogue's "89" is stale). Family breakdown from the registry: itt (incl. joint/floored variants), gain_factors (gf), level_factors (lf), aligned (al), did, mechanism (mech), mediation (med), dose_response (dose), lcsm, growth/historical (hg/jc), horseshoe (hs), corr_factor/mm, adjusted (adj), block_exposure (bx), concurrent (ca), survival (surv), long_corr_factor (lcf), growth-coupling (gc).
- Config: `reporting` = 6000 draws × 6000 tune × 6 chains, `target_accept 0.95` (confirmed in each fit's `config.json`).
- Machine: local macOS, 16 cores; env `dse-language-reading-predictors` (PyMC 6.1.0).

## Method — resumable per-model driver

Prior experience (memory: "long fit sweeps get reaped") is that a monolithic `fit_statistical_model.py all --config reporting --render` background run gets killed part-way, and because `all --render` batches all renders to the very end, an interrupted sweep leaves _no_ rendered reports. So instead of `all`, a **resumable per-model driver** (`scratchpad/driver.sh`) was used:

- Iterates the 176 ids; for each runs `python scripts/fit_statistical_model.py <id> --config reporting --render` (env python by absolute path; `QUARTO_PYTHON` is pinned inside the fit script since PR #351, so the render works in a non-activated shell).
- **Skips** any model whose final `output/statistical_models/models/<id>-reporting/index.html` already exists → safe to re-launch after a reap; it resumes where it left off.
- Fits are atomic: each fit stages to a hidden sibling dir and is promoted only on success (`OutputTransaction`), so an interrupted fit never leaves a half-written final dir to be mistaken for complete.
- Logs one timestamped line per model (`SKIP`/`FIT`/`DONE`/`FAIL`) to a driver log, with each model's full stdout/stderr captured separately for debugging.
- Launched with the Bash tool's `run_in_background` (notifies on exit); a persistent Monitor heartbeats progress every 5 min and flags stalls (driver reaped).

## Smoke test (pre-flight)

`lrp-rli-itt-001 --config reporting --render` end-to-end before committing to the full sweep: ~25 s, exit 0, `index.html` (83 KB) rendered, gate **passed** (r̂ = 1.0003, min ESS = 18 509, 0 divergences, BFMI all > 1.0). Confirmed `config.json` sampling = 6000/6000/6 chains/0.95. Pipeline validated.

## Report-template gap found and filled

The sweep surfaced a pre-existing repo gap: **9 `mech` models had no `docs/models/<id>/index.qmd`** report template, so they fit fine but could not render an `index.html` (`_copy_report_template` is a literal path check with no parent fallback). The nine, and the templates drafted for them (thin-template convention; LRP71 shape for the moderation models, LRP58 shape for the knee-tests; Quarto AI-authorship callout on each):

- **Joint-readiness moderation** (letter-sound $L \to W$ HSGP curve + a vocabulary moderator): `mech-093` ($\times R$), `mech-094` ($\times TR$), `mech-095` ($\times TE$) — companions of the existing LRP71 ($\times E$).
- **GP knee-tests** (nonparametric HSGP re-attempts, testing for a "knee"): `mech-156` ($R\to W$), `mech-157` ($E\to W$), `mech-188` ($TR\to W$), `mech-189` ($TE\to W$), `mech-190` (blending $B\to W$, new), `mech-191` (intervention sessions $IS\to W$, new continuous-covariate dose curve).

All nine templates spell-check clean and are **render-validated end-to-end** against their real fits (both structural patterns). Corrected one slip: receptive vocab $R$ is `rowpvt` (ROWPVT), not `bpvs`.

## Progress / results

- **Sweep:** 2026-07-19 **11:08 → 18:11 UTC (~7 h wall-clock)**, sequential, one fit at a time (6 chains on 16 cores). The resumable driver ran the whole sweep without being reaped.
- **Fits:** **176 / 176 fitted and rendered** — 176 `config.json`, 176 `index.html`, none missing. (The 4 `FAIL` lines in `driver.log` — mech-093/094/095/156 — are cosmetic: fit before their templates existed, then rendered manually. Real fit failures: **0**.)
- **Long pole:** the three two-mediator g-formula models (`med-064/066/075`) each ran ~35–40 min — their `sensitivity_sweep_two_mediator` (#335) re-runs 2 legs × 21 δ = 42 full counterfactual decompositions over all 36 000 posterior draws. Inherent to reporting draws, not a bug; left at the reporting standard (no `n_deltas`/subsampling tuning).

### Convergence gate: 159 / 176 pass

Full per-model table in `output/statistical_models/_sweep_gate_summary.csv`. The 17 flagged split cleanly:

- **13 divergence-only** — pass r̂ ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.3, fail _only_ on divergences, all well within the METHODS ≤ 1 % guidance (worst is `mech-190` at 31/36 000 = **0.086 %**): the six knee-tests `mech-156/157/188/189/190/191` (for these the small divergence count **is** the finding — the HSGP curve is demanding and the "knee" stays only weakly identified, exactly as the templates warn), plus `mech-095`, the dose models `dose-077/083/084/177`, `did-007` and `hs-001`. **Usable, noted.**
- **4 genuine concerns** — all the corr_factor / measurement-model family, the known latent-factor funnel: `mm-001` (r̂ 1.019, ESS 354), `mm-002` (r̂ 1.048, ESS 64), `mm-101` (r̂ 1.021, ESS 260), `rlm-mm-001` (r̂ 1.029, ESS 64, 143 div). Their **domain correlations remain usable; the structural/latent legs are held** pending a non-centred reparameterisation / higher `target_accept` — the recurring latent-factor funnel in these correlation models.

### Headline ITT τ — risk-difference scale, 89 % equal-tailed CI (house standard)

Positive τ = intervention helps; only τ is causal. Labels per the evidence ladder (P ≥ 0.75/0.91/0.97/0.99).

| Outcome                           | Model   | τ (median) | 89 % CI         | P(τ>0) | Evidence        |
| --------------------------------- | ------- | ---------- | --------------- | ------ | --------------- |
| Letter-sound knowledge (L)        | itt-007 | +0.110     | [0.053, 0.166]  | 0.999  | **very strong** |
| Phoneme blending (B)              | itt-008 | +0.099     | [0.022, 0.174]  | 0.980  | strong          |
| Word reading (W)                  | itt-010 | +0.030     | [0.009, 0.051]  | 0.986  | strong          |
| Taught expressive vocab (TE)      | itt-002 | +0.064     | [0.018, 0.111]  | 0.985  | strong          |
| Taught receptive vocab (TR)       | itt-001 | +0.057     | [0.008, 0.105]  | 0.968  | moderate        |
| Not-taught receptive vocab (UR)   | itt-003 | +0.050     | [−0.002, 0.103] | 0.937  | moderate        |
| Nonword reading (N, off-floor)    | itt-011 | +0.100     | [−0.038, 0.237] | 0.877  | suggestive      |
| Not-taught expressive vocab (UE)  | itt-004 | +0.026     | [−0.029, 0.080] | 0.773  | suggestive      |
| Phonetic spelling (P, off-floor)  | itt-009 | +0.041     | [−0.071, 0.155] | 0.724  | inconclusive    |
| Standardised receptive vocab (R)  | itt-005 | +0.001     | [−0.022, 0.025] | 0.539  | inconclusive    |
| Standardised expressive vocab (E) | itt-006 | +0.001     | [−0.018, 0.020] | 0.534  | inconclusive    |

This is the expected coherent picture: strong-to-very-strong ITT benefit on the code-related and directly-taught skills (L, B, W, TE), tapering to moderate/suggestive on the not-taught vocabulary transfer measures, and **inconclusive-and-probably-negligible on broad standardised vocabulary (R, E)** — the standardised measures barely move.

### Mediation (g-formula)

The word-reading gain is **mediated by letter-sound knowledge**, not by the vocabulary route:

- `med-059` (count mediator = letter sounds): NIE = **+1.68 words** [0.58, 3.21], P = 0.997 (**very strong**); NDE ≈ +0.17 [−1.70, 2.07], P = 0.56 (inconclusive); proportion mediated ≈ 0.82.
- `med-062` (Gaussian reading-route composite): NIE = +0.92 words [0.07, 2.19], P = 0.961 (strong).
- `med-064` (two-mediator): NIE via letter-sound = +1.98 [0.62, 3.63], P = 0.996; via expressive vocab ≈ +0.08 [−0.47, 0.75], P = 0.58 — the indirect effect runs through letter-sound, not vocabulary.

### Cross-model comparison

`scripts/compare_statistical_models.py --config reporting` → `output/statistical_models/comparison/` (exit 0): `itt_vs_joint_tau.csv`, `tau_forest.png`, `mediation_family.csv`+forest, `mechanism_loo_compare.csv`, `phonics_route_loo_compare.csv`, `age_moderation_loo_compare.csv`, `tier1_*` decoding-specificity contrast + negative-control forest, `triangulation_consistency.csv`. Two sub-comparisons were **conservatively skipped**: the dose-response and DiD-L-dose LOO compares, because their inputs (`dose-077`, `did-007`) carry divergence-only gate-REVIEW flags; and the mechanism forest, on a pre-existing `mech-058` obs-count mismatch (157 vs 156 — confounder-only missingness the keep-mask does not model). None caused by this run.

### Artefact locations

- Per-model outputs + reports: `output/statistical_models/models/<id>-reporting/` (176 dirs, each with `index.html`).
- Gate roll-up: `output/statistical_models/_sweep_gate_summary.csv`.
- Comparison: `output/statistical_models/comparison/`.
- New report templates: `docs/models/lrp-rli-mech-{093,094,095,156,157,188,189,190,191}/index.qmd`.
- Not uploaded/published (public site is preliminary — confirm scope before publishing). Traces (`trace.nc`) are on disk but excluded from any upload by default.
