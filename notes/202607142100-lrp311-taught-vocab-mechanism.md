<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# Taught-vocabulary and phonological-memory mechanism models: TR → W, TE → W and RW → W (#311)

Date: 2026-07-14

Related: #311, #314 (descriptive-association workstream), #265 (HSGP reparameterisation, closed), #269 (treatment-affected conditioning), #309 (the IS recanting-witness sign-off pattern), `dag/dag-language-reading.dagitty` (2026-07-10 revision).

## What was added

Two new mechanism-family models, `lrp-rli-mech-088` (TR → W) and `lrp-rli-mech-089` (TE → W), giving the taught-vocabulary dose-response into word reading as adjusted associations — "an additional p taught spoken words is associated with o additional words read". Both follow the LRP56/57 vocabulary-exposure precedent and enter the mechanism **linearly** (the HSGP curve has known sampler-geometry pathology on vocabulary predictors; DAG-required adjusters are never dropped to buy convergence).

## Adjustment-set derivations (contemporaneous DAG, 2026-07-10 revision)

**TR → W (mech-088).** Parents of TR: {A, GA, HS, IG, IS, RW}. Adjusted: G (always, via beta_G), A (linear), HS (`hs`/`hs_missing`), RW (`erbto`/`erbto_missing`), W_pre (autoregressive baseline). TR has no measured skill parent, so no concurrent measure confounder enters. TE and RV are TR-descendants that also cause W — deliberately NOT conditioned on (they carry legitimate indirect paths such as TR → TE → W).

**TE → W (mech-089).** Parents of TE: {A, GA, HS, IG, IS, RW, SP, TR}. As above plus SP (`deapp_c`/`deapp_c_missing`) and the one measured skill confounder **TR at its concurrent post score** (TR → TE and TR → WR), the LRP56/57 idiom. EV/EG/EI are TE-descendants and stay out.

**IS (intervention sessions) — the open backdoor, flagged not adjusted.** IS → TR/TE and IS → WR make session dose a genuine common cause in both models, but IS is treatment-affected (IG → IS): conditioning on it would insert a post-treatment collider between the randomised arm and latent ability (IG → IS ← GA → W). Per the family precedent (no mechanism model adjusts IS; #269) and the #309 sign-off pattern (2026-07-14), the primary models leave IS unadjusted and the reports name shared session dose as a plausible inflator of the slope. A dose-adjusted sensitivity companion is deferred. Both models also carry the standing caveats: latent GA confounding (child intercept is a partial proxy), and — specific to these exposures — taught-vocabulary variation is largely intervention-generated, so the slopes describe covariation within a treated system.

## RW → W: the covariate-exposure decision (#311's first design decision) — implemented via route (b)

#311 offered two routes for phonological memory (RW, `erbto`): (a) register `erbto` as a bounded-count `Measure`, or (b) extend `build_mechanism_model` to take a standardised-covariate exposure. `erbto` is integer-valued (observed range 1–36, all four waves), but **the ERB total's documented test maximum is recorded nowhere in the repo** — registering a denominator guessed from the observed maximum would fabricate a scale property, and the bounded-count logit machinery is exactly wrong if the true maximum differs. Decision: **route (b)**.

Update (2026-07-14, follow-up PR): route (b) is now implemented as `lrp-rli-mech-090` (RW → W). `build_mechanism_model` takes a `mechanism_is_covariate` flag: the exposure is a standardised `prepared.covariates` key entered as `beta_mech * z(exposure)` (linear only — the HSGP curve, its priors and the readiness-threshold post-processing all assume a bounded-count logit input, so the flag requires `linear_mechanism=True`). The pipeline loads the exposure covariate, `require_observed=("erbto",)` drops the mean-imputed rows (imputation-plus-indicator is an _adjuster_ policy, never acceptable for the exposure), and `config.json` records the raw-units anchor `mechanism_exposure_sd_raw` so `beta_mech` (per +1 SD) can be read back in raw ERB points. The mechanism curve is written on the raw score axis (`mech_x`, observed 3–36) rather than the logit axis. The adjustment set is the cleanest of the #311 trio: RW's parents are {A, GA, HS} only, so it conditions on G, A, HS and W_pre — and critically there is **no IG→RW or IS→RW edge**, so the session-dose backdoor LRP88/LRP89 flag does not exist here. If the ERB manual's maximum can be sourced later, route (a) becomes available and is simpler, but route (b) is faithful to what the repo actually knows about the instrument.

## Ids

086/087 are claimed by the open PR #309 (mediation code-route); 083–085 were taken by the #228 sweep (dose-083/084, gc-085). The next free bare-family numbers were therefore 088/089 (and 090 reserved in intent for RW → W).

## Reporting fits (2026-07-14)

All three models fitted at `--config reporting` (6000 draws × 6 chains, target_accept 0.95) and **pass the convergence gate** (r̂ ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.3, 0 divergences): LRP88 max r̂ 1.0025, LRP89 1.0008, LRP90 1.0007; min ESS ≈ 3.9k; BFMI 0.91–0.98; zero divergences throughout. `mechanism_curve.csv` and the rendered `index.html` (items-scale translation) are written for each. Slopes as adjusted associations (`beta_mech`, logit scale): TR → W 0.225, 95% CrI [0.118, 0.332], P(>0) = 1.000 (very strong evidence of a positive association); TE → W 0.213, 95% CrI [0.064, 0.364], P(>0) = 0.998 (very strong); RW → W 0.099 per +1 SD of the ERB total (= +9.12 points; mean 21.5), 95% CrI [−0.017, 0.218], P(>0) = 0.952 (moderate evidence — the band still spans zero). n = 157/157/149 rows across 53/53/52 children (LRP90 drops 13 mean-imputed exposure rows via `require_observed`). Never causal — the child intercept only proxies the time-invariant part of latent general ability, and TR/TE variation is largely intervention-generated.

Render note: the pipeline's `--render` step needs `QUARTO_PYTHON` pointed at the conda env interpreter — Quarto otherwise executes the `.qmd` with the system Python and fails on `import arviz`. The fits themselves are unaffected (all artefacts write before the render step).
