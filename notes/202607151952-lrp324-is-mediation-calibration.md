> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Intervention-session calibration of the mediation NIE sensitivity sweep (#324)

## Decision

The existing #289 sensitivity sweep stays unchanged: it subtracts a non-negative bias magnitude $\delta$ from the fitted mediator→outcome coefficient in the direction of the null, then re-runs the g-formula. The named intervention-session (`IS` / `attend`) calibration locates a scenario on that one-dimensional surface with the single-omitted-variable approximation $\delta_{IS} \approx |\beta_{IS\to M_{std}}\beta_{IS\to Y_{logit}}|$. Both dose associations are first put on the mediation sample's one-standard-deviation phase-1 session scale; the mediator association is additionally divided by the fitted mediator logit's standard deviation because `b_M` acts on a standardised mediator. This is a reporting calculation over existing model/data outputs, not a new likelihood, prior or registered model.

The primary point uses the gate-passed period-1 slopes from `lrp-rli-dose-083` for `IS→L` and, where one exists, the matching gate-passed outcome dose-response fit (`lrp-rli-dose-077` for W; `lrp-rli-dose-084` for B). No registered off-floor N dose-response fit exists, so MED-086 combines the fitted `IS→L` slope with a descriptive phase-1 Bernoulli-logit `IS→off-floor N` association adjusted for arm and age. Every target also computes an observed-data cross-check on its exact mediation sample: `IS→L` adjusts for arm, age and L baseline; `IS→W/B` adjusts for arm, age and the outcome baseline; off-floor N follows the suite floor rule and omits its near-degenerate baseline.

The reported 90% scenario band is the envelope of the separate marginal slope endpoints plus the observed-data cross-check. It is deliberately not called a credible or confidence interval: the slopes come from separate fits and their joint dependence is not represented. Treating the whole `IS→outcome` dose slope as mediator–outcome confounding is conservative only in a limited sense, because that slope can contain a genuine `IS→L→outcome` component as well as the recanting-witness backdoor; latent general ability also confounds the dose associations. The calibration therefore asks whether an IS-sized bias is compatible with the NIE tipping point. It does not identify a causal session effect, validate sequential ignorability or repair the natural-effect non-identification.

## Off-floor scale

MED-086's sensitivity parameter $\delta$ is necessarily a shift in the Bernoulli outcome's log-odds coefficient per one-standard-deviation mediator. There is no constant logit-to-risk-difference conversion because the change depends on baseline risk and the covariate distribution. The existing sweep already performs the correct conversion: each $\delta$ is passed through the Bernoulli g-formula, which returns the NIE on the off-floor risk-difference scale. `mediation_is_calibration.csv` interpolates that computed curve at the IS point and reports the resulting risk difference and 95% interval.

## Scope and artefacts

The required targets are MED-086 (L→off-floor N) and MED-087 (L→B). The same generic implementation also covers MED-059 (L→W), because its identical `L ← IS → W` structure has both dose-response sources already fitted. MED-092 is not included: its period-stacked on-intervention exposure and ignorability frame do not share the phase-1 randomised mediation calibration scale. The two-mediator MED-064/066 extension remains with #335, where each mediator leg needs its own sweep before calibration.

Each supported fit writes `mediation_is_calibration.csv`, containing the rescaled source slopes, observed cross-check, point and band on the $\delta$ surface, tipping comparison, response-scale NIE at the IS point, verdict and one computed sentence. `scripts/regenerate_mediation_calibration.py` backfills the file from existing sensitivity/dose artefacts and the source data without posterior sampling. A missing or gate-failed dose source produces an explicit `not_available` row rather than silently using an unconverged estimate or aborting the mediation fit.
