> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne within-child joint companion: specification and promotion checks

## Decision

Register `lrp-rlm-jc-002` as the C2(ii) companion to `lrp-rlm-jc-001` from #409. It fits the confirmed-ceiling measures `basread`, `bpvs` and `basdig` jointly and separates their stable between-child correlation from wave-specific within-child co-movement.

## Analysis frame

Use the jointly complete waves 1–3 core only: 71 children and exactly three rows per child. Exclude extension waves 4–5 because they would weight better-observed children more heavily and mix the common three-group period with an attrition-selected tail whose fifth wave contains only the Down syndrome group. The matched between- and within-child estimates therefore use the same balanced rows.

## Model

Retain the RLMJC01 group-by-wave population means, group-specific stable child scales and LKJ-correlated stable child offsets. Add one LKJ-correlated logistic-normal deviation for each child-wave row. Double-centre these deviations so they average to zero within child and within group-by-wave cell; this separates the new term from the stable child offsets and fitted population cells. Use a Binomial count likelihood because this residual supplies the extra-Binomial variance. Report the within-child correlation on the latent logit scale, the matched stable-level correlation and their posterior difference.

Both correlation matrices are shared across reading groups. Group-specific matrices would estimate six correlation parameters for each layer from group sizes of 22, 31 and 18 children and only three repeated waves, which is not supported here.

## Interpretation

The within-child question is symmetric: when a child is above their own stable level in one skill at a wave, are they also above their stable level in another? It does not identify which skill changes first, whether one causes the other, or whether a common time-varying factor produced both departures. `readgrp` remains an observational cohort factor and `causal_status="none"` applies throughout.

## Development probe and identification decision

The first development fit retained RLMJC01's Beta-Binomial concentration and added the logistic-normal residual. It reproduced the known failure mode from the dependence-aware joint-model work: the within scales collapsed to approximately 0.04–0.07 logits and every correlation retained an approximately −0.7 to +0.7 interval. Both variance mechanisms were competing for the same row-level variation, so that formulation was rejected rather than promoted.

The registered companion follows the repository's `joint_mechanism` precedent: use Binomial counts and let the correlated logistic-normal residual supply the extra-Binomial variance. The registered prior is `sigma_within ~ HalfNormal(0.5)` on the logit scale. Before promotion beyond a draft model, fit a `HalfNormal(1.0)` sensitivity and require the residual-scale identifiability decisions to agree; for any pair that is resolvable under both priors, require the correlation medians to retain direction and their 89% intervals to overlap materially.

## Correlation identifiability rule

A correlation is not substantively interpretable when either latent residual standard deviation collapses to zero. Mark a measure's scale as resolvable only when the posterior probability that `sigma_within > 0.05` logits is at least 0.95, and interpret a pair only when both scales resolve. The 0.05-logit boundary is deliberately small: at the most sensitive probability of 0.5 it moves the expected score by at most 1.25 percentage points, approximately 1.1 `basread` items or 0.4 `bpvs`/`basdig` items. It is a practical identifiability threshold, not a minimum scientifically important effect.

## Test-tier fit evidence, 2026-08-11

The four-chain `test` fit (2,000 tuning + 2,000 retained draws per chain, target acceptance 0.90) passed the automatic computational gate: zero divergences, maximum R-hat 1.008 and minimum bulk/tail effective sample size 494. The residual-scale medians were 0.315 logits for `basread`, 0.032 for `bpvs` and 0.043 for `basdig`; posterior probabilities above the 0.05-logit boundary were 1.000, 0.291 and 0.428 respectively. Only word reading resolved, so no measure pair had two estimable residual scales and none of the wide within-correlation posteriors is interpreted. This is a test-tier diagnostic result, not a publication result. The release decision also remained `inputs_unresolved` because the 96-versus-97 source-provenance discrepancy is open.

## Promotion checks

- The balanced-frame assertion holds: 71 children, 213 rows and three waves per child.
- Simulation recovery distinguishes the signs of the between- and within-child correlations under the registered priors.
- The reporting-tier fit passes R-hat, effective-sample-size, BFMI and zero-divergence gates.
- Posterior-predictive checks are acceptable for all three bounded measures.
- The wider within-scale prior does not materially reverse or sharpen the within-child conclusions.
- Publication remains withheld while the Byrne source-provenance discrepancy tracked in #409 is unresolved; clean computation does not override that input gate.
