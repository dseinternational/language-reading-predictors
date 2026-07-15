<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Waitlist-crossover redesign: arm-by-wave estimands

- **Date:** 2026-07-15
- **Status:** implementation decision
- **Issue:** #340

## Decision

Replace the binary waitlist-crossover model's single common-current-treatment coefficient with a three-wave arm-by-wave parameterisation for the t1, t2 and t3 outcome levels. The model has an observed-t1 logit intercept anchor, t2-vs-t1 and t3-vs-t1 period offsets, and a separate immediate-minus-waitlist gap at each wave. Report four distinct quantities rather than calling one coefficient a causal “DiD treatment effect”:

1. `arm_gap_t1`, the immediate-minus-waitlist contrast at t1. This is a pre-treatment balance quantity, not an intervention effect.
2. `tau_t2`, the immediate-minus-waitlist contrast at t2. This is the clean randomised contrast and may be interpreted causally under the trial's usual assumptions.
3. `arm_gap_t3`, the immediate-minus-waitlist contrast at t3. Both arms have crossed into treatment by then, with different treatment histories, so this is a post-crossover association.
4. `delta_crossover = tau_t2 - arm_gap_t3`, the amount by which the immediate-arm advantage closes from t2 to t3. This describes waitlist catch-up after crossover; it is not a second randomised treatment effect.

The bounded outcomes retain their Beta-Binomial likelihood, and the heavily floored outcomes retain a Bernoulli likelihood on period-end off-floor status. For the latter, all contrasts concern off-floor **prevalence**, not the transition from floor to off floor.

## Why the earlier specification was rejected

The earlier current-treatment coding used the same treatment coefficient for the immediate arm at t2, the immediate arm at t3 and the waitlist arm at t3. It therefore forced the adjusted arms to be equal at t3 and forced the randomised t2 arm difference to equal the crossover change contrast. Those restrictions were not consequences of randomisation and required implausibly strong assumptions about no cumulative exposure, carryover, block-content or diminishing-return effects. Including t1 as a modelled outcome now also exposes any pre-randomisation arm imbalance instead of absorbing it into a restricted post-only intercept.

The earlier design documentation also described each child as their own control. A child random intercept does not perform exact fixed-effect differencing: it partially pools stable between-child differences under a distributional model. Only waitlist children change treatment status, while immediate-arm children are treated in both analysed periods.

## Period-start outcome

Do not condition on each period's start score. The t1 score is pre-randomisation, but the t2 score used as the P2 baseline is already affected by treatment in the immediate arm. A common period-start coefficient would therefore condition one arm on a treatment-induced intermediate, block part of the earlier treatment effect and change the target away from a total-effect interpretation. Removing that term is appropriate, but it means the model is a longitudinal arm-by-wave model of period-end levels rather than a conditional-gain model.

The binary models instead include t1 as the first response wave. The dose models retain two transition rows but use the single pre-randomisation t1 outcome as a common precision covariate for both rows. Neither design may silently reintroduce the treatment-affected t2 outcome as the P2 baseline.

## Dose variants

Session dose remains observational. The dose variants retain the two P1/P2 transition rows and include randomised arm, period, current treatment, the pre-randomisation t1 outcome and t1 age. Sessions are centred among treated rows and enter only through a treated × centred-dose term, separating the current-treatment cell contrast from residual variation in session intensity. Comparisons of pooled and period-varying slopes answer only whether one observational dose parameterisation predicts these rows better.

## Heterogeneity variant

LRPDID13 remains exploratory. It adds a child-specific deviation only to each observed waitlist child's t3 outcome and defines the corresponding catch-up quantity as `delta_crossover_i = delta_crossover + v_delta_i`; it does not fit an immediate-arm random treatment slope. Each deviation is informed by one post-crossover observation, so its variance may absorb heterogeneous maturation, treatment history, period shocks and measurement noise as well as response heterogeneity. It must not be reported as a clean test that “true non-responders” exist. The report gives the fitted waitlist-sample average on the logit scale and deliberately omits a scalar item/probability translation that would fail to integrate the child-specific t3 deviations.

## Reporting and validation requirements

- Show the six arm-by-wave fitted cells or their standardised margins before derived contrasts, plus cell-stratified posterior-predictive checks of means and zero rates.
- Label only `tau_t2` causal; label `arm_gap_t1` as a balance quantity and `arm_gap_t3`, `delta_crossover`, dose and heterogeneity quantities associational or exploratory as appropriate.
- State that the child random intercept partially pools stable heterogeneity rather than making every child their own control.
- Describe the no-period-start-outcome decision and the resulting level-based estimand.
- For floored outcomes, use “off-floor prevalence” or “probability of being off floor”, never “coming off the floor”.
- Exclude convergence-gate failures from interpretive LOO comparisons and verify pointwise observation identities and order, not just row counts.
- Maintain exact algebraic tests, deterministic bounded-count recovery and misspecification simulations, reporting tests for every derived contrast, fitted-row metadata tests, cell-stratified posterior-predictive tests and LRPDID13 catch-up marginalisation tests.
- Refit all affected models before interpreting or publishing revised estimates; prior reports from the restricted current-treatment model are stale.

## Consequence for the evidence hierarchy

The dedicated ITT models remain the primary causal analyses. The waitlist-crossover family now decomposes what the t1, t2 and t3 data say without imposing a single treatment effect across incompatible treatment histories. Agreement of its randomised t2 contrast with the ITT is a software and parameterisation check on shared data, not independent causal replication; post-crossover convergence is descriptive triangulation.
