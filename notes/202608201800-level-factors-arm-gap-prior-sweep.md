# Level-factors balance-prior (`arm_gap_t1`) sweep — reporting config, 2026-08-20

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

**What ran.** The balance-prior sensitivity recommended by the 2026-08-20 level-factors code review (finding 1, `notes/202608201500-level-factors-code-review.md`): `scripts/level_factors_prior_sensitivity.py --config reporting --axis arm_gap`, one invocation per outcome, in three same-day batches — the default sweep set (W, L, P, B, N — the #389 review's five), then the three remaining outcomes whose reporting fits also flag `arm_gap_t1` as prior-dominated (F, TR, TE), then the three psense-clear outcomes (R, E, T) for completeness — so **all eleven registered LF outcomes** are covered. Each cell rebuilds the registered primary's typed run plan with only the `arm_gap_t1` prior scale moved across the grid σ ∈ {0.3 (registered), 0.5, 1.0} and refits under the primaries' own sampling contract (6 chains × 6000 draws, `target_accept` 0.95, nutpie, seed 20260701). All **33 cells converged** on the full all-free-variable criteria (max R-hat ≤ 1.003, min ESS ≈ 4,000–14,000, BFMI clear, zero divergences), and every row is hash-bound to its current `-reporting` primary (`primary_config_sha256` / `primary_trace_sha256`). Results live in `output/statistical_models/level_tau_prior_sensitivity/level_arm_gap_prior_sensitivity.csv` with content-addressed cell traces under `traces/level-reporting/trace_<outcome>_armgap-*.nc`; the output root is untracked, so the full table is reproduced below. This sweep is a sensitivity companion, **not** release-gate evidence — the gate's evidence contract remains the treatment-prior (tau) sweep, and the runner refuses `--attach` on this axis.

**Question.** The reporting fits' power-scaling flags `arm_gap_t1` as "strong prior / weak likelihood" in 8 of 11 fits, and the balance term trades off directly against the released `d_grp_time[t2]` contrast: if the tight `Normal(0, 0.3)` prior were suppressing the t1-imbalance subtraction, widening it should move the causal headline. Does it?

**Answer: no material sensitivity — and where any exists, the registered scale is the conservative side.** The focal `d_grp_time[t2]` posterior (logit scale, mean and 89% equal-tailed interval; `pd` = P(AME > 0); items/risk-difference scale from the same fit's t2 AME):

| Outcome                                          |   σ | `d_grp_time[t2]` mean [89%] |    pd | AME (outcome scale)   |
| ------------------------------------------------ | --: | --------------------------- | ----: | --------------------- |
| W (word reading, 79 items)                       | 0.3 | +0.346 [+0.036, +0.662]     | 0.964 | +2.29 [+0.24, +4.35]  |
|                                                  | 0.5 | +0.345 [+0.035, +0.656]     | 0.962 | +2.28 [+0.23, +4.32]  |
|                                                  | 1.0 | +0.340 [+0.021, +0.658]     | 0.955 | +2.25 [+0.14, +4.33]  |
| L (letter sounds, 32 items)                      | 0.3 | +0.477 [+0.135, +0.817]     | 0.987 | +2.83 [+0.80, +4.85]  |
|                                                  | 0.5 | +0.472 [+0.123, +0.823]     | 0.985 | +2.80 [+0.72, +4.89]  |
|                                                  | 1.0 | +0.465 [+0.116, +0.813]     | 0.983 | +2.75 [+0.68, +4.82]  |
| B (phoneme blending, 10 items)                   | 0.3 | +0.314 [−0.069, +0.694]     | 0.904 | +0.65 [−0.14, +1.42]  |
|                                                  | 0.5 | +0.319 [−0.068, +0.708]     | 0.908 | +0.66 [−0.14, +1.45]  |
|                                                  | 1.0 | +0.320 [−0.079, +0.710]     | 0.903 | +0.66 [−0.16, +1.46]  |
| P (phonetic spelling, off-floor risk difference) | 0.3 | +0.010 [−0.666, +0.682]     | 0.510 | +0.1 pp [−8.4, +8.6]  |
|                                                  | 0.5 | +0.038 [−0.637, +0.726]     | 0.534 | +0.5 pp [−8.0, +9.1]  |
|                                                  | 1.0 | +0.082 [−0.596, +0.762]     | 0.574 | +1.0 pp [−7.5, +9.6]  |
| N (nonword reading, off-floor risk difference)   | 0.3 | +0.163 [−0.475, +0.805]     | 0.660 | +2.9 pp [−8.4, +14.3] |
|                                                  | 0.5 | +0.198 [−0.447, +0.837]     | 0.688 | +3.5 pp [−7.9, +15.0] |
|                                                  | 1.0 | +0.238 [−0.419, +0.889]     | 0.721 | +4.2 pp [−7.4, +15.8] |
| F (CELF basic concepts, 18 items)                | 0.3 | +0.211 [−0.072, +0.492]     | 0.886 | +0.80 [−0.27, +1.86]  |
|                                                  | 0.5 | +0.219 [−0.064, +0.501]     | 0.891 | +0.83 [−0.24, +1.89]  |
|                                                  | 1.0 | +0.226 [−0.057, +0.511]     | 0.899 | +0.85 [−0.21, +1.92]  |
| TR (taught receptive vocabulary, 24 items)       | 0.3 | +0.219 [−0.045, +0.483]     | 0.909 | +1.17 [−0.24, +2.57]  |
|                                                  | 0.5 | +0.233 [−0.035, +0.500]     | 0.919 | +1.24 [−0.18, +2.66]  |
|                                                  | 1.0 | +0.235 [−0.035, +0.503]     | 0.918 | +1.25 [−0.19, +2.68]  |
| TE (taught expressive vocabulary, 24 items)      | 0.3 | +0.278 [−0.012, +0.569]     | 0.938 | +1.30 [−0.06, +2.65]  |
|                                                  | 0.5 | +0.304 [+0.007, +0.598]     | 0.949 | +1.42 [+0.03, +2.78]  |
|                                                  | 1.0 | +0.315 [+0.017, +0.614]     | 0.955 | +1.47 [+0.08, +2.85]  |
| R (receptive vocabulary, 170 items)              | 0.3 | +0.007 [−0.144, +0.156]     | 0.531 | +0.19 [−4.24, +4.55]  |
|                                                  | 0.5 | +0.012 [−0.138, +0.162]     | 0.548 | +0.34 [−4.05, +4.75]  |
|                                                  | 1.0 | +0.015 [−0.137, +0.166]     | 0.563 | +0.45 [−4.04, +4.86]  |
| E (expressive vocabulary, 170 items)             | 0.3 | +0.006 [−0.149, +0.161]     | 0.525 | +0.16 [−3.88, +4.20]  |
|                                                  | 0.5 | +0.011 [−0.145, +0.167]     | 0.546 | +0.29 [−3.76, +4.33]  |
|                                                  | 1.0 | +0.011 [−0.146, +0.167]     | 0.544 | +0.28 [−3.81, +4.34]  |
| T (TROG receptive grammar, 32 items)             | 0.3 | +0.084 [−0.137, +0.307]     | 0.724 | +0.63 [−1.04, +2.32]  |
|                                                  | 0.5 | +0.091 [−0.136, +0.315]     | 0.742 | +0.69 [−1.03, +2.38]  |
|                                                  | 1.0 | +0.091 [−0.131, +0.313]     | 0.744 | +0.69 [−0.99, +2.36]  |

**Reading.**

- **Graded outcomes (W, L, B): the headline is invariant.** Moving the balance prior from the registered 0.3 to a weakly-informative 1.0 shifts `d_grp_time[t2]` by at most 0.012 logits (≲ 0.06 posterior SD), with L drifting very slightly _down_ rather than up — so the joint posterior is not simply leaking the raw gap back in; the cross-wave pooling through the `d_grp_time` priors also acts on `arm_gap_t1`, and the two mechanisms roughly cancel here. The psense "strong prior / weak likelihood" flag on `arm_gap_t1` is real (the balance term itself is prior-shaped) but does **not** propagate materially into the released causal contrast.
- **Floored outcomes (P, N): the registered prior is mildly conservative.** Widening the balance prior lets `arm_gap_t1` track the (negative) adjusted t1 off-floor gap more closely, and the t2 change strengthens accordingly — by ≈ +0.07 logits (P) and ≈ +0.08 logits (N) across the grid, ≈ 0.17–0.19 posterior SD. Direction matches the review's leak analysis (negative t1 gaps ⇒ tight balance prior attenuates benefit). Neither qualitative reading changes: P stays centred on no effect at every scale (pd 0.51 → 0.57), and N stays an uncertain positive (pd 0.66 → 0.72) with the 89% interval spanning zero throughout.
- **F, TR: same mild conservatism, no reading change.** Both strengthen by ≈ +0.015 logits (≈ 0.09–0.10 posterior SD) across the grid, pd moving 0.886 → 0.899 (F) and 0.909 → 0.918 (TR); the 89% intervals span zero at every scale and the direction reading is unchanged.
- **TE is the largest mover, and it sits on the interval boundary.** Taught expressive vocabulary strengthens by ≈ +0.036 logits across the grid — ≈ 0.20 posterior SD, the biggest shift of the eight — and the 89% lower edge crosses zero as the balance prior widens: [−0.01, +0.57] at the registered 0.3 versus [+0.01, +0.60] at 0.5 and [+0.02, +0.61] at 1.0 (items scale +1.30 [−0.06, +2.65] → +1.47 [+0.08, +2.85]). The house evidence language, which reads direction probability rather than an interval-zero rule, is stable (pd 0.938 → 0.955, the same round-odds band) — but a reader applying an interval-excludes-zero rule would call the registered fit equivocal and the wider-prior fits positive, so TE is the one outcome where the balance-prior choice touches a presentational boundary. The registered scale is the _conservative_ side of that boundary.
- **R, E, T (the psense-clear fits): no sensitivity, as their power-scaling predicted.** Shifts of ≤ 0.01 logits (≈ 0.05–0.09 posterior SD) across the grid; R and E stay centred on no effect (pd 0.53–0.56) and T stays an uncertain positive (pd 0.72–0.74) at every scale. The psense diagnosis and the sweep agree: where the balance term is not prior-dominated, the balance-prior scale does not matter at all.
- **Anchor check.** The σ = 0.3 cells refit the registered specification under a different seed and reproduce the primaries' released contrasts (e.g. W +0.346 vs the primary's ≈ +0.34 median), so the sweep is measuring the prior move, not a pipeline difference.

**Consequence for the review's finding 1.** The empirical half of the finding is resolved across the **whole family** — all eleven registered LF outcomes are swept. Wherever the balance prior matters at all, the registered scale errs **conservative** (the direction the review's leak analysis predicted, given the uniformly negative adjusted t1 gaps), the shifts are at most ≈ 0.2 posterior SD, and the psense-clear fits (R, E, T) are flat, so the power-scaling diagnosis and the sweep agree everywhere. No direction reading changes; the one presentational sensitivity is TE, where the 89% interval's zero-crossing flips across the grid while the round-odds direction label does not. The report-prose softening (the subtraction is partial in a small sample) stays correct and in place. The remaining open item is a policy choice, not an empirical one — whether the balance term _should_ carry a wider prior on principle (jointly with the DiD family, which shares the `arm_gap_t1` idiom). This sweep says the level family's released numbers change little either way and the current choice is the cautious side; TE is the outcome to look at again if that policy discussion is taken up.

## Related

- `notes/202608201500-level-factors-code-review.md` (finding 1 and the fix batch), `notes/202608191900-level-factors-t1-gap-reference.md` (#552), #389 / #482 (the robustness gate whose evidence contract this deliberately does not touch).
