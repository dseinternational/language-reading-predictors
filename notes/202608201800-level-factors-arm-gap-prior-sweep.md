# Level-factors balance-prior (`arm_gap_t1`) sweep — reporting config, 2026-08-20

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

**What ran.** The balance-prior sensitivity recommended by the 2026-08-20 level-factors code review (finding 1, `notes/202608201500-level-factors-code-review.md`): `scripts/level_factors_prior_sensitivity.py --config reporting --axis arm_gap`, one invocation per outcome over the default sweep set (W, L, P, B, N — the #389 review's five). Each cell rebuilds the registered primary's typed run plan with only the `arm_gap_t1` prior scale moved across the grid σ ∈ {0.3 (registered), 0.5, 1.0} and refits under the primaries' own sampling contract (6 chains × 6000 draws, `target_accept` 0.95, nutpie, seed 20260701). All **15 cells converged** on the full all-free-variable criteria (max R-hat ≤ 1.003, min ESS ≈ 4,000–14,000, BFMI clear, zero divergences), and every row is hash-bound to its current `-reporting` primary (`primary_config_sha256` / `primary_trace_sha256`). Results live in `output/statistical_models/level_tau_prior_sensitivity/level_arm_gap_prior_sensitivity.csv` with content-addressed cell traces under `traces/level-reporting/trace_<outcome>_armgap-*.nc`; the output root is untracked, so the full table is reproduced below. This sweep is a sensitivity companion, **not** release-gate evidence — the gate's evidence contract remains the treatment-prior (tau) sweep, and the runner refuses `--attach` on this axis.

**Question.** The reporting fits' power-scaling flags `arm_gap_t1` as "strong prior / weak likelihood" in 8 of 11 fits, and the balance term trades off directly against the released `d_grp_time[t2]` contrast: if the tight `Normal(0, 0.3)` prior were suppressing the t1-imbalance subtraction, widening it should move the causal headline. Does it?

**Answer: no material sensitivity.** The focal `d_grp_time[t2]` posterior (logit scale, mean and 89% equal-tailed interval; `pd` = P(AME > 0); items/risk-difference scale from the same fit's t2 AME):

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

**Reading.**

- **Graded outcomes (W, L, B): the headline is invariant.** Moving the balance prior from the registered 0.3 to a weakly-informative 1.0 shifts `d_grp_time[t2]` by at most 0.012 logits (≲ 0.06 posterior SD), with L drifting very slightly _down_ rather than up — so the joint posterior is not simply leaking the raw gap back in; the cross-wave pooling through the `d_grp_time` priors also acts on `arm_gap_t1`, and the two mechanisms roughly cancel here. The psense "strong prior / weak likelihood" flag on `arm_gap_t1` is real (the balance term itself is prior-shaped) but does **not** propagate materially into the released causal contrast.
- **Floored outcomes (P, N): the registered prior is mildly conservative.** Widening the balance prior lets `arm_gap_t1` track the (negative) adjusted t1 off-floor gap more closely, and the t2 change strengthens accordingly — by ≈ +0.07 logits (P) and ≈ +0.08 logits (N) across the grid, ≈ 0.17–0.19 posterior SD. Direction matches the review's leak analysis (negative t1 gaps ⇒ tight balance prior attenuates benefit). Neither qualitative reading changes: P stays centred on no effect at every scale (pd 0.51 → 0.57), and N stays an uncertain positive (pd 0.66 → 0.72) with the 89% interval spanning zero throughout.
- **Anchor check.** The σ = 0.3 cells refit the registered specification under a different seed and reproduce the primaries' released contrasts (e.g. W +0.346 vs the primary's ≈ +0.34 median), so the sweep is measuring the prior move, not a pipeline difference.

**Consequence for the review's finding 1.** The empirical half of the finding is resolved: the "partial subtraction" concern does not materially move any of the five swept headlines, and for P/N the registered scale errs conservative. The report-prose softening (the subtraction is partial in a small sample) stays correct and in place. The remaining open item is a policy choice, not an empirical one — whether the balance term _should_ carry a wider prior on principle (jointly with the DiD family, which shares the `arm_gap_t1` idiom); this sweep says the answer changes little for the level family's released numbers, so there is no urgency. The three flagged outcomes not in the default sweep set (F, TR, TE) can be added with `--outcomes F TR TE --axis arm_gap` if the same reassurance is wanted there; the runner now supports all eleven.

## Related

- `notes/202608201500-level-factors-code-review.md` (finding 1 and the fix batch), `notes/202608191900-level-factors-t1-gap-reference.md` (#552), #389 / #482 (the robustness gate whose evidence contract this deliberately does not touch).
