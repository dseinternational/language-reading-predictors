> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# #391 finding 5: the cross-design triangulation moves to the canonical items-scale marginals

**Date:** 2026-08-07. **Issue:** #391 (final remaining item). **Decision provenance:** the 2026-07-22 decision comment, finding 5 — deliberately sequenced after the findings 2+3 respecification (#490) settled what the canonical gain-factor estimand is.

## What changed

`triangulation_consistency.csv` (from `scripts/compare_statistical_models.py`) previously compared three **logit coefficients**: the ITT `tau`, the crossover `tau_t2` and the gain-factor `beta_trt` read raw from `factor_summary.csv`. The review's finding named the wrong-column defect (`beta_trt` is not the gain-factor headline estimand); the decision named the deeper one: the logit link is **non-collapsible**, so the gain-factor conditional coefficient — conditioned on the child intercept, own baseline, ability, upstream skills and exogenous confounders — is systematically larger in magnitude than the marginal ITT `tau` even under an identical truth, identical data and perfect randomisation. A logit-scale magnitude comparison across these designs is therefore apples-to-oranges no matter which gain-factor column it reads, and no labelling fixes it.

The triangulation now consumes each family's **canonical items-scale average marginal effect over its randomised comparison**, the same artefacts the model reports headline:

| Design          | Estimand               | Source artefact                                                                                                               | Averaging population                            |
| --------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| ITT             | `itt_t2_ame_items`     | `tau_summary.csv` `tau_prob_*` × the measure's item count (an exact per-draw linear transform of every quantile)              | t2 available-case children (both arms)          |
| Crossover (DiD) | `did_t2_arm_gap_items` | `did_summary.csv` `tau_t2_items_*` (wave-standardised pushforward)                                                            | t2 wave rows (both arms)                        |
| Gain factors    | `gf_period1_ame_items` | `treatment_marginal.csv` `trt_items_*` (interaction-free since #490, so its direction coincides with `beta_trt` draw-by-draw) | period-1 randomised transition rows (both arms) |

Every design row carries explicit `scale` and `population` columns — the marginals share the items scale but **not** the averaging population (t2 levels vs period-1 transition), which is stated rather than implied and is why the flags remain qualitative. The direction/overlap verdict is computed on this scale over converged designs; a legacy artefact with no items emission (the old DiD `delta`, or a gain-factor fit predating `treatment_marginal.csv`) stays readable but cannot enter the verdict pool (`n_ame_verdict_pool` records the pool size, and the verdict is NA rather than a logit/items hybrid when fewer than two remain). The raw logit coefficients survive only as clearly-labelled `*_logit_*` **appendix columns** — the decision's "clearly-caveated appendix, if at all" — with the non-collapsibility rationale in the builder's docstring. No separate logit forest is produced for the triangulation (none existed; the `tau_forest.png` ITT-vs-joint overlay compares like-for-like conditional taus within one family and is untouched).

## Regenerated results (reporting config)

Six of seven outcomes triangulate consistently on the items scale (direction agreement + mutual interval overlap): W, E, L, TR, TE, F. Word reading shows the intended effect of the scale move most clearly — the three marginals are now directly comparable: ITT +2.37, crossover +2.22, gain-factor +2.63 items.

R is flagged `direction_agree = False`, and it is the documented null-boundary artefact, not a contradiction: ITT +0.23 items [−3.75, +4.26] (P(>0) = 0.54), crossover −0.06 [−5.11, +5.00] (0.49), gain-factor −1.78 [−5.63, +1.96] (0.22). All three are essentially null with heavily overlapping wide intervals (`intervals_overlap = True`); the direction flag trips because the medians straddle zero. The builder's docstring says exactly this — a `direction_agree = False` with wide overlapping intervals is a null-result artefact — and the receptive-vocabulary reading across all three designs is "no resolved effect", consistent with the family reports.

## Test coverage

`tests/test_compare_triangulation.py` was rewritten onto the new schema. Beyond the ported verdict/gate/catalogue tests, three new pins hold the finding-5 acceptance criteria: the gain-factor verdict follows the canonical marginal even when a fixture makes `beta_trt` disagree with it in sign (the coefficient appearing only in the appendix columns); the `scale`/`population` columns are explicit per design; and a logit-only legacy design is excluded from the AME verdict pool rather than mixed across scales.
