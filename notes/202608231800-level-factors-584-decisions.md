> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Level-factors #584: the four decisions, 2026-08-23

Frank settled the four decisions #584 asks for on 2026-08-23, after being walked through the evidence reproduced from the eleven stored `-reporting` traces. This note records what was decided, the evidence behind each choice, and what implementing it requires. The decision-free repairs are separate and already landed (`notes/202608231600-level-factors-584-decision-free-fixes.md`, PR #593).

## Decision 1 — the natural-scale estimand: the arm-free standardised average marginal effect

**Decided.** The items / risk card becomes the average, over the fitted t2 children **each evaluated at their own arm-free profile**, of the effect of the randomised t2 change in the adjusted arm gap.

Concretely: net the _whole_ group contribution (`arm_gap_t1` + `d_grp_time[t2]` + `gamma_grp_ability x ability`) out of every t2 row's linear predictor, then add back only the focal contrast and average the probability difference. The published transform removes the focal contrast and the moderation increment but leaves `arm_gap_t1` in the immediate arm's operating point and never adds it to the waiting arm's — a hybrid over observed-arm operating points that is not a nameable estimand.

**Why this one, and not the alternatives.** Four candidates were compared on the same stored draws:

| Outcome | Published card | Arm-free standardised | Response-scale DiD | Full t2 contrast |
| ------- | -------------: | --------------------: | -----------------: | ---------------: |
| W       |          +2.30 |             **+2.27** |              +2.37 |            +2.53 |
| R       |          +0.23 |             **+0.24** |              −0.08 |            −4.50 |
| E       |          +0.14 |             **+0.14** |              −0.20 |            −3.06 |
| L       |          +2.84 |             **+2.86** |              +2.83 |            +3.25 |
| TE      |          +1.30 |             **+1.34** |              +1.08 |            +0.31 |

- The arm-free standardisation is a **named estimand** with an explicit population (the fitted t2 children), an explicit random-effect convention (each child's own fitted intercept) and an explicit treatment of effect modification (the moderation increment held at centred ability, since `gamma_grp_ability` is time-invariant and identified mostly off the non-randomised waves).
- It **keeps the card sign-coherent with the coefficient the report flags causal**. Both the published transform and the arm-free one add the same focal draw to every row, so `expit(eta0 + d) - expit(eta0)` carries the sign of `d` in every row and `P(card > 0)` equals `P(d > 0)` to three decimals. A marginal response-scale difference-in-differences is not a per-draw monotone transform of `d`: it disagrees in about 5% of draws and moves the direction probability across 0.5 for both vocabulary outcomes (R 0.535 to 0.488, E 0.522 to 0.469), so adopting it would require the direction probability, the ROPE and the release gate's focal statistics all to be recomputed from the new functional rather than from the logit draws.
- It **matches the ITT and gain families' AME idiom** (`_itt_ame_draws`: net the treatment term out of each row, add it back, average), so the three families' cards remain the same kind of quantity.
- It **moves no published number by more than 0.04 items** and no direction probability at all, so the change is a definitional repair, not a results correction.

The full t2 arm contrast stays where it is — a levels-view row — because it carries the covariate-adjusted chance imbalance present before randomisation and answers a different question.

**Scope note.** The change bites only under the t1-centred parameterisation (#552). Under the `arm_gap_reference="free"` comparator the focal `b_grp_time[1]` _is_ the whole t2 arm gap, so netting the full group contribution is what the current code already does there.

## Decision 2 — LF-006 gets a required guessing-floor companion

**Decided.** Add a `score_mean_link` setting to the level family and register a paired phoneme-blending companion (proposed `lrp-rli-lf-106`), gated for release as a pair in the way `lrp-rli-itt-008` / `lrp-rli-itt-108` already are.

**Evidence.** The ordinary Beta-Binomial inverse-logit mean permits blending expectations below the known three-choice chance floor, and the stored LF-006 posterior uses that room: 24 of 215 rows have posterior-mean expected proportions below one third, 13.7% of all row-by-draw expected proportions are below it, and at t2 it is 8 of 54 row means and 16.0% of the mass. The ordinary card is about +0.644 items on a ten-item test; mechanically applying `mu = 1/3 + (2/3) * inverse_logit(eta)` to the same latent draws gives about +0.429, which is a diagnostic of the sensitivity, not a substitute for the refit.

METHODS' wording — "any headline B interpretation" — is not family-specific, and the score-mean-link machinery (`likelihood.beta_binomial_from_score_mean_link`, `apply_score_mean_link`) is already generic. Only the factory validation and `release._blending_pair_release_failures` are hard-scoped to `kind == "itt"`.

**Known gap carried forward.** `lrp-rli-gf-006` has the same defect and is not covered by this decision. It needs its own issue; the release policy for B should end up uniform across every family that publishes it.

## Decision 3 — the four-wave fit stays the model of record, with a t1/t2-only comparator

**Decided.** Keep the four-wave levels model as the model of record and add a t1/t2-only comparator (proposed `lrp-rli-lf-201`–`211`), reporting its difference from the four-wave estimate.

**Evidence.** `b_grp_time[t] = arm_gap_t1 + d_grp_time[t]`, so the post-crossover likelihood informs the shared `arm_gap_t1` and `d_grp_time[t2]` trades off against that anchor; the child random intercept, the dispersion and the single time-invariant `gamma_grp_ability` also pool across all four waves. Posterior correlations between `arm_gap_t1` and `d_grp_time[t2]` run from −0.07 to −0.44 across the eleven fits. The coefficient is randomisation-anchored but longitudinal-model-dependent, and "the clean randomised contrast" overstates the separation.

**Why not decouple the later waves structurally.** Giving t3/t4 free gaps would cut one borrowing path and leave the others — the child intercept and `kappa` still pool — so it buys a partial fix at the cost of the levels view's whole purpose, which is to show all four waves on one scale. A two-wave comparator isolates the randomised window completely, and no existing family provides one: the `did` family fits t1–t3 with per-wave gaps.

## Decision 4 — the dispersion and child-SD priors change for this family, with the current-prior fits kept as comparators

**Decided.** Adopt the `1/sqrt(kappa)` dispersion parameterisation for the level family, with its scale re-derived so it stays calibration-preserving at RLI denominators up to 170, and widen `sigma_child` for the level family. Back up the current eleven reporting fits and report the before/after.

**Evidence.**

| Parameter     | Prior                        | Level posteriors   | Power scaling       |
| ------------- | ---------------------------- | ------------------ | ------------------- |
| `kappa`       | HalfNormal(50), 99th pct 129 | 170 (R), 198 (E)   | flagged in 8 of 9   |
| `sigma_child` | HalfNormal(0.5), 99th pct    | 1.39 (W), 1.67 (P) | flagged in 11 of 11 |

`inv_sqrt_kappa_prior`'s own docstring gives the argument: a HalfNormal on the concentration cannot reach the near-Binomial limit `kappa >> n`, which for a bounded count is the ordinary hypothesis "no extra-Binomial dispersion beyond the child random intercept". It scoped itself to the RLM historical families and deferred the high-denominator RLI outcomes. This is the evidence that was missing for them, and it is the same defect in a sharper form: the RLM posteriors piled against the prior's ceiling, whereas the level family's vocabulary posteriors sit _past_ the prior's 99th percentile — the data want less over-dispersion than the prior permits.

`sigma_child` is the clearer misfit and the higher priority. A levels model has **no own-baseline term**, so its child intercept must absorb the entire between-child spread in level, where a gain or ANCOVA model conditions that spread away. HalfNormal(0.5) is a gain-model scale, and every one of the eleven posteriors exceeds its 99th percentile. The new scale must be chosen against the prior-predictive rather than fitted to the posteriors.

Exploratory wider-prior refits during the review did not change the focal conclusions but did change posterior dispersion and uncertainty, so the comparators matter: the current-prior artefacts are the before, and both go in the reports.

## What implementing this takes

Ordered, because the estimand must be settled before anything is regenerated:

1. **Decision 1** — one functional in `reporting.level_t2_marginal_effect`, reused by the posterior summary, ROPE, prior pushforward, plot and key findings; run-plan estimand prose, key-findings wording and the `_results_factors` "what the causal effect holds fixed" block; synthetic tests where logit DiD, probability DiD and the t2 arm difference deliberately differ. Can be backfilled over the stored traces without a refit.
2. **Decision 2** — `score_mean_link` through settings, plan, factory and reporting; register LF-106 and its report; extend `BLENDING_LINK_MODELS` and relax the ITT-only guard in `release._blending_pair_release_failures`; open the `gf-006` gap as its own issue.
3. **Decision 3** — a two-wave setting on the level plan and eleven registered comparators, plus the cross-fit comparison the reports quote.
4. **Decision 4** — the re-derived `1/sqrt(kappa)` scale (a calibration exercise against the RLI denominators, not a copied constant), a level-family `sigma_child` scale justified on the prior-predictive, and registered sensitivity axes for both.
5. **The refit** — about 23 reporting fits (eleven re-prior'd primaries, LF-106, eleven two-wave comparators), preceded by backing up the current directories, then regenerating convergence, PPC, LOO, sensitivity, release, key-finding and report artefacts, and re-reading the cross-family syntheses from the regenerated artefacts.
