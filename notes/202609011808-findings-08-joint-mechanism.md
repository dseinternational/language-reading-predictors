> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `joint_mechanism` family — is the letter-sound route specific to decoding?

**Read `findings-00-overview` first**, and ideally the `mechanism` note, which this family refines. Nothing here is causal. Both models pass the convergence gate with zero divergences and are publishable. Both were refitted late on 2026-09-01 by the #653 K-fold work at commit `30be4c0a`, from a working tree with uncommitted changes (`dirty: true` in their provenance) — the only two fits in the batch carrying that flag. Their slopes, contrasts and key-findings sentences are identical to the `b18ea944` fits kept beside them (`.pre-jm-kfold-20260901`); what the refit added is the K-fold estimate below.

## The data

**RLI trial only**, used two ways. `jm-001` is a **levels** design: one row per child at a single timepoint, fitted once per wave (53 children at waves 1–3, 52 at wave 4). `jm-002` is a **transition ANCOVA**: all period transitions stacked, 53 children contributing 153 rows, each row carrying both outcomes' own starting scores.

## What the model is for

To ask whether letter-sound knowledge is _more_ tied to nonword decoding than to word reading, the two slopes must come from one posterior with an explicit cross-outcome dependence block; subtracting two separate fits imposes a covariance of zero. Both models fit word reading and nonword reading together on the same standardised letter-sound exposure and report Δ = slope(letter sounds → nonword) − slope(letter sounds → word reading). Nonword reading can only be done by decoding, so a decoding-specific route predicts Δ > 0. `jm-001` additionally reports the letter-sound-to-word-reading slope holding _latent_ decoding fixed, as a ratio to its unconditional value.

## What was found — the two designs disagree, and that is informative

| Model    | Design                      | Δ (logit per SD)                                                                                   | Reading                                                        |
| -------- | --------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `jm-002` | transition (post given pre) | **+0.81** (89% +0.49 to +1.13), P(Δ > 0) = 1.00                                                    | letter sounds track **nonword** reading more                   |
| `jm-001` | levels, per wave            | t1 −0.47 [−0.97, +0.04]; t2 −0.17 [−0.61, +0.26]; t3 −0.26 [−0.68, +0.15]; t4 +0.06 [−0.29, +0.41] | no stable sign across waves; P(Δ > 0) = 0.07, 0.27, 0.15, 0.61 |

In `jm-002` the two slopes are nonword +1.05 (89% +0.74 to +1.36) against word reading +0.24 (+0.14 to +0.34) on one commensurate scale, and the between-child correlation of the two outcomes is +0.25 (89% −0.60 to +0.82) — the dependence block is doing some work but is weakly determined. The factorised cross-check from the separate `mech-096`/`101` fits gives +0.78 (+0.47 to +1.10), agreeing to within 0.03; the comparison contract marks the two as not strictly comparable because rows and exposure scaling differ slightly (152/156/153 rows, exposure SD 1.39/1.43/1.41).

In `jm-001` the per-wave word-reading slopes are large (0.76 to 1.01 per SD) and the nonword slopes are larger after wave 1 (0.29, 0.84, 0.53, 0.82) as more children come off the nonword floor (72%, 64%, 52% and 40% floored at t1–t4), so Δ moves from negative at t1 (P(Δ < 0) = 0.93) to about zero at t4. All four waves are reported; none is selected as a headline.

**These are different estimands and need not agree in sign.** A levels contrast absorbs every stable difference that makes a child score high on both measures at once, including general ability and reading development; a transition contrast conditions each outcome on its own starting score and removes much of that component. For the narrower question — whether letter-sound level is more strongly associated with post-period nonword than word reading after baseline adjustment — the transition estimand is the relevant one, and on its own terms the specificity test passes clearly. The levels result is what a shared reading-development component, exaggerated by the nonword floor, would produce.

**Holding latent decoding fixed** removes 10%, 24%, 18% and 40% of the letter-sound-to-word-reading level slope at the four waves (ratios 0.90, 0.76, 0.82, 0.60; absolute reductions 0.07, 0.24, 0.15, 0.30 logit per SD, the last three with P > 0.98). The ratio is an unbounded conditional-to-marginal slope ratio, not a mediated share; the denominator-stability rule holds at every wave. Most of the letter-sound association with word reading survives conditioning on decoding, which is what a two-route (decoding plus sight-word) reading process would show and also what differential measurement of the two tests would show.

## What this family cannot settle

Both Δs are **operational contrasts between two adjusted test-score associations**. The tests differ in item count (79 against 6), floor, discrimination and reliability, and neither model calibrates them to a common latent scale, so a single general ability loading differently on the two tests would produce a non-zero Δ with no causal letter-sound route at all. Power scaling flags the nonword slope, Δ and the residual correlation in `jm-002` and all three slopes in every wave of `jm-001` for prior sensitivity, so read the magnitudes as prior-dependent. The nonword variance components in `jm-002` are not well determined and should not be quoted.

## The new-child prediction target

Neither model publishes a validated importance-sampling estimate of its new-child predictive score: `jm-002` has one child above the Pareto-k threshold (maximum 1.14) and `jm-001`, whose per-wave residual _is_ the child effect, has fifteen (maximum 6.12). Both therefore take the grouped child-level K-fold route PR #653 added for this family. For `jm-001` all five folds converged and the new-child expected log predictive density is −258.0 (SE 12.8), a complete estimate. For `jm-002` one of the five folds failed the convergence gate (minimum ESS 195, R-hat 1.014), so its four-fold value (−595.0, SE 32.6) is recorded as incomplete and is not the declared estimate: a partial K-fold is a selection, not a smaller sample. Withholding the importance-sampling estimates is the policy working, not a defect in the fits.

## Model inventory

Both models pass the convergence gate with zero divergences and are publishable: `jm-001` (per-wave levels, constructed against the `concurrent` family's minimal-adjustment pair) and `jm-002` (phase-stacked transitions, constructed against `mech-096`/`mech-101`). `jm-001`'s priors table names the `Normal(0, 0.3)` slope prior it actually fits; the 2026-08-31 regeneration that first corrected that row (`notes/202608311600-prior-descriptor-findings-637.md`) has been superseded by the refit, which writes the row by construction.
