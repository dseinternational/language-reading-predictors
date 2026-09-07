<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# A second non-verbal ability indicator: what it buys, and what it does not

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

Date: 2026-09-07 — **Status: DECIDED** (author instruction, 2026-09-07).

## The question

Committing the deposited trial archive (#664) brought in **WPPSI-III Object Assembly at t1**, the second non-verbal subtest the original trial reported. The suite adjusts for measured ability with **Block Design alone**, and `notes/202607172345-design-lessons-for-future-studies.md` names single-indicator constructs and the absence of a real general-ability battery among the study's measurement walls. So: what should we do with it?

## What the data say

Object Assembly is complete for all 54 analysed children, exactly as Block Design is, with no floor or ceiling pile-up (4 children at zero, none at the 37 maximum, mean 9.48, SD 6.77 against Block Design's 12.59 and 6.26).

**The two subtests correlate at 0.664** (Spearman 0.680). Treating them as two indicators of what they share, that correlation *is* the reliability of a single subtest as a measure of the common factor, and the Spearman-Brown reliability of their sum is **0.798**. So a single-subtest adjustment leaves about a third of the shared factor's variance unadjusted, and the composite about a fifth.

Note what the published subtest alphas (0.84 Block Design, 0.85 Object Assembly) are and are not: they are internal consistency *of each subtest*, which includes subtest-specific variance that is not the common factor. Quoting them as the reliability of the ability adjustment overstates it, and this note is partly here so that mistake is not made in the report.

## What it would change: nothing material

A screening regression over the fitted mechanism rows (clustered least squares on the logit scale, ability broadcast from t1 as the models do it) put the letter-sound → word-reading slope at:

| Ability adjustment      | Letter-sound slope | Ability slope    |
| ----------------------- | ------------------ | ---------------- |
| none                    | +0.284 (0.092)     | —                |
| Block Design alone      | +0.250 (0.090)     | +0.147 (0.068)   |
| Object Assembly alone   | +0.273 (0.090)     | +0.116 (0.052)   |
| two-subtest composite   | +0.257 (0.089)     | +0.152 (0.062)   |
| both, separately        | +0.253 (0.089)     | +0.115 (0.082)   |

**The composite absorbs no more than Block Design alone** — very slightly less, and the difference is under a tenth of a standard error. The negative control moves from +0.075 to +0.074. The ability coefficient itself changes by a factor of 1.03, 1.00 and 0.87 on word reading, receptive and expressive vocabulary. The one real gain is precision: standard errors on the ability term narrow by 9 to 14 %.

The registered fit confirms the screen: `lrp-rli-mech-311` at `dev` tier returns `beta_mech` = 0.242 against its parent's recorded +0.245.

Extrapolating matters here. Moving reliability from 0.66 to 0.80 moved the slope by 0.007 logits, so there is no reason to expect perfect reliability to move it materially either. That is an argument against spending a full sweep on an errors-in-variables upgrade.

## The deeper point, which cuts against doing more

**Block Design and Object Assembly are both perceptual-organisation subtests.** What they share is a *narrow visuospatial factor*, and visuospatial processing is the relative strength in the Down syndrome profile. Measuring that better does not bring the adjustment closer to the DAG's latent general ability `GA`, which stays unmeasured and structurally unblockable. A composite-adjusted association is better adjusted for one domain; it is not closer to a causal quantity.

This is why the design-lessons note's request stands unmet: two subtests from one domain is not a general-ability battery, and nothing in the archive changes that.

## Decision

Two of the four options considered, as agreed:

1. **Take the reporting gain.** `METHODS.md` now states the 0.664 correlation and what it implies about the reach of a single-subtest adjustment, warns against quoting the subtest alphas for that purpose, and records the visuospatial-factor ceiling. This costs nothing and improves the report immediately.
2. **Register targeted companions**, so the anticipated criticism — that the ability adjustment rests on one noisy subtest — has a fitted answer rather than an argument. Seven models, `LRP306`–`LRP312`, one per ability-adjusted parent, each differing from its parent in exactly one declared setting.

Rejected, and why:

- **Swapping the composite in everywhere.** Around 120 model modules carry the ability covariate, so it means a full re-sweep and a new artefact set, to move coefficients by well under a tenth of their standard error. Poor value, and it would invalidate the current artefacts for no scientific gain.
- **A latent non-verbal factor / errors-in-variables model.** Cleanest methodologically, but two indicators is the bare minimum for identification and the notes already record the measurement model as fragile and prior-dependent at this sample size (`notes/202608061500-default-prior-recalibration-383.md`). It also would not improve identification, since the factor is still a proxy for `GA`.

## Why the whole panel and not just the headline

The panel is what makes the argument. `LRP196`–`201` is a specificity comparison — the letter-sound slope on the written-code outcomes against the oral-language negative controls — so re-reading the word-reading cell alone would break its logic. `LRP312` is included because `LRP258` is the ability-adjusted version of the family's **headline estimand**, the flexible curve whose interquartile contrast the mechanism family publishes, and that claim ("+6.8 to +6.5 items, a 4 % shift") is load-bearing.

| New     | Parent  | Outcome                    | Role                                |
| ------- | ------- | -------------------------- | ----------------------------------- |
| LRP306  | LRP196  | nonword reading (N)        | the decoding-specific channel       |
| LRP307  | LRP197  | receptive vocabulary (R)   | oral-language negative control      |
| LRP308  | LRP198  | expressive vocabulary (E)  | oral-language negative control      |
| LRP309  | LRP199  | receptive grammar (T)      | oral-language negative control      |
| LRP310  | LRP200  | basic concepts (F)         | oral-language negative control      |
| LRP311  | LRP201  | word reading (W)           | the mixed sight/decoding channel    |
| LRP312  | LRP258  | word reading (W), HSGP     | the family's headline curve         |

Each is registered `Status.COMPANION` against its parent, and a test asserts the one-knob property: same outcome, mechanism, adjustment set, baseline symbol, outcomes tuple, linearity and random-intercept setting, differing only in `ability_covariate`.

## Implementation

- **Data.** `objass` added to `rli_data_long.csv` (t1 only, as Block Design is) and `objass1` to `rli_data_wide.csv`, by `scripts/derive_object_assembly.py`. The deposit and the analysis files carry different anonymised labels, so the join is on the same 71-field row fingerprint the missingness loader already reconciles them by; the fingerprint is computed, used and discarded, and no identifier crosswalk is written. `--check` recomputes and compares without writing, and a test runs it, so the committed column cannot drift from the deposit unnoticed.
- **The column was appended textually, not by a `read_csv`/`to_csv` round trip.** Re-serialising the frames rewrote two float columns in each file in the last bit — a ~2e-16 relative change, numerically meaningless but a pointless edit to committed research data. Verified afterwards that both files are identical to their previous contents apart from the new column.
- **`RLI_LOCAL_WIDE_SHA256` updated** for the new wide file. The 71-field reconciliation digest is unaffected, because Object Assembly is not one of the reconciled fields.
- **Composite derived, not stored** (`preprocessing.derive_nonverbal_ability_composite`), following the `HEARING_C` precedent so the definition lives in one place. It is the raw sum: the subtests have near-equal spread here, so the sum agrees with the average of their standardised scores to a correlation of 0.99984, every consumer standardises it downstream, and a raw sum has no reference-sample ambiguity. Returns `None` when either component is absent, so the historical cohort is a no-op rather than getting a silent single-subtest fallback.
- **Both new names are in `DEFAULT_EXCLUDED`,** as `blocks` already was, so the 50 gradient-boosting models' predictor sets are untouched and no GB refit is implied. A test asserts it.
- **`objass_c` added to `SUPPORTED_ABILITY_COVARIATES`**; registry counts and the catalogue updated (mechanism family 46 → 53, registry 269 → 276).
- **Eleven tests** in `tests/test_nonverbal_ability_composite.py`, including that the composite is complete wherever Block Design is — otherwise a companion would silently analyse fewer children than its parent and the comparison would not be like-for-like.

## Not fitted here

These seven need a `reporting`-tier fit, which will happen in the next sweep on the machine that holds the artefacts. Until then they are registered and their reports are templates. Nothing existing is invalidated: no fitted model changes, because the composite is only named by the new companions.

## One thing found while doing this

A **third** copy of the POSIX-only render-test environment arrived with #663 after the #664 fix was written, so two of that PR's new tests were failing on `main`. Fixed the same way, and there are now no remaining copies.

## Related

- `notes/202609071500-incorporate-deposited-trial-archive.md` — the change that made the subtest available, and the finding that flagged this.
- `notes/202607172345-design-lessons-for-future-studies.md` — the measurement walls, including the general-ability battery this does not supply.
- `notes/202607172330-tier1-decoding-specificity-spec.md` — the panel these companions check.
- `notes/202609071300-technical-report-plan-v2.md` — the report plan whose limitations chapter carries the reliability figure.
