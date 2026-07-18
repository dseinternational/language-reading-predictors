<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Findings: is letter-sound knowledge _used for decoding_? (Tier-1 decoding-specificity mini-suite)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Results from the Tier-1 mini-suite specced in `notes/202607172330-tier1-decoding-specificity-spec.md`. Numbers are the **`reporting`** fits (6×6000; all six mechanism models clear the convergence gate) and are reported to the house standard — the posterior **median** with an **89% equal-tailed** credible interval (`notes/202607172359-credible-interval-standard.md`). Everything here is an **adjusted association** under the revised DAG, never a causal skill→skill effect — general ability (`GA`) is latent and unblockable, and the child random intercept does not stand in for it. The only randomised warrant in the suite is the ITT arm.

## The question and the design in one paragraph

Do gains in letter-sound knowledge (`LS`) actually get _used to decode_, or do they merely _travel alongside_ reading gains through general ability and shared teaching? "Used for decoding" has a signature the confounds cannot fake: the effect must run through the alphabetic operation, whose fingerprint is **nonword reading (`N`)** — a string that cannot be sight-read. Three designs test this: **1A** contrasts the letter-sound slope on pure decoding (`N`) against mixed word reading (`W`); **1B** checks that letter sounds move the written-code outcomes but **not** oral-language "negative-control" outcomes; **1C** triangulates the arm-anchored `LS → N → WR` cascade from existing models. All slopes are the linear `beta_mech`, on a common **logit-per-SD-of-letter-sounds** scale.

## 1A — convergent–discriminant contrast: letter sounds feed decoding far more than word reading

| slope (logit per SD of `LS`)          | median    | 89% CI                          |
| ------------------------------------- | --------- | ------------------------------- |
| `LS → N` (nonword decoding, mech-096) | **+1.03** | [0.74, 1.34]                    |
| `LS → W` (word reading, mech-101)     | **+0.25** | [0.15, 0.35]                    |
| **Δ = `LS→N` − `LS→W`**               | **+0.78** | **[0.47, 1.10]**, P(Δ>0) ≈ 1.00 |

Letter sounds predict **pure decoding about four times as strongly as they predict word reading**, and the contrast Δ excludes zero decisively. This is the decoding-use signature: a pure-`GA`-confounding account gives no reason for letter sounds to predict `N` _more_ than `W` (if anything `GA` should predict the broader word-reading skill at least as much), so the natural explanation for `LS→N` ≫ `LS→W` is that letter-sound knowledge is being converted into decoding. **Decoding-use supported.**

## 1B — negative-control outcomes: written code moves, oral language barely does

| letter-sound slope on…               | median    | 89% CI       | role                       |
| ------------------------------------ | --------- | ------------ | -------------------------- |
| nonword decoding `N` (mech-096)      | **+1.03** | [0.74, 1.34] | positive (written code)    |
| word reading `W` (mech-101)          | **+0.25** | [0.15, 0.35] | positive (written code)    |
| receptive vocabulary `R` (mech-097)  | +0.11     | [0.06, 0.16] | negative control           |
| expressive vocabulary `E` (mech-098) | +0.10     | [0.06, 0.15] | negative control           |
| receptive grammar `T` (mech-099)     | +0.12     | [0.04, 0.20] | negative control           |
| basic concepts `F` (mech-100)        | +0.29     | [0.16, 0.42] | negative control (weakest) |

The vocabulary and grammar controls (`R`, `E`, `T`) sit at **+0.10 to +0.12** — roughly a **ninth** of the `LS→N` slope and about **half** of `LS→W`. Letter sounds act overwhelmingly on the written code, exactly as the decoding account predicts. **Two honest qualifications:**

1. **The oral-language slopes are small but not zero** (each P(>0) ≈ 0.99). A genuinely decoding-specific `LS` should be null here; the residual ≈ +0.11 is the expected fingerprint of the `GA` / shared-teaching confounding that no observational design in this study can remove. So the negative controls _attenuate_ toward zero rather than reaching it — supportive of specificity, not a clean null.
2. **`F` (CELF basic concepts) is not a clean control**: at **+0.29** it behaves like a written-code outcome, not an oral-language one. This was the a-priori-weakest control (18 items; basic-concept items like _before/after, more/less_ plausibly share teaching/method variance with the intervention). Set `F` aside as uninformative rather than reading it as evidence against specificity.

### The scale-artefact this exposes (a reusable lesson)

On the **items scale** the same fits _look_ like letter sounds drive vocabulary strongly: over the letter-sound range they predict **+16 receptive-vocabulary items** and **+14 expressive-vocabulary items**, versus only **+3.7 nonwords** and **+9.9 words**. That impression is a **scale artefact** — the vocabulary tests have 170 items, so a small per-SD logit slope multiplies into many items, while nonword reading has only 6. As a fraction of each test's range, `N` moves ≈ 62% but `R` only ≈ 10%. The **logit slope is the commensurate quantity** for a cross-outcome specificity comparison, and on it the ranking is unambiguous: `N` ≫ `W` > oral-language controls. (Recorded because the items scale is what a reader meets first in the per-model reports.)

## 1C — the arm-anchored `LS → N → WR` cascade (triangulation, not one model)

The DAG designates nonword reading the **"code route (mediator)"** between phonics knowledge and word reading (`LS → NW → WR`). A single chained two-mediator model (`lrp-rli-med-081`) was specced but **withdrawn at build time**: the two-mediator factory needs each mediator's autoregressive baseline, and `N` is post-only (its t1 score is ~72% floored and deliberately not co-loaded), so the fit cannot construct an `N` baseline — the floor caveat surfacing as an infrastructure block (see spec §6). The cascade is instead read from the existing **arm-anchored** g-formula models (randomised exposure, already in the suite):

- **`med-086`** — arm → `LS` → `N`: does the intervention raise nonword decoding _via letter sounds_? (letters feed decoding)
- **`med-074`** — arm → `N` → `WR`: does it raise word reading _via nonword decoding_? (decoding feeds reading)
- **`med-059`** — arm → `LS` → `WR`: the total letter-sound route (the cascade's endpoints).

Read together these are the two links of `LS → N → WR`; numbers and their floor/`GA` caveats are in the mediation findings note (`notes/202607161800-findings-mediation.md`). The `N`-floor keeps both links wide, so 1C corroborates rather than clinches. The genuinely missing piece — the single within-model joint `{LS, N}` decomposition + the `LS→N` coupling coefficient — needs a floor-tolerant second-mediator leg in `build_two_mediator_model`, deferred.

## Overall verdict

**Decoding-use is supported.** The primary contrast (1A) is decisive — letter sounds feed pure decoding roughly four times as strongly as mixed word reading — and the negative-control panel (1B) shows the letter-sound effect concentrated on the written code, with oral-language associations attenuated to about a ninth of the decoding slope. The result is not a perfectly clean dissociation: a small residual `LS`→oral-language association (≈ +0.11) is consistent with the irreducible `GA`/teaching leakage this design cannot remove, and one control (`F`) is contaminated. Together with the existing phonics-route synergy (`mech-072`: `L×B → N`) and the arm-anchored cascade (`med-086`, `med-074`), the weight of evidence is that letter-sound gains in this cohort **are predominantly converted into decoding**, not merely correlated with reading through other mechanisms — while honestly acknowledging that the last increment of "purely decoding" cannot be certified observationally at n ≈ 54 with a latent ability confound.

## Caveats (governing)

- **Adjusted associations, not causal.** `GA` is unblockable; every slope here is an adjusted association. Only the ITT arm is randomised.
- **`N` floor.** Nonword reading is ≈ 57–72% floored; its slopes and the 1C cascade are power-limited (a small/uncertain value is floor-limited, not "no effect").
- **Matched conditioning.** All six models share the adjustment set `{G, A, HS, IS, SP}` + own baseline (revised-DAG parents of `LS`, #245), so the panel is like-for-like; only the outcome changes.
- **No item-level data**, so the gold-standard within-child "decode nonwords built from letters you personally know" test remains impossible (see `notes/202607172345-design-lessons-for-future-studies.md`).
- These are **`reporting`** fits (6×6000), re-fit under the 89%/50% credible-interval standard and re-published with the full suite.

## Cross-references

- Spec + adjustment-set derivations + triangulation truth-table: `notes/202607172330-tier1-decoding-specificity-spec.md`.
- Comparison artefacts: `output/statistical_models/comparison/tier1_1a_contrast.csv`, `tier1_negative_control_forest.{csv,png}`.
- Models: `docs/models/lrp-rli-mech-{096,097,098,099,100,101}/`.
- The DS sight-word-over-decoding literature the question sits in: Cupples & Iacono (2000, [doi:10.1044/jslhr.4303.595]); Roch & Jarrold (2012, [doi:10.1016/j.jcomdis.2011.11.001]).
