<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 07 — the mechanism family (skill-to-skill couplings)

Reports every model in the `mechanism` family (34) and the two `joint_mechanism` models from the 2026-08-04/05 `reporting` refit. **36 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

The ITT family establishes _that_ the intervention works. This family asks _what travels with what_: how strongly is one skill associated with another, across all phases of the study, adjusting for the causal diagram's adjustment set.

**Design.** The outcome's post-score given its own pre-score (Beta-Binomial, logit link) with the exposure skill entering either **linearly** or as a **flexible curve** (a Hilbert-space Gaussian process, "HSGP") when the question is whether the relationship has a knee or threshold. A child random intercept handles repeated observations. Some models add a linear moderator to ask whether the coupling differs by a third variable.

**Nothing in this family is causal.** Not one coefficient. General ability is latent and unblockable, the child random intercept does **not** stand in for it, and the only randomised warrant anywhere in the suite is the ITT arm. These are adjusted associations, and the phrase "X drives Y" must not be used of any of them.

## The core reading couplings

Effect is the items-scale change in the outcome across the observed range of the exposure, median with 89% range. **Items scales differ between measures, so read down a column, not across rows of different outcomes.**

| Model      | Coupling                               | Shape  | Effect over exposure range       | P(>0) |
| ---------- | -------------------------------------- | ------ | -------------------------------- | ----: |
| `mech-058` | Letter sounds → word reading           | curve  | **+6.8** W items (+2.6 to +11.0) | 0.997 |
| `mech-101` | Letter sounds → word reading           | linear | +9.9 W items (+6.1 to +13.5)     | 1.000 |
| `mech-096` | Letter sounds → nonword reading        | linear | +3.7 NW items (+2.8 to +4.5)     | 1.000 |
| `mech-088` | Taught receptive vocab → word reading  | curve  | +9.0 W items (+4.9 to +13.1)     | 1.000 |
| `mech-089` | Taught expressive vocab → word reading | curve  | +8.6 W items (+3.3 to +13.4)     | 0.994 |
| `mech-057` | Expressive vocabulary → word reading   | linear | +5.3 W items (−0.6 to +10.9)     | 0.926 |
| `mech-090` | Phonological memory → word reading     | curve  | +3.1 W items (+0.2 to +5.9)      | 0.954 |
| `mech-056` | Receptive vocabulary → word reading    | linear | +2.9 W items (−2.6 to +8.2)      | 0.798 |
| `mech-102` | Phonological memory → nonword reading  | linear | +2.5 NW items (+1.7 to +3.3)     | 1.000 |
| `mech-103` | Speech production → nonword reading    | linear | +2.3 NW items (+1.4 to +3.1)     | 1.000 |
| `mech-190` | Phoneme blending → word reading        | curve  | +0.6 W items (−0.8 to +4.6)      | 0.732 |
| `mech-156` | Receptive vocabulary → word reading    | curve  | +0.2 W items (−1.6 to +3.7)      | 0.634 |
| `mech-157` | Expressive vocabulary → word reading   | curve  | +0.6 W items (−1.0 to +5.5)      | 0.729 |

**Letter-sound knowledge is the strongest and best-resolved coupling to word reading**, and it is the one the mediation family (note 08) takes up.

**Linear and curve versions disagree for vocabulary, and the curve wins on design.** `mech-056`/`mech-057` (linear) give R→W +2.9 and E→W +5.3; the flexible-curve versions `mech-156`/`mech-157` give +0.2 and +0.6, both inconclusive. A linear term forced through a relationship that is mostly flat will report the average slope as if it were a consistent effect. Where the two disagree, the curve models are the more honest description, and the vocabulary→word-reading couplings should be read as **weak and poorly resolved**, not as the linear numbers suggest.

The cross-model mechanism forest puts the three main couplings on one comparable per-SD scale: letter sounds → word reading **+0.238** [0.085, 0.404], expressive vocabulary → word reading **+0.122** [−0.013, 0.257], receptive vocabulary → word reading **+0.064** [−0.057, 0.185]. Only the letter-sound coupling is clearly resolved.

## Decoding specificity — the sharpest result in this family, and its limit

The question: does letter-sound knowledge actually get _used to decode_, or does it merely travel alongside reading through shared teaching and general ability? The test is that decoding-use has a signature confounding cannot easily fake — the association should be strongest on **nonword reading**, a string that cannot be sight-read.

On a common **logit-per-SD-of-letter-sounds** scale:

| Letter-sound slope on…             |    Median | 89%          | Role                             |
| ---------------------------------- | --------: | ------------ | -------------------------------- |
| Nonword reading (`mech-096`)       | **+1.03** | 0.74 to 1.34 | positive control (written code)  |
| Word reading (`mech-101`)          | **+0.25** | 0.15 to 0.35 | positive control (written code)  |
| Basic concepts (`mech-100`)        |     +0.29 | 0.16 to 0.42 | negative control (oral language) |
| Receptive grammar (`mech-099`)     |     +0.12 | 0.04 to 0.20 | negative control                 |
| Receptive vocabulary (`mech-097`)  |     +0.11 | 0.06 to 0.16 | negative control                 |
| Expressive vocabulary (`mech-098`) |     +0.10 | 0.06 to 0.15 | negative control                 |

**The contrast is now identified, which it was not before.** The earlier reading rested on a product-of-marginals sensitivity computed under a working independence assumption — nonword and word reading share children, so that Δ was not a proper posterior contrast. `jm-002` fits both outcomes **jointly** on the same exposure with a bivariate child random intercept, making Δ a within-model deterministic:

- **Identified Δ = β(LS→N) − β(LS→W) = +0.81 [0.50, 1.14], P(Δ>0) = 0.9999** (`jm-002`)
- Product-of-marginals sensitivity: +0.78 [0.47, 1.10] (historical comparator)

The two agree closely, so the specificity conclusion now rests on an identified contrast. Letter sounds are associated with pure decoding about **four times** as strongly as with word reading, which is hard to explain by general ability alone — a purely ability-driven account gives no reason for letter sounds to predict the _narrower_ skill more than the broader one.

**Two limits that must travel with this result.**

**The negative controls do not come out clean.** All four oral-language outcomes have clearly positive letter-sound slopes (every P(>0) ≈ 0.99+), and basic concepts (+0.29) is actually _larger_ than word reading (+0.25). A genuinely decoding-specific letter-sound skill should be null on oral language. The controls **attenuate** toward zero (vocabulary and grammar at about a ninth of the nonword slope) rather than reaching it. That residual is the expected fingerprint of general-ability and shared-teaching confounding that no observational design here can remove. The right claim is "supportive of specificity", not "specificity demonstrated".

**The specificity appears in change, not in levels.** `jm-001` fits the same joint W/N structure as a **per-wave levels** model — no own-baseline conditioning — and the contrast reverses:

| Wave | Δ = β(LS→N) − β(LS→W)  | P(Δ>0) |
| ---- | ---------------------- | -----: |
| t1   | −0.49 (−0.98 to +0.02) |   0.06 |
| t2   | −0.15 (−0.60 to +0.29) |   0.29 |
| t3   | −0.26 (−0.67 to +0.15) |   0.16 |
| t4   | +0.06 (−0.29 to +0.40) |   0.61 |

In levels, letter sounds predict word reading _at least_ as strongly as nonword reading, and at t1 the favoured direction is the opposite of the specificity claim. This is not a contradiction so much as a statement about what is being asked: in levels both outcomes reflect accumulated ability, and letter sounds correlate with everything accumulated; conditional on where a child started, the incremental association is specific to decoding. **The decoding-specificity finding is a claim about conditional change and should always be stated that way.** Quoting "+0.81, P ≈ 1.00" without that qualifier would misrepresent it, because the levels view of the same children does not show it.

## Moderation — the recurring null

Eight models ask whether the letter-sound → word-reading coupling differs by a third variable: phoneme blending (`mech-061`), nonword decoding (`mech-063`), expressive vocabulary (`mech-071`), receptive vocabulary (`mech-093`), taught receptive (`mech-094`), taught expressive (`mech-095`), age (`mech-073`) and phonological memory (`mech-104`).

Three of the eight have a purpose-built no-interaction baseline — `mech-061`/`mech-161`, `mech-063`/`mech-163` and `mech-104`/`mech-204`; `mech-071` is compared against plain `mech-058`; the remaining four (`mech-073/093/094/095`) have no matched comparator. (`mech-172` is a baseline too, but for `mech-072`, which models letter sounds → _nonword_ decoding and so is not one of the eight.) The pattern across the family is consistent: **apparent moderations collapse once subject random effects and the adjustment set are in**. The main coupling stays in the +4.8 to +8.0 items range in every one of these models; the interaction terms do not resolve. Age moderation (`mech-073`) is the clearest example — no credible age moderation of the letter-sound → word-reading association.

Nested PSIS-LOO comparisons between the interaction models and their baselines were repaired by exact refit where possible (`mech-058/071`, `mech-061/161`, `mech-063/163`, `mech-104/204`, one refit each; `mech-072/172` needed none). Every one of the five contrasts remains inconclusive under the |elpd_diff| < 4 rule, which at this sample size is the expected outcome rather than a failure.

## Robustness

`mech-158` is a complete-case comparator to `mech-058` with no imputed covariates: +6.9 items against +6.8. The letter-sound → word-reading coupling is not an artefact of covariate imputation.

`mech-191` relates **intervention sessions** to word reading as a flexible curve (+2.2 W items over 0–94 sessions, P = 0.97). Dose is not randomised, so this is an association with a known confounding path — children least able to learn tended to attend least.

## Caveats

- **No causal terms anywhere in this family.**
- **Items are not comparable across measures.** The per-SD forest is the comparable view.
- **Curve versus linear matters**, and where they disagree the curve is the better description.
- **Eight models in this family** (`mech-093/094/095/156/157/188/189/191`) required the thin-support HSGP reparameterisation to reach zero divergences at reporting tier; the curve amplitude rises slightly under it and no parameter moves more than 0.064 posterior SD, so the readouts are unaffected (run record note).
- **Predictive calibration.** 50% bands cover about 73% of observations.
