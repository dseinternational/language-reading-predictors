<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 11 — the concurrent-association family

Reports every model in the `concurrent` family from the 2026-08-04/05 `reporting` refit. **11 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

The simplest question in the suite, asked carefully: **at a single point in time, which skills go together?** For each outcome, every other skill measured at the same wave enters as a predictor, and the model reports both the **adjusted** association (all other skills held) and the **bivariate** one (that skill alone), at each of the four waves.

**Design.** Per-wave Beta-Binomial regression of the outcome on the concurrent skills, with trait covariates (non-verbal ability, hearing, speech production, phonological memory). Reporting both adjusted and bivariate side by side is the point — the gap between them is the substantive information.

**No causal content whatsoever, and no temporal ordering either.** These are contemporaneous correlations. They cannot distinguish "letter sounds help word reading" from "word reading helps letter sounds" from "general ability drives both", and unlike the LCSM family (note 10) they do not even have lead–lag ordering to lean on. They are descriptive structure.

## Results — the strongest adjusted associations per outcome

Slopes are on the logit scale per SD of the predictor, pooled across the four waves (median of the per-wave estimates). `adj` holds every other skill; `biv` is that predictor alone.

| Model    | Outcome                 | Strongest adjusted associations (adj / biv)                                                               |
| -------- | ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `ca-001` | Word reading            | **Letter sounds +0.46 / +0.73** (P=1.00); taught expressive +0.27 / +0.55; expressive vocab +0.22 / +0.63 |
| `ca-002` | Letter sounds           | **Word reading +0.40 / +0.67** (P=0.99); expressive vocab +0.25 / +0.43; blending +0.20 / +0.42           |
| `ca-007` | Phoneme blending        | Word reading +0.31 / +0.44 (P=0.97); letter sounds +0.24 / +0.44                                          |
| `ca-004` | Taught expressive vocab | Expressive vocab +0.31 / +0.47 (P=0.98); taught receptive +0.21 / +0.41                                   |
| `ca-008` | Basic concepts          | Receptive vocab +0.24 / +0.44 (P=0.94); expressive vocab +0.22 / +0.52                                    |
| `ca-003` | Taught receptive vocab  | Receptive vocab +0.17 / +0.30 (P=0.92); taught expressive +0.16 / +0.32                                   |
| `ca-006` | Expressive vocabulary   | Taught expressive +0.15 / +0.26 (P=0.97); receptive vocab +0.09 / +0.21                                   |
| `ca-009` | Receptive grammar       | Taught receptive +0.14 / +0.27 (P=0.91); receptive vocab +0.12 / +0.21                                    |
| `ca-005` | Receptive vocabulary    | Expressive vocab +0.13 / +0.24 (P=0.94); taught receptive +0.11 / +0.17                                   |

**Two structural facts stand out.**

**The skills cluster into a reading group and a language group.** Word reading, letter sounds and blending are each other's strongest partners; the vocabulary and grammar measures pair with each other. The letter-sound ↔ word-reading pairing is the strongest single association in the family in both directions (+0.46 and +0.40), which is consistent with — but does not add causal weight to — the mechanism and mediation findings.

**Adjustment roughly halves the associations.** In each of the nine multi-predictor models the _strongest_ adjusted slope is about 53–70% of its bivariate counterpart (median 58%); across all predictors, including the weak ones, the median ratio is about 0.42 with a much wider spread — weak bivariate associations can vanish entirely or flip sign on adjustment. The shrinkage is therefore a strong general tendency rather than a uniform factor. That pattern is the signature of a **shared general factor**: much of what any two skills have in common is what all of them have in common. It is the same phenomenon the measurement models (note 14) quantify directly — a single dimension carries about 83% of the variance among this cohort's three domain factors, and about 91% among the historical cohort's four.

The practical implication is a warning about the bivariate column: a raw correlation between two skills in this cohort substantially overstates their specific relationship.

## The two minimal-adjustment models

`ca-010` and `ca-011` strip the adjustment set to the letter-sound → word-reading relationship alone, matched to the mechanism family's parameterisation so the two can be compared.

| Model    | Predictors                       | Adjusted slope                                                      |
| -------- | -------------------------------- | ------------------------------------------------------------------- |
| `ca-010` | Letter sounds only               | **+0.77** (P = 1.00)                                                |
| `ca-011` | Letter sounds + nonword decoding | Letter sounds **+0.59**, nonword decoding **+0.39** (both P = 1.00) |

With no other skills competing, the letter-sound → word-reading association is +0.77 — against +0.46 when the full skill set is held. Adding nonword decoding takes letter sounds down to +0.59 and gives nonword decoding +0.39; the two written-code skills share a substantial part of their association with word reading, which is what a common alphabetic process would predict.

These models exist so the concurrent view can be compared like-for-like with the mechanism family's conditional-change view, and they are the right rows to use for that comparison rather than `ca-001`.

## Caveats

- **Contemporaneous associations only** — no causal content, no temporal ordering.
- **Read the adjusted column**, and treat the bivariate column as showing how much of a raw correlation is shared-general-factor rather than specific.
- **Per-wave estimates are pooled here for readability**; the fits report all four waves separately and the per-wave pattern is stable rather than trending.
- **Predictive calibration.** 50% bands cover about 71% of observations.
