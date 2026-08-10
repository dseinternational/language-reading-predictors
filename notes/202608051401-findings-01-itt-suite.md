<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Available-case modified ITT terminology added by a LLM-based AI tool (Codex/GPT-5).

# Findings 01 — the available-case modified ITT suite

Reports every available-case modified ITT model in the `itt` family from the full `reporting` refit of 2026-08-04/05 (run record: `notes/202608050649-reporting-refit-predictive-checks.md`). **28 models, all passing the convergence gate.** Reading conventions are in note 00. Preliminary research data — all estimates provisional.

## What these models do

The single most important family: the **available-case modified ITT estimate of the assigned-arm contrast**, one model per outcome. Randomisation identifies the assigned-arm contrast in the full cohort in principle; a causal reading of the fitted available-case estimate additionally requires the stated selection and missing-outcome assumptions.

**Design.** Each model uses the randomised t1→t2 window — one row per child, before anyone crossed over. The score is a bounded count, modelled with a **Beta-Binomial** likelihood on a logit linear predictor (a Binomial that allows more between-child spread than a plain Binomial). The linear predictor carries the assigned-arm term τ, the child's **own** t1 score on that outcome, and **linear age**.

**The adjustment set is deliberately almost empty**, and this trips people up. Own-baseline and age are in the model as _precision_ terms — they sharpen the estimate — not as confounder control. Randomisation identifies the full-cohort assigned-arm contrast without a confounder adjustment set, but the fitted result remains an **available-case modified ITT estimate** because of loss to follow-up and model-specific observation requirements. The empty adjustment set does not repair that selection. No cross-baselines from other skills are included; adding them would risk conditioning on things the intervention itself moves.

**What is causal here, and what is not.** The available-case modified ITT estimate τ has a causal reading only under the stated selection assumption. Every other coefficient in these models is an adjusted association and reading them as levers is the Table-2 fallacy. The causal interpretation holds for the **fitted available-case sample** only if archive inclusion and outcome observation do not depend jointly on assigned arm and potential outcomes. Typically 53–54 children are fitted out of 57 randomised, so this is not the effect estimate for every randomised child.

**Floored outcomes get a different estimand.** Phonetic spelling (PS) and nonword reading (NW) sit hard on the floor — most children score zero — so a graded count model is the wrong instrument. Those two use the registered **floor rule**: a Bernoulli on the "off the floor at post" indicator, with the treatment effect reported as a **risk difference in percentage points**, not items. Their rows below are marked accordingly.

## Results — the thirteen models of record

Effect is the intervention-minus-waitlist contrast on the items scale (percentage points for the two floor-rule outcomes), median with the 89% credible range. "Evidence" is the fixed ladder applied to the direction probability. δ is the project's pre-agreed minimally-important difference; `P(≥δ)` is the separate _size_ claim; ROPE is the posterior mass too small to matter either way.

| Model     | Outcome                               | Effect (89% range)            | P(>0) | Evidence     |     δ | P(≥δ) | ROPE |
| --------- | ------------------------------------- | ----------------------------- | ----: | ------------ | ----: | ----: | ---: |
| `itt-007` | Letter sounds (LS)                    | **+3.5** items (+1.7 to +5.3) | 0.999 | very strong  |     2 |  0.91 | 0.09 |
| `itt-010` | Word reading (WR)                     | **+2.4** items (+0.7 to +4.1) | 0.986 | strong       |     1 |  0.90 | 0.10 |
| `itt-002` | Taught expressive vocab (TE)          | **+1.5** items (+0.4 to +2.7) | 0.985 | strong       |     1 |  0.78 | 0.22 |
| `itt-001` | Taught receptive vocab (TR)           | **+1.4** items (+0.2 to +2.5) | 0.968 | moderate     |     1 |  0.69 | 0.31 |
| `itt-008` | Phoneme blending (PA)                 | **+1.0** items (+0.2 to +1.7) | 0.980 | strong       |     1 |  0.49 | 0.51 |
| `itt-025` | Basic concepts (LF)                   | +0.9 items (−0.3 to +2.0)     | 0.891 | suggestive   |     1 |  0.43 | 0.57 |
| `itt-026` | Receptive grammar (RG)                | +0.7 items (−0.8 to +2.1)     | 0.760 | suggestive   |     1 |  0.35 | 0.61 |
| `itt-003` | Not-taught receptive vocab (UR)       | +0.6 items (−0.0 to +1.2)     | 0.937 | moderate     |     1 |  0.16 | 0.84 |
| `itt-004` | Not-taught expressive vocab (UE)      | +0.3 items (−0.3 to +1.0)     | 0.773 | suggestive   |     1 |  0.05 | 0.95 |
| `itt-005` | Receptive vocabulary (RV)             | +0.2 items (−3.7 to +4.3)     | 0.539 | inconclusive |     2 |  0.24 | 0.58 |
| `itt-006` | Expressive vocabulary (EV)            | +0.2 items (−3.1 to +3.5)     | 0.534 | inconclusive |     2 |  0.19 | 0.67 |
| `itt-009` | Phonetic spelling (PS) — floor rule ‡ | +4 pp (−7 to +16)             | 0.724 | inconclusive | 10 pp |  0.20 | 0.78 |
| `itt-011` | Nonword reading (NW) — floor rule ‡   | +10 pp (−4 to +24)            | 0.877 | suggestive   | 10 pp |  0.50 | 0.49 |

‡ **Withheld from release** under the robustness gate adopted on 2026-08-05 — see below. The numbers are shown because this note is a technical record, not a published finding.

## Two rows are not releasable, and four carry an attenuation caveat

The key-findings release gate introduced after these fits were made (`notes/202608051500-decision-key-findings-robustness-release-gate.md`, `statistical_models/release.py`) classifies every ITT fit on the power-scaling sensitivity of its own τ, over and above the sampling gate. Re-running it across the 28 fits gives **26 release, 2 withhold**, and nothing in the prior-dominant class.

**The two withholds are the floor-rule outcomes**, `itt-009` (PS) and `itt-011` (NW). Both sit in the prior-data-conflict class with the highest prior sensitivities in the family (0.13 and 0.16), and the six-cell treatment-prior grid the policy requires for a floor-rule headline is not present and provenance-aligned for these fits. Under the gate their causal headline is withheld, and their result tables are suppressed in the rendered report. **Do not quote +4 pp or +10 pp as findings.** Building the grid would lift the withhold; nothing needs refitting.

**Four rows in the table release with a caveat rather than cleanly** — letter sounds (`itt-007`), phoneme blending (`itt-008`), not-taught receptive vocabulary (`itt-003`) and basic concepts (`itt-025`) — along with seven of the robustness models. In that class power-scaling flags a prior-data conflict on τ but the likelihood moves the posterior too, so the reading is that the suite's deliberately conservative zero-centred prior is **attenuating** a real effect rather than manufacturing one. The consequence for a reader: **the direction is the more reliable part of these four results, and the size is best read as a lower bound.** It cuts the same way as the winner's-curse caveat below, so the two do not compound.

The remaining rows — word reading, both taught vocabulary sets, not-taught expressive, receptive grammar and both broad vocabulary measures — are clear on this statistic.

**How to read this table.** The ordering is by strength of evidence for a positive effect, and it recovers the intervention's own logic: the two skills it teaches most directly — letter sounds and word reading — carry the strongest evidence, and the taught word sets follow. Broad standardised vocabulary sits at the bottom, essentially flat.

Three things deserve emphasis because they are easy to misread.

**Direction and size are separate claims, and they diverge here.** Phoneme blending has _strong_ evidence of a positive effect (P = 0.98) but only a **49%** chance the benefit reaches the 1-item threshold, with 51% of the posterior inside the ROPE. "Probably positive" and "probably big enough to matter" are different statements, and for PA only the first is supported. The same pattern, less sharply, applies to the taught vocabulary sets.

**The not-taught contrast is the transfer question.** Taught receptive vocabulary gains +1.4 items while the matched not-taught words gain +0.6, and taught expressive +1.5 against not-taught +0.3. The intervention moves the words it teaches; the matched words it does not teach move much less. The `joint` family (note 02) tests that contrast directly rather than by eyeballing two separate models.

**Broad vocabulary is inconclusive, not null.** RV and EV have wide intervals (±3–4 items on a 170-item test) and direction probabilities near 0.54 — the data simply do not resolve the sign. That is different from evidence of no effect, though the ROPE mass (0.58, 0.67) does suggest that if an effect exists it is probably small.

## Phoneme blending is response-link sensitive — read the pair

`itt-008` cannot be read alone, and the suite enforces this: neither blending fit will release its findings unless the validated trace-backed paired bundle is present.

Phoneme blending is **ten three-alternative forced-choice items**, so a child guessing at random scores about 3.3. The ordinary logit link happily models expected scores below chance, which is mechanically impossible. `itt-108` refits the identical structure with a guessing-floor link, mean = 1/3 + (2/3)·logit⁻¹(η).

| Link                       | Effect (89% range)        | P(>0) | Evidence   |
| -------------------------- | ------------------------- | ----: | ---------- |
| Ordinary logit (`itt-008`) | +1.0 items (+0.2 to +1.7) | 0.980 | strong     |
| Guessing floor (`itt-108`) | +0.5 items (−0.1 to +1.1) | 0.893 | suggestive |

Respecting the guessing floor **roughly halves the effect and moves the evidence from strong to suggestive**. Under the floor link, movement in the sub-chance region is attributed to guessing noise rather than ability. PSIS-LOO cannot separate the two links, so the data do not adjudicate — the argument for the floor link is mechanical (the instrument _is_ three-alternative forced choice), not predictive. **The pair is the result. Quoting +1.0 items on its own overstates what the blending data support.**

## Robustness — the estimate does not depend on the adjustment choices

Fourteen further models re-fit the headline outcomes under different adjustments. The question each answers is "does τ survive this?", and the answer throughout is yes.

| Check                                | Models                                               | Result                                                                                             |
| ------------------------------------ | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **General ability** (block design)   | `itt-017`–`itt-024` (TR, TE, UR, UE, RV, EV, LS, WR) | Every effect within ~0.15 items of its base model. LS 3.52→3.54, WR 2.37→2.23, TE 1.55→1.47.       |
| **SES adjustment**                   | `itt-013` (WR), `itt-113` (LS)                       | WR 2.37→2.54, LS 3.52→3.41. Intervals widen slightly; direction unchanged.                         |
| **Matched complete-case** comparator | `itt-014` (WR), `itt-114` (LS)                       | WR 2.37, LS 3.93 — the SES-adjusted result is not an artefact of the smaller complete-case sample. |
| **Site (area) adjustment**           | `itt-027` (WR), `itt-028` (LS)                       | WR 2.57 (P = 0.992, very strong), LS 3.51. No site confounding signature.                          |

The letter-sound and word-reading results are the most heavily checked in the suite and are stable across every adjustment tried. Note that the site adjustment slightly _raises_ WR's direction probability rather than eroding it (0.986 → 0.992), and the general-ability adjustment barely moves it (0.986 → 0.980) — these are not fragile findings propped up by a particular covariate set.

## Caveats that apply to the whole family

- **Available-case, not all-randomised.** 53–54 of 57 randomised children are fitted. The causal reading needs missingness not to depend jointly on arm and potential outcomes; extending to all 57 needs further assumptions. `analysis_set.csv` in each fit records the exact numbers, and the joint family (note 02) carries attrition bounds.
- **Items are not equal-interval, and not comparable across measures.** "+3.5 letter sounds out of 32" and "+2.4 words out of 79" cannot be ranked against each other as amounts of learning. Three items at the hard end of a ladder represent more progress than three at the easy end.
- **δ is post-hoc.** The minimally-important differences were agreed after the first results review, so `P(≥δ)` should be read beside the threshold-sensitivity curve (`rope_benefit_curve.csv`) rather than as a pre-registered test.
- **Winner's curse.** At n ≈ 54, any estimate that just clears a threshold is on average too large. Lead with the interval.
- **Predictive calibration.** These fits' 50% prediction bands cover about 66% of observations — overcoverage that is substantially mechanical (discrete counts, small denominators) rather than a likelihood defect, established by the Conway–Maxwell-binomial probe in `notes/202607261405-binomial-exchangeability-item-difficulty-review.md`. The ITT family is among the mildest in the suite on this statistic — only the measurement models (0.58) and the aligned family (0.63) sit closer to nominal.

## Where this leads

The available-case modified ITT suite supplies the primary assigned-arm estimates; the rest of the suite asks whether they replicate under other designs and what mechanism carries them. The DiD crossover models (note 05) and the gain-factor ANCOVA (note 03) re-estimate related effects from different rows and a different identification argument; the mediation family (note 08) tests whether the word-reading gain runs through letter-sound knowledge.
