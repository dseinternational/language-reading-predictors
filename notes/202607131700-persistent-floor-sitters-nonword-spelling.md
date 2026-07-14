<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Persistent floor-sitters on nonword reading and phonetic spelling: the descriptive cut

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8). Every figure below was recomputed directly from `data/rli_data_long.csv` by `scripts/descriptive/floor_sitters.py`; re-run that script to reproduce the tables and figures before they enter a report.

Date: 2026-07-13 — relates to issue #230 §5 (persistent floor-sitters), companion to `notes/202607091318-future-predictor-exploration-directions.md`.

## The question

The education team asked about the children still scoring zero on the phonics measures at the end of the study. This note is the descriptive groundwork: how many children are still at floor on nonword reading (`nonword`, /6) and phonetic spelling (`spphon`, /92) at the final wave, whether that is a small hard tail or a large slow-moving group, how much of it is a genuinely stuck group versus boundary noise, and what distinguishes the floored children. It fixes the definitions that the discrete-time survival model (Phase 2) then estimates a _when_ for. It is purely descriptive; it fits no model and makes no causal claim.

Throughout, **at floor** means a recorded score of exactly zero.

## The floor is nearly half the cohort, and it is still falling

The floor rate falls steadily across the four waves for both measures — the tests are still developmentally hard for roughly half the cohort at the end, and that half is still in motion rather than stuck at a hard ceiling of ability.

| Measure           | t1          | t2          | t3          | t4          |
| ----------------- | ----------- | ----------- | ----------- | ----------- |
| Nonword reading   | 72% (36/50) | 64% (34/53) | 52% (27/52) | 40% (21/52) |
| Phonetic spelling | 78% (42/54) | 64% (34/53) | 57% (30/53) | 48% (26/54) |

At t4, 16 children (of the 52 with both scores observed) are at floor on **both** measures. The floor is arm-balanced at t4, so there is no crude earlier-start / extra-period signal in who remains: on observed t4 scores, nonword floor is 10/27 in the immediate arm (group 1) versus 11/25 in the waitlist arm (group 2); phonetic spelling is 13/28 versus 13/26.

## A sustained-event definition is needed, because some movement is boundary flicker

A single crossing above zero is noise-prone at these low ceilings. Among the children observed at all four waves, a meaningful minority come off the floor and then fall back to zero by t4, so the fixed-t4 snapshot both over- and under-counts the stuck group depending on the rule used.

| Measure           | four-wave completers | never off floor | off then back to floor by t4 | off floor at t4 |
| ----------------- | -------------------- | --------------- | ---------------------------- | --------------- |
| Nonword reading   | 48                   | 12              | 7                            | 29              |
| Phonetic spelling | 53                   | 24              | 1                            | 28              |

For the survival model (Phase 2) this note fixes three definitions:

- **Persistent floor-sitter** — never records a score above zero across all observed waves (nonword: 12 of 48 completers; phonetic spelling: 24 of 53). This is the noise-proof stuck group.
- **Off-floor event (survival primary)** — the first wave at which a child records a score above zero. Simple and uses no future information, but treats the 7 flicker children (nonword) as events.
- **Sustained off-floor event (survival sensitivity)** — the first wave above zero that is _not_ followed by a return to zero at any later observed wave. This removes flicker but is only defined for children with enough follow-up. A graded **near-floor band** (for example, counting a score of 0–1 as still effectively floored) is the third option and damps one-point boundary movement without discarding follow-up; the choice between the sustained rule and the near-floor band is deferred to the model fit.

## The floor mostly reflects incomplete letter-sound prerequisites, not an inability to decode

The single strongest correlate of staying at the nonword floor is how far the child has got with letter sounds — the direct teaching target and the outcome the intervention moves most. Cross-tabulating the t4 nonword floor against concurrent letter-sound knowledge (YARC-LSK, /32):

| Concurrent letter sounds | at nonword floor |
| ------------------------ | ---------------- |
| < 16                     | 6/7 (86%)        |
| 16–25                    | 11/23 (48%)      |
| 26–32                    | 4/22 (18%)       |

The floored children (n = 21 at t4) sit early on the decoding path rather than at a hard limit: median letter sounds 20/32, median blending 5/10 with only 7/21 reaching blending ≥ 6, and yet many sight-read some real words (median word reading 5, maximum 34 of 79) without decoding. Floor status is a prerequisite-timing signal, not evidence that decoding is unreachable.

## A four-child production puzzle

Four children have near-complete letter sounds (≥ 26/32) and some blending and word reading, yet score zero on nonword reading — and one of them spells 53 words phonetically while reading no nonwords aloud.

| subject_id          | letter sounds | blending | word reading | nonword | phonetic spelling |
| ------------------- | ------------- | -------- | ------------ | ------- | ----------------- |
| ID_0D23DC86A37B807D | 31            | 10       | 21           | 0       | 53                |
| ID_43A40173C81CBD21 | 28            | 9        | 12           | 0       | 0                 |
| ID_BB12CED680895743 | 27            | 10       | 9            | 0       | 44                |
| ID_C722B3B63C6E496F | 26            | 5        | 7            | 0       | 0                 |

Spelling words phonetically while scoring zero on _reading_ nonwords aloud points at the speech-production demand of the nonword task (reading aloud) rather than at missing decoding knowledge. This is a case review at n = 4, not a model (Phase 4), and it connects to the speech battery (`deapp*`, `lsam*`) that currently enters no model.

## Missingness and reconciling the "stuck" counts

The dataset already carries subject-level flags `nonword_none` and `spphon_none`, both constant within a child. These flag more children than the four-wave-completer count of persistent floor-sitters, because they include children with incomplete follow-up who never scored above zero on the waves they _were_ observed:

| Indicator           | subjects flagged | completer "never off floor" |
| ------------------- | ---------------- | --------------------------- |
| `nonword_none == 1` | 15               | 12 (of 48 completers)       |
| `spphon_none == 1`  | 25               | 24 (of 53 completers)       |

Any report figure must state which denominator it uses. This note uses **non-missing scores per wave** for the floor-rate table, and **four-wave completers** for the sustained/flicker/persistent split. One subject has no nonword data at any wave (four all-missing rows) and drops out of every nonword cut.

## The causal caveat, kept visible

By t4 both arms have been treated, so "still at floor despite intervention" is **prognostic, not evidence of failure**: the untreated counterfactual is unknowable and could be worse. Floor status mostly reflects incomplete letter-sound foundations — the outcome the intervention moves most (ITT τ = +0.110, `P(τ>0) = 0.999`) — which points towards continue-and-extend, not screen-out. The survival model in Phase 2 inherits this framing: treatment enters as a hazard shift on time-to-off-floor, read as a prognostic association anchored on the immediate arm's randomised period, not as a licence to gate-keep.

## Reproducing this cut

```bash
python scripts/descriptive/floor_sitters.py
```

Tables print to stdout and are written as CSVs; the three figures (floor-rate trajectories, the letter-sound-band cross-tab, and per-child nonword trajectories coloured by completer category) are written to `output/descriptive/` (gitignored). Only the script and this note are committed.
