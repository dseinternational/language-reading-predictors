<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Decision: drop the vocabulary → letter-sound analyses (issue #405, PR #407)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Records a scoping decision taken 2026-07-24: the taught- and standardised-vocabulary → letter-sound analyses are dropped, not deferred. Issue #405 is closed as won't-do and PR #407 closed unmerged; the reusable factory capability is salvaged separately.

## What was dropped

Four `kind="mechanism"` models specified in issue #405 and built in PR #407, all with letter-sound knowledge `L` as the outcome and taught vocabulary as the exposure:

| Model              | Path     | Readout                                 |
| ------------------ | -------- | --------------------------------------- |
| `lrp-rli-mech-102` | `TR → L` | concurrent (`TR_post → L_post ∣ L_pre`) |
| `lrp-rli-mech-103` | `TR → L` | lagged (`TR_pre → L_post ∣ L_pre`)      |
| `lrp-rli-mech-104` | `TE → L` | concurrent                              |
| `lrp-rli-mech-105` | `TE → L` | lagged                                  |

The decision extends beyond those four to the class: no analysis in this project takes vocabulary — taught or standardised, receptive or expressive — as a directional predictor of letter-sound knowledge. Nothing had been merged, so no model, report, or finding is withdrawn; the branch is the only casualty.

## Why

**The committed DAG posits no vocabulary → letter-sound edge, in either direction of the plausible mechanism.** In `dag/dag-language-reading-lagged.dagitty` the parents of `LS` at each wave are speech `SP`, hearing `HS`, age `A`, latent general ability `GA` and the intervention, plus `LS_t` and (since 2026-07-17) `WR_t`. No vocabulary node is a parent of `LS`. The literature route that would motivate one — lexical restructuring, in which vocabulary growth sharpens phonological representations and thence phoneme awareness — is committed the other way round in our graph: `LS_t → PA_t`, not `PA → LS`. So the question is not merely unsupported by the DAG; the DAG takes the opposite position on the only mechanism that would license it.

**The indirect routes that do exist are already decomposed by existing models.** The lagged DAG _does_ contain directed paths from taught vocabulary to letter sounds — `TR_t → WR_t → LS_t1`, `TE_t → WR_t → LS_t1`, `TE_t → PA_t → WR_t → LS_t1` — but every one runs through word reading, via the reverse `WR_t → LS_t1` edge added on 2026-07-17. Both legs are already fitted: `TR`/`TE` → word reading in `mech-088`/`089`/`188`/`189`, and word reading → letter sounds in `med-176` (`W_t2 → L_t4`), the model built specifically to exercise that reverse edge. A marginal vocabulary → letter-sound slope would not add a leg; it would collapse an already-decomposed composite into a single coefficient with no posited generating mechanism behind it.

**Identification consumed the phenomenon.** The review of PR #407 established that the lagged models' adjustment set — copied from the concurrent siblings — left 28 open backdoor paths for `TE` and 14 for `TR`, all of the form `X_t ← {TR_t ∣ RW_t} → … → WR_t → LS_t1`. A verified re-derivation against the lagged DAG found minimal sufficient sets of `{A, HS, RW}` + `L_pre` for `TR → LS_t1` and `{A, HS, RW, SP, TR}` + `L_pre` for `TE → LS_t1`. Both require conditioning on the phonological-memory and (for `TE`) taught-receptive paths — that is, blocking precisely the routes carrying whatever substance the association had. What survives is a slope whose remaining content is latent-`GA` confounding plus shared intervention dose. When the adjustment set needed for identification is the set that removes the phenomenon, the analysis is reporting that the question is ill-posed rather than that the answer is null.

**Independent measurement reason.** `TR`/`TE` are the block-1 taught word lists, whose denominators are unconfirmed. This is why the horseshoe ranking cross-checks (`hs-003`/`hs-004`) already exclude taught vocabulary from their predictor sets. That is poor footing for a headline analysis irrespective of the DAG.

## The framing to use if this is raised again

The descriptive pass that motivated #405 is not in dispute: taught receptive and expressive vocabulary are strong concurrent correlates of letter-sound knowledge (pooled _r_ ≈ 0.62 and 0.65, rising across waves) and weak-to-negative predictors of subsequent letter-sound gain. The correct account of that pattern is shared latent general ability plus shared intervention dose — both children's vocabulary and their letter-sound knowledge respond to the same teaching and the same underlying ability — with, in the lagged graph, a genuine indirect contribution routed through word reading.

State it as: **no direct edge is posited; the indirect route is fully covered by existing models; a marginal slope would be an uninterpretable composite.** Avoid the shorter "vocabulary does not affect letter sounds", which is true only of a _direct_ edge and reads as contradicting `med-176`, built to estimate exactly the reading-feeds-back-into-the-code path.

## Scope: what was not dropped

Vocabulary still appears as one predictor among many where letter sounds are the outcome. These are not directional vocabulary → letter-sound analyses, and removing vocabulary from them would break their designs. They stand:

- **`ca-002`** — concurrent conditional associations, `L` on `{W, B, TR, TE, R, E}`. The family describes the conditional joint distribution of the core skill set, each measure focal in turn; every coefficient is already flagged an adjusted association subject to the Table-2-fallacy caveat.
- **`hs-003` / `hs-004`** — regularised-horseshoe ranking cross-checks for letter-sound gain and level, with `R`/`E` among the predictors (taught vocabulary already excluded).
- **`gbg-009` / `gbl-009`** — the gradient-boosting letter-sound models, whose predictor sets come from `Predictors.DEFAULT_*`.

Issues #404 (letter-sound → word-reading moderation by blending / decoding) and #421 (letter-sound → word-reading follow-ups) ask the opposite-direction question and are unaffected.

## What was kept from PR #407

The `mechanism_at_pre` capability on `build_mechanism_model` — a default-off flag that takes the mechanism regressor from the period-start logit, giving the lagged form `mechanism_pre → outcome_post`, with the reporting writers and the `PreparedData.pre_counts` companion made alignment-aware so the artefacts describe the vector actually fitted. It is model-agnostic and generic to the mechanism family; #421 may want lagged readouts. Salvaged to its own branch with its tests; the four model modules, their report templates and their registry entries are not.
