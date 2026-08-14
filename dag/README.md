<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Byrne/RLM DAG documentation added by a LLM-based AI tool (Codex/GPT-5).

# Causal DAGs

The authoritative causal DAG(s) for the study. This is the design artefact the **step-2 Bayesian models** (ITT, joint, mechanism, mediation, dose-response, gain/level-factor, …) are built against; it encodes causal **structure only** — exposure/outcome roles are assigned per analysis.

## Files

| File                                                | What it is                                                                                                                                                                                                                                                                       |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dag-language-reading.dagitty`                      | Authoritative machine-readable base DAG (contemporaneous / single-wave). Paste into [dagitty.net](https://www.dagitty.net/dags.html) to render or inspect adjustment sets.                                                                                                       |
| `dag-language-reading.dot`                          | Graphviz source for a colour-coded, left-to-right rendering.                                                                                                                                                                                                                     |
| `dag-language-reading.svg`                          | Rendered figure (regenerated from the `.dot`; self-contained, safe to embed).                                                                                                                                                                                                    |
| `dag-language-reading-lagged.dagitty`               | Time-lagged / wave-unrolled companion graph (#250) — a two-slice `_t → _t1` template encoding **Option A (base DAG copied per wave), adopted 2026-07-13**. Rationale + design decisions: [`../notes/202607131200-time-lagged-dag.md`](../notes/202607131200-time-lagged-dag.md). |
| `dag-language-reading-lagged-per-wave.dot` / `.svg` | Rendered **Option A** — the adopted structure (with `A`/`GA`/`HS`/`IG`/`IS` summarised in a note for legibility).                                                                                                                                                                |
| `dag-language-reading-lagged.dot` / `.svg`          | Rendered **Option B (pure-lagged)** — considered and **not adopted**; kept for the decision record.                                                                                                                                                                              |
| `dag-reading-language-memory.dagitty`               | Adopted working contemporaneous graph for the observational Byrne/RLM cohort (`study_id="rlm"`); no arrow or group contrast is causal.                                                                                                                                           |
| `dag-reading-language-memory.dot` / `.svg`          | Rendered contemporaneous Byrne/RLM graph.                                                                                                                                                                                                                                        |
| `dag-reading-language-memory-lagged.dagitty`        | Adopted annual two-slice Byrne/RLM companion: copied-per-wave structure, autoregressive carry-over, and the three source-anchored reverse edges from prior word reading to later receptive vocabulary, receptive grammar and digit recall.                                       |
| `dag-reading-language-memory-lagged.dot` / `.svg`   | Rendered Byrne/RLM lagged companion; common-cause edges are summarised for legibility.                                                                                                                                                                                           |

The prose exposition of the structure — its assumptions, the TD/DS/IDD evidence, honest weaknesses and alternatives considered — is maintained as a review draft in [`../notes/202607101444-dag-explanation-review-draft.md`](../notes/202607101444-dag-explanation-review-draft.md).

Regenerate the figure after editing the `.dot`:

```bash
dot -Tsvg dag/dag-language-reading.dot -o dag/dag-language-reading.svg
dot -Tsvg dag/dag-reading-language-memory.dot -o dag/dag-reading-language-memory.svg
dot -Tsvg dag/dag-reading-language-memory-lagged.dot -o dag/dag-reading-language-memory-lagged.svg
```

Keep the `.dagitty` file and the `.dot`/`.svg` in step: the `.dagitty` file is the source of truth for structure; the Graphviz files are a view of it (with the two universal parents `A` and `GA` summarised in a note rather than drawn, for legibility — the only place the picture departs from the `.dagitty` block).

## Provenance

The current graph is the **2026-07-10 team revision**; see [`../notes/202607101100-dag-revision-team-decisions.md`](../notes/202607101100-dag-revision-team-decisions.md) for the decision record and the follow-up model-adjustment issues, and [`../notes/202606231600-dag-revision-consolidated.md`](../notes/202606231600-dag-revision-consolidated.md) for the superseded 2026-06-23 structure and its full deliberation.

## Time-lagged (wave-unrolled) companion — Option A adopted 2026-07-13

A **time-lagged DAG** (decision 7 of the 2026-07-10 revision) makes measurement occasions explicit, so maturation, the direction of change and reciprocal edges become representable and each model's adjustment set is readable as the prior-wave parents. The graph: [`dag-language-reading-lagged.dagitty`](dag-language-reading-lagged.dagitty), a two-slice `_t → _t1` template (assessments ~20 weeks apart). Its structure and design decisions are recorded in [`../notes/202607131200-time-lagged-dag.md`](../notes/202607131200-time-lagged-dag.md); it supports the mediation adjustment question in #264 and enables the cross-lagged / LCSM tests in #229, and does not yet drive any fitted models.

**Structure decision — resolved.** Two ways to unroll were drawn and explained for the team in [`../notes/202607131300-time-lagged-dag-options.md`](../notes/202607131300-time-lagged-dag-options.md): **Option A (base DAG copied per wave)** — the full within-wave cascade in each wave, joined by carry-over plus the lagged reverse edges from word reading (`WR → TE`, `WR → TR`, `WR → PA`, `WR → RW`; target set directed 2026-07-13, with `WR → LS` added 2026-07-17 to enable the reverse-direction test — see [`../notes/202607172100-reverse-mediation-wr-ls-direction-spec.md`](../notes/202607172100-reverse-mediation-wr-ls-direction-spec.md), models LRP-RLI-MED-176/276 — and `WR → NW` adopted as a provisional working-DAG assumption on 2026-08-08 — see [`../notes/202608081900-decision-wr-nw-lagged-edge.md`](../notes/202608081900-decision-wr-nw-lagged-edge.md)) — and **Option B (pure-lagged)** — no within-wave skill→skill edges. **Option A was adopted 2026-07-13**; the reasoning (plausibility at the 20-week interval, identification, honesty of the record, adjustment sets) is recorded in the PR #288 discussion. Figures: [`dag-language-reading-lagged-per-wave.svg`](dag-language-reading-lagged-per-wave.svg) (Option A, adopted) and [`dag-language-reading-lagged.svg`](dag-language-reading-lagged.svg) (Option B, kept for the record). Still open: the per-edge reverse-edge justifications and a crossover-aware slice.

## Byrne/RLM annual lagged companion — adopted 2026-08-14

The Byrne/RLM graph is a separate observational-cohort structure, not a relabelled intervention DAG. Its lagged companion [`dag-reading-language-memory-lagged.dagitty`](dag-reading-language-memory-lagged.dagitty) uses the copied-per-wave structure at an annual interval and pre-specifies exactly `basread_t → {bpvs_t1, trog_t1, basdig_t1}`. These edges encode Byrne, MacDonald and Buckley's language-and-auditory-memory question; the visual-memory variables reported in the paper are absent from the repository. Latent general ability and hearing keep every coupling residually confounded, `readgrp` is not a treatment, and the reading-matched group was selected on `basread`.

The graph is approved as a structural working hypothesis only. No reciprocal model is registered by the DAG PR: `notes/202608141700-byrne-lagged-dag-decision.md` requires a simulation gate over the real missingness patterns first, and distinguishes the paper's waves 1–3 from later prepared-data sensitivities.
