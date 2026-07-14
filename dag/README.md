<!-- SPDX-License-Identifier: CC-BY-4.0 -->

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

The prose exposition of the structure — its assumptions, the TD/DS/IDD evidence, honest weaknesses and alternatives considered — is maintained as a review draft in [`../notes/202607101444-dag-explanation-review-draft.md`](../notes/202607101444-dag-explanation-review-draft.md).

Regenerate the figure after editing the `.dot`:

```bash
dot -Tsvg dag/dag-language-reading.dot -o dag/dag-language-reading.svg
```

Keep the `.dagitty` file and the `.dot`/`.svg` in step: the `.dagitty` file is the source of truth for structure; the Graphviz files are a view of it (with the two universal parents `A` and `GA` summarised in a note rather than drawn, for legibility — the only place the picture departs from the `.dagitty` block).

## Provenance

The current graph is the **2026-07-10 team revision**; see [`../notes/202607101100-dag-revision-team-decisions.md`](../notes/202607101100-dag-revision-team-decisions.md) for the decision record and the follow-up model-adjustment issues, and [`../notes/202606231600-dag-revision-consolidated.md`](../notes/202606231600-dag-revision-consolidated.md) for the superseded 2026-06-23 structure and its full deliberation.

## Time-lagged (wave-unrolled) companion — Option A adopted 2026-07-13

A **time-lagged DAG** (decision 7 of the 2026-07-10 revision) makes measurement occasions explicit, so maturation, the direction of change and reciprocal edges become representable and each model's adjustment set is readable as the prior-wave parents. The graph: [`dag-language-reading-lagged.dagitty`](dag-language-reading-lagged.dagitty), a two-slice `_t → _t1` template (assessments ~20 weeks apart). Its structure and design decisions are recorded in [`../notes/202607131200-time-lagged-dag.md`](../notes/202607131200-time-lagged-dag.md); it supports the mediation adjustment question in #264 and enables the cross-lagged / LCSM tests in #229, and does not yet drive any fitted models.

**Structure decision — resolved.** Two ways to unroll were drawn and explained for the team in [`../notes/202607131300-time-lagged-dag-options.md`](../notes/202607131300-time-lagged-dag-options.md): **Option A (base DAG copied per wave)** — the full within-wave cascade in each wave, joined by carry-over plus the lagged reverse edges from word reading (`WR → TE`, `WR → TR`, `WR → PA`, `WR → RW`; target set directed 2026-07-13) — and **Option B (pure-lagged)** — no within-wave skill→skill edges. **Option A was adopted 2026-07-13**; the reasoning (plausibility at the 20-week interval, identification, honesty of the record, adjustment sets) is recorded in the PR #288 discussion. Figures: [`dag-language-reading-lagged-per-wave.svg`](dag-language-reading-lagged-per-wave.svg) (Option A, adopted) and [`dag-language-reading-lagged.svg`](dag-language-reading-lagged.svg) (Option B, kept for the record). Still open: the per-edge reverse-edge justifications and a crossover-aware slice.
