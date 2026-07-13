<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Causal DAGs

The authoritative causal DAG(s) for the study. This is the design artefact the **step-2 Bayesian models** (ITT, joint, mechanism, mediation, dose-response, gain/level-factor, …) are built against; it encodes causal **structure only** — exposure/outcome roles are assigned per analysis.

## Files

| File                                                | What it is                                                                                                                                                                                                            |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dag-language-reading.dagitty`                      | Authoritative machine-readable base DAG (contemporaneous / single-wave). Paste into [dagitty.net](https://www.dagitty.net/dags.html) to render or inspect adjustment sets.                                            |
| `dag-language-reading.dot`                          | Graphviz source for a colour-coded, left-to-right rendering.                                                                                                                                                          |
| `dag-language-reading.svg`                          | Rendered figure (regenerated from the `.dot`; self-contained, safe to embed).                                                                                                                                         |
| `dag-language-reading-lagged.dagitty`               | **Draft (#250)** time-lagged / wave-unrolled companion graph — a two-slice `_t → _t1` template. Rationale + design decisions: [`../notes/202607131200-time-lagged-dag.md`](../notes/202607131200-time-lagged-dag.md). |
| `dag-language-reading-lagged.dot` / `.svg`          | Rendered **Option B (pure-lagged)**, the current draft, in house style.                                                                                                                                               |
| `dag-language-reading-lagged-per-wave.dot` / `.svg` | Rendered **Option A (base DAG copied per wave)**, the alternative under review.                                                                                                                                       |

The prose exposition of the structure — its assumptions, the TD/DS/IDD evidence, honest weaknesses and alternatives considered — is maintained as a review draft in [`../notes/202607101444-dag-explanation-review-draft.md`](../notes/202607101444-dag-explanation-review-draft.md).

Regenerate the figure after editing the `.dot`:

```bash
dot -Tsvg dag/dag-language-reading.dot -o dag/dag-language-reading.svg
```

Keep the `.dagitty` file and the `.dot`/`.svg` in step: the `.dagitty` file is the source of truth for structure; the Graphviz files are a view of it (with the two universal parents `A` and `GA` summarised in a note rather than drawn, for legibility — the only place the picture departs from the `.dagitty` block).

## Provenance

The current graph is the **2026-07-10 team revision**; see [`../notes/202607101100-dag-revision-team-decisions.md`](../notes/202607101100-dag-revision-team-decisions.md) for the decision record and the follow-up model-adjustment issues, and [`../notes/202606231600-dag-revision-consolidated.md`](../notes/202606231600-dag-revision-consolidated.md) for the superseded 2026-06-23 structure and its full deliberation.

## Time-lagged (wave-unrolled) companion — draft, under review

A **time-lagged DAG** (decision 7 of the 2026-07-10 revision) makes measurement occasions explicit, so maturation, the direction of change and reciprocal edges become representable and each model's adjustment set is readable as the prior-wave parents. First-cut draft: [`dag-language-reading-lagged.dagitty`](dag-language-reading-lagged.dagitty), a two-slice `_t → _t1` template. Its structure and design decisions are recorded in [`../notes/202607131200-time-lagged-dag.md`](../notes/202607131200-time-lagged-dag.md); it supports the mediation adjustment question in #264 and enables the cross-lagged / LCSM tests in #229, and does not yet drive any fitted models.

**Open decision — two ways to unroll.** The draft `.dagitty` currently encodes **Option B (pure-lagged):** no within-wave skill→skill edges, so every skill effect acts across a wave. The alternative is **Option A (base DAG copied per wave):** keep the full within-wave cascade in each wave and add carry-over plus the reverse edges. Option A is the recommendation — only the reverse reading→language edges truly need the time gap. Both are drawn and explained for the team in [`../notes/202607131300-time-lagged-dag-options.md`](../notes/202607131300-time-lagged-dag-options.md), with figures [`dag-language-reading-lagged-per-wave.svg`](dag-language-reading-lagged-per-wave.svg) (Option A) and [`dag-language-reading-lagged.svg`](dag-language-reading-lagged.svg) (Option B).
