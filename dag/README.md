<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Causal DAGs

The authoritative causal DAG(s) for the study. This is the design artefact the **step-2 Bayesian models** (ITT, joint, mechanism, mediation, dose-response, gain/level-factor, …) are built against; it encodes causal **structure only** — exposure/outcome roles are assigned per analysis.

## Files

| File                                | What it is                                                                                                                                                                 |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dag-language-reading.dagitty`      | Authoritative machine-readable base DAG (contemporaneous / single-wave). Paste into [dagitty.net](https://www.dagitty.net/dags.html) to render or inspect adjustment sets. |
| `dag-language-reading.dot`          | Graphviz source for a colour-coded, left-to-right rendering.                                                                                                               |
| `dag-language-reading.svg`          | Rendered figure (regenerated from the `.dot`; self-contained, safe to embed).                                                                                              |
| `dag-language-reading-explained.md` | Prose exposition: the structure, its assumptions, the TD/DS/IDD evidence, honest weaknesses, and the alternatives considered.                                              |

Regenerate the figure after editing the `.dot`:

```bash
dot -Tsvg dag/dag-language-reading.dot -o dag/dag-language-reading.svg
```

Keep the `.dagitty` file and the `.dot`/`.svg` in step: the `.dagitty` file is the source of truth for structure; the Graphviz files are a view of it (with the two universal parents `A` and `GA` summarised in a note rather than drawn, for legibility — the only place the picture departs from the `.dagitty` block).

## Provenance

The current graph is the **2026-07-10 team revision**; see [`../notes/202607101100-dag-revision-team-decisions.md`](../notes/202607101100-dag-revision-team-decisions.md) for the decision record and the follow-up model-adjustment issues, and [`../notes/202606231600-dag-revision-consolidated.md`](../notes/202606231600-dag-revision-consolidated.md) for the superseded 2026-06-23 structure and its full deliberation.

## Planned

A **time-lagged DAG** (and its matching models) is a separate workstream and will be recorded here alongside the contemporaneous graph (e.g. `dag-language-reading-lagged.dagitty`).
