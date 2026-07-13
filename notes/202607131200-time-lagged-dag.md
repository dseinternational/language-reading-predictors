<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Time-lagged (wave-unrolled) DAG — structure and rationale (#250)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8); revised for the adopted Option A structure by Claude Code/Fable 5.

Separate workstream agreed at the 2026-07-10 DAG revision (decision 7; `notes/202607101100-dag-revision-team-decisions.md`), and the highest-value structural upgrade flagged in the critical review (`notes/202607091430-dag-critical-review-td-atypical-literature.md`, §"wave-unrolled companion DAG"). The graph is `dag/dag-language-reading-lagged.dagitty`; this note records what it is, the design decisions, and what it buys.

**Structure adopted 2026-07-13: Option A** (the base DAG copied at each wave). The A-vs-B considerations — plausibility, identification, honesty of the record, adjustment sets — are recorded in the PR #288 discussion and in `notes/202607131300-time-lagged-dag-options.md`. Still open for review: the per-edge reverse-edge justifications and a crossover-aware slice.

## Why unroll

The authoritative `dag/dag-language-reading.dagitty` is **contemporaneous** (single-wave): every skill→skill edge is a concurrent association whose direction is assigned by theory, and it cannot express maturation or the direction of change over the four waves. Making measurement occasions explicit does four things at once:

1. **Reciprocal edges become representable** instead of confessed-away — the forward `PA → WR` path (within-wave under the adopted Option A) _and_ `WR_t → PA_{t+1}` can both appear, so which direction dominates is an empirical (cross-lagged) question rather than an acyclicity casualty.
2. **Reading → language coexists with language → reading** without a cycle — `WR_t → EV_{t+1}` (the founding RLI hypothesis) alongside the within-wave `EV → PA → WR` cascade.
3. **Adjustment sets become readable off the graph** — each outcome's confounders are its **parents at the prior wave**. This would make confounder choices derivable rather than judged case-by-case. Note the #247 sweep is already complete under the contemporaneous graph, so this helps future sweeps, not that one.
4. **Temporal precedence is explicit**, which supports the #264 resolution that baseline vocabulary is an admissible mediation confounder. It does not settle it on its own: the temporal-precedence argument that resolved #264 does not depend on this file.

## Convention: a two-slice template

The file is a **template**, not a full unroll: suffix `_t` is a wave and `_t1` the next wave, and the same structure repeats over each adjacent transition (t1→t2, t2→t3, t3→t4). Assessments are ~20 weeks apart — the 40-week programme spans two transitions. A full t1–t4 unroll is causally equivalent for d-separation but ~4× the nodes and unreadable. The one thing the template cannot show — that the intervention's active window is arm-specific — is recorded below.

## Design decisions

- **Within-wave cascade (Option A — adopted 2026-07-13).** Each wave holds the full base-DAG skill cascade; waves are joined by carry-over (`X_t → X_{t+1}`) and the lagged reverse `WR` edges only. Forward influence of one skill on a _different_ skill at the next wave is mediated (`X_t → X_{t+1} → Y_{t+1}` and `X_t → Y_t → Y_{t+1}`) rather than drawn as direct cross-lagged edges — the standard unrolled treatment. Rationale: assessments are ~20 weeks apart, and the tightly coupled cascade steps (letter-sounds feeding blending and decoding) act on a timescale of days to weeks; a pure-lagged graph would assert those dynamics do not exist. The within-wave paths are part of the causal description, **not** an estimation demand: at n≈54 the fitted models estimate lagged, pooled couplings only (see the model-plan note), and the gap between graph and model is documented rather than hidden. The previously flagged within-wave `LS→WR` carve-out is moot under A — the whole cascade is within-wave. The pure-lagged alternative (Option B) is kept drawn in `dag/dag-language-reading-lagged.svg` for the decision record.
- **Reverse / reciprocal edges added** (all lagged): `WR→EV`, `WR→RV` (reading → vocabulary — the founding RLI hypothesis and the print-exposure/Matthew effect; the critical review's "most consequential missing structure"), `WR→PA` (reciprocal with `PA→WR` — enables the cross-lagged dominance test), and `WR→RW` (reading → phonological memory — **the most tentative; easy to drop** **[flagged for review]**).
- **`HS` kept a stable exogenous root.** `A→HS` (age-related conductive hearing change) is biologically defensible, but `hearing_c` is recorded time-invariantly in this cohort (the same 44/54 non-missing at every wave — one baseline classification carried across waves), so a time-varying `HS_t` or an `A→HS` edge is unidentifiable here. Revisit only if per-wave audiology data appears.
- **Intervention timing is arm-specific.** `IG` (randomised at baseline) drives the window's sessions `IS` and, ITT, the taught skills at the next wave. The **active window differs by arm** — immediate arm t1→t2, waitlist arm t2→t3 — which the generic template shows as a single active transition; models must apply it to the correct window per arm.

Verified acyclic (36 nodes, 195 edges).

## What it newly identifies

These are claims about what the graph makes _derivable_. None is exercised by a fitted model in #250; they land with the #264 and #229 follow-ons.

- **#264 (mediation E/R).** With waves explicit, `E_t1`/`R_t1` are unambiguously **pre-treatment** for the t1→t2 window, so the descendant-of-treatment argument that #246 used (and that was overturned) does not apply to the baseline value. Reading the graph, a mediator `M_{t+1}`'s admissible confounders are its prior-wave parents, which _include_ baseline vocabulary where vocabulary is a parent — so `E_t1`/`R_t1` come out **in** for the vocabulary-adjacent mediators, on a derivation rather than an assertion. The per-mediator derivation is #264's deliverable.
- **Cross-lagged dominance tests** — the forward `PA→WR` path vs the lagged `WR_t→PA_{t+1}`; reading→language (`WR→EV`) — become first-class hypotheses the LCSM machinery can probe.
- **Maturation / direction of change** — expressible via autoregression + per-wave age.

## Scope of #250 vs follow-ons

Per the issue, #250 delivers the **graph + this note + a model-family scope**, not fitted models. The lagged graph is the natural substrate for the **cross-lagged panel / LCSM** family (#229) and resolves the mediation adjustment question (#264). Building those models is their own work; #250 makes them derivable.
