<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Time-lagged (wave-unrolled) DAG — draft and rationale (#250)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8). The causal structure is the team's to revise; this is a first cut for red-lining.

Separate workstream agreed at the 2026-07-10 DAG revision (decision 7; `notes/202607101100-dag-revision-team-decisions.md`), and the highest-value structural upgrade flagged in the critical review (`notes/202607091430-dag-critical-review-td-atypical-literature.md`, §"wave-unrolled companion DAG"). The graph is `dag/dag-language-reading-lagged.dagitty`; this note records what it is, the design decisions, and what it buys.

## Why unroll

The authoritative `dag/dag-language-reading.dagitty` is **contemporaneous** (single-wave): every skill→skill edge is a concurrent association whose direction is assigned by theory, and it cannot express maturation or the direction of change over the four waves. Making measurement occasions explicit does four things at once:

1. **Reciprocal edges become representable** instead of confessed-away — `PA_t → WR_{t+1}` _and_ `WR_t → PA_{t+1}` can both appear, so which direction dominates is an empirical (cross-lagged) question rather than an acyclicity casualty.
2. **Reading → language coexists with language → reading** without a cycle — `WR_t → EV_{t+1}` (the founding RLI hypothesis) alongside `EV_t → PA_{t+1} → WR_{t+2}`.
3. **Adjustment sets become readable off the graph** — each outcome's confounders are its **parents at the prior wave**, which removes the judgement the contemporaneous graph forced onto the gain/level-factor sweep (#247).
4. **Temporal precedence is explicit**, which settles whether baseline vocabulary is an admissible mediation confounder (#264, below).

## Convention: a two-slice template

The file is a **template**, not a full unroll: suffix `_t` is a wave and `_t1` the next wave, and the same structure repeats over each adjacent transition (t1→t2, t2→t3, t3→t4). A full t1–t4 unroll is causally equivalent for d-separation but ~4× the nodes and unreadable. The one thing the template cannot show — that the intervention's active window is arm-specific — is recorded below.

## Design decisions

- **Pure lagged.** No within-wave causal edges among skills; every skill→skill effect is `X_t → Y_{t+1}`, with only residual correlation within a wave. Only the exogenous drivers — age `A`, latent ability `GA`, hearing `HS` — act within a wave (they are not skills whose direction is in question). Rationale: temporal precedence is the entire point of unrolling; a within-wave arrow would re-assign direction by theory, the exact flaw being fixed. **Cost, stated plainly:** a "lag" bundles any faster-than-a-wave dynamics (e.g. letter-sounds converting to decoding inside one ~annual wave) into the lagged coefficient rather than a separate within-wave path. **[Flagged for review]** — a within-wave `LS→WR`/decoding exception is the one plausible carve-out if sub-wave ordering must be separated.
- **Reverse / reciprocal edges added** (all lagged): `WR→EV`, `WR→RV` (reading → vocabulary — the founding RLI hypothesis and the print-exposure/Matthew effect; the critical review's "most consequential missing structure"), `WR→PA` (reciprocal with `PA→WR` — enables the cross-lagged dominance test), and `WR→RW` (reading → phonological memory — **the most tentative; easy to drop** **[flagged for review]**).
- **`HS` kept a stable exogenous root.** `A→HS` (age-related conductive hearing change) is biologically defensible, but `hearing_c` is recorded time-invariantly in this cohort (the same 44/54 non-missing at every wave — one baseline classification carried across waves), so a time-varying `HS_t` or an `A→HS` edge is unidentifiable here. Revisit only if per-wave audiology data appears.
- **Intervention timing is arm-specific.** `IG` (randomised at baseline) drives the window's sessions `IS` and, ITT, the taught skills at the next wave. The **active window differs by arm** — immediate arm t1→t2, waitlist arm t2→t3 — which the generic template shows as a single active transition; models must apply it to the correct window per arm.

Verified acyclic (37 nodes, 157 edges).

## What it newly identifies

- **#264 (mediation E/R).** With waves explicit, `E_t1`/`R_t1` are unambiguously **pre-treatment** for the t1→t2 window, so the descendant-of-treatment argument that #246 used (and that was overturned) does not apply to the baseline value. Reading the graph, a mediator `M_{t+1}`'s admissible confounders are its prior-wave parents, which _include_ baseline vocabulary where vocabulary is a parent — so `E_t1`/`R_t1` come out **in** for the vocabulary-adjacent mediators, on a derivation rather than an assertion. The per-mediator derivation is #264's deliverable.
- **Cross-lagged dominance tests** — `PA_t→WR_{t+1}` vs `WR_t→PA_{t+1}`; reading→language (`WR→EV`) — become first-class hypotheses the LCSM machinery can probe.
- **Maturation / direction of change** — expressible via autoregression + per-wave age.

## Scope of #250 vs follow-ons

Per the issue, #250 delivers the **graph + this note + a model-family scope**, not fitted models. The lagged graph is the natural substrate for the **cross-lagged panel / LCSM** family (#229) and resolves the mediation adjustment question (#264). Building those models is their own work; #250 makes them derivable.
