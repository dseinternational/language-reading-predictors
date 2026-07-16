# Findings — dose_response family (outcome vs amount of intervention)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary.

## What these models ask

"Do children who did **more** intervention (more sessions) score higher?" The models relate the outcome to the standardised number of sessions, per period. **Session dose is not randomised** — how much a child attended reflects ability, attendance, and availability — and it is a **partial collider** in the causal diagram, so the dose slope is strictly an **observational association** and a _sensitivity view_, never a treatment effect.

## Convergence gate

`dose-277` (pooled comparator) **passed**. The four period-resolved fits are flagged for **divergences only**, all well under the 1% guideline and with healthy R̂/ESS/BFMI — **usable with a caveat**: `dose-077` (0.047%), `dose-177` (0.036%), `dose-083` (0.011%), `dose-084` (0.006%).

## Results

The cleanest single number is the **pooled** slope (`dose-277`, word reading): a **1-SD increase in sessions was associated with +1.3 items** (95% range +0.4 to +2.2; 99.8% positive — very strong _association_). The period-resolved models (`dose-077` W, `dose-083` L, `dose-084` B, `dose-177` ability-adjusted W) let the slope differ by period; their per-period slopes are in each model's `dose_slope_summary.csv`.

Because a gate-flagged model is excluded from cross-model LOO comparison (the #340 discipline), the dose-response LOO comparison (`dose-077` vs its baseline) was **correctly skipped** this run — see the [DiD note](202607161800-findings-did.md) and `output/statistical_models/comparison/`.

## The one-paragraph story

More sessions go with higher scores, and for word reading the pooled association is clearly positive. But this is the one family where the direction of causation is genuinely ambiguous — abler or better-attending children both do more sessions _and_ score higher — so it is reported as an association and used only to triangulate, not to claim a dose effect.

## What is causal

**Nothing.** Session dose was not assigned; the slope is observational and partly a collider. Treat it as "children who did more also scored higher", not "more sessions cause higher scores".
