<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Gain/level-factor adjustment sets re-derived on the revised DAG (#247)

Date: 2026-07-12

Related: #247 (this change), #251 (DAG-revision tracker), #243/#233 (the revised DAG), #245/#258 (the mechanism `adjust_for` path this reuses), #224/#225 (the taught-vocabulary and nonword factor models, absorbed here), #232 (the closed-unmerged branch carried forward), #264/#250 (the time-lagged DAG that may revisit these).

## What changed and why

The 2026-07-10 DAG revision (`dag/dag-language-reading.dagitty`) reversed speech production `SP` into a cause, added hearing `HS` as a root, widened phonological memory `RW`, and added direct vocabulary→reading edges (`TE→WR`, `EV→WR`). Randomisation still identifies the trial effect, so the **causal** term in every gain/level-factor model is unaffected. But the **observational** adjustment sets — the "upstream DAG skills" each factor model conditions on to report its internal slopes as honest _adjusted associations_ — are now different. This change re-derives them against the revised graph and lands the whole gain/level-factor family (including the previously-unmerged taught-vocabulary and nonword models) on one graph in one change.

## The re-derivation

For each outcome the adjustment set is its DAG parents, minus age/general-ability (carried by `gamma_A` + the ability covariate + the child intercept) and the intervention nodes (the randomised `beta_trt`). The remaining parents split into **measured bounded-count skills → `skill_symbols`** (entered as period baselines) and **non-measure confounders `HS`/`SP`/`RW` → `adjust_for`** (`hs`, `deapp_c`, `erbto` and their missing indicators).

| Model     | Outcome                          | `skill_symbols`       | `adjust_for` (confounders) |
| --------- | -------------------------------- | --------------------- | -------------------------- |
| gf/lf-001 | W (word reading)                 | TR, TE, R, E, L, N, B | —                          |
| gf/lf-002 | R (receptive vocab)              | TR                    | HS, RW                     |
| gf/lf-003 | E (expressive vocab)             | R, TR, TE             | HS, SP, RW                 |
| gf/lf-004 | L (letter sounds)                | —                     | HS, SP                     |
| gf/lf-005 | P (phonetic spelling, off-floor) | L, B                  | RW                         |
| gf/lf-006 | B (blending / PA)                | L, E, TE              | HS, SP, RW                 |
| gf/lf-007 | F (CELF basic concepts)          | R, TR                 | —                          |
| gf/lf-008 | T (TROG receptive grammar)       | R, TR                 | —                          |
| gf/lf-009 | TR (taught receptive)            | —                     | HS, RW                     |
| gf/lf-010 | TE (taught expressive)           | TR                    | HS, SP, RW                 |
| gf/lf-011 | N (nonword reading, off-floor)   | L, B                  | SP, RW                     |

Note nonword reading (N) has **no** direct `IG→NW` edge — treatment reaches it only through the code skills — so its on-intervention term is a controlled contrast, and word reading (W) has no `HS`/`SP`/`RW` parent, so it takes no `adjust_for`.

## Decisions worth recording

1. **Full DAG parents for the gain family (not a curated subset).** Every measured DAG-parent skill enters `skill_symbols`, so each internal slope is a direct-effect-flavoured adjusted association. For word reading this is the full seven-skill set. We verified this is fittable: the complete-case intersection of all seven baselines keeps 153 of 154 rows, so the "full parents" choice costs essentially no sample size, and nonword reading — flagged elsewhere as post-only — does carry a usable period baseline in the `phase_mode="all"` gain layout (`pre_logit` finite on 100% of rows), so keeping it in W's set is not degenerate.

2. **Level factors take confounders only, no measure-skill adjusters.** The level model is a group×time trajectory model. Its `HS`/`SP`/`RW` adjusters are exogenous (`IG` has no edge to them), so adjusting for them does not touch the randomised t2 contrast. But the _measured_ skill parents are themselves treatment-affected; conditioning a levels model on another skill's contemporaneous level would condition on a post-treatment mediator and bias the very trajectory the model estimates. So the level family gets the `adjust_for` path only.

3. **The treatment marginal effect is now averaged over period-1 rows only (#247 P2).** Previously the gain family's items-scale average marginal effect was averaged over all three transitions — including the post-crossover ones, which carry no untreated observations and baselines that may already be treatment-affected — and described as reproducing the ITT effect. That average is a model-based transported contrast, not the randomised estimand. The reporting core (`_itt_ame_draws` and the `treatment_marginal_effect`/`rope_summary`/`prior_pushforward`/`rope_sensitivity` wrappers) now take an optional `row_mask`; the gain pipeline passes the period-1 mask (`phase == 0`). The logit-scale `beta_trt` posterior is unchanged — only its probability/items-scale marginalisation is restricted, and the reports now say so. The level family already did the analogous thing (its causal marginal restricts to the t2 rows).

4. **Off-floor estimand relabelled, not restricted.** The floored outcomes P and N keep the pooled "off the floor **at post** (`post > 0`)" Bernoulli — the reports now state plainly that this pools zero→positive movement, persistence above zero, and return-to-floor, rather than describing it as "coming off the floor". Restricting to `pre == 0` (true floor exit) was considered and deferred; it would change the target population and is not needed for the association read these models support.

5. **The child random intercept does not control latent general ability.** Every report and docstring in the family now says this explicitly: the intercept is a partial, shrunken stand-in for between-child heterogeneity, so the internal slopes stay descriptive associations. Latent-ability confounding remains; the randomised claim lives only in `beta_trt` (gain) / `b_grp_time[1]` (level).

## Recording what was actually fitted

Each fit now writes an `effective_adjustment` block to `config.json` (`requested` / `fitted` / `dropped_constant`), reusing the mechanism family's helper. A `_missing` indicator that is constant on the fitted rows is dropped by the loader and recorded as dropped, so a covariate that received no coefficient can no longer masquerade as an estimated adjuster.

## Scope and follow-up

This is PR 1: the gain/level-factor factories (`adjust_for` path), the pipeline wiring + period-1 marginal fix, all `gf/lf-001–011` specs (with the taught-vocabulary and nonword models carried forward from `origin/feat/lrp224-taught-vocab-factors`), the registry/report updates, and tests. Verification is dev-tier only (each model builds, samples, and emits the new `gamma_{c}` terms + `effective_adjustment`); no reporting-tier refit. **Closes #224 and #225.**

The remaining DAG-dependent family sweep — `adjusted` (ADJ-065), `dose_response` (DOSE-077), `lcsm` (LCSM-067), `growth` (GC-069/070) — is the explicit second stage of #247 and lands in a follow-up PR. `mediation_multi` (MED-064) already carries `HS`/`SP`/`RW` and is held under #264/#250. If the time-lagged DAG (#250) changes the graph again, this family is re-derived against it.
