<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore Byrne MacDonald Buckley readgrp basread basspel bpvs trog woco basdig bassim basnum basmat rlm rlmhg rlmjc xsbr xspg dagitty lcsm clpm riclpm ancova -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> This is a **forward-looking plan for discussion**, not results and not fitted models. It maps the RLI Layer-2 model families onto the Byrne cohort given the proposed Byrne DAG (`dag/dag-reading-language-memory.dagitty`) and the measures/waves actually present in the extract. Most items are gated on human/data-owner decisions (instrument ceilings, group scope, and green-lighting the lagged companion) — those gates are called out per model.

# Comparable models for the Byrne, MacDonald & Buckley (2002) cohort — plan

Companion to the DAG proposal (`notes/202607131500-byrne-dag-proposal.md`) and the earlier follow-up plan (`notes/202607021052-issue-164-byrne-followup-plan.md`), which this supersedes on the modelling question specifically. The aim: decide **which of the sixteen RLI statistical-model families port to Byrne, in what adapted form, and in what order** — the same way `lrp-rlm-hg-001` already ported the `historical_growth` family.

## The governing fact: no randomisation, so nothing is causal

The RLI suite's one causal quantity is the randomised intention-to-treat `τ` (and its within-person DiD replica). Byrne has **no intervention**: `readgrp` is an observational cohort factor. So **every comparable Byrne model is descriptive or adjusted-associational — `causal_status="none"` throughout** (as `lrp-rlm-hg-001` already declares). The practical consequence is clean: the Byrne suite is the **RLI _observational_ subset, ported** (`adjusted`, `mechanism`, `mediation`, `lcsm`, `growth`, `corr_factor`, `horseshoe`, plus the descriptive `historical_growth`/levels view), and the **four randomised/intervention families do not port at all** (`itt`, `did`, `aligned`, `dose_response`). Metadata for every Byrne model: `study_id="rlm"`, `design="historical_cohort"`, `causal_status="none"`, `estimand_type` ∈ {`descriptive`, `association`}.

## Family-by-family portability

| RLI family (`kind`)             | Ports?              | Adapted form for Byrne                                                                                                                   | `estimand_type` |
| ------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `historical_growth`             | ✅ **live**         | `lrp-rlm-hg-001` exists (BAS reading, w1–3); extend per measure                                                                          | descriptive     |
| `level_factors`                 | ✅ subsumed         | group×time levels = exactly what `historical_growth` fits here; no separate family needed                                                | descriptive     |
| `growth`                        | ✅ strong           | multivariate trajectories + does **baseline ability** predict trajectory shape (README already earmarks Byrne replication of GC-069/070) | descriptive     |
| `joint`                         | ✅                  | correlated group-by-wave trajectories over a small measure set (the `rlmjc` model in the follow-up plan)                                 | descriptive     |
| `corr_factor`                   | ✅ strong           | correlated **domain-factor** measurement model — the direct analogue of the paper's reading/language/memory correlation tables           | descriptive     |
| `lcsm`                          | ✅ strong           | within-child latent change-score: prior-wave language/memory → reading _change_ (analogue of `lcsm-067`)                                 | association     |
| _(cross-lagged / CLPM)_         | ⚠️ new, reduced     | the **reciprocal reading↔language↔memory** question proper; needs a lagged Byrne DAG; sample-limited (see constraints)                   | association     |
| `adjusted`                      | ✅                  | between-child: which wave-1 skills go with more subsequent reading gain, mutually adjusted (analogue of `adj-065`)                       | association     |
| `horseshoe`                     | ✅                  | regularised-horseshoe predictor-ranking cross-check for a Byrne outcome                                                                  | association     |
| `mechanism`                     | ⚠️ assumption-heavy | adjusted skill→skill dose-response (e.g. `basdig → basread`); `GA` only _partially_ controllable via `bassim`/`basmat`                   | association     |
| `mediation` / `mediation_multi` | ⚠️ optional         | g-formula NDE/NIE with an **observational** exposure — very assumption-laden without randomisation; defer                                | association     |
| `gain_factors`                  | ⚠️ partial          | the ANCOVA (post ~ pre + covariates) ports, but its _causal_ group term does not; collapses into `adjusted`                              | association     |
| `itt`                           | ❌                  | needs a randomised group — `readgrp` is observational                                                                                    | —               |
| `did`                           | ❌                  | needs the waitlist crossover — no intervention                                                                                           | —               |
| `aligned`                       | ❌                  | needs intervention onset — no intervention                                                                                               | —               |
| `dose_response`                 | ❌                  | needs intervention **sessions** — none exist                                                                                             | —               |

## The binding constraints (they shape every choice)

1. **Sample size.** 24 Down syndrome, 42 average readers, 31 reading-matched (97 total). The DS group — the scientifically central one — is **n = 24**. RLI already judged a free RI-CLPM **not estimable at n ≈ 54** and shipped the parsimonious `lcsm-067` instead; the Byrne DS panel is less than half that. **Any longitudinal coupling model must be parsimonious**: a few pre-specified, pooled couplings with informative priors, not a free cross-lagged system.
2. **Estimable wave window (measured, from the extract).** Waves **1–4 carry all three groups**; **wave 5 is Down-syndrome-only** (17 children; average and reading-matched both drop to zero). So between-group contrasts live in **w1–w4**, and w5 extends only the DS within-group trajectory. Group sizes shrink over waves through attrition (DS 24→17; Avg 42→0 by w5; RM 31→0 by w5).
3. **`basmat` is wave-3+; `basnum` is absent at wave 5.** No baseline (w1) non-verbal-reasoning score, so the GC-style "does baseline non-verbal ability predict trajectory shape?" question must fall back on **`bassim`** (verbal reasoning, available w1) as the baseline-ability proxy, and flag `basmat` as a later-wave-only covariate.
4. **Instrument ceilings are unconfirmed (the hard gate).** The Beta-Binomial likelihood needs a confirmed denominator per measure. Only `basread` (87) is treated as confirmed (and even that is the _observed_ maximum, not manual-verified). Observed maxima in the extract: `basspel` 18, `woco` 31, `bpvs` 29, `trog` 20, `basdig` 34, `basnum` 60, `bassim` 18, `basmat` 22. **Every non-`basread` bounded-count model is blocked** until these are confirmed against the instrument manuals, or a Normal/Student-t likelihood on the raw score is chosen where a bounded count is inappropriate (follow-up-plan decision 3).
5. **Reading-matched selection.** Group 3 is selected on reading level, so `basread` is a **selection variable** for any analysis using it (a collider). Between-group contrasts and any cross-lagged edge touching group 3 must treat this explicitly.

## Adjustment sets, read off the Byrne DAG

For the associational (adjusted) models, the confounders are readable from `dag/dag-reading-language-memory.dagitty` the usual way — common causes of the exposure and outcome, never a mediator:

- **`basdig → basread`** (memory → reading): common causes are `{age, GA, readgrp}` → adjust `age`, `readgrp`, and the `GA` proxies `bassim`/`basmat`. **Do not** adjust `bpvs` — it is a _mediator_ (`basdig → bpvs → basread`), not a confounder.
- **`bpvs → basread`** (vocabulary → reading): add `basdig` to the set (it is a common cause of both), giving `{age, readgrp, GA-proxies, basdig}`.
- **`trog → woco`, `basread → woco`** (Simple-View routes into comprehension): adjust `{age, readgrp, GA-proxies}` plus upstream memory/vocabulary as common causes.
- **Longitudinal (lcsm / cross-lagged):** confounders are the **prior-wave parents** of the changing node, read off the lagged Byrne DAG (below), consistent with how RLI derives lagged adjustment sets.

`GA` adjustment is only **partial** here (two noisy indicators, `basmat` late) — so these stay _adjusted associations_, reported with the residual-confounding caveat, never "X drives Y". This is the same rule as the whole RLI observational layer.

## Proposed model set, phased

Ids follow the registered scheme (`lrp-rlm-{family}-NNN`, `study_id="rlm"`), extending the `hg`/`jc` prefixes already in the follow-up plan.

**Phase A — per-measure descriptive growth (mechanical on the live family).** `lrp-rlm-hg-002…` : one `historical_growth` model per measure (`basspel`, `woco`, `bpvs`, `trog`, `basdig`, `basnum`; `bassim`; `basmat` w3+ handled separately), group-by-wave levels + within-group growth, w1–w4 three-group with a w5 DS-only extension where the panel supports it. _Gate: instrument ceilings (constraint 4)._ Each still needs its thin `docs/models/{id}/index.qmd` (parent-fallback template keeps this light).

**Phase B — joint & measurement structure.** (i) `lrp-rlm-jc-001` : a `joint`/`growth` model of correlated group-by-wave trajectories over a small set (reading + a vocabulary + a memory measure) — how the trajectories move together within group. (ii) `lrp-rlm-mm-001` : a `corr_factor` **domain-factor measurement model** (reading = `basread`/`basspel`/`woco`; language = `bpvs`/`trog`; memory = `basdig`; ability = `bassim`/`basmat`/`basnum`) — the modern analogue of the paper's correlation tables, and the cleanest way to summarise the construct structure before any coupling model. _Gate: ceilings; measurement-invariance assumptions across groups stated up front._

**Phase C — the reciprocal question (the scientifically distinctive model).** `lrp-rlm-lcsm-001` (change-score) and, if the sample stretches, a reduced two-variable `lrp-rlm-clpm-001` (cross-lagged): does prior-wave **reading** predict later **language/memory** _change_, over and above the forward path? This is Byrne et al.'s founding hypothesis — and their reported _null_ — put to a modern coupled model. It **requires a lagged Byrne DAG** (a two-slice wave-unrolled companion mirroring `dag/dag-language-reading-lagged.dagitty`, with reverse edges `basread_t → {bpvs, basdig, trog}_{t+1}`), which the DAG proposal scoped but did not draw. Byrne's **5 waves** (deeper than RLI's 4) favour this, but the **DS n = 24** forces the same parsimony RLI used (pooled coupling across transitions, informative priors); a free RI-CLPM stays parked, exactly as in RLI. _Gate: green-light + draft the lagged companion; confirm which reverse edges are pre-specified vs exploratory._

**Phase D — predictor views (associational).** `lrp-rlm-adj-001` : between-child, which wave-1 skills go with more subsequent reading gain, mutually adjusted (analogue of `adj-065`). `lrp-rlm-hs-001` : a regularised-horseshoe ranking cross-check over the same predictor set. Both are honest "what predicts reading" summaries with no causal claim. _Gate: ceilings; DS-only vs pooled decision._

**Phase E — optional, assumption-heavy.** `mechanism`-style adjusted skill→skill slopes (e.g. `basdig → basread`) if a specific pairwise question is wanted; `mediation` only if a decomposition is genuinely justified despite the observational exposure. Recommend **deferring** both until Phases A–D are in and the team wants a specific pairwise/decomposition estimand.

**Separate, larger track — cross-study (from the follow-up plan).** `lrp-xs-br-001` (bridge/calibration: link Byrne BAS measures to the RLI measures under explicit linking assumptions) and `lrp-xs-pg-001` (pooled cross-study growth) remain a distinct methodological commitment, only after the bridge assumptions are explicit. Out of scope for the single-study Byrne suite above.

## Decisions this plan needs

1. **Instrument ceilings / likelihood** per non-`basread` measure — the gate on Phases A, B, D (follow-up-plan decision 3).
2. **Group scope** — three-group (fuller picture) vs Down-syndrome-vs-reading-matched (the sharper developmental contrast) as the primary framing, per measure and per phase (follow-up-plan decision 4). Note w5 is DS-only regardless.
3. **Green-light the lagged companion** and the reduced cross-lagged/LCSM model (Phase C) — the distinctive analysis; without it Byrne is only descriptive growth + measurement structure.
4. **Reading-matched selection handling** — confirm `basread` is treated as a selection variable wherever group 3 enters.
5. **Random-effects heterogeneity** — index the subject-intercept SD (and possibly `κ`) by group, per the follow-up plan's decision 7, before the per-measure growth sweep so it propagates.

## What is explicitly _not_ proposed

No ITT, DiD, aligned per-protocol, or dose-response models — all four require the randomised intervention, sessions, or onset alignment that this cohort does not have. Stating this closes the "why isn't there a Byrne ITT?" question up front: there cannot be one.

## Addendum (2026-07-15) — post-plan families and a Phase E prerequisite

> [!NOTE]
> Addendum drafted by a LLM-based AI tool (Claude Code/Fable 5), 2026-07-15. Tracked in #338.

This plan mapped the sixteen Layer-2 families as of 2026-07-13. Two descriptive families landed in the RLI suite immediately afterwards (both in the #314 descriptive-association workstream), and #335 added a mediation-robustness requirement — none of which the table above covers. Recorded here so this note stays the single mapping document.

| RLI family (`kind`)                          | Ports?    | Adapted form for Byrne                                                                                                                                                                                                                                                                            | `estimand_type` |
| -------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `concurrent` (#312, `lrp-rli-ca-*`)          | ✅ strong | per-wave mutually adjusted conditional associations over {`basread`, `basspel`, `woco`, `bpvs`, `trog`, `basdig`, `basnum`} + age, `readgrp` per the group-scope decision; `basread`-focal and `bpvs`-focal first (`lrp-rlm-ca-001`/`002` presumptive), cross-check partner of the `rlm-mm` model | association     |
| `long_corr_factor` (#313, `lrp-rli-lcf-001`) | ⚠️ defer  | per-wave latent domain correlations; already fragile at RLI n ≈ 54, not the right first instrument at Down-syndrome n = 24 — fit the cross-sectional `rlm-mm-001` (Phase B) first and revisit after                                                                                               | descriptive     |

Three adaptation notes for the `concurrent` port, beyond the standing constraints (ceilings, group scope, selection):

- **The vocabulary side collapses and changes meaning.** Byrne has receptive vocabulary only (`bpvs`; no expressive measure), so the RLI receptive/expressive focal pair (#336) becomes a single `bpvs`-focal model. And with no code-route measures, the RLI reading "vocabulary ↔ reading given phonics" cannot be asked here — the Byrne conditional is "given verbal memory (`basdig`), grammar (`trog`), the ability proxies and age", with the collapsed decoding route (the latent `DEC` extension above) named in the limitations.
- **Conditioning meets the selection hazard head-on.** The family deliberately conditions on contemporaneous skills, and `basread` is a selection variable wherever group 3 enters — so the group-scope and selection decisions bind harder here than for the growth family.
- **Power.** The RLI ca fits run n ≈ 53 per wave with seven predictors and lean on regularising priors; Down-syndrome-only Byrne is n = 24 per wave, pooled three-group n ≈ 97 (w1–w4). The DS-only vs pooled framing is the binding decision; either way the slopes stay strongly regularised and the adjusted-vs-bivariate gap is reported per the family convention.

**Phase E prerequisite (from #335/#324).** If Phase E `mechanism`/`mediation` models are ever green-lit, they ship with the #289 unmeasured-confounding sensitivity sweep and a named-confounder calibration from day one — with residual `GA` (beyond the `bassim`/`basmat` proxies, plus the latent `DEC`/`HS` extensions) playing the role RLI's intervention-sessions confounder plays in #324/#335. The measured ability indicators give Byrne an empirical anchor for "GA-strength confounding" that RLI never had; a med-079-style negative control (a DAG-severed pair, e.g. `trog → basnum`) is the natural calibration companion.
