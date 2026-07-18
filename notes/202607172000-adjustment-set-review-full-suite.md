<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Full-suite adjustment-set review against the revised DAG

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

## Why this review

The mech-088/089 correction (`notes/202607171700-mech-intervention-sessions-adjustment-collider-review.md`) fixed a mistaken read from the DAG: a genuine confounder (intervention sessions, `IS`/`attend`) had been dropped from two taught-vocabulary mechanism models because a collider concern was misapplied. That is the kind of error that can hide elsewhere, so this review audits **every** registered statistical model (168 across 16 families) and its adjustment/conditioning set against the revised base DAG (`dag/dag-language-reading.dagitty`, 2026-07-10) and — for the longitudinal families — the crossover-aware unroll used by #250/#264.

## Method

1. A backdoor d-separation checker (networkx) was built and validated: it reproduces the existing DAG-reviewed mechanism sets (mech-058, mech-056) exactly. For a total-effect adjusted association of exposure `X` on outcome `Y` it computes the minimal observed `Z` such that `Z ∪ {GA}` d-separates `X` and `Y` in the backdoor graph (latent general ability `GA` held blockable, since no observed set can close it), and it flags (a) **under-adjustment** — an observable backdoor left open — and (b) **over-adjustment** — conditioning on a descendant of `X`.
2. The suite was partitioned by family and audited by seven independent worker passes, each reading the actual factory + module code so it could resolve the question the static graph cannot: **whether each adjuster enters at the period start (pre-exposure, safe) or the period end (post-exposure, a treatment descendant).** That temporal distinction is decisive for the randomised estimands and for the mediation cross-world condition.
3. Every non-trivial flag was then adversarially re-verified against the DAG and the governing notes (#246, #247, #264, #269/#309) before being recorded here.

The audit script and the shared reference bundle are preserved with this review.

## Headline

**Every family that carries a causal claim is clean.** The mech-088/089 omission was the real bug; it is fixed, and no other model repeats that failure. Specifically verified:

- **ITT / joint** (randomisation-identified): no model conditions on any post-randomisation variable; the only regressors are the pre-randomisation own baseline (t1), linear t1 age, and — in the robustness companions — pre-treatment SES / block-design / area. Randomisation identification is intact throughout.
- **gain_factors**: the `skill_symbols` covariates were verified to enter at the **period start (pre)**, so the randomised period-1 τ is unbiased; every skill and exogenous adjuster is a true DAG parent of its outcome, and each set is exactly the complete measured-parent set (arm mediator `IS` correctly excluded).
- **level_factors**: exogenous-only (`HS`/`SP`/`RW`); no contemporaneous skill is conditioned; per-outcome parent sets exact; treatment-movable `SP`/`RW` routed to baseline as an extra guard.
- **mechanism**: all 21 backdoor sets are sufficient and valid; the mech-088/089 fix is confirmed (`IS` now present and in the minimal set).
- **dose_response / did**: the IS-exposure sets adjust the arm-fork `{G, A}`; DiD `tau_t2` is purely randomised (no baseline/skill/period-start adjuster); the cumulative-dose collider is kept off by default.
- **block_exposure**: arm and age are handled by the event-study structure, and `delta` is labelled an association.
- **corr_factor / lcsm / growth / horseshoe / survival / historical**: no DAG-derived backdoor claim; associations and measurement quantities are correctly labelled.

Two issues surfaced, both narrower than mech-088, plus minor metadata hygiene.

## Finding 1 — mediation family: adjustment-set inconsistency + inaccurate docstrings (MEDIUM)

**This is not an open-backdoor bug of the mech-088 class.** Per #264 (`notes/202607142340-lrp264-mediation-adjustment-dsep.md`, CI-locked in `tests/test_lagged_dag_adjustment_sets.py`), the whole mediation family is _unidentified regardless_ of these covariates: every model has an unblockable witness `M ← IS → WR` — intervention dose is a treatment-induced mediator-outcome confounder — so all models are honestly labelled ID-2 / `GA`-confounded g-formula estimates under stated assumptions.

The issue is **consistency of observable-confounder handling**. The revised DAG makes hearing (`HS`), speech (`SP`), and phonological memory (`RW`) exogenous common causes of several mediators and word reading. The #259 revision added these to the flagship route models and the letter-sound companions, and #264's own pre-/post comparison shows they do real numerical work (they absorbed the direct path: proportion-mediated rose 0.62 → 0.82 for MED-059, 0.64 → 0.92 for MED-064). But six exploratory siblings were never updated:

| model                                     | mediator(s)        | DAG-parent exogenous confounders | had them? | missing            |
| ----------------------------------------- | ------------------ | -------------------------------- | --------- | ------------------ |
| med-059 / 062 / 064, 086 / 087, 186 / 187 | L / L+PA / L+E / L | HS, SP (, RW)                    | yes       | —                  |
| med-066, med-075                          | L + PA             | HS, SP, RW                       | no        | hs, deapp_c, erbto |
| med-068                                   | TE                 | HS, SP, RW                       | no        | hs, deapp_c, erbto |
| med-080                                   | TR                 | HS, RW                           | no        | hs, erbto          |
| med-074                                   | NW                 | SP, RW                           | no        | deapp_c, erbto     |
| med-076                                   | L (t4 outcome)     | HS, SP                           | no        | hs, deapp_c        |

Consequences: (a) the six siblings' route-share estimates were on a different adjusted footing than the flagships they are implicitly compared with; (b) three docstrings were factually wrong — med-076 claimed its set was "identical to LRP59" (which carries `hs`/`deapp_c`; it did not), and med-066/med-075 claimed "the same set LRP62 adjusts for"; (c) med-068/med-080 named the other skills (`L`/`R`/`E`) as "the mediator-outcome confounders" when the DAG parents are `HS`/`SP`/`RW` (the named skills are admissible baseline precision/proxy terms per #264, not the exogenous confounders). The most likely cause is that these models predate the #259 flagship revision and were not carried forward.

### Decision (this review)

Reconcile the six sibling sets by **adding** the missing exogenous confounders (mirroring exactly what #259 did to the flagships — add, do not remove the admissible baseline skills), correct the docstrings, and re-fit the six at `reporting`. This does **not** change any identification status (the models remain ID-2, the `IS` witness is untouched) and it does not affect the CI guard, whose assertions are structural (no IG-descendant in the set; the witness stays unblockable). It puts the family's observable-confounder handling on one consistent footing and makes the docstrings true. Post-reconciliation adjustment sets:

- med-066 / med-075: `[G, A, E, R, W_pre, L_t1, B_t1, hs, hs_missing, deapp_c, deapp_c_missing, erbto, erbto_missing]`
- med-068: `[G, A, L, R, W_pre, TE_t1, hs, hs_missing, deapp_c, deapp_c_missing, erbto, erbto_missing]`
- med-080: `[G, A, L, E, W_pre, TR_t1, hs, hs_missing, erbto, erbto_missing]`
- med-074: `[G, A, E, R, W_pre, N_t1, deapp_c, deapp_c_missing, erbto, erbto_missing]`
- med-076: `[G, A, E, R, W_pre, L_t1, hs, hs_missing, deapp_c, deapp_c_missing]` (now genuinely identical to MED-059)

Following #264's own comparison, the substantive expectation is that the **direct** path shrinks toward zero and the mediated proportion rises, while the indirect (letter-sound / code-route) share is largely unchanged. New numbers will be recorded in `notes/202607161800-findings-mediation.md` after the re-fit, and #264's per-model table gets a dated addendum noting the extension.

## Finding 2 — mech-072 / mech-172: the moderator is a DAG-descendant of the exposure (LOW / interpretation)

mech-072 (letter-sound `L` → decoding `N`, moderated by phoneme blending `B`) and its no-interaction companion mech-172 condition on the moderator `B` = `PA` at its contemporaneous value. The revised DAG has `LS → PA → NW`, so `PA` is a descendant of the exposure `LS` and a mediator of the `L → N` effect. Conditioning on it (the `gamma_mod · z(B)` main effect) makes the reported mechanism slope a **controlled-direct effect**, not the total effect. The module docstring already excludes word reading `W` for exactly this reason ("sibling/descendant … over-control") but frames `L` and `B` as "parallel prerequisites" whose ordering "does not [need] resolving" — which contradicts the committed `LS → PA` edge that the family's own mech-190 relies on (it uses `LS` as a confounder-parent of `PA`).

This is defensible **if read as effect-modification of the `L → N` relationship at fixed `B`**, but the "parallel" framing is inconsistent with the DAG. The estimand is not wrong so much as mis-described.

### Decision (this review)

Docstring-only: reframe mech-072/172 to state that under the committed DAG `B` = `PA` is a descendant of `L`, so the mechanism slope is a controlled-direct (not total) effect and the interaction is effect-modification by a downstream skill. No adjustment-set change and no re-fit (the fit is unaffected by docstring text).

## INFO (hygiene; no bias, not actioned here)

- `estimand_type` / `causal_status` structured metadata is unset on the nine `aligned` specs, `adj-065`, and several RLI structural families (corr_factor / lcsm / long_corr_factor / RLI horseshoe). The association/measurement labelling lives instead in the factory docstrings and report partials, which are unambiguous, and `context.py` backfills the causal defaults only for `itt`/`joint`. Worth closing for parity with the families that set the fields explicitly, but it is a completeness gap, not a mis-derived claim.
- gain_factors' non-causal skill associations (`gamma_s`) additionally carry `IS` (dose) confounding that cannot be removed without biasing the randomised τ — a documented interpretive limit, correctly prioritising the unbiased τ. A one-line reminder in those reports would help readers.

## Scope note

The audit covered adjustment/conditioning _structure_ against the DAG. It did not re-check convergence, prior calibration, or likelihood choice (those are the refit's own gate) and it takes the symbol→node and covariate→node mappings as given by `measures.py` and the DAG header.
