<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Mediation adjustment sets settled by time-indexed d-separation (#264)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

## Decision

**Baseline expressive and receptive vocabulary (`E`/`R`, the models' t1 values of `EV`/`RV`) stay in the mediation-family adjustment sets.** The wave-unrolled derivation below confirms the 2026-07-12 interim decision and closes the question #259 opened: with measurement occasions explicit, `E_t1`/`R_t1` are **not** descendants of the randomised treatment, dropping them changes **no** model's backdoor-blocking status, and in five of the thirteen models they do real work (one is a member of the strictly-valid parent set; four are the admissible pre-treatment proxies of treatment-affected contemporaneous confounders). **No adjustment set changes, so no model is refitted and no headline number moves** — the `rep-lite` sweep note (`notes/202607111100-replite-full-statistical-fit.md`) and the reporting sweep note (`notes/202607131300-full-statistical-refit-reporting.md`) stand as written.

Every claim below was checked mechanically with networkx d-separation runs; the script is preserved at [`assets/202607142340-med-adjustment-dsep.py`](assets/202607142340-med-adjustment-dsep.py) and the load-bearing checks are promoted to CI in `tests/test_lagged_dag_adjustment_sets.py`, parametrised over the live `SPEC.adjustment` lists so an edit to either the `.dagitty` or a MED model's set re-triggers the derivation.

## Why #259's descendant argument fails

#259 proposed dropping `E`/`R` because `EV`/`RV` are descendants of the intervention on the contemporaneous graph (`IG → TE → EV`, `IG → TR → RV`), which would make conditioning on them a treatment-affected-confounder (recanting-witness) violation. The contemporaneous graph carries no measurement occasions, so it cannot distinguish a baseline measurement from a post-treatment one. On the crossover-aware unroll of `dag/dag-language-reading-lagged.dagitty` (#250) the question is answerable, and the answer is unambiguous: `EV_1` and `RV_1` are **not** in the descendant set of `IG` on any unroll (three-slice or four-slice) — the intervention delivered in the t1→t2 window cannot affect a t1 measurement. The 2026-07-12 hold was right, and the descendant relation read off the single-wave graph was the wrong object.

## Method

The graphs and conventions are those of the #250 design note (`notes/202607141030-time-lagged-model-designs.md`): the two-slice template `dag/dag-language-reading-lagged.dagitty` (Option A, adopted 2026-07-13) unrolled into a crossover-aware **three-slice** graph (window 1 active for the immediate arm, window 2 for both arms), extended to **four slices** for MED-076's t4 outcome (window 3 is continued delivery for both arms, per the session records). Latent general ability `GA` is removed before every check — no measured set can block it, so each check asks the honest question ("`GA` aside, does this set block every mediator → outcome backdoor?") and the family keeps labelling its decompositions g-formula estimates under stated assumptions. The fitted models enter age once, at baseline; on the unroll the fitted `A` is granted `{A_1, A_2}` because age at later waves is deterministic given the fixed wave spacing (the `.dagitty` header records `A_t → A_t1` as a placeholder for maturation, not a stochastic cause). Missing-data indicator columns (`*_missing`) are estimator policy, not graph nodes, and are not mapped.

For each model the mediator(s) sit at wave 2 and the outcome at wave 2 (wave 4 for MED-076); `ModelSpec.adjustment` maps onto the unroll with baselines at wave 1 (`G → IG`, `hs → HS`, `deapp_c → SP_1`, `erbto → RW_1`, bare symbols and `*_t1`/`W_pre` markers → the wave-1 skill nodes).

## The derivation, model by model

Three facts hold for **every** one of the thirteen models (each is asserted per-model in the CI tests):

1. **The fitted set is cross-world-admissible.** No fitted adjustment set contains any descendant of `IG` — every adjuster precedes treatment (or is the time-invariant `HS`).
2. **The fitted all-baseline set does not strictly block the mediator → outcome backdoors, and dropping `E`/`R` changes nothing** — the set is `[NOT-VALID]` with them and `[NOT-VALID]` without them. The graph wants contemporaneous wave-2 states (`SP_2`, `RW_2`, `TR_2`, …), several of which are treatment-affected. The `E`/`R` question was therefore never an identification question: retention is a precision-and-proxy call, which is exactly how the family's honesty boxes already read.
3. **No treatment-non-descendant set can exist.** Each model has a **witness**: a collider-free mediator → outcome backdoor path every one of whose interior nodes is a descendant of `IG`, so no admissible set can block it. For most models the witness is the window-1 session node — e.g. `LS_2 ← IS_1 → WR_2` — i.e. dose is a treatment-induced mediator–outcome confounder, precisely the VanderWeele, Vansteelandt & Robins (2014, doi:10.1097/EDE.0000000000000034) obstacle the module docstrings already record. Natural effects stay unidentified whatever happens to `E`/`R`; the g-formula-under-stated-assumptions framing, the interventional companion (MED-078), and the #323/#324 follow-ups are unchanged.

Per model (fitted set in unrolled-graph nodes; `A` = `{A_1, A_2}` by the determinism grant):

| Model   | M → Y               | Fitted set (mapped)                           | Blocks M→Y? | Witness (unblockable backdoor)   |
| ------- | ------------------- | --------------------------------------------- | ----------- | -------------------------------- |
| MED-059 | LS_2 → WR_2         | IG, A, EV_1, RV_1, LS_1, WR_1, HS, SP_1       | no          | LS_2 ← IS_1 → WR_2               |
| MED-062 | {LS_2, PA_2} → WR_2 | IG, A, EV_1, RV_1, WR_1, HS, RW_1, SP_1       | no          | LS_2 ← IS_1 → WR_2               |
| MED-064 | {LS_2, EV_2} → WR_2 | IG, A, RV_1, WR_1, LS_1, EV_1, HS, RW_1, SP_1 | no          | LS_2 ← IS_1 → WR_2               |
| MED-066 | {LS_2, PA_2} → WR_2 | IG, A, EV_1, RV_1, WR_1, LS_1, PA_1           | no          | LS_2 ← IS_1 → WR_2               |
| MED-068 | TE_2 → WR_2         | IG, A, LS_1, RV_1, WR_1, TE_1                 | no          | TE_2 ← IS_1 → WR_2               |
| MED-074 | NW_2 → WR_2         | IG, A, EV_1, RV_1, WR_1, NW_1                 | no          | NW_2 ← LS_2 ← IS_1 → WR_2        |
| MED-075 | {LS_2, PA_2} → WR_2 | IG, A, EV_1, RV_1, WR_1, LS_1, PA_1           | no          | LS_2 ← IS_1 → WR_2               |
| MED-076 | LS_2 → WR_4         | IG, A, EV_1, RV_1, WR_1, LS_1                 | no          | LS_2 ← IS_1 → WR_2 → WR_3 → WR_4 |
| MED-078 | LS_2 → WR_2         | IG, A, EV_1, RV_1, LS_1, WR_1, HS, SP_1       | no          | LS_2 ← IS_1 → WR_2               |
| MED-079 | RG_2 → WR_2         | IG, A, EV_1, RV_1, WR_1, RG_1                 | no          | RG_2 ← TR_2 → WR_2               |
| MED-080 | TR_2 → WR_2         | IG, A, LS_1, EV_1, WR_1, TR_1                 | no          | TR_2 ← IS_1 → WR_2               |
| MED-086 | LS_2 → NW_2         | IG, A, LS_1, PA_1, HS, SP_1, RW_1             | no          | LS_2 ← IS_1 → PA_2 → NW_2        |
| MED-087 | LS_2 → PA_2         | IG, A, PA_1, LS_1, HS, SP_1                   | no          | LS_2 ← IS_1 → PA_2               |

For each model the parent set `pa(M)` of the mediator block (which is strictly valid by construction, `GA` aside, but contains `IG`-descendants — the named recanting witnesses) was also checked: it is `[VALID]`, and it **stays `[VALID]` with `EV_1`/`RV_1` added** — conditioning on baseline vocabulary opens no backdoor anywhere in the family, so `E`/`R` are harmless as well as admissible.

## What `E`/`R` actually do, model by model

The issue asked whether baseline vocabulary is a confounder, a neutral covariate, or a collider/descendant to exclude. Three roles emerge (mechanically, from membership of `pa(M)` and of its treatment-affected members):

| Model               | `E` (EV_1)                                                              | `R` (RV_1)                                     |
| ------------------- | ----------------------------------------------------------------------- | ---------------------------------------------- |
| MED-064             | **member of the valid set** — `EV_1` is a parent of the mediator `EV_2` | proxy for treatment-affected `RV_2 ∈ pa(EV_2)` |
| MED-062, -066, -075 | proxy for treatment-affected `EV_2 ∈ pa(PA_2)`                          | precision-only                                 |
| MED-079             | precision-only                                                          | proxy for treatment-affected `RV_2 ∈ pa(RG_2)` |
| all others          | precision-only                                                          | precision-only                                 |

"Proxy" means the contemporaneous wave-2 vocabulary state is a genuine member of the strictly-valid set but is itself treatment-affected (a recanting witness that cannot be conditioned without breaking the cross-world assumption); its baseline value is the closest admissible stand-in. So in the models where the mediator block includes blending or vocabulary (MED-062/064/066/075) and in the negative control (MED-079), `E` and/or `R` are doing partial confounder-blocking work, and in the rest they are neutral precision terms. Nowhere are they colliders or descendants. **Per-mediator verdict: keep, everywhere they currently appear.** MED-068 and MED-080 (which carry only one of the pair) and MED-086/087 (which carry neither) are also correct as fitted — see below.

## MED-086/087 (the 2026-07-14 provisional sign-off)

The two letter-sound mediation companions added in #309 carried no `E`/`R`, signed off as provisional pending this derivation. Confirmed: `EV`/`RV` do not enter `pa(LS_2)`, and neither `EV_2` nor `RV_2` is a parent of the mediator or outcome blocks in either model, so there is no open backdoor for baseline vocabulary to block — their exclusion stands. The `IS` recanting-witness structure the sign-off flagged (`LS ← IS → PA`) is exactly what the time-indexed witness makes explicit: `LS_2 ← IS_1 → PA_2` (MED-087) and `LS_2 ← IS_1 → PA_2 → NW_2` (MED-086) are blockable only at `IG`-descendants, so the caveat-plus-companions handling (#323 interventional companions, #324 sensitivity calibration) remains the right response, not adjustment.

## The pre- vs post-revision decomposition comparison (#246, task 4)

#246's final task — report the change in the direct/indirect split after the revised-DAG re-specification — was deferred to this issue when #246 closed. The comparison is between the last pre-revision fits (old adjustment sets, `rep-lite` sweep of 2026-07-11) and the revised-set fits of #259 (`+ HS, SP` for MED-059, `+ HS, RW, SP` for MED-062/064, with missing-indicator handling; `reporting` sweep of 2026-07-13). Words are out of test length; posterior medians with `P(> 0)`:

| Model   | Quantity         | Pre-revision (rep-lite, 2026-07-11) | Post-revision (reporting, 2026-07-13) |
| ------- | ---------------- | ----------------------------------- | ------------------------------------- |
| MED-059 | Total            | +2.85, P=0.992                      | +1.95 [−0.69, +4.51], P=0.929         |
|         | NIE (via L)      | +1.78, P=0.998                      | +1.68 [+0.38, +3.61], P=0.997         |
|         | NDE              | +1.07, P=0.856                      | +0.17 [−2.13, +2.52], P=0.558         |
|         | Prop. mediated   | ≈0.62                               | ≈0.82                                 |
| MED-062 | Total            | +2.65, P=0.988                      | +1.63 [−1.00, +4.18], P=0.894         |
|         | NIE (code route) | +1.01, P=0.965                      | +0.92 [−0.10, +2.56], P=0.961         |
|         | NDE              | +1.64, P=0.933                      | +0.62 [−1.92, +3.21], P=0.684         |
| MED-064 | Total            | +2.98, P=0.992                      | +2.00 [−0.89, +4.98], P=0.913         |
|         | NIE_L            | +1.94, P=0.998                      | +1.89 [+0.39, +4.10], P=0.996         |
|         | NIE_E            | −0.02, P=0.468                      | +0.03 [−0.69, +1.02], P=0.583         |
|         | NDE              | +1.06, P=0.848                      | −0.07 [−2.47, +2.39], P=0.477         |

The revision moved the **direct** path, not the indirect one: the letter-sound (and code-route) NIE is essentially unchanged in every model, while the NDE shrinks toward zero and the total shrinks with it — the added confounders (`HS`, `SP`, `RW`) absorb variance that had been loading on the direct path, raising the proportion mediated (0.62 → 0.82 for MED-059; 0.64 → 0.92 for MED-064). The substantive claim is untouched and, if anything, sharper: the word-reading gain routes through letter-sound knowledge, with no independent vocabulary route (`NIE_E` ≈ 0 on both sides of the revision). Caveat for the record: the pre-revision numbers come from the `rep-lite` sampling tier and the post-revision from `reporting` — both clear the same convergence gate, and a sampling-tier difference cannot produce shifts of this size or pattern, so the deltas are read as the specification change.

## Consequences

- **No model refits.** The scope item "if they come out, re-derive and refit at rep-lite" is moot: nothing comes out.
- **`notes/202607111100-replite-full-statistical-fit.md` needs no update** — no headline mediation number moves.
- The MED module comments that read "retained pending the time-indexed d-separation (#264)" now cite this note; the MED-086/087 "provisional pending" caveats are discharged.
- The derivation is CI-guarded: `tests/test_lagged_dag_adjustment_sets.py` re-runs it against the live `.dagitty` and the live `SPEC.adjustment` lists.
