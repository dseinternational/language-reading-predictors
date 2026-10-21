<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Reverse longitudinal mediation: design record

> [!NOTE]
> Interpretation and status corrected by a LLM-based AI tool (Codex/GPT-6), 21 October 2026. The implementation record is historical. Current reporting follows [METHODS.md](../METHODS.md); the decompositions do not identify causal direction.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). **Status: approved and implemented 2026-07-17.** The `WR_t → LS_t1` edge is in the lagged DAG, the CI guard was bumped and re-run clean (Step 1b result below), and `lrp-rli-med-176` is built + registered + dev-smoke-fit; the `reporting` fit and the forward/reverse contrast (written into the mediation findings note) complete the run. **Step 1b result:** adding the edge changed the witness/validity of **no** existing mediation model — all 54 prior CI assertions still pass, plus 3 new ones for med-176 (57 total). Implementation also fixed a latent naming bug (`build_mediation_model` hardcoded the outcome-baseline PyMC node as `"W_pre_logit"`, which collided when the mediator is itself `W`; now parameterised by `outcome_symbol`, byte-identical for every W-outcome model).

## Why

At the July proposal stage, the forward mediation models assumed the direction letter sounds → word reading (`LS → WR`): the contemporaneous base DAG hard-codes `LS → WR` with no reverse edge, and the lagged DAG (`dag/dag-language-reading-lagged.dagitty`) adds reverse edges `WR_t → {TE, TR, PA, RW}_{t+1}` — even annotating `WR_t → PA_t1` as _"reciprocal with the within-wave PA → WR — the cross-lagged test of which direction dominates"_ — but pointedly **omits `WR_t → LS_t1`**. So the reverse hypothesis (learning to read words feeds back into letter-sound knowledge) is ruled out a priori and cannot be tested. Roch & Jarrold (2012, [doi:10.1016/j.jcomdis.2011.11.001](https://doi.org/10.1016/j.jcomdis.2011.11.001)) reported irregular word reading as a longitudinal predictor of later nonword reading in twelve participants with Down syndrome. This concerns nonword reading, not letter-sound knowledge; it motivates a hypothesis rather than establishing a causal `WR → LS` edge.

This spec is the smallest concrete step: reuse the existing longitudinal-mediation machinery (as in `med-076`, `L_t2 → W_t4`) to fit the **mirror** `W_t2 → L_t4`, and report the two side by side. It is the randomisation-anchored, low-build first move; the fuller RI-CLPM / cross-lagged programme (see `notes/202607172000-adjustment-set-review-full-suite.md`'s companion discussion) can follow.

**Interpretation.** Latent general ability and treatment-induced dose confounding remain in both specifications. Adding a reverse edge does not identify either natural indirect effect. The two models assume different directions and predict different outcomes; neither their raw indirect-effect magnitudes nor their direction probabilities establish which causal direction dominates. Assignment continues to randomise treatment schedules after crossover, but it does not randomise either skill as a mediator.

## Step 1 — DAG edit (one line)

In `dag/dag-language-reading-lagged.dagitty`, change the reverse block:

```
WR_t -> { TE_t1 TR_t1 PA_t1 RW_t1 }      →      WR_t -> { TE_t1 TR_t1 PA_t1 RW_t1 LS_t1 }
```

and extend the header comment (the reverse/reciprocal section, ~lines 43–51) to record `WR_t → LS_t1` as the reading-feeds-back-into-the-code cross-lagged test, mirroring the existing `WR → PA` reciprocal. The edit is structurally identical to the reverse edges already present and stays acyclic (all edges run `t → t1`, never back). Node count is unchanged (`LS_t1` already exists); edge count 195 → 196.

## Step 1b — CI / identification ripple (REQUIRED before interpreting anything)

The new edge changes the graph that `tests/test_lagged_dag_adjustment_sets.py` uses to validate **every** mediation adjustment set, so this is not a cosmetic edit:

1. Update `assert template.number_of_edges() == 195` → `196` (node assertion unchanged at 36).
2. Add `"LS"` to `REVERSE = ["TE", "TR", "PA", "RW"]` so the hand-coded unroll still mirrors the `.dagitty` (the `test_unroll_slices_mirror_the_dagitty_template` guard).
3. **Re-run the full test and re-derive.** Adding `WR_t → LS_t1` can change the witness / validity status of the models whose mediator or outcome touches `LS` or `WR` — `med-059/076/078/086/087/186/187`. For each, confirm (or update `MED_WITNESSES` / `MED_ER_ROLES`) that: the fitted set still contains no `IG`-descendant, the mediator-outcome backdoors are still not strictly blockable (the `IS` witness), and no _new_ observable backdoor opened that the fitted set leaves open. Do **not** interpret the new model until this is green and any change is documented.

These checks were prerequisites for the July implementation recorded above.

## Step 2 — the reverse model (mirror of med-076)

New module `src/language_reading_predictors/statistical_models/lrp_rli_med_176.py` (id `lrp-rli-med-176`; register in `definitions.MODEL_REGISTRY`):

- `kind="mediation"`, exposure = randomised arm `IG`.
- **Mediator** `mechanism_symbol="W"` → `W_t2` (Beta-Binomial on `W_t1`).
- **Outcome** `outcome_symbol="L"`, `extra={"outcome_time": 4}` → `L_t4` (Beta-Binomial on `L_t1`), reusing `load_and_prepare_lagged_outcome` exactly as med-076 does.
- **Adjustment set — DERIVE against the revised lagged DAG with the finder; do not hand-copy.** Candidate (for review): `[G, A, L_pre, W_t1, hs, hs_missing, deapp_c, deapp_c_missing]` — outcome own-baseline `L_pre`, mediator baseline `W_t1`, and the exogenous common causes of `WR` (mediator) and `LS` (outcome): hearing `HS` (`hs`) and speech `SP` (`deapp_c`). Note the asymmetry vs med-076: phonological memory `RW`/`erbto` is **not** a DAG parent of `LS`, so it is _not_ a mediator-outcome confounder here and should be omitted; likewise baseline vocabulary `E`/`R` cause `WR` but not `LS`, so they are not mediator-outcome confounders for this route (unlike med-076, where they were precision terms for the `WR` outcome). Confirm all of this with the d-separation finder against the edited DAG.
- Config: `dev` smoke-test → `reporting` (mediation family runs no PSIS-LOO; 6×6000 draws).

## Step 3 — the contrast and readouts

Report `med-176` (`W_t2 → L_t4`, reverse) beside `med-076` (`L_t2 → W_t4`, forward):

- NIE (indirect), NDE (direct), total and **proportion mediated**, with the median, inner 50% and outer 89% equal-tailed credible intervals and named tail probabilities required by current reporting rules;
- the separate period-1 assigned-arm contrast, with its available-case and timing qualifications; it does not identify the mediation decomposition;
- the unmeasured-confounding tipping point for each NIE.

**Do _not_ compare the two by PSIS-LOO / ELPD.** They predict _different outcomes_ (`W` vs `L`), so their pointwise log-likelihoods are on different scales and are not comparable — LOO is only valid within a same-outcome nested comparison (that belongs to the later RI-CLPM step, not this one). The contrast here is the NIE magnitudes, direction probabilities, and the ceiling-adjusted interpretation below. (This corrects the "NIE contrast + LOO" phrasing in the originating discussion.)

## Ceiling caveat on the reverse outcome (decisive for interpretation)

`LS` is bounded 0–32 and rises across waves: at **t4, 12% of children are at the ceiling (32) and 25% are ≥30** (t1 mean 14.3 → t4 mean 23.7). So the reverse model's _outcome_ has less room to move than med-076's `WR` outcome (79-item scale, observed max 64). A near-zero reverse NIE therefore has **two** explanations that must be separated — a genuinely absent `WR → LS` effect, or a ceiling artefact — so:

- report the share of children at/near the `LS` ceiling at t4 alongside the estimate;
- fit a **t3 sensitivity** (`W_t2 → L_t3`; only ~6% at ceiling at t3) — a cleaner-ceiling read of the same direction;
- optionally, a non-ceiling-subset sensitivity (drop children at `L_t4 = 32`).

## Limits of the comparison

Compare each decomposition with its own outcome scale, fitted population, uncertainty and confounding sensitivity. Positive indirect components in both specifications do not demonstrate reciprocal causation. A near-zero component does not rule out that direction. Changing the outcome from t4 to t3 changes both ceiling support and follow-up time, so it cannot isolate ceiling effects. Selecting children by their observed t4 outcome also changes the population through post-randomisation selection.

## Honest limits (state in the report)

- Both directions remain subject to latent-ability and treatment-induced confounding.
- The design randomises a multi-component intervention, not an isolated experimental change in letter sounds or word reading. It does not by itself establish greater power for one mediator direction.
- Sample size alone does not determine interval width; read uncertainty and simulation diagnostics from each fit.
- Different instruments can have different measurement errors. Item counts alone do not establish their reliabilities, and a latent-variable model needs its own measurement assumptions.

## Status

The July design is recorded above as approved and implemented. Current model definitions, run plans and publication decisions govern reuse. The interpretation corrections added on 21 October change documentation, not the fitted specification or stored results.
