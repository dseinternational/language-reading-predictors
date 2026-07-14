<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

# Mediation beyond word reading: the code route to nonword decoding and blending (#228 item 12)

Date: 2026-07-14

Related: #228 (suite-gap analysis, item 12), #225 (nonword gain-factor gf-011), #246/#264 (mediation adjustment sets), LRP59/64/66/75 (word-reading mediation), mech-072/172 (the L→N association).

## What this adds

The mediation family targeted word reading only. Item 12 extends the code route to two further outcomes, via letter-sounds (L):

- **`lrp-rli-med-086` — nonword reading (N), off-floor.** The code route on the purest decoding outcome. N is ~57% floored, so — as everywhere in the suite — the outcome is the off-floor indicator ($N_{t2}>0$), and NIE/NDE are reported on the **off-floor risk-difference** scale.
- **`lrp-rli-med-087` — phoneme blending (B), graded.** The downstream chain link, a standard graded Beta-Binomial mediation (mirrors med-059 with the outcome swapped).

Both are single-mediator (L alone). The refit found blending is downstream of letter-sounds and adds no independent code route (LRP66/75), so the L+B decomposition on N is a deferred follow-up.

## The opt-in off-floor g-formula extension

The mediation g-formula only supported a graded Beta-Binomial outcome. This adds an **opt-in** `outcome_kind="bernoulli_offfloor"` path (default `"beta_binomial"` → the 11 existing med models are **byte-identical**; the `test_mediation.py` regression passes unchanged):

- `factories._build_outcome_leg` / `build_mediation_model`: the off-floor outcome leg is a Bernoulli on $(post>0)$ (node `y_offfloor`, no `kappa_Y`) and **drops the own-baseline term** `b_W` (the Normal(1,·) autoregressive prior does not transfer to a binary indicator, and a floored baseline logit is degenerate — the off-floor ITT/DiD/gf convention). `MediationData.off_floor=True`, `n_trials_W=1`.
- `mediation.decompose`: an off-floor branch reads no `b_W` and, with `n_trials_W=1`, the `words_*` columns collapse onto the risk difference (`== prob_*`); an `off_floor` flag on each row lets the report label the scale. `sensitivity_sweep` inherits it.
- `pipeline.fit_mediation`: reads `outcome_kind`, uses the `y_offfloor` obs node for PPC/prior-predictive.
- New unit test `test_decompose_offfloor_outcome`: proves decompose returns finite risk-difference NIE/NDE and does not read `b_W`.

## Dev-tier read (reporting folds into the next refit)

Both fit with **0 divergences**; the code-route story is coherent:

- **med-086 (N via L):** NIE via L = **+0.086 off-floor risk difference, P(NIE>0)=0.985**; NDE ≈ 0 (P=0.40); proportion mediated ≈ 0.84. The nonword-decoding gain runs **through letter-sounds**, and the near-zero direct effect **corroborates the DAG's no-`IG→NW` exclusion restriction** (the intervention reaches nonword reading only via the code skills).
- **med-087 (B via L):** NIE via L = +0.25 (P=0.91, moderate) with a substantial direct path (NDE P=0.82) — blending is partly, not wholly, transmitted through letter-sounds.

Wide intervals at n≈50–53; a model-based g-formula decomposition under stated (cross-world) assumptions, not an identified natural effect (same caveats as med-059).

## Adjustment sets — flagged for DAG review

Mediation adjustment sets are owned by the DAG review (#246/#258/#259). Proposed, from the revised DAG + gf-011/mech-072/med-064 precedent:

- **med-086 (L→N):** SP (`deapp_c`) + RW (`erbto`); **HS excluded** (not a nonword parent — the gf-011 rule); no outcome own-baseline (off-floor).
- **med-087 (L→B):** HS (`hs`) + SP (`deapp_c`); **RW excluded** (not a letter-sound parent). **Flag:** intervention sessions `IS`(`attend`) are a common cause of L and B but treatment-affected (IG→IS) → a recanting-witness risk (the reason E/R were dropped); proposed **not** adjusted, pending confirmation.

## Files

- `factories.py` (`_build_outcome_leg`, `build_mediation_model`, `MediationData`), `mediation.py` (`decompose` off-floor branch), `pipeline.py` (`fit_mediation`) — the opt-in off-floor extension.
- `lrp_rli_med_086.py`, `lrp_rli_med_087.py` + `_MECH` registry entries.
- `docs/models/lrp-rli-med-08{6,7}/index.qmd` + a guarded off-floor callout in `_partials/_results_mediation.qmd`.
- `tests/statistical_models/test_mediation.py` — the off-floor decompose test.

This is preliminary, exploratory work in progress.
