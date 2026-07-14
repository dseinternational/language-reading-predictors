<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

# Block-2 taught-vocabulary block-active exposure family (`bx`, #228 item 5)

Date: 2026-07-14

Related: #228 (suite-gap analysis, item 5), #224 (block-1 taught-vocabulary gain/level factors + the block-active exposure-coding proposal), #214 (taught/not-taught denominators), #247 (revised-DAG adjustment sets).

## What this adds

The block-2 taught-vocabulary measures — `b2extau`/`b2retau` (taught) and `b2exnt`/`b2rent` (not-taught) — are measured at t2–t4 for every child and were **completely unused** by the statistical suite. This adds a new `block_exposure` (`bx`) family that models them: four models `lrp-rli-bx-001..004` for TE2 (taught expressive, the informative outcome), TR2 (taught receptive), UE2 and UR2 (the not-taught expressive/receptive comparators).

## Why a new family, and what the estimand is

Block 2 is introduced in phase 2, so it has **no t1 baseline and no randomised contrast** — the block-1 approach (an ITT/gain/level model anchored on the randomised t2 contrast) does not apply. The only identifying variation is the **staggered block-2 teaching**: the immediate arm reaches block 2 in phase 2 (measured at t3) while the wait-list arm is still on block 1, and the wait-list arm reaches block 2 in phase 3 (t4). The data show the signature clearly — `b2extau` is equal across arms at t2 (immediate 6.32 / wait-list 6.27 of 24), diverges at t3 (9.89 / 8.46), and converges by t4 (10.92 / 10.12).

`bx` is a **staggered-adoption / event-study** model over the per-timepoint levels frame: `logit μ = α + α_time[t] + δ·exposed + γ_A·age + γ_ability·z(blocks) + Σ γ_c·z(confounder) + u_child`, where `exposed` switches on once a child has been taught block 2 (immediate from t3, wait-list from t4). The focal `δ` is the shift in the block-2 taught-word level attributable to block 2 being actively taught.

**It is an association (parallel-trends), not a randomised effect.** There is no period in which block-2 exposure was randomised, so `δ` is causal only if block-2 trajectories would have been parallel across arms absent block-2 teaching. Crucially, over t2→t3 the wait-list arm is on **block 1**, not idle, so the contrast is "block-2-active _vs block-1-active_", never treated-vs-untreated. The child random intercept absorbs stable between-child (hence between-arm) differences and `α_time` the shared maturation trend; age-at-block-2 (immediate children reach block 2 younger) and any arm-specific slope difference remain confounded. Status `ASSOCIATION`, and the factor summary flags no causal term. This is the block-2 realisation of the block-active exposure-coding variant proposed (but not built) on #224.

The block-1 approach was ruled out on both statistical and code grounds: the level-factors pipeline is welded to the t2 randomised card (which block 2 lacks), and `build_did_model` hard-codes the P1/P2 crossover and refuses any other window. `bx` reuses the shared helpers (levels frame, `α_time`, the non-centred child intercept, the `η_base + δ·exposed` split and the treatment-tier `δ` prior) but is its own factory (`build_block_exposure_model`) and pipeline (`fit_block_exposure`), leaving the shipped families untouched.

## Findings (reporting-tier fit, 6 chains × 6000 draws)

The specificity pattern is as hoped — a positive exposure signal on the directly-taught expressive words, and inconclusive/null on the not-taught comparators and on the near-ceiling receptive words:

| Model    | Outcome                   | `δ` (logit, median) | items AME | P(δ>0)    | evidence                                         |
| -------- | ------------------------- | ------------------- | --------- | --------- | ------------------------------------------------ |
| `bx-001` | TE2 taught expressive     | **+0.144**          | **+0.72** | **0.834** | suggestive — the block-2 exposure signal         |
| `bx-002` | TR2 taught receptive      | −0.158              | −0.74     | 0.155     | inconclusive (near-ceiling; as block 1, TE ≫ TR) |
| `bx-003` | UE2 not-taught expressive | −0.114              | −0.28     | 0.251     | inconclusive (placebo behaves as it should)      |
| `bx-004` | UR2 not-taught receptive  | +0.061              | +0.15     | 0.640     | inconclusive (placebo behaves as it should)      |

All four **clear the convergence gate** (0 divergences, R̂ ≤ 1.002, min ESS ≥ 4,571). Power is thin — n ≈ 26–28/arm, one informative timepoint (t3), an item-scale divergence of ~1.4 taught words — so the family is powered to read the _direction_ and the taught-vs-not-taught contrast, not to pin a tight interval. Read it as an internal replication of the taught-word effect on fresh (block-2) data, not a standalone estimate.

## Data caveats

- **Denominators confirmed for block 2 (2026-07-14).** TE2/TR2 = 24 and UE2/UR2 = 12, checked directly against the block-2 word list: each block is 9 words × 4 word-classes = 36, split 6 taught + 3 not-taught per class → 24 taught / 12 not-taught per modality, block 2 exactly as block 1 (the #214 logic). The observed maxima are consistent (b2extau 21, b2retau 24, b2exnt 10, b2rent 12 excluding one corrupt cell). Marked `n_trials_confirmed=True`; `measures.unconfirmed_ceilings()` is now empty.
- **One corrupt source cell.** `b2rent` has a single impossible value (subject `ID_0D60E282E4368506` at t4: `b2rent = 31` against the 12-item ceiling, with `b2reto = 10 < b2retau = 21` — total below taught). `bx-004` sets that one cell to NaN (dropped, logged) via the opt-in `drop_ceiling_violations=("UR2",)` loader flag; the denominator stays 12. The global fail-loud ceiling guard is unchanged for every other measure. Flagged here for a source-data fix.

## Files

- `measures.py` — TE2/TR2/UE2/UR2 registered (denominators 24/12, `n_trials_confirmed=True`, confirmed against the word list); `TAUGHT_BLOCK2_OUTCOMES`; distal/ROPE membership.
- `factories.py` — `build_block_exposure_model` (new).
- `pipeline.py` — `fit_block_exposure` (+ `_bx_coef_names`/`_bx_diag_vars`).
- `reporting.py` — `block_exposure_summary`.
- `preprocessing.py` — opt-in `drop_ceiling_violations` loader flag.
- `definitions.py` / `model_ids.py` — `block_exposure`→`bx` family + `_BX` registry entries.
- `lrp_rli_bx_001..004.py` — the four model modules.
- `docs/models/lrp-rli-bx-00{1..4}/index.qmd` + `_partials/_results_block_exposure.qmd` — report chapters.

## Next steps

1. ~~Confirm the block-2 word counts against the source word list, then flip `n_trials_confirmed` to `True`.~~ Done (2026-07-14): the block-2 word list confirms 24 taught / 12 not-taught, flags flipped to `True`.
2. Fix the corrupt `b2rent` cell at source.
3. The four chapters render cleanly and all four fits clear the gate at reporting draws; they fold into the next full-suite refit + publish.

This is preliminary, exploratory work in progress.
