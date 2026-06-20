# LRP74–LRP76 — taught-vocabulary outcome models

_2026-06-19_

## Why

The suite's vocabulary models so far use only the **standardised** tests
(ROWPVT/EOWPVT; LRP53/LRP54), both of which were null. But the published trial's
headline vocabulary result was on **directly-taught** vocabulary: Burgoyne et al.
(2012) found a significant gain on taught *expressive* vocabulary while
standardised vocabulary did not move. Until now the suite could not speak to that
result. These three models fill the gap — the long-flagged "intervention-fidelity
/ positive-control" work (see `notes/2026-05-12-project-review.md`, §D, and
`notes/202604181453-review-next-steps.md`, LRP61 sketch).

## The measures (`measures.py`, new symbol family — NOT in `ITT_OUTCOMES`)

Block-1 taught-vocabulary tests (words explicitly taught in phase 1, weeks 1–20;
baseline t1, randomised post t2 — the ITT window). Block 2 is introduced in phase
2 and has no t1 baseline, so it is not modelled here.

| Symbol | Column     | n_trials | Confirmed | Note                                   |
| ------ | ---------- | -------- | --------- | -------------------------------------- |
| `TE`   | `b1extau`  | 24       | yes       | Taught expressive (6 words × 4 types)  |
| `TR`   | `b1retau`  | 24       | yes       | Taught receptive                       |
| `UE`   | `b1exnt`   | 12       | **no**    | Not-taught expressive (observed max)   |
| `UR`   | `b1rent`   | 12       | **no**    | Not-taught receptive (observed max)    |

- **Taught = 24** is documented: "Six words of each type (nouns, adverbs,
  adjectives, prepositions)" and tabulated as `(24)` in Burgoyne et al. (2012),
  Table 3.
- **Not-taught = 12** is the *observed maximum* (both modalities top out at 12,
  consistent with a half-size 3-words-×-4-types control set). The paper tabulates
  only the taught tests, so this is flagged `n_trials_confirmed=False`. The
  *sign* of the taught-vs-not-taught difference is robust to this (taught is a
  strong effect, not-taught little), but the **magnitude** is not — `tau[UE]`
  shifts with the assumed not-taught ceiling, so the difference and any
  probability-scale summary for the not-taught outcomes should be treated as
  approximate pending the data dictionary. Confirm the ceiling, or refit with an
  alternative (e.g. N=24) as a sensitivity check, before quoting the size.
  (Cf. the active `fix/lrp80` ceiling-confirmation work — W=79, P=92.)
- Data: n=54 (28 control, 26 intervention); all four Block-1 measures complete at
  both t1 and t2, so LRP74/75 lose **zero** cases.

## The models (ids 74–76 — the clean gap below the in-flight 78–85 dev branches)

_Sign convention (suite-wide): `G = group − 1`, and group 2 is the waiting-control
arm in the randomised window, so a **negative τ = intervention benefit** (see
`notes/202604181600-lrp52-58-findings.md`). "Expectation: positive" below means a
positive *benefit*, i.e. τ < 0._

- **LRP74 — ITT, taught expressive (`TE`).** Headline / positive control. Mirrors
  the LRP52 ITT (phase-0 window, Beta-Binomial, own-baseline autoregression).
  Conditioned on its own taught baseline **plus the matched standardised
  expressive baseline `E`** (not all eight ITT baselines — parsimony at n≈54, and
  no dropped cases). _Expectation: positive_ (the trial's significant effect).
- **LRP75 — ITT, taught receptive (`TR`).** Same template, conditioned on
  standardised receptive `R`. _Expectation: weaker / near-null_ — the trial found
  gains on taught expressive "though not on the equivalent receptive vocabulary
  measure". The expressive≠receptive asymmetry is itself the finding; 74 and 75
  are reported as a pair.
- **LRP76 — joint, taught vs not-taught expressive (`TE` vs `UE`).** The
  generalisation contrast. Two-outcome joint Beta-Binomial; headline parameter is
  `tau[UE] − tau[TE]` (new `reporting.tau_difference_summary`; with negative-τ =
  benefit, this equals benefit-on-taught minus benefit-on-not-taught, so a
  *positive* value means the taught words moved more — limited generalisation).
  A 2×2 LKJ residual
  correlation models within-child taught/not-taught dependence (identifiable at
  K=2, unlike the prior-dominated 8-outcome LRP55 block) — toggle off if it
  destabilises sampling (the difference is then conservative). _Expectation:
  positive difference_ — taught moved more than untaught (little transfer).

## Framing & honesty

The contrast, not a single "vocabulary improved" claim: the programme **teaches
what it teaches (expressively), with little transfer** — not to the receptive
knowledge of those same words (LRP75), not to untaught words (LRP76), and not to
standardised vocabulary (LRP53/54). Taught and standardised vocabulary are kept as
**separate outcomes** throughout. Same caveats as the other ITT models: n≈54, wide
intervals, causal claim limited to the randomised phase-0 window.

## Implementation (additive — LRP52–73 untouched; defaults preserve behaviour)

- `measures.py`: +4 `Measure` entries + `TAUGHT_BLOCK1_OUTCOMES`.
- `factories.build_itt_model`: new optional `cross_symbols` (default `None` →
  the existing every-other-ITT-outcome behaviour).
- `pipeline.fit_itt`: reads `outcomes` / `cross_symbols` from `spec.extra`,
  validates `outcome_symbol ∈ outcomes`.
- `pipeline.fit_joint`: loads `spec.extra["outcomes"]`; computes the difference
  summary when `spec.extra["difference"]` is set.
- `reporting.tau_difference_summary`: per-draw `tau[a] − tau[b]` with interval and
  `P(>0)`.
- New modules `lrp74/75/76.py`, registered in `scripts/fit_statistical_model.py`.
- Report stubs `docs/models/lrp74–76/index.qmd`; factory + difference tests.

## Build 3 (inventory parity) — taught-vocabulary ML models LRP23/LRP24

Added the gain + achievement-level LightGBM/SHAP models for taught vocabulary,
parallel to lrp03/lrp04 for standardised expressive vocabulary (a different
subsystem: `src/language_reading_predictors/models/`, not the Bayesian
`statistical_models/`). Next free ML ids, continuing the odd-gain/even-level
pairing (lrp23↔lrp03, lrp24↔lrp04):

- **LRP23 — taught expressive-vocabulary *gain*** (`b1extau_gain`, `GainModel`),
  analogue of lrp03.
- **LRP24 — taught expressive-vocabulary *level*** (`b1extau`, `LevelModel`),
  analogue of lrp04.

Key decisions:

- **Leakage exclusion.** The taught/not-taught block sub-scores (b1extau, b1exnt,
  b1retau, …) are already in `DEFAULT_EXCLUDED`, so the only block columns in the
  default predictor sets are the two *totals*. `b1exto` (= `b1extau + b1exnt`)
  contains the target, so both models set `exclude=[V.B1EXTO]` — the single
  deviation from the lrp03/lrp04 predictor set. `b1reto` (receptive total) is not
  a superset of the expressive taught score and is retained for the baseline.
- **Hyperparameters** are borrowed from the lrp03/lrp04 standardised analogues as
  a starting baseline (the targets are closely matched: `b1extau` skew 0.35,
  `b1extau_gain` skew 0.71, ~20% negative). A target-specific Optuna tune
  (`scripts/tune_model.py lrp23/lrp24`) and iterative feature selection are the
  follow-ups — exactly the state lrp03/lrp04 shipped in. Importance rankings (the
  point of these exploratory models) are robust to reasonable params.
- New: `models/lrp23.py`, `models/lrp24.py` (registered in `models/__init__.py`),
  `docs/models/lrp23,24/index.qmd`, `tests/test_models.py` (registration +
  leakage guard).

Still open (acknowledged, low priority): target-specific tuning, iterative
feature selection, and the receptive/Block-2 taught variants if wanted.
