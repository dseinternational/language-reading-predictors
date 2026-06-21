# LRP level models — same-skill "construct-reduced" variants

**Date:** 2026-06-21
**Scope:** the 11 gradient-boosting *level* models (lrp02, 04, 06, 08, 10, 12,
14, 16, 18, 20, 22). Defines and implements the **same-skill variant rule**
that factors concurrent same-construct measurement into the exploratory layer,
following the replication finding that level-model R² is largely concurrent
same-construct correlation (`notes/202606201500-gb-replication-findings.md`, §3.3).

## Motivation

The replication showed level-model R² is repeatedly carried by a co-administered,
same-construct (sometimes same-instrument) test — e.g. `aptinfo`↔`aptgram`
(scored from the same Action Picture Test picture descriptions),
`deappin`↔`deappfi` (scored from the same DEAP picture-naming sample). Predicting
an outcome from a *parallel scoring of the same performance* is **criterion
contamination**, not a developmental finding (it is not train/test leakage —
GroupKFold handles that — but shared-method variance).

The gradient-boosting feature selection (distance-correlation redundancy +
permutation importance) **cannot** catch this: it scores predictor↔predictor
redundancy and predictor importance, neither of which penalises a predictor for
being a parallel form of the *outcome*. Worse, that parallel form has the
highest importance, so a CV-optimising filter anchors on it and prunes around it.
The contamination is therefore invisible to — and protected by — the predictive
selection. It has to be addressed with substantive (curated) knowledge of which
predictors restate the outcome, not a statistic.

## Decision

- **Primaries unchanged.** The primary level models remain the best predictive
  sets (they keep concurrent siblings). They are the predictive / convergent-
  validity artifact, read as concurrent correlation per the replication note.
- **Add a same-skill-reduced variant** (`lrpNN_noconstruct`) for each level model
  that *has* a same-skill sibling among its selected predictors. The variant
  drops that sibling and is re-tuned (Optuna 150-trial MAE, 10-fold GroupKFold,
  seed 47), answering: *what predicts the outcome beyond a concurrent measure of
  the same skill?*

### "Same skill", not "same domain"

The exclusion unit is the **skill**, which is finer than the schema's
`Variables.CONSTRUCTS` families. A predictor is dropped from a variant only when
it is a concurrent restatement of the outcome — either:

1. **same elicited sample** — APT information & grammar (same picture
   descriptions); DEAP initial/vowel/final (same picture-naming sample); or
2. **a different instrument of the identical skill** — e.g. expressive
   vocabulary measured by the standardised EOWPVT *and* the bespoke taught test
   (`b1exto`); receptive vocabulary by ROWPVT *and* `b1reto`.

**Different skills are kept visible**, even within one `CONSTRUCTS` domain:

- letter-sound knowledge (`yarclet`) keeps nonword decoding (`nonword`), phoneme
  blending (`blending`) and phonetic spelling (`spphon`) — these are distinct
  skills, and their (cross-skill) association with letter-sound knowledge is of
  interest, not contamination;
- expressive vs receptive **grammar** are treated as separate (different
  modalities — a live question in DS research), so `trog` stays visible to
  `aptgram`/`aptinfo` and vice versa.

This is why the coarse-`CONSTRUCTS` variants from the earlier FS pass were
**removed** (see below): `reading_decoding` and `language_composite` lump
genuinely different skills.

## Variants (primaries unchanged)

Primary → variant, pooled out-of-fold R² (51-fold GroupKFold, reporting fit) and
tuner-inner CV MAE (10-fold):

| variant | outcome | drops (same skill) | pooled OOF R² | tuner MAE |
|---|---|---|---|---|
| `lrp08_noconstruct` | receptive vocab (`rowpvt`) | `b1reto` | 0.64 → 0.65 | 6.90 → 7.04 |
| `lrp18_noconstruct` | expressive grammar (`aptgram`) | `aptinfo` | 0.68 → 0.48 | 2.33 → 3.32 |
| `lrp20_noconstruct` | expressive info (`aptinfo`) | `aptgram` | 0.80 → 0.75 | 2.69 → 3.06 |
| `lrp22_noconstruct` | final-consonant articulation (`deappfi`) | `deappin` | 0.55 → −0.03 | 9.96 → 16.13 |

- **lrp08**: receptive vocabulary is *still* well-predicted (R² ≈ unchanged)
  after dropping the bespoke receptive sibling — the signal is carried by
  expressive-vocabulary and broader language measures (`aptinfo`, `eowpvt`,
  `trog`, `celf`), not by a parallel receptive-vocab test. The most informative
  of the four variants.
- **lrp18 / lrp20**: dropping the APT same-sample sibling costs ≈ 0.20 / 0.05 of
  R². The remainder is genuine cross-skill signal (top remaining predictors
  `erbnw`, `b1exto`); `trog` and other constructs stay visible.
- **lrp22**: null (see below).

`lrp18`/`lrp20` variants existed already and drop exactly the right APT sibling;
only their rationale wording was corrected (they were mislabelled
"language_composite"-reduced — they never dropped `celf`/`trog`). `lrp08`/`lrp22`
variants are new (re-tuned 2026-06-21).

### deappfi (lrp22) — the null is the finding

Dropping `deappin` (DEAP initial-consonant accuracy, scored from the *same*
picture-naming sample as the final-consonant target `deappfi`) collapses CV
(tuner-inner MAE 9.96 → 16.13; the chosen model is ~3 trees). **There is no
non-articulation predictor of final-consonant accuracy at this n.** Per decision,
the within-DEAP primary (`lrp22`, MAE ≈ 9.96) is kept as a convergent-validity
reference and this variant documents the null — rather than reducing the primary to it.

## Models with no variant (no same-skill sibling in the predictors)

- `lrp02` (`ewrswr`): word-reading composite; its EWR/SWR components are excluded
  as composite members, so no same-skill sibling is present.
- `lrp04` (`eowpvt`): the primary already dropped `b1exto` (2026-04 construct
  decision), so it is *already* same-skill-clean — no separate variant needed.
- `lrp06` (`yarclet`), `lrp10` (`celf`), `lrp12` (`trog`), `lrp14` (`nonword`),
  `lrp16` (`blending`): their R² comes from *different* skills, which are kept
  visible. Specifically removed in this pass: **`lrp12_noconstruct`** (had
  dropped expressive grammar + concept knowledge from receptive grammar) and
  **`lrp14_noconstruct`** (had dropped letter-sound knowledge from nonword) —
  both dropped different-skill measures and so contradicted the same-skill rule.

## Caveats

- `lrp08`'s primary is still **under-pruned** (8 dcor ≥ 0.70 pairs in the
  expressive-vocabulary / APT cluster — see the replication note's open
  "re-prune LRP08/LRP09" item). `lrp08_noconstruct` inherits that residual
  redundancy; the lrp08/lrp09 re-prune remains a separate follow-up.
- Same exploratory caveats as the primaries: gain models untouched here; reduced
  *rankings* are reliable only for the top one or two predictors; the pinned
  hyperparameters are CV-equivalent optima, not unique.

## Reproduce

```bash
# retune the two new variants (writes output/tuning/<id>/best_params.json):
conda run -n dse-language-reading-predictors --no-capture-output \
  python output/replication/tune_variants.py
# fit + render the four affected variants (reporting config):
conda run -n dse-language-reading-predictors --no-capture-output \
  python scripts/fit_model.py lrp08_noconstruct --config reporting --render
#   ... lrp22_noconstruct / lrp18_noconstruct / lrp20_noconstruct
```
