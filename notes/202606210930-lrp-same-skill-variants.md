# LRP level models — same-skill "construct-reduced" variants

**Date:** 2026-06-21
**Scope:** the 11 gradient-boosting _level_ models (lrp02, 04, 06, 08, 10, 12,
14, 16, 18, 20, 22). Defines and implements the **same-skill variant rule**
that factors concurrent same-construct measurement into the exploratory layer,
following the replication finding that level-model R² is largely concurrent
same-construct correlation (`notes/202606201500-gb-replication-findings.md`, §3.3).

## Motivation

The replication showed level-model R² is repeatedly carried by a co-administered,
same-construct (sometimes same-instrument) test — e.g. `aptinfo`↔`aptgram`
(scored from the same Action Picture Test picture descriptions),
`deappin`↔`deappfi` (scored from the same DEAP picture-naming sample). Predicting
an outcome from a _parallel scoring of the same performance_ is **criterion
contamination**, not a developmental finding (it is not train/test leakage —
GroupKFold handles that — but shared-method variance).

The gradient-boosting feature selection (distance-correlation redundancy +
permutation importance) **cannot** catch this: it scores predictor↔predictor
redundancy and predictor importance, neither of which penalises a predictor for
being a parallel form of the _outcome_. Worse, that parallel form has the
highest importance, so a CV-optimising filter anchors on it and prunes around it.
The contamination is therefore invisible to — and protected by — the predictive
selection. It has to be addressed with substantive (curated) knowledge of which
predictors restate the outcome, not a statistic.

## Decision

- **Primaries unchanged.** The primary level models remain the best predictive
  sets (they keep concurrent siblings). They are the predictive / convergent-
  validity artifact, read as concurrent correlation per the replication note.
- **Add a same-skill-reduced variant** (`lrpNN_noconstruct`) for each level model
  that _has_ a same-skill sibling among its selected predictors. The variant
  drops that sibling and is re-tuned (Optuna 150-trial MAE, 10-fold GroupKFold,
  seed 47), answering: _what predicts the outcome beyond a concurrent measure of
  the same skill?_

### "Same skill", not "same domain"

The exclusion unit is the **skill**, which is finer than the schema's
`Variables.CONSTRUCTS` families. A predictor is dropped from a variant only when
it is a concurrent restatement of the outcome — either:

1. **same elicited sample** — APT information & grammar (same picture
   descriptions); DEAP initial/vowel/final (same picture-naming sample); or
2. **a different instrument of the identical skill** — e.g. expressive
   vocabulary measured by the standardised EOWPVT _and_ the bespoke taught test
   (`b1exto`); receptive vocabulary by ROWPVT _and_ `b1reto`.

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

| variant             | outcome                                  | drops (same skill) | pooled OOF R² | tuner MAE    |
| ------------------- | ---------------------------------------- | ------------------ | ------------- | ------------ |
| `lrp04_noconstruct` | expressive vocab (`eowpvt`)              | `b1exto`           | 0.70 → 0.65   | 6.16 → 7.08  |
| `lrp18_noconstruct` | expressive grammar (`aptgram`)           | `aptinfo`          | 0.70 → 0.48   | 2.33 → 3.32  |
| `lrp20_noconstruct` | expressive info (`aptinfo`)              | `aptgram`          | 0.80 → 0.75   | 2.69 → 3.06  |
| `lrp22_noconstruct` | final-consonant articulation (`deappfi`) | `deappin`          | 0.55 → −0.03  | 9.96 → 16.13 |

- **lrp04**: expressive vocabulary stays well-predicted (0.70 → 0.65) after
  dropping the bespoke expressive sibling `b1exto` — the signal is carried by
  other constructs (receptive vocabulary, reading, articulation), not by a
  parallel expressive-vocab test. The most informative of the four variants.
- **lrp18 / lrp20**: dropping the APT same-sample sibling costs ≈ 0.20 / 0.05 of
  R². The remainder is genuine cross-skill signal (top remaining predictors
  `erbnw`, `b1exto`); `trog` and other constructs stay visible.
- **lrp22**: null (see below).

`lrp18`/`lrp20` variants existed already and drop exactly the right APT sibling;
only their rationale wording was corrected (they were mislabelled
"language_composite"-reduced — they never dropped `celf`/`trog`). The `lrp04` and
`lrp22` variants are new; the variant **set** was re-derived by the subsequent
uniform feature-selection pass (`notes/202606211200-uniform-gb-fs.md`).

### deappfi (lrp22) — the null is the finding

Dropping `deappin` (DEAP initial-consonant accuracy, scored from the _same_
picture-naming sample as the final-consonant target `deappfi`) collapses CV
(tuner-inner MAE 9.96 → 16.13; the chosen model is ~3 trees). **There is no
non-articulation predictor of final-consonant accuracy at this n.** Per decision,
the within-DEAP primary (`lrp22`, MAE ≈ 9.96) is kept as a convergent-validity
reference and this variant documents the null — rather than reducing the primary to it.

## Models with no variant (no same-skill sibling in the predictors)

- `lrp02` (`ewrswr`): word-reading composite; its EWR/SWR components are excluded
  as composite members, so no same-skill sibling is present.
- `lrp08` (`rowpvt`): the uniform feature selection already drops the bespoke
  receptive sibling `b1reto`, so the primary is same-skill-clean — no variant
  needed.
- `lrp06` (`yarclet`), `lrp10` (`celf`), `lrp12` (`trog`), `lrp14` (`nonword`),
  `lrp16` (`blending`): their R² comes from _different_ skills, which are kept
  visible. The earlier coarse-construct **`lrp12_noconstruct`** and
  **`lrp14_noconstruct`** variants were removed (they dropped different-skill
  measures — expressive grammar / concept knowledge from receptive grammar, and
  letter-sound knowledge from nonword).

## Caveats

- The variant **set** documented here was re-derived by the later uniform
  feature-selection pass (`notes/202606211200-uniform-gb-fs.md`): `lrp04` gained
  a variant and `lrp08` lost the need for one (uniform pruning drops `b1reto`,
  and also cleared lrp08's earlier redundancy).
- Same exploratory caveats as the primaries: gain models untouched here; reduced
  _rankings_ are reliable only for the top one or two predictors; the pinned
  hyperparameters are CV-equivalent optima, not unique.

## Reproduce

```bash
# retune the two new variants (writes output/tuning/<id>/best_params.json):
conda run -n dse-language-reading-predictors --no-capture-output \
  python output/replication/tune_variants.py
# fit + render the four affected variants (reporting config):
conda run -n dse-language-reading-predictors --no-capture-output \
  python scripts/fit_model.py lrp04_noconstruct --config reporting --render
#   ... lrp22_noconstruct / lrp18_noconstruct / lrp20_noconstruct
```
