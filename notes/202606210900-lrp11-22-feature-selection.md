# LRP11–22 — feature selection implemented

**Date:** 2026-06-21 (selection derived in the 2026-06-20 replication;
see `notes/202606201500-gb-replication-findings.md`).

The twelve models LRP11–LRP22 previously carried the **full** default predictor
set (32 level / 34 gain) with **no feature selection** — each with ~12
distance-correlation ≥ 0.70 pairs and 18–28 predictors at/below the importance
noise floor. This applies a standard, refit-CV-driven selection to all twelve
and adds construct-reduced variants for the level models where a same-construct
sibling dominates.

## Method (standard, refit-CV driven)
1. **Correlation filter** — rank predictors by full-set out-of-fold (neg-RMSE)
   permutation importance; keep a predictor unless it has distance-correlation
   ≥ 0.70 with an already-kept (higher-importance) predictor. The baseline
   measure is force-kept for gain models.
2. **Noise-floor cut** — drop remaining predictors at/below 0.005 importance.
3. **Standardised-instrument swap** — prefer the standardised test over its
   intervention-taught sibling (`eowpvt`←`b1exto`, `rowpvt`←`b1reto`) **only**
   when it does not cost CV *and* does not reintroduce a ≥ 0.70 pair (the swap
   was skipped for lrp12/lrp20, where `eowpvt`↔`rowpvt` = 0.72 would have).
4. **Decision** — for the two flat-importance models (lrp12, lrp13) the smaller
   of the corr-filter / corr-filter+floor sets was chosen by re-tuned tuner-inner
   CV MAE within 3 % of the full set; the rest took the corr-filter+floor set.
5. Each final set was **re-tuned** (Optuna 150-trial MAE, 10-fold GroupKFold,
   seed 47) and the params pinned in the module. Encoded as `SelectionStep`s so
   the provenance is persisted to `config.json`.

> Tooling (gitignored): `output/replication/implement_fs.py` (retune),
> `gen_edits.py` + `apply_edits.py` + `cleanup_docs.py` + `set_notes.py` (encode),
> `verify_fits.py` (check). Manifest: `output/replication/findings/fs_implementation.json`.

## Result — every model, redundancy eliminated (0 dcor ≥ 0.70 pairs)
Pooled OOF R² / MAE from the encoded models (real 51-fold GroupKFold):

| model | target | n full→final | R² full→final | MAE full→final | note |
|---|---|---|---|---|---|
| lrp11 | trog_gain | 34→5 | 0.18→0.19 | 3.13→3.04 | |
| lrp12 | trog | 32→26 | 0.46→0.48 | 2.88→2.86 | resists pruning (flat importance) |
| lrp13 | nonword_gain | 34→2 | 0.24→0.15 | 1.00→0.96 | ⚠ R²↓ but MAE↓; ~48 % zeros, near-degenerate |
| lrp14 | nonword | 32→3 | 0.38→0.49 | 0.92→0.77 | |
| lrp15 | blending_gain | 34→2 | 0.13→0.15 | 1.53→1.48 | |
| lrp16 | blending | 32→2 | 0.36→0.32 | 1.71→1.75 | coarse 0–10 target caps R² |
| lrp17 | aptgram_gain | 34→6 | 0.07→0.22 | 3.20→2.94 | noise removal helps markedly |
| lrp18 | aptgram | 32→4 | 0.68→0.70 | 2.63→2.43 | |
| lrp19 | aptinfo_gain | 34→5 | 0.09→0.09 | 3.30→3.20 | |
| lrp20 | aptinfo | 32→8 | 0.80→0.81 | 2.94→2.74 | |
| lrp21 | deappfi_gain | 34→6 | 0.10→0.11 | 8.23→8.21 | |
| lrp22 | deappfi | 32→7 | 0.55→0.55 | 9.95→10.09 | |

For 10/12 the reduction holds or **improves** pooled R²; lrp13 and lrp16 lose a
little R² but improve/hold MAE on near-degenerate targets (heavy zero mass / coarse
scale), so the parsimonious set was kept. CV deltas above are at the model's full
51-fold splits; the per-model `SelectionStep` records the tuner-inner 10-fold MAE
used for the selection decision.

## Construct-reduced variants (level models — "what predicts X beyond its sibling tests")
| variant | drops | R² primary→variant |
|---|---|---|
| lrp12_noconstruct | aptgram, celf | 0.48→0.44 |
| lrp14_noconstruct | yarclet | 0.49→0.46 |
| lrp18_noconstruct | aptinfo | 0.70→0.48 |
| lrp20_noconstruct | aptgram | 0.81→0.75 |

**lrp22 (deappfi level) has no construct variant by design:** removing the sibling
DEAP sub-score `deappin` collapses CV to R² ≈ −0.09 (replication finding) — there is
no non-articulation predictor of final-consonant accuracy, so a construct-reduced
deappfi model is not viable; recorded as a null result rather than shipped.

## Caveats (carry over from the replication)
- These remain **exploratory**. Gain models are near-noise (R² ≤ ~0.22); in every
  model only the dominant predictor is robustly above the importance floor, so the
  reduced *rankings* are not reliable beyond the top one or two.
- Level-model performance is largely concurrent same-construct correlation; the
  construct variants exist to expose that.
- Hyperparameters are not identifiable (flat surface); the pinned params are one of
  many CV-equivalent optima.

## To regenerate official artifacts
```bash
conda run -n dse-language-reading-predictors --no-capture-output \
  python scripts/fit_model.py all --include-variants --config reporting --render
```
