# How well can we predict word-reading gain, and which baseline factors carry the signal?

> [!WARNING]
> AI-assisted analysis. Numbers reproducible from `fit_model.py` (reporting config)
> and `predictability_readout.py`; interpretation should be reviewed by the study team.

Date: 2026-06-17

## Context

Handoff question: _can we predict an individual child's rate of word-reading progress
from baseline information, and how well?_ The project already has the model to answer
this — `LRP01` (exploratory, MAE-tuned) and `LRP01Prediction` (prediction-focused,
RMSE-tuned) predict `ewrswr_gain` (word-reading gain over a wave-to-wave transition)
from the same 6 predictors with a LightGBM, cross-validated `GroupKFold`-by-child
(53 splits, n = 157 pooled transitions). This note turns their existing
cross-validation output into a plain predictability readout. **No new model is fitted.**

## Method

- Refit `lrp01` and `lrp01_prediction` at `--config reporting`.
- `scripts/predictability_readout.py` reads each model's `metrics.json`,
  `oof_predictions.csv`, `permutation_importance.csv` and
  `shap_direction_diagnostics.csv` and produces:
  - the **skill vs a predict-the-mean baseline** — the pooled out-of-fold
    `cv_pooled_r2` (= $1 - \mathrm{SS_{res}}/\mathrm{SS_{tot}}$ with `SS_tot` taken
    against each fold's _training mean_), i.e. the fraction of individual variation
    explained out of sample. An explicit `DummyRegressor(strategy="mean")` run
    through the same folds confirms the baseline (pooled R² = 0.000);
  - a **baseline-only skill row** — the same out-of-fold CV re-run on the predictor
    set with concurrent/period-related features removed (`Variables.PERIOD_RELATED` —
    here just `attend`), so the number reflects only information available at baseline.
    The recomputed full-set value reproduces `metrics.json` exactly, confirming the
    baseline-only figure comes from the same machinery;
  - a **predicted-vs-actual calibration plot** (`calibration.png`): out-of-fold
    predicted gain (x) versus observed gain (y), with the y = x line;
  - the **top predictors** paired with **direction** (permutation importance for
    _how much_; SHAP-feature Spearman sign + monotonicity flag for _which way_ and
    _how consistently_), per CLAUDE.md's interpretation rule.
- Why pooled R² rather than the per-fold mean: at 53 group folds each held-out fold
  has only ~3 rows, so the per-fold `cv_r2_mean` is degenerate and wildly negative
  (−2.56 ± 6.60). The pooled out-of-fold R² is the honest "variance explained" number.

## Results

### Out-of-sample skill (n = 157, target SD = 4.31 reading-words)

| Model                  | tuning | pooled OOF **R²** | pooled OOF RMSE | mean-baseline RMSE | RMSE reduction | CV RMSE (per-fold) | in-sample R² |
| ---------------------- | ------ | ----------------: | --------------: | -----------------: | -------------: | -----------------: | -----------: |
| **`lrp01_prediction`** | RMSE   |         **0.211** |            3.86 |               4.34 |            11% |        3.37 ± 1.91 |        0.358 |
| `lrp01`                | MAE    |             0.117 |            4.08 |               4.34 |             6% |        3.36 ± 2.35 |        0.224 |

So the prediction-focused model explains **about 21% of the between-child variation in
word-reading gain out of sample** — modestly predictable, well above predict-the-mean
but far from individual forecasting. (The MAE-tuned exploratory model trades pooled R²
for robustness, hence its lower 0.117.) In-sample R² (0.36) exceeds out-of-sample,
the expected optimism gap at this sample size.

### Skill from baseline information alone (the actual question)

The headline 0.211 **includes `attend`** — intervention sessions attended _during the
gain window_ — which is a concurrent dose, not something known at baseline. The clean
answer to "can we forecast from **baseline** info" drops `attend` (and any other
period-related feature; here only `attend`) and re-runs the identical out-of-fold CV on
the remaining baseline-available features (`lrp01_prediction`, recomputed with the same
splits/params — the full-set value reproduces `metrics.json` exactly):

| predictor set                       | n predictors | pooled OOF **R²** | RMSE reduction vs mean |
| ----------------------------------- | -----------: | ----------------: | ---------------------: |
| full (includes concurrent `attend`) |            6 |             0.211 |                    11% |
| **baseline-only (no `attend`)**     |            5 |         **0.167** |                     9% |

**From baseline information alone, word-reading progress is weakly-to-modestly
predictable: pooled OOF R² ≈ 0.17 (~17% of individual variation), driven mainly by
(younger) age.** The ~4 pp gap to the full set is the share of predictability carried by
_concurrent attendance_ (dose / intervention timing) rather than anything knowable up
front. (For the MAE-tuned `lrp01` the same split is 0.117 → 0.106, a ~1 pp gap.)

### Calibration (`output/models/lrp01_prediction/calibration.png`)

Predicted gains are compressed into roughly 0–8 words while observed gains run from −5
to +21. The model **never forecasts the largest-gain children** and under-predicts the
top end, with wide vertical scatter at every predicted value. The observed-on-predicted
slope (0.97) is close to 1, so there is little _slope_ bias — the limitation is
**resolution** (the predictor set cannot separate who will make a large gain), not
systematic over/under-shooting.

### Which factors carry the signal (`lrp01`; see `shap_summary.png`)

| rank | predictor  | what it is                                       | perm. importance (Δ held-out RMSE) | SHAP–feature ρ | direction                  |
| ---: | ---------- | ------------------------------------------------ | ---------------------------------: | -------------: | -------------------------- |
|    1 | `attend`   | intervention sessions attended **in the window** |                              0.066 |          +0.95 | more sessions → more gain  |
|    2 | `age`      | age                                              |                              0.019 |          −0.86 | **younger → more gain**    |
|    3 | `blending` | phonological blending                            |                              0.004 |          +0.67 | higher → more gain (noisy) |
|    4 | `celf`     | receptive language                               |                             −0.003 |          +0.85 | ~0 out-of-sample value     |
|    5 | `yarclet`  | letter-sound knowledge                           |                             −0.009 |          +0.88 | ~0 out-of-sample value     |
|    6 | `b1exto`   | expressive vocabulary                            |                             −0.016 |          +0.78 | ~0 out-of-sample value     |

The SHAP directions are all sensible (younger age, and higher baseline ability on every
language/phonics measure, point to more gain). But out-of-fold **permutation importance
tells the harder truth**: only `attend` and `age` carry real held-out predictive value;
the baseline ability measures point the expected direction yet add ≈ 0 (or slightly
negative) skill beyond age once age and attendance are in the model. `lrp01_prediction`
agrees on this ordering (`attend` > `age` ≫ the rest).

## Verdict

- **From baseline information alone, word-reading progress is only weakly-to-modestly
  predictable: pooled OOF R² ≈ 0.17 (~17% of individual variation), driven mainly by
  (younger) age.** The full-feature 0.21 is higher only because it also uses `attend`,
  a concurrent dose. Either way the model says _who gains a little vs a lot on average_
  but cannot pick out the children who make the largest gains (calibration).
- **`attend` is the single strongest contributor by permutation importance, but it is a
  within-window intervention-_dose_, not baseline information** — it counts sessions
  attended _during the very transition the gain spans_ (a period-related construct,
  excluded from the level-model predictor set for this reason). Removing it costs ~4 pp
  of R² (0.211 → 0.167): a real but modest share. So the headline 0.21 is **not** a pure
  "from baseline alone" forecast; the baseline-only 0.17 is.
- **Among genuine baseline features, only younger `age` carries clear out-of-sample
  signal.** Baseline vocabulary, receptive language, letter-sound knowledge and
  blending point the expected direction (SHAP) but add little held-out skill beyond age
  (permutation importance ≈ 0 or negative). "Mostly general starting ability" is too
  generous on these data: it is **mostly younger age**, with baseline ability
  directionally right but weak.
- This is the empirical face of the handoff's concern that `ewrswr_gain` pools all
  transitions and **mixes natural development with intervention timing** — the `attend`
  contribution _is_ that mixing. Cleanly separating "response to the programme" from
  development is exactly what **Step 2** (a model restricted to the randomised phase-0
  wave-1→2 window) is designed to do; it is planned and staged for a follow-up.

## Caveats

- n is small (53 children, 157 pooled transitions); the CV spread is wide (per-fold CV
  RMSE ± ~1.9–2.4) and a single fit's importance ranking is fold-sensitive — see the
  model's `stability_selection.csv`.
- This is the exploratory ML layer: it reports _which factors associate with gain and
  how predictable gain is_, **not** causal effects. `attend` is associative here, not an
  estimated treatment effect (the ITT contrast lives in the Bayesian `statistical_models`
  layer).
- Skill is reported as pooled out-of-fold R²; the per-fold mean R² is not usable at this
  fold size.

## Reproduce

```
conda activate dse-language-reading-predictors
python scripts/fit_model.py lrp01            --config reporting
python scripts/fit_model.py lrp01_prediction --config reporting
python scripts/predictability_readout.py lrp01_prediction lrp01 --top-n 6
# → output/models/{id}/calibration.png, predictability_readout.json
# → output/models/{id}/{metrics.json, shap_summary.png, permutation_importance.csv}
```
