> [!NOTE]
> Drafted by a LLM-based AI tool (OpenAI Codex/GPT-5).

# Gradient-boosting reporting fit review

Date reviewed: 2026-07-01

Source artefacts: `output/models/{model_id}/` from the full reporting run.
I reviewed the tabular artefacts that drive the reports: `metrics.json`,
`permutation_importance.csv`, `construct_importance.csv`,
`shap_direction_diagnostics.csv`, `predictor_ranking.csv`, and
`stability_selection.csv`. All 50 gradient-boosting model directories had a
rendered `index.html` and the expected reporting diagnostics.

`output/review/gb_reporting_summary.csv` contains the consolidated extraction
used for this note.

## Introduction

This project is trying to understand which starting skills and background
features are related to later progress in language and reading for children with
Down syndrome. The gradient-boosting models are the exploratory first step. They
are useful because they can screen many predictors at once and can pick up
non-linear patterns, such as "this predictor matters more at low scores than at
high scores".

The important limitation is that these models are prediction tools. If a
predictor is important, that means it helped the model guess the outcome for
held-out data. It does not mean that changing that predictor would necessarily
change the child's progress. That causal question is what the later Bayesian
models are for.

The two broad model types answer different questions:

- Gain models ask, "Who made more progress from one timepoint to the next?"
- Level models ask, "Who had a higher score at a timepoint?"

Level models are usually easier to predict because current reading and language
scores are strongly related to other current reading and language scores. Gain
models are harder but often more relevant to the study's question about
progress.

## How to read this note

- `pooled CV R2` is the pooled out-of-fold R2 from grouped cross-validation. It
  is the main predictive-performance number here. In plain words, it asks how
  well the model predicted children or timepoints it had not already fitted. A
  value near 0 means little improvement over predicting the average. A negative
  value means worse than predicting the average. Values closer to 1 are better.
- `RMSE` and `MAE` are average prediction-error measures in the outcome's own
  units. Lower is better. They are useful for scale, but they are harder to
  compare across outcomes with different score ranges.
- `in-sample gap` is in-sample R2 minus pooled CV R2. A large gap means the model
  fits the fitted rows much better than held-out rows; it is a warning about
  optimism, especially with small samples. Another way to say this is that the
  model may have learned details of this dataset that will not repeat well.
- Permutation importance asks what happens when one predictor is shuffled. If
  shuffling a predictor damages prediction, the model was relying on it.
- SHAP direction is descriptive. `+` means larger predictor values generally
  push predictions upward; `-` means they generally push predictions downward;
  `mixed` means no simple one-way direction. SHAP helps with "which way did the
  model use this predictor?", not "what would happen if we intervened on this
  predictor?".
- Stability selection checks whether the same predictors keep appearing near
  the top when the model is refitted on resampled data. A stable predictor is
  more reassuring than a predictor that appears only once.
- These are exploratory prediction models. They rank associations useful to the
  model. They do not estimate causal effects.

## Cross-model findings

Gain models predict change only weakly to modestly. Most gain-model pooled CV R2
values sit between about 0.01 and 0.23. The best gain models were `lrpgbg13`
(`nonword_gain`, CV R2 0.297), `lrpgbg09` (`yarclet_gain`, 0.250), `lrpgbg14`
(`celf_gain`, 0.226), `lrpgbg21` (`deappvo_gain`, 0.214), and `lrpgbg01`
(`b1retau_gain`, 0.206). `lrpgbg22` (`deappav_gain`) was below zero, so its
out-of-fold predictions were worse than a baseline mean model on this metric.

The dominant gain-model pattern is an own-baseline or same-domain predictor with
a negative SHAP direction. In plain terms, children starting higher on the
relevant measure often have less predicted gain. That is compatible with ceiling
limits and regression-to-the-mean patterns; it should not be read as the skill
"causing" lower growth.

Level models are much more predictable, but this is partly because they predict
current level from other current levels and closely related composites. The very
high R2 models are mostly composite or near-composite outcomes: `lrpgbl19`
(`erbto`, CV R2 0.976), `lrpgbl23` (`deapp_c`, 0.968), and `lrpgbl22`
(`deappav`, 0.935). These results are useful for measurement structure, but they
are not evidence of developmental mechanisms by themselves.

Several models have large in-sample gaps and should be interpreted cautiously.
The largest gaps were `lrpgbl21` (`deappvo`, gap 0.660), `lrpgbl28` (`lsamto`,
0.513), `lrpgbg08` (`aptgram_gain`, 0.472), `lrpgbg14` (`celf_gain`, 0.460),
and `lrpgbg03` (`b1rent_gain`, 0.448). These may still rank useful predictors,
but the exact ordering is less secure.

Stability selection was clearer for level models than gain models. Gain models
usually had only two or three predictors appearing in the top five in at least
half the bootstraps; level models usually had around four. The least stable
headline rankings were `lrpgbg03`, `lrpgbg14`, `lrpgbg16`, and `lrpgbl21`, each
with only one predictor above the 50 percent top-five threshold.

## Summary

- The models can describe which skills travel together, but they cannot by
  themselves explain why children improved.
- Predicting current skill level is much easier than predicting change.
- Starting score is often the strongest gain-model predictor. This probably
  reflects score limits, regression to the mean, and measurement scale as much
  as development.
- Repeated signals across related outcomes are more credible than any single
  model's top predictor.
- Conclusions should be carried forward only when they also make sense in the
  Bayesian models and in the study design.

## Gain models

The gain-model rows below summarise models of change. Read these rows as
"predictors associated with more or less observed progress", not as causes of
progress.

- **LRPGBG01** (`b1retau_gain`, n=161, pooled CV R2=0.206, RMSE=2.82, MAE=2.19,
  in-sample gap=0.338). Permutation top: `b1retau`, `time`, `age`. SHAP
  top/direction: `b1retau` (-, monotonic), `rowpvt` (+, monotonic), `attend` (+,
  monotonic). Stable top-k features: `b1retau`, `age`, `rowpvt`.
- **LRPGBG02** (`b1extau_gain`, n=161, pooled CV R2=0.084, RMSE=2.96, MAE=2.26,
  in-sample gap=0.312). Permutation top: `b1extau`, `deappvo`, `blending`. SHAP
  top/direction: `b1extau` (-, monotonic), `erbword` (+, monotonic), `eowpvt` (+,
  monotonic). Stable top-k features: `b1extau`, `deappvo`.
- **LRPGBG03** (`b1rent_gain`, n=161, pooled CV R2=0.070, RMSE=1.77, MAE=1.34,
  in-sample gap=0.448). Permutation top: `b1rent`, `time`, `celf`. SHAP
  top/direction: `b1rent` (-, monotonic), `attend` (+, noisy), `rowpvt` (+,
  monotonic). Stable top-k features: `b1rent` only.
- **LRPGBG04** (`b1exnt_gain`, n=161, pooled CV R2=0.080, RMSE=1.48, MAE=1.11,
  in-sample gap=0.289). Permutation top: `b1exnt`, `attend`, `aptinfo`. SHAP
  top/direction: `b1exnt` (-, monotonic), `attend` (+, noisy), `aptinfo` (+,
  monotonic). Stable top-k features: `b1exnt`, `attend`.
- **LRPGBG05** (`rowpvt_gain`, n=161, pooled CV R2=0.072, RMSE=9.35, MAE=6.98,
  in-sample gap=0.089). Permutation top: `rowpvt`, `trog`, `celf`. SHAP
  top/direction: `rowpvt` (-, monotonic), `trog` (-, monotonic), `celf` (+,
  monotonic). Stable top-k features: `rowpvt`, `celf`, `attend`.
- **LRPGBG06** (`eowpvt_gain`, n=161, pooled CV R2=0.086, RMSE=6.69, MAE=5.13,
  in-sample gap=0.199). Permutation top: `eowpvt`, `deappvo`, `aptinfo`. SHAP
  top/direction: `eowpvt` (-, monotonic), `deappvo` (+, monotonic), `trog` (+,
  monotonic). Stable top-k features: `eowpvt`, `trog`, `deappvo`, `aptinfo`.
- **LRPGBG07** (`aptinfo_gain`, n=160, pooled CV R2=0.094, RMSE=4.24, MAE=3.34,
  in-sample gap=0.414). Permutation top: `aptinfo`, `b1reto`, `aptgram`. SHAP
  top/direction: `aptinfo` (-, monotonic), `erbword` (+, monotonic), `b1reto`
  (+, monotonic). Stable top-k features: `aptinfo`, `erbword`, `age`.
- **LRPGBG08** (`aptgram_gain`, n=158, pooled CV R2=0.078, RMSE=4.17, MAE=3.22,
  in-sample gap=0.472). Permutation top: `aptgram`, `blending`, `spphon`. SHAP
  top/direction: `aptgram` (-, monotonic), `erbword` (+, monotonic), `age` (-,
  monotonic). Stable top-k features: `aptgram`, `erbword`, `trog`, `spphon`.
- **LRPGBG09** (`yarclet_gain`, n=160, pooled CV R2=0.250, RMSE=4.43, MAE=3.28,
  in-sample gap=0.183). Permutation top: `yarclet`, `time`, `age`. SHAP
  top/direction: `yarclet` (-, monotonic), `attend` (+, monotonic), `agespeak`
  (mixed). Stable top-k features: `yarclet`, `time`.
- **LRPGBG10** (`blending_gain`, n=161, pooled CV R2=0.089, RMSE=2.05, MAE=1.55,
  in-sample gap=0.261). Permutation top: `blending`, `eowpvt`, `erbword`. SHAP
  top/direction: `blending` (-, monotonic), `eowpvt` (+, noisy), `rowpvt` (+,
  monotonic). Stable top-k features: `blending`, `eowpvt`, `rowpvt`.
- **LRPGBG11** (`spphon_gain`, n=159, pooled CV R2=0.011, RMSE=16.12, MAE=9.59,
  in-sample gap=0.160). Permutation top: `aptgram`, `attend`, `mumedupost16`.
  SHAP top/direction: `ewrswr` (+, monotonic), `attend` (+, monotonic),
  `mumedupost16` (+, monotonic). Stable top-k features: `ewrswr`, `aptgram`.
- **LRPGBG12** (`ewrswr_gain`, n=157, pooled CV R2=0.093, RMSE=4.14, MAE=3.07,
  in-sample gap=0.367). Permutation top: `attend`, `age`, `time`. SHAP
  top/direction: `attend` (+, monotonic), `age` (-, monotonic), `eowpvt` (+,
  monotonic). Stable top-k features: `attend`, `age`, `yarclet`.
- **LRPGBG13** (`nonword_gain`, n=153, pooled CV R2=0.297, RMSE=1.39, MAE=0.94,
  in-sample gap=0.173). Permutation top: `nonword`, `erbnw`, `spphon`. SHAP
  top/direction: `nonword` (-, monotonic), `erbnw` (+, monotonic), `spphon` (+,
  monotonic). Stable top-k features: `nonword`, `erbnw`, `ewrswr`.
- **LRPGBG14** (`celf_gain`, n=160, pooled CV R2=0.226, RMSE=2.82, MAE=2.16,
  in-sample gap=0.460). Permutation top: `celf`, `nonword`, `spphon`. SHAP
  top/direction: `celf` (-, monotonic), `spphon` (+, monotonic), `yarclet` (+,
  noisy). Stable top-k features: `celf` only.
- **LRPGBG15** (`trog_gain`, n=161, pooled CV R2=0.133, RMSE=3.90, MAE=3.15,
  in-sample gap=0.277). Permutation top: `trog`, `celf`, `eowpvt`. SHAP
  top/direction: `trog` (-, monotonic), `eowpvt` (+, monotonic), `celf` (+,
  monotonic). Stable top-k features: `trog`, `celf`, `eowpvt`.
- **LRPGBG16** (`deappfi_gain`, n=152, pooled CV R2=0.101, RMSE=12.57, MAE=8.57,
  in-sample gap=0.137). Permutation top: `deappfi`, `attend`, `ewrswr`. SHAP
  top/direction: `deappfi` (-, monotonic), `ewrswr` (-, monotonic), `agebooks`
  (+, monotonic). Stable top-k features: `deappfi` only.
- **LRPGBG17** (`erbnw_gain`, n=147, pooled CV R2=0.105, RMSE=2.98, MAE=2.37,
  in-sample gap=0.445). Permutation top: `erbnw`, `aptinfo`, `attend`. SHAP
  top/direction: `erbnw` (-, monotonic), `erbword` (+, monotonic), `earinf` (-,
  monotonic). Stable top-k features: `erbnw`, `yarclet`.
- **LRPGBG18** (`erbword_gain`, n=148, pooled CV R2=0.155, RMSE=2.76, MAE=1.89,
  in-sample gap=0.237). Permutation top: `erbword`, `b1reto`, `deappfi`. SHAP
  top/direction: `erbword` (-, monotonic), `deappfi` (+, monotonic), `b1reto`
  (-, noisy). Stable top-k features: `erbword`, `deappfi`.
- **LRPGBG19** (`erbto_gain`, n=147, pooled CV R2=0.112, RMSE=4.75, MAE=3.60,
  in-sample gap=0.314). Permutation top: `erbnw`, `erbto`, `age`. SHAP
  top/direction: `erbnw` (-, monotonic), `erbto` (-, monotonic), `deappfi` (+,
  monotonic). Stable top-k features: `erbnw`, `erbto`.
- **LRPGBG20** (`deappin_gain`, n=152, pooled CV R2=0.034, RMSE=6.37, MAE=4.93,
  in-sample gap=0.271). Permutation top: `deappin`, `time`, `b1reto`. SHAP
  top/direction: `deappin` (-, monotonic), `time` (+, noisy), `blending` (+,
  monotonic). Stable top-k features: `deappin`, `b1reto`, `time`.
- **LRPGBG21** (`deappvo_gain`, n=152, pooled CV R2=0.214, RMSE=5.09, MAE=3.20,
  in-sample gap=0.197). Permutation top: `deappvo`, `time`, `b1exto`. SHAP
  top/direction: `deappvo` (-, monotonic), `deappfi` (+, monotonic), `time` (+,
  noisy). Stable top-k features: `deappvo`, `deappfi`, `time`.
- **LRPGBG22** (`deappav_gain`, n=152, pooled CV R2=-0.054, RMSE=4.79, MAE=3.76,
  in-sample gap=0.438). Permutation top: `deappav`, `deappvo`, `eowpvt`. SHAP
  top/direction: `deappav` (-, monotonic), `deappvo` (-, mixed), `deappfi` (-,
  noisy). Stable top-k features: `deappfi`, `deappav`.

## Level models

The level-model rows below summarise models of attained score level. High
prediction accuracy here often means that closely related measures overlap
strongly. That is valuable for understanding the measurement structure, but it
is not the same as explaining progress.

- **LRPGBL01** (`b1retau`, n=215, pooled CV R2=0.631, RMSE=2.58, MAE=2.00,
  in-sample gap=0.215). Permutation top: `b1exto`, `trog`, `celf`. SHAP
  top/direction: `b1exto` (+, monotonic), `trog` (+, monotonic), `celf` (+,
  monotonic). Stable top-k features: `trog`, `celf`, `b1exto`, `aptinfo`,
  `yarclet`, `eowpvt`.
- **LRPGBL02** (`b1extau`, n=215, pooled CV R2=0.737, RMSE=2.45, MAE=1.95,
  in-sample gap=0.165). Permutation top: `aptinfo`, `eowpvt`, `b1reto`. SHAP
  top/direction: `aptinfo` (+, monotonic), `eowpvt` (+, monotonic), `b1reto` (+,
  monotonic). Stable top-k features: `aptinfo`, `eowpvt`, `b1reto`, `aptgram`.
- **LRPGBL03** (`b1rent`, n=215, pooled CV R2=0.451, RMSE=1.57, MAE=1.27,
  in-sample gap=0.339). Permutation top: `aptinfo`, `erbword`, `blending`. SHAP
  top/direction: `aptinfo` (+, monotonic), `b1exto` (+, monotonic), `rowpvt` (+,
  monotonic). Stable top-k features: `aptinfo`, `b1exto`, `trog`, `rowpvt`,
  `eowpvt`.
- **LRPGBL04** (`b1exnt`, n=215, pooled CV R2=0.588, RMSE=1.59, MAE=1.25,
  in-sample gap=0.248). Permutation top: `eowpvt`, `aptinfo`, `spphon`. SHAP
  top/direction: `eowpvt` (+, monotonic), `aptinfo` (+, monotonic), `spphon`
  (+, monotonic). Stable top-k features: `eowpvt`, `aptinfo`, `celf`, `rowpvt`.
- **LRPGBL05** (`rowpvt`, n=215, pooled CV R2=0.613, RMSE=8.84, MAE=7.12,
  in-sample gap=0.172). Permutation top: `aptinfo`, `eowpvt`, `b1reto`. SHAP
  top/direction: `b1reto` (+, monotonic), `aptinfo` (+, monotonic), `eowpvt` (+,
  monotonic). Stable top-k features: `aptinfo`, `b1reto`, `eowpvt`, `celf`,
  `b1exto`, `trog`.
- **LRPGBL06** (`eowpvt`, n=215, pooled CV R2=0.696, RMSE=7.83, MAE=6.12,
  in-sample gap=0.217). Permutation top: `b1exto`, `aptinfo`, `celf`. SHAP
  top/direction: `b1exto` (+, monotonic), `aptinfo` (+, monotonic), `rowpvt`
  (+, monotonic). Stable top-k features: `b1exto`, `aptinfo`, `rowpvt`, `celf`,
  `ewrswr`.
- **LRPGBL07** (`aptinfo`, n=214, pooled CV R2=0.782, RMSE=3.75, MAE=2.91,
  in-sample gap=0.195). Permutation top: `aptgram`, `b1exto`, `eowpvt`. SHAP
  top/direction: `aptgram` (+, monotonic), `b1exto` (+, monotonic), `eowpvt`
  (+, monotonic). Stable top-k features: `b1exto`, `aptgram`, `eowpvt`,
  `b1reto`, `rowpvt`.
- **LRPGBL08** (`aptgram`, n=211, pooled CV R2=0.656, RMSE=3.76, MAE=2.68,
  in-sample gap=0.233). Permutation top: `aptinfo`, `nonword`, `yarcsi`. SHAP
  top/direction: `aptinfo` (+, monotonic), `erbword` (+, monotonic), `erbnw`
  (+, monotonic). Stable top-k features: `aptinfo`, `erbword`, `erbnw`,
  `deappin`.
- **LRPGBL09** (`yarclet`, n=214, pooled CV R2=0.541, RMSE=6.03, MAE=4.85,
  in-sample gap=0.234). Permutation top: `ewrswr`, `b1exto`, `time`. SHAP
  top/direction: `ewrswr` (+, monotonic), `b1exto` (+, monotonic), `eowpvt`
  (+, monotonic). Stable top-k features: `ewrswr`, `b1exto`, `eowpvt`,
  `b1reto`.
- **LRPGBL10** (`blending`, n=215, pooled CV R2=0.301, RMSE=2.15, MAE=1.77,
  in-sample gap=0.074). Permutation top: `b1reto`, `ewrswr`, `spphon`. SHAP
  top/direction: `eowpvt` (+, monotonic), `ewrswr` (+, monotonic), `spphon` (+,
  monotonic). Stable top-k features: `ewrswr`, `b1reto`, `yarclet`, `aptinfo`,
  `spphon`.
- **LRPGBL11** (`spphon`, n=214, pooled CV R2=0.677, RMSE=16.45, MAE=10.35,
  in-sample gap=0.166). Permutation top: `yarcsi`, `ewrswr`, `age`. SHAP
  top/direction: `ewrswr` (+, monotonic), `yarcsi` (+, monotonic), `age` (+,
  monotonic). Stable top-k features: `ewrswr`, `yarcsi`, `age`.
- **LRPGBL12** (`ewrswr`, n=210, pooled CV R2=0.552, RMSE=9.73, MAE=6.35,
  in-sample gap=0.167). Permutation top: `spphon`, `yarclet`, `b1exto`. SHAP
  top/direction: `yarclet` (+, monotonic), `spphon` (+, monotonic), `yarcsi`
  (+, monotonic). Stable top-k features: `spphon`, `yarclet`, `b1exto`,
  `nonword`.
- **LRPGBL13** (`nonword`, n=207, pooled CV R2=0.456, RMSE=1.35, MAE=0.87,
  in-sample gap=0.242). Permutation top: `ewrswr`, `yarclet`, `aptinfo`. SHAP
  top/direction: `spphon` (+, monotonic), `ewrswr` (+, monotonic), `yarclet` (+,
  monotonic). Stable top-k features: `ewrswr`, `yarclet`, `spphon`.
- **LRPGBL14** (`celf`, n=214, pooled CV R2=0.462, RMSE=3.14, MAE=2.57,
  in-sample gap=0.200). Permutation top: `eowpvt`, `b1reto`, `aptinfo`. SHAP
  top/direction: `eowpvt` (+, monotonic), `b1reto` (+, noisy), `rowpvt` (+,
  monotonic). Stable top-k features: `eowpvt`, `b1reto`, `rowpvt`, `b1exto`.
- **LRPGBL15** (`trog`, n=215, pooled CV R2=0.476, RMSE=3.53, MAE=2.89,
  in-sample gap=0.193). Permutation top: `b1reto`, `b1exto`, `rowpvt`. SHAP
  top/direction: `b1exto` (+, monotonic), `b1reto` (+, monotonic), `rowpvt` (+,
  monotonic). Stable top-k features: `b1reto`, `b1exto`, `rowpvt`.
- **LRPGBL16** (`deappfi`, n=207, pooled CV R2=0.549, RMSE=14.25, MAE=10.07,
  in-sample gap=0.292). Permutation top: `deappin`, `ewrswr`, `time`. SHAP
  top/direction: `deappin` (+, monotonic), `erbword` (+, monotonic), `erbnw`
  (+, monotonic). Stable top-k features: `deappin`, `erbword`, `erbnw`.
- **LRPGBL17** (`erbnw`, n=202, pooled CV R2=0.738, RMSE=2.46, MAE=1.92,
  in-sample gap=0.127). Permutation top: `erbword`, `aptinfo`, `yarcsi`. SHAP
  top/direction: `erbword` (+, monotonic), `deappin` (+, monotonic), `aptinfo`
  (+, monotonic). Stable top-k features: `erbword`, `deappin`, `deappfi`,
  `aptinfo`, `yarcsi`.
- **LRPGBL18** (`erbword`, n=203, pooled CV R2=0.752, RMSE=2.64, MAE=2.03,
  in-sample gap=0.180). Permutation top: `erbnw`, `rowpvt`, `spphon`. SHAP
  top/direction: `erbnw` (+, monotonic), `deappfi` (+, monotonic), `deappin`
  (+, monotonic). Stable top-k features: `erbnw`, `deappin`, `deappfi`,
  `aptgram`.
- **LRPGBL19** (`erbto`, n=202, pooled CV R2=0.976, RMSE=1.48, MAE=0.53,
  in-sample gap=0.003). Permutation top: `erbnw`, `erbword`, `nonword`. SHAP
  top/direction: `erbword` (+, monotonic), `erbnw` (+, monotonic), `nonword`
  (+, monotonic). Stable top-k features: `erbnw`, `erbword`, `deappin`,
  `aptgram`.
- **LRPGBL20** (`deappin`, n=207, pooled CV R2=0.575, RMSE=8.39, MAE=6.31,
  in-sample gap=0.254). Permutation top: `deappvo`, `deappfi`, `erbnw`. SHAP
  top/direction: `deappfi` (+, monotonic), `erbword` (+, monotonic), `deappvo`
  (+, monotonic). Stable top-k features: `deappfi`, `deappvo`, `erbword`,
  `erbnw`.
- **LRPGBL21** (`deappvo`, n=207, pooled CV R2=0.195, RMSE=5.18, MAE=3.54,
  in-sample gap=0.660). Permutation top: `deappin`, `time`, `nonword`. SHAP
  top/direction: `deappin` (+, monotonic), `time` (+, noisy), `erbword` (+,
  monotonic). Stable top-k features: `deappin` only.
- **LRPGBL22** (`deappav`, n=207, pooled CV R2=0.935, RMSE=2.93, MAE=1.63,
  in-sample gap=0.044). Permutation top: `deappfi`, `deappin`, `deappvo`. SHAP
  top/direction: `deappfi` (+, monotonic), `deappin` (+, monotonic), `deappvo`
  (+, monotonic). Stable top-k features: `deappfi`, `deappin`, `deappvo`,
  `erbword`.
- **LRPGBL23** (`deapp_c`, n=207, pooled CV R2=0.968, RMSE=6.25, MAE=3.43,
  in-sample gap=0.020). Permutation top: `deappfi`, `deappin`, `deappvo`. SHAP
  top/direction: `deappfi` (+, monotonic), `deappin` (+, monotonic), `deappvo`
  (+, monotonic). Stable top-k features: `deappvo`, `deappin`, `deappfi`,
  `erbword`, `erbnw`.
- **LRPGBL24** (`lsammlu`, n=106, pooled CV R2=0.435, RMSE=0.55, MAE=0.44,
  in-sample gap=0.319). Permutation top: `erbword`, `aptinfo`, `deappin`. SHAP
  top/direction: `erbword` (+, monotonic), `aptinfo` (+, monotonic), `deappin`
  (+, monotonic). Stable top-k features: `deappin`, `erbword`, `aptinfo`,
  `erbnw`.
- **LRPGBL25** (`lsammax`, n=106, pooled CV R2=0.281, RMSE=1.86, MAE=1.35,
  in-sample gap=0.306). Permutation top: `deappin`, `erbword`, `erbnw`. SHAP
  top/direction: `deappin` (+, monotonic), `erbword` (+, monotonic), `b1reto`
  (+, monotonic). Stable top-k features: `deappin`, `erbword`.
- **LRPGBL26** (`lsamint`, n=106, pooled CV R2=0.353, RMSE=15.46, MAE=11.34,
  in-sample gap=0.379). Permutation top: `time`, `area`, `deappin`. SHAP
  top/direction: `deappin` (+, monotonic), `area` (+, monotonic), `aptinfo` (+,
  monotonic). Stable top-k features: `area`, `deappin`, `erbword`.
- **LRPGBL27** (`lsamun`, n=106, pooled CV R2=0.476, RMSE=15.19, MAE=11.62,
  in-sample gap=0.372). Permutation top: `erbnw`, `aptinfo`, `trog`. SHAP
  top/direction: `deappin` (+, monotonic), `aptinfo` (+, monotonic), `trog` (+,
  monotonic). Stable top-k features: `deappin`, `aptinfo`, `erbword`, `rowpvt`.
- **LRPGBL28** (`lsamto`, n=106, pooled CV R2=0.289, RMSE=44.32, MAE=32.85,
  in-sample gap=0.513). Permutation top: `deappvo`, `deappin`, `trog`. SHAP
  top/direction: `deappin` (+, monotonic), `erbword` (+, monotonic), `trog` (+,
  monotonic). Stable top-k features: `deappin`, `trog`, `aptinfo`.

## Follow-up checks

- Treat `lrpgbg22`, `lrpgbg11`, and `lrpgbg20` as low predictive-signal gain
  models unless another metric or scientific reason justifies keeping them in
  the main narrative.
- For gain models where own baseline dominates with a negative SHAP direction,
  report this as "starting higher predicted less gain" and explicitly mention
  ceiling/regression-to-the-mean as plausible explanations.
- For very high level-model R2 values, especially composite outcomes, state the
  likely measurement-overlap explanation before interpreting the ranked
  predictors.
- Before making substantive claims, compare these rankings with the Bayesian
  model results and prioritise conclusions that agree in direction and
  uncertainty.
