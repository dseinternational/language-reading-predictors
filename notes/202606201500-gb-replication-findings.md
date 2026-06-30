# Gradient-boosting models — independent replication & critical review

**Date:** 2026-06-20
**Scope:** All 22 primary LRP gradient-boosting models (LRP01–LRP22): independent
replication of **tuning**, **training**, and **feature selection**, with a
critical eye on correlated predictors. Recorded notes were deliberately _not_
read first and recorded findings in source were treated as unverified.

> Working artefacts (gitignored): driver `output/replication/replicate.py`,
> analysis script `output/replication/analyze.py`, per-model evidence
> `output/replication/findings/<id>.facts.md`, cross-model table
> `output/replication/findings/SUMMARY.md`. Re-fitted models + diagnostics in
> `output/replication/models/<id>{,_full}/`; re-tunes in `output/tuning/`.

---

## 1. Executive summary

The replication **runs cleanly and the recorded numbers reproduce** at the level
that matters (CV error), but the exercise surfaces several issues that change how
these models should be read. In order of importance:

1. **Gain (change-score) models carry almost no signal.** Pooled out-of-fold R²
   for the 11 gain models is 0.07–0.25 (median ≈ 0.14); for the 11 level models
   it is 0.36–0.80 (median ≈ 0.61). Importance rankings and feature-selection
   decisions for the gain models are therefore built on models barely better than
   predicting the mean.

2. **"Predictors of progress" are, empirically, mostly "baseline level of the
   same skill."** In 9 of 11 gain models the single dominant predictor — usually
   the _only_ one above the noise floor — is the baseline value of the very
   measure whose gain is being predicted (e.g. `rowpvt` for `rowpvt_gain`,
   importance 1.28). This is mechanical regression-to-the-mean (gain = post − pre,
   and pre is in the predictor set), not a substantive predictor of change.

3. **Level models' high R² is concurrent-measurement tautology, not
   developmental prediction.** The top predictor of a level model is repeatedly a
   co-administered, same-construct test — twice a _subtest of the same
   instrument_: `aptinfo`↔`aptgram` (Action Picture Test; LRP18/LRP20) and
   `deappin`↔`deappfi` (DEAP; LRP22). These "predict" a score largely from a
   sibling score measured in the same session.

4. **Permutation importance is statistically fragile here.** In **every one of
   the 22 models**, _no_ predictor has an out-of-fold permutation importance that
   is more than 1 SD above zero (the per-fold spread dwarfs the mean). Rankings
   below the top one or two features are essentially noise, and the 3-decimal
   importance values quoted in the recorded selection notes over-state the
   precision available.

5. **Correlated-predictor redundancy is pervasive and largely un-addressed in
   LRP11–22.** A recurring cluster of mutually correlated language/vocabulary
   tests (`b1exto`,`eowpvt`,`aptinfo`,`b1reto`,`rowpvt`,`aptgram`,`celf`; pairwise
   distance-correlation 0.6–0.81) plus speech pairs (`erbnw`↔`erbword` 0.84,
   `deappin`↔`deappfi` 0.76) and reading pairs (`spphon`↔`ewrswr` 0.78–0.80)
   appears in almost every model. The 12 models that have had **no feature
   selection** (LRP11–22) each retain ~12 pairs at dcor ≥ 0.7 and ~25 more at
   0.6–0.7, alongside 18–28 predictors with ≤0 importance.

6. **Tuning does not reproduce at the parameter level — only the objective
   does.** Re-tuning each model (Optuna 150 trials, same seed/regime) lands within
   ~1–3 % of the recorded CV MAE but with wildly different hyperparameters
   (`n_estimators` swings 2–19×, e.g. LRP01 118→19, LRP20 445→258). The objective
   surface is flat/multimodal at this sample size; the specific recorded
   hyperparameters are not a meaningful, identifiable object.

7. **Where feature selection was actually done (LRP01–10), it mostly did not
   discard real signal** (≈0 "questionable drops"). The drops that look alarming
   under permutation importance are either documented construct-driven choices
   (LRP04 dropping `b1exto`, LRP10 dropping `eowpvt`/`b1reto`) or redundancy-
   justified — and the refit CV confirms they cost little. The bigger problems are
   _retained_ noise/redundancy (LRP08, LRP09) and the structural points above.

**Bottom line:** the pipeline is sound and the recorded models reproduce, but the
gradient-boosting layer is best read as _exploratory and largely confirmatory of
regression-to-the-mean plus concurrent test correlations_. It does not, on this
evidence, identify robust non-baseline predictors of progress. Treat gain-model
importance rankings with great caution, treat level-model R² as concurrent
correlation rather than prediction, and finish (and tighten) feature selection on
LRP11–22 before any of those rankings are quoted.

---

## 2. Method (what was replicated, and how)

- **Environment.** conda env `dse-language-reading-predictors` (Python 3.14.6) at
  `V:\miniconda3\envs\`. Must be run through `conda run -n …` (or activation):
  running the env's `python.exe` directly leaves `…\Library\bin` off `PATH` and
  matplotlib/numpy fault natively with Windows delay-load error `0xC06D007F`.
- **Tooling.** The project's own `scripts/tune_model.py` and
  `EstimatorPipeline.fit()` (reporting config) were used unmodified — this is a
  faithful replication, not a reimplementation.
- **Per model:** (1) re-tune the registered predictor set (Optuna 150 trials,
  `scoring=mae`, `objective=mae`, 10-fold `GroupKFold`, seed 47); (2) reporting
  fit of the registered set with the **recorded** hyperparameters (honest
  reproduction) → CV + OOF permutation importance + construct importance +
  distance-correlation clustering + stability selection + SHAP direction; (3) for
  the **pruned** models (LRP01–10), additionally tune + fit the **full default
  predictor set** (`*_full`, no selection steps) to judge the pruning from
  scratch. All 32 fits + 32 tunes completed without error.
- **Metrics convention used below.** Pooled out-of-fold **R²** (SS aggregated
  across folds vs each fold's _training_ mean) is the honest generalisation
  metric and the one quoted here. **Ignore the per-fold "R²(fold) mean"** in the
  artefacts: with `cv_splits` ≈ 51–53 over ~150–215 rows the folds hold ~1
  subject (3–4 rows) each — effectively leave-one-subject-out — so per-fold R² is
  wildly negative and uninformative (e.g. LRP02 −17, LRP21 −22). "z" below =
  importance mean ÷ its across-fold SD; z<1 means "not robustly above zero".

### Method caveats discovered (apply to the pipeline generally)

- **`metrics.json` headline `cv_r2_mean` is the noisy per-fold average**, not the
  pooled value — easy to misread as "the model is broken". Prefer `cv_pooled_r2`.
- **`stability_selection.csv` uses a different importance metric** (scikit-learn's
  default R² scorer, in-bag on each bootstrap) than
  `permutation_importance.csv` (OOF, neg-RMSE). They frequently _disagree_ — e.g.
  LRP22 `erbword` has OOF importance −0.106 yet appears in the top-5 in 100 % of
  bootstraps. The OOF neg-RMSE table is the trustworthy one for held-out value;
  the stability table should not be read as corroborating it.
- **Permutation importance over-states the necessity of redundant features.**
  LRP04: `b1exto` is rank-1 in the full set (imp 0.562) yet removing it and
  refitting barely moves pooled R² (0.71→0.70) because `aptinfo`/`celf`/`rowpvt`
  carry the same information. Importance-on-a-fixed-model ≠ leave-one-out refit
  value; selection should lean on the refit CV (as the project's CV deltas do),
  not the importance magnitude.

---

## 3. Cross-model patterns (the substantive findings)

### 3.1 Predictability: gains ≈ noise, levels ≈ concurrent correlation

|                   | pooled R² range | median |
| ----------------- | --------------- | ------ |
| Gain models (11)  | 0.07–0.25       | ~0.14  |
| Level models (11) | 0.36–0.80       | ~0.61  |

The gain/level split is the dominant signal in the whole suite. It is exactly
what one expects: a difference score removes the stable between-child variance
that makes concurrent levels predictable, leaving mostly measurement noise.

### 3.2 Baseline level dominates every gain model (regression to the mean)

Top predictor (OOF importance) of each gain model = its own baseline measure:
`eowpvt`(LRP03 .40), `yarclet`(LRP05 .66), `rowpvt`(LRP07 1.28), `celf`(LRP09 .40),
`trog`(LRP11 .45), `nonword`(LRP13 .28), `blending`(LRP15 .20), `aptgram`(LRP17 .35),
`aptinfo`(LRP19 .46), `deappfi`(LRP21 .82). In most, it is the _only_ predictor
above the 0.005 noise floor. LRP01 is the lone exception — its recorded selection
_dropped_ the baseline `ewrswr`, leaving `attend` as a (barely-positive) top
feature. The practical reading: once you know where a child started, the other
measures add little to predicting how much they change.

### 3.3 Level-model R² is concurrent same-construct/same-instrument correlation

Strongest cases (top predictor → target, both measured the same session):

- **Same instrument:** LRP20 `aptgram`→`aptinfo` (R² 0.80); LRP18 `aptinfo`→`aptgram`
  (0.68); LRP22 `deappin`→`deappfi` (0.55).
- **Same domain:** LRP06 `ewrswr`→`yarclet` (0.64); LRP02 `spphon`→`ewrswr` (0.61);
  LRP04 `aptinfo`/`b1exto`→`eowpvt` (0.70); LRP08 `aptinfo`/`eowpvt`→`rowpvt` (0.64).
  These are not predictions of progress; they are "test X correlates with sibling
  test Y." Useful as data-quality / convergent-validity checks, misleading if read
  as identifying influences on attainment.

### 3.4 Permutation importance is noise-dominated

Across all 22 models, **0 predictors** reach z ≥ 1. The top feature typically has
z ≈ 0.4–0.7 (directionally real), but everything below is statistically
indistinguishable from zero. Counts of retained predictors with ≤0 OOF importance:
0 (LRP04) up to 28 (LRP13); the un-pruned LRP11–22 average ~20. The honest
statement these models support is "feature A is the dominant signal; the rest is
noise at this n," not a ranked list of secondary predictors.

### 3.5 Correlated predictors

A persistent redundancy structure recurs everywhere:

- **Language/vocabulary cluster:** `b1exto`–`eowpvt` 0.81, `aptinfo`–`b1exto` 0.77–0.81,
  `aptinfo`–`eowpvt` 0.75–0.77, `b1exto`–`b1reto` 0.74–0.76, `aptgram`–`aptinfo` 0.72–0.76,
  `eowpvt`–`rowpvt` 0.71–0.72.
- **Speech:** `erbnw`–`erbword` 0.84; `deappin`–`deappfi` 0.76.
- **Reading:** `spphon`–`ewrswr` 0.78–0.80; `spphon`–`yarcsi` 0.75–0.77.

Where careful FS was applied the redundancy is gone (LRP01/02/04/06/10: 0–1 pairs
≥0.7). Where FS was light or absent it dominates: **LRP08** (17 predictors, 8 pairs
≥0.7 — under-pruned, kept the whole language cluster) and **all of LRP11–22**
(~12 pairs ≥0.7 each). This is the clearest actionable gap.

### 3.6 Time-invariant baselines retained as clutter

The 8 within-child-constant baselines (`agebooks`,`agespeak`,`dadedupost16`,
`earinf`,`hearing`,`mumedupost16`,`numchil`,`vision`) are retained in all 12
un-pruned models; in every case they sit at/below the noise floor. Harmless to
predictions but they inflate the predictor count and (for level models) carry a
mild leakage/inflation risk flagged in `Variables.TIME_INVARIANT_BASELINES`.

### 3.7 Tuning reproduces in objective, not in parameters

Re-tuned CV MAE ≈ recorded (within ~1–3 %) for all models, but hyperparameters
diverge sharply (`n_estimators` examples: LRP01 118→19, LRP02 53→271, LRP06
277→45, LRP17 52→360, LRP20 445→258, LRP22 385→90). Flat objective surface + small
n + variable recorded trial budgets (50 vs 150). **Consequence:** do not interpret
specific hyperparameters; they are interchangeable among near-equivalent optima.
For exploratory importance work this is fine, but it means "the tuned model" is
not unique, which compounds the importance-instability in 3.4.

---

## 4. Per-model findings

Format: pooled R² (registered) · headline · specific issues. "≤0" = retained
predictors with non-positive OOF importance.

### Outcome 1 — word reading (EWRSWR)

- **LRP01 `ewrswr_gain` — R² 0.12 (weak).** Only `attend` clears the floor
  (0.066); `celf`,`yarclet`,`b1exto` ≤0. Selection (uniquely) dropped the baseline
  `ewrswr`; full-34 set is no better (R² 0.09). Pruning kept the top-4 full-set
  features — defensible, but the model is near-noise. Retained `b1exto`↔`yarclet`
  0.65, `b1exto`↔`celf` 0.64.
- **LRP02 `ewrswr` — R² 0.61 (strong).** Top `spphon` 0.123, `eowpvt`, `yarclet`.
  But 5/13 ≤0, including `aptinfo` (rank 13, and redundant with `eowpvt` dcor 0.77 — a
  questionable _keep_). Pruning _helped_ (full-32 R² 0.51). Dropped `b1exto`/`yarcsi`
  rank 3–4 in the full set, but on redundancy grounds (defensible). 4 time-invariant
  retained.

### Outcome 2 — expressive vocabulary (EOWPVT)

- **LRP03 `eowpvt_gain` — R² 0.14 (weak).** Baseline `eowpvt` dominates (0.40),
  then `deappvo` 0.21, `trog` 0.10 (both z≈0.2–0.3). Heavy retained redundancy (7
  pairs ≥0.7, including `b1exto`↔`eowpvt` 0.81). Carries `b1exto` (rank 8, ≈0 imp,
  redundant with `eowpvt`) — a questionable keep. Same-construct retained:
  `b1exto`,`eowpvt`.
- **LRP04 `eowpvt` — R² 0.70 (strong).** Cleanest pruned model (0 ≤0, 0 pairs ≥0.7).
  Recorded deliberately dropped `b1exto` (another expressive-vocab test) for
  interpretability; full-set confirms `b1exto` is rank-1 (0.562) but removing it
  costs ~nothing (R² 0.71→0.70) — the documented choice is sound and a good
  example of refit-CV beating raw importance. Top retained `aptinfo` 0.71 is itself
  language-adjacent.

### Outcome 3 — letter-sound knowledge (YARCLET)

- **LRP05 `yarclet_gain` — R² 0.21 (moderate).** Baseline `yarclet` 0.66; rest
  weak (z<0.2). 5/15 ≤0 (including `deappvo` worst, kept). Retained `deappin`↔`deappfi`
  0.76 etc. Full-set no better. Two mild "questionable drops" (`spphon`,`erbnw`),
  both low-importance.
- **LRP06 `yarclet` — R² 0.64 (strong).** Top `ewrswr` 0.875 (word reading →
  letter sounds, concurrent), `b1exto` 0.33, `time` 0.17. Pruning helped (full
  0.58). 3/10 ≤0. `nonword` (same construct) retained.

### Outcome 4 — receptive vocabulary (ROWPVT)

- **LRP07 `rowpvt_gain` — R² 0.19 (weak).** Baseline `rowpvt` 1.28 is essentially
  the whole model; everything else ≤0.07 or negative (5/12 ≤0). Pruning helped a
  lot (full-34 R² 0.06). Retained `deappin`↔`deappfi` 0.76, `b1exto`↔`b1reto` 0.74.
- **LRP08 `rowpvt` — R² 0.64 (strong) — most under-pruned model.** 17 predictors,
  **8 pairs ≥0.7 + 20 at 0.6–0.7**, 6/17 ≤0. Kept the entire mutually-correlated
  language cluster (`aptinfo`,`eowpvt`,`b1reto`,`b1exto`,`celf`,`trog`). Full-32 R²
  0.62 ≈ registered 0.64 → could be pruned much further (to ~4–6 features) with
  negligible loss. Prime candidate for redundancy-driven re-selection.

### Outcome 5 — concept knowledge (CELF)

- **LRP09 `celf_gain` — R² 0.25 (moderate) but baseline-only in substance.**
  `celf` baseline 0.40; **10/17 ≤0, 14/17 below floor** — 16 of 17 predictors are
  noise. 5 pairs ≥0.7. Effectively `celf_gain ~ celf` dressed with noise; the
  17-feature set is far larger than the evidence supports.
- **LRP10 `celf` — R² 0.49 (strong).** Construct-driven: dropped top-2 vocab
  (`eowpvt` rank-1, `b1reto` rank-3 in full set) deliberately; R² barely changed
  (full 0.47) → redundancy again. Retains `trog` at the _worst_ importance
  (−0.052, rank 10/10) as a "language control" — actively unhelpful; reasonable to
  drop. 4 time-invariant retained.

### Outcome 6 — receptive grammar (TROG) — _no FS done_

- **LRP11 `trog_gain` — R² 0.18 (weak).** Baseline `trog` 0.45; `celf` 0.055,
  `eowpvt` 0.025; **22/34 ≤0.** 12 pairs ≥0.7, 8 time-invariant. Baseline-plus-noise.
- **LRP12 `trog` — R² 0.46 (strong).** Top `b1exto`/`b1reto` (vocab→grammar,
  correlated); 19/32 ≤0; 12 pairs ≥0.7. `rowpvt` shows the stability-vs-OOF
  contradiction (OOF −0.013 but top-5 in 90 % of bootstraps). Needs FS.

### Outcome 7 — nonword reading (NONWORD) — _no FS done_

- **LRP13 `nonword_gain` — R² 0.24 (moderate).** Baseline `nonword` 0.28; **28/34
  ≤0** (the most noise of any model). 13 pairs ≥0.7. Target is ~48 % zeros — a
  near-degenerate change score; importance beyond the baseline is meaningless.
- **LRP14 `nonword` — R² 0.38 (moderate).** Top `ewrswr`/`spphon` (reading,
  concurrent); 22/32 ≤0; 14 pairs ≥0.7. Target 57 % floor — consider a
  hurdle/zero-inflated or quantile treatment (already flagged in source).

### Outcome 8 — phoneme blending (BLENDING) — _no FS done_

- **LRP15 `blending_gain` — R² 0.13 (weak).** Baseline `blending` 0.20; rest ≤0.03
  (18/34 ≤0). 12 pairs ≥0.7.
- **LRP16 `blending` — R² 0.36 (moderate).** Weakest level model; top `b1reto`
  0.018, `ewrswr` 0.016 — even the leaders are at the floor (coarse 0–10 scale caps
  R²). 13/32 ≤0; 12 pairs ≥0.7.

### Outcome 9 — expressive grammar (APTGRAM) — _no FS done_

- **LRP17 `aptgram_gain` — R² 0.07 (weak; lowest in suite).** Baseline `aptgram`
  0.35; **23/34 ≤0.** Essentially no signal beyond baseline.
- **LRP18 `aptgram` — R² 0.68 (strong).** Top `aptinfo` 0.535 — **same instrument
  (APT) as the target.** 18/32 ≤0; 12 pairs ≥0.7. High R² is within-instrument
  correlation.

### Outcome 10 — expressive information (APTINFO) — _no FS done_

- **LRP19 `aptinfo_gain` — R² 0.09 (weak).** Baseline `aptinfo` 0.46; `b1reto`
  0.07; 20/34 ≤0. 12 pairs ≥0.7.
- **LRP20 `aptinfo` — R² 0.80 (strongest in suite).** Top `aptgram` 0.62 — again
  the **sibling APT subtest** — then `b1exto`,`eowpvt`,`rowpvt`,`b1reto` (vocab
  cluster). 11/32 ≤0; 8 pairs ≥0.7. The 0.80 is concurrent same-instrument +
  vocab correlation; not developmental.

### Outcome 11 — fine articulation (DEAPPFI) — _no FS done_

- **LRP21 `deappfi_gain` — R² 0.095 (weak).** Baseline `deappfi` 0.82 only;
  **25/34 ≤0.** Ceiling-driven regression dominates.
- **LRP22 `deappfi` — R² 0.55 (strong).** Top `deappin` 0.43 — **same instrument
  (DEAP) sub-score** — but very noisy (z 0.18). Stark OOF-vs-stability contradiction:
  `erbword`/`erbnw` have large negative OOF importance yet appear top-5 in
  73–100 % of bootstraps. 22/32 ≤0; 11 pairs ≥0.7.

---

## 5. Recommendations

1. **Reframe the gain models honestly.** Report them as showing
   regression-to-the-mean (baseline dominates) with little evidence for
   non-baseline predictors of change at this n. Do not publish secondary
   importance rankings from gain models — they are noise (3.4). Consider whether
   _baseline-adjusted_ gains or a level-at-t+1 controlling for level-at-t
   formulation is more informative than raw change scores.
2. **Reframe the level models as concurrent correlation.** Their R² is real but
   reflects co-administered same-construct (sometimes same-instrument) tests.
   Where the question is "what predicts attainment," the within-instrument
   predictors (`aptgram`↔`aptinfo`, `deappin`↔`deappfi`) and same-domain reading
   pairs should arguably be excluded, as LRP04/LRP10 already do for vocabulary.
3. **Finish feature selection on LRP11–22** and **re-prune LRP08/LRP09.** Each
   carries ~12 dcor≥0.7 pairs and 18–28 ≤0-importance features. The refit-CV
   evidence (3.4, LRP04/LRP08/LRP10) shows aggressive redundancy pruning costs
   almost nothing. Use refit-CV deltas, not importance magnitude, to drive it.
4. **Stop interpreting specific hyperparameters** (3.7) and consider fixing a
   single sensible default regime per model family rather than per-model Optuna
   optima, given they are non-identifiable and CV-equivalent.
5. **Fix two reporting hazards:** surface `cv_pooled_r2` (not the per-fold mean)
   as the headline R², and add a caveat that `stability_selection` uses a
   different (in-bag R²) importance metric than the OOF table, so the two can
   legitimately disagree.
6. **Trust, with caveats, the recorded selection decisions for LRP01–10.** They
   did not discard genuine signal; the documented construct-driven drops are
   sound. The residual issues are _retained_ clutter (e.g. LRP02 `aptinfo`, LRP09's
   16 noise features, LRP10 `trog`), not bad drops.

---

## 6. Reproduction

```bash
# all compute (idempotent, ~1.5 h):
conda run -n dse-language-reading-predictors --no-capture-output \
  python output/replication/replicate.py lrp01 lrp02 … lrp22
# evidence + summary:
conda run -n dse-language-reading-predictors --no-capture-output \
  python output/replication/analyze.py lrp01 … lrp22
conda run -n dse-language-reading-predictors --no-capture-output \
  python output/replication/summarize.py
```
