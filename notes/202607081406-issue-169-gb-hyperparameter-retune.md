> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Issue #169 — LightGBM hyperparameter retune for all GB models

Date: 2026-07-08. Retuned all 50 gradient-boosting models (`lrp-rli-gbg-001`–`lrp-rli-gbg-022`, `lrp-rli-gbl-001`–`lrp-rli-gbl-028`) on their current full `DEFAULT_GAIN` / `DEFAULT_LEVEL` predictor sets, superseding the earlier pruned-set / borrowed parameters left `retune-pending` after #116 Phase D retired hard feature selection.

## Tuning policy and command

Per the issue's proposed policy (unchanged after the pilot exposed no problems):

```bash
python scripts/tune_model.py {model_id} --n-trials 150 --scoring mae --lgbm-objective mae --seed 47
```

`GroupKFold` by `subject_id` at each model's own `cv_splits`, with an inner `GroupShuffleSplit` early-stopping slice carved from each training fold (the outer val fold is never shown to early stopping). The mean best iteration across folds becomes the tuned `n_estimators`. All 50 GB models use `LGBMPipeline` (target transform `none`) and the MAE objective, so the policy applies uniformly.

## Run manifest and orchestration

Added `scripts/tune_models_batch.py` — a resumable batch runner (family selection `gain`/`level`/`core`/`exploratory`/`all` or explicit `--models`; skips models already complete under the same policy unless `--force`; continues past failures; `--dry-run`). It tunes each model in its own subprocess sequentially — each Optuna trial already saturates all cores via LightGBM's `n_jobs=-1`, so running models concurrently oversubscribes and is slower.

- Run manifest: `output/tuning/retune169_manifest.json` (per-model command, git commit, start/end, wall-clock, status, headline CV metric).
- Per-model logs: `output/tuning/_logs/{model_id}.log`; tuned params in `output/tuning/{model_id}/best_params.json`.
- Review table: `output/tuning/review_169.csv` (old-vs-new CV MAE ± fold-std, `n_estimators`, boundary/pathology flags, verdict).
- Batch wall-clock: ~4.0 h (mean 5.2 min/model), **0 failures**. The tuning run was executed at commit `040ff26`; the promoted params were then re-targeted onto the canonical model-id rename (#168 Phase 2, #212) before this PR — the rename is content-preserving apart from the `model_id` string, so the tuned values transfer verbatim.

## Results — every change is within fold-noise

Comparing each new tune against the previously committed params evaluated under the same `GroupKFold` protocol:

- **All 50 verdicts `accept-neutral`**: the absolute MAE change is below 0.25 × the fold-std for every model. Fold std (≈ 2–7 MAE units) dwarfs the mean differences.
- 43/50 new MAE ≤ committed, 7/50 marginally worse — all within noise. Median MAE change −3.1 % (range −15.5 % to +9.1 %).
- This matches the issue's own risk note: with small grouped folds, TPE mostly re-lands in the same region. The retune's value is **provenance and parsimony, not predictive gain** — e.g. `lrp-rli-gbg-012` drops from 580 to 193 trees at the same MAE.

## Pathology scan — clean

- No failures; no `n_estimators` ceiling hits (max 1097 vs the 2000 ceiling → early stopping always engaged); new `n_estimators` range 3–1097, median ~127.
- **`lrp-rli-gbg-017` (nonword-repetition gain) collapses to `n_estimators = 3`** — a near-constant model. Params are fine; the _outcome_ carries little signal, so its ranking is low-information regardless. Flagged, not rerun (gain models are expected to be near-noise).
- 46/50 touch a search-space boundary, but the dominant boundary (44 models) is `reg_alpha` / `reg_lambda` at the 1e-3 log-floor — i.e. models prefer near-zero explicit L1/L2, regularising via tree size / subsampling instead. At 1e-3 this is effectively "no regularization", a benign search-floor artefact rather than a per-model problem. A handful additionally touch `num_leaves` / `max_depth` / `subsample` / `colsample_bytree` bounds mildly. No params were rejected on this basis.

## Borrowed-parameter groups — retired

The four borrowing relationships (`lrp-rli-gbg-002→001/003/004`, `lrp-rli-gbg-009→011`, `lrp-rli-gbl-002→001/003/004`, `lrp-rli-gbl-009→011`) are **retired**: every model now carries target-specific tuned params, and all former borrowers differ from their old source. `tests/test_borrowed_params.py` was rewritten to assert the relationships stay _broken_ (guarding against accidental re-copy) rather than that they match.

## Promotion and doc updates

- Promoted the tuned params into `_LGBM_MAE_PARAMS` in all 50 `models/lrp_rli_gbg_*.py` and `models/lrp_rli_gbl_*.py` modules (values only; each module's existing key schema — some carry `random_state`, some don't — was preserved to keep the diff to parameter values).
- Removed all `retune-pending` / borrowed / frozen-snapshot prose from module docstrings, param comments, and `notes` strings, replacing it with the #169 tuned wording.
- Updated `docs/models/README.md` and the affected per-model `docs/models/{id}/index.qmd` templates (the 10 formerly-borrowed models plus 12 exploratory models whose prose still claimed "not been through a target-specific tune").

## Exceptions / not promoted

None. All 50 tuned params were accepted and promoted. No model was rejected or rerun. `lrp-rli-gbg-017`'s tiny tree count is recorded above as a data property (near-noise outcome), not a tuning defect.

## Validation

`pytest tests/test_models.py tests/test_borrowed_params.py`, `python scripts/fit_model.py all --config dev`, `ruff check src/`, `npm run format:check`, and `npm run spellcheck` — see the PR for the run results.

## Reporting fit

After promotion, all 50 GB models were refit under the production config with reports rendered (`python scripts/fit_model.py all --config reporting --render`) — **50/50 fitted, 0 failed, 50/50 Quarto reports rendered**, running the full SHAP-interaction analysis and cluster-first predictor ranking that `dev` skips. The production fit confirms the retune's within-fold-noise verdict: no model degraded and no pooled R² is negative. Quality splits by model type as the DAG predicts — level (`gbl`) pooled R² median ~0.62 (up to 0.98 for the near-tautological composites `erbto`/`deapp_c`/`deappav`), gain (`gbg`) pooled R² 0.02–0.27 (change scores carry little predictable signal beyond regression-to-the-mean; `gbg-017` erbnw_gain stays a near-constant 3-tree model). Rankings are dominated by each outcome's own baseline plus `time`/`age` (autoregression + maturation), with `trog`, `aptinfo`, `spphon`/`blending`, and `yarclet` recurring as the substantive cross-predictors.

Artefacts uploaded to Azure Blob Storage under `<outputs container>/language-reading-predictors/output/{tuning,models}/` (models: 6776 files, ~640 MB). Output is git-ignored, so nothing beyond this note is committed for the reporting fit.
