<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# GB feature selection retired — #116 Phase D

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

**Date:** 2026-07-01 · **Issue:** [#116](https://github.com/dseinternational/language-reading-predictors/issues/116) Phase D

## What changed

The gradient-boosting discovery layer no longer performs hard feature
**selection**. Every GB model (`LRPGBG##` / `LRPGBL##`) now fits the **full**
`Predictors.DEFAULT_GAIN` (34) / `DEFAULT_LEVEL` (33) predictor set; the ordered,
clustered **ranking** (`scripts/rank_predictors.py`, and the per-model
"Predictor ranking and clustering" report section) is the deliverable — not a
pruned subset.

Concretely:

- Removed the `SelectionStep` mechanism: the class (`models/common.py`), the
  `ModelConfig.selection_history` field, `_collect_selection_history` and the
  selection-step application in `models/base_model._build_predictors`, the
  `selection_steps` class attribute, and every model's `_SELECTION_STEPS`
  constant. `_build_predictors` now returns `base − target − exclude + include`.
- `save_config` no longer writes `selection_history` to `config.json`.
- `scripts/rank_predictors.py` drops the selection-vs-ranking comparison
  (`kind="sel"`, `ranking_vs_selected`); the full-set ranking and the
  `ranking_excluding_same_skill.csv` view remain. `conditional_dropout_check`
  (a full-set ranking robustness check) is retained.
- Dropped the four `_noconstruct` variants (`lrpgbl06/07/08/16_noconstruct`).
  Their same-skill contrast is now the ranking's `ranking_excluding_same_skill.csv`
  + the curated `SAME_SKILL_SIBLINGS` annotation (registry 54 → 50).
- Deleted `scripts/period_resolved_gb_diagnostic.py` (its pruned-vs-full premise
  is retired).

## Why

#116's thesis is that a defensible, reproducible **ranking** — not a hard prune —
is the honest output of the discovery layer. The prior "uniform feature
selection" pass (2026-06-21/23, distance-correlation redundancy filter + noise
floor) reduced each model to 3–11 predictors, which biases the very importances
the layer exists to report and hides candidates from the ranking.

## Consequence: results change; params are retune-pending

Switching from pruned (3–11) to full (33–34) predictor sets **changes every
formerly-pruned model's committed results**. Hyperparameters were Optuna-tuned on
the pruned sets and are **retained as a full-set baseline, retune-pending** — a
deliberate, documented choice (#116: cluster-level rankings are robust to
reasonable hyperparameters; re-tuning ~40 models × 150-trial Optuna is deferred).
`scripts/tune_model.py` remains for an optional later refresh.
