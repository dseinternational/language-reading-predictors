> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Retirement of the boosting variant comparison script

`scripts/compare_variants.py` compared fold scores from different versions of the same boosting model. Its examples paired word-reading fits using selected predictors and log-transformed targets. The current registry has no such variants, and the script has no callers in the fit, reporting or test workflows. The [selection retirement decision](202607011000-gb-selection-retired-phase-d.md) explains the move to ranking predictors on the full set.

For the current question about same-skill predictors, `scripts/rank_predictors.py` refits eligible models after removing the designated predictors and reports their held-out performance. To reproduce the former paired-fold analysis on archived score files, use the [script at the last revision before removal](https://github.com/dseinternational/language-reading-predictors/blob/742fe07b4f54cbe2f57a2b2e2dc19c44ec8c649f/scripts/compare_variants.py) with fits that used the same outcome, children and folds.
