# Exploratory analysis scripts

Descriptive, pre-model exploratory passes that inform the gated modelling decisions.
Purely descriptive — **no models, no causal language.** Each script loads a study's
data directly and writes one figure per file (PNG + SVG + CSV) under
`output/exploratory/` (gitignored), so commit the scripts, not the generated figures.

## `rlm_associations.py` — Byrne reading-language-memory descriptive pass (#409 item A)

Mirrors the RLI descriptive work for the observational Byrne, MacDonald & Buckley
(2002) cohort (`study_id="rlm"`). It answers the two associational questions the
Byrne suite is currently thin on, before any of the gated Beta-Binomial or
group-contrastive models are fitted. Run:

```bash
python scripts/exploratory/rlm_associations.py
```

Writes to `output/exploratory/rlm/`:

- **Per-wave level correlations** (`per_wave_corr_w{1..5}`): a cross-sectional
  correlation matrix over the battery plus age at each wave (pooled across groups).
- **Between-child vs within-child decomposition** (`between_child_corr`,
  `within_child_corr`, `reading_between_vs_within`): the between-child matrix
  correlates children's mean levels ("do children who read well tend to score well
  elsewhere"); the within-child matrix correlates each wave's departures from the
  child's own mean ("is a good year for reading a good year for the other skill").
  They answer different questions and are reported separately. Word reading couples
  with age much more strongly within-child (developmental growth) than between-child
  (where cohort composition flattens it).
- **RTM-corrected baseline → gain partials** (`rtm_partial_{pooled,group1,group2,group3}`,
  with the uncorrected `raw_baseline_gain_*` alongside): for every predictor–outcome
  pair, the correlation of the predictor's wave-1 level with the outcome's w1→w3 gain,
  **conditioning on the outcome's own wave-1 level**. Raw baseline→gain correlations
  are regression-to-the-mean-confounded by construction; the partial is the honest
  descriptive analogue (the correction that flipped the taught-vocabulary reading in
  the RLI strand, #405). The predictor-equals-outcome diagonal is undefined after
  conditioning and is left blank.
- **Within-group age check** (`age_within_group_check`): whether the pooled
  age → word-reading-gain signal (`lrp-rlm-adj-001`'s headline) survives inside each
  `readgrp`. The pooled correlation is stronger than any within-group one, so the
  pooled age signal is partly cohort composition rather than a purely within-child
  effect.

Groups follow the catalogue labels (1 = Down syndrome, 2 = Average readers,
3 = Reading-matched). Correlations use Pearson's r throughout, for coherence with
the between/within variance decomposition and the linear RTM residualisation;
Spearman gives the same qualitative pattern and is a cheap robustness check.

**Not causal.** `readgrp` is an observational cohort factor and every number here is
a descriptive correlate carrying the usual residual-confounding caveat. These figures
are inputs to the still-gated modelling decisions in #338/#409 (instrument ceilings,
group scope, the reading-matched selection collider), not publication estimates.
