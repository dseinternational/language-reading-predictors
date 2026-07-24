# Exploratory analysis scripts

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

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
  They answer different questions and are reported separately. Restricted to the
  prespecified common w1–w3 window so child means are comparable across children
  (the later waves are progressively Down-syndrome-only); within that window children
  still contribute unequal numbers of observed waves, so read the within-child matrix
  as a descriptive decomposition, not a balanced variance partition. Word reading
  couples with age much more strongly within-child (developmental growth) than
  between-child (where cohort composition flattens it).
- **RTM-corrected baseline → gain partials** (`rtm_partial_{pooled,group1,group2,group3}`,
  with the uncorrected `raw_baseline_gain_*` alongside): for every predictor–outcome
  pair, the correlation of the predictor's wave-1 level with the outcome's w1→w3 gain,
  **conditioning on the outcome's own wave-1 level**. Raw baseline→gain correlations
  are regression-to-the-mean-confounded by construction; the partial is the honest
  descriptive analogue (the correction that flipped the taught-vocabulary reading in
  the RLI strand, #405). The predictor-equals-outcome diagonal is undefined after
  conditioning and is left blank.
- **Within-group age check** (`age_within_group_check`): a crude age → word-reading-gain
  diagnostic, conditioning only on baseline word reading, pooled and within each
  `readgrp`. This is deliberately crude and is **not** a reproduction of
  `lrp-rlm-adj-001`: that model uses a different analytic sample (n ≈ 69) and a full
  covariate adjustment set (`bpvs`, `trog`, `basdig`, `bassim`, `basnum` and group
  nuisance terms), whereas this diagnostic uses the pooled n ≈ 77 with baseline word
  reading alone, and the group-specific rows are child-level (between-child), not
  within-child. Read the pattern — the pooled correlation running stronger than any
  within-group one — as a descriptive prompt that the pooled age signal may be partly
  cohort composition, not as an adjusted effect estimate.

Groups follow the catalogue labels (1 = Down syndrome, 2 = Average readers,
3 = Reading-matched). Correlations use Pearson's r throughout, for coherence with
the between/within variance decomposition and the linear RTM residualisation;
Spearman gives the same qualitative pattern and is a cheap robustness check.

**Not causal.** `readgrp` is an observational cohort factor and every number here is
a descriptive correlate carrying the usual residual-confounding caveat. These figures
are inputs to the still-gated modelling decisions in #338/#409 (instrument ceilings,
group scope, the reading-matched selection collider), not publication estimates.
