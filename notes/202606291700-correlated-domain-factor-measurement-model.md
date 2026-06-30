# Correlated-domain-factor measurement model (LRPMM01)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

**Issue:** #134 (the DAG-sanctioned successor to the closed LRP66 / #97).
**Model:** `lrpmm01` (kind `corr_factor`).

## Question

Replace the single latent general ability `g` of LRP66 with **correlated domain
factors** as an _identification-neutral but better-fitting_ measurement match for
the observed same-construct clustering, and recover the one output no merged model
produces: an explicit **factor-loadings / communality** table plus a partial
**measurement-error correction** of the skill->gain slopes.

## Design

A reflective CFA + a structural Beta-Binomial leg (between-child, one row per
child, `phase_mode="span"`):

- **Measurement.** Correlated latent factors over the standardised T1 skill
  indicators, with an **LKJ** prior on the factor correlation matrix. Factor
  variances are **fixed to 1** (standard normals pushed through the correlation's
  Cholesky) and loadings are positive (`HalfNormal`, fixing orientation). The
  indicator residual variance is **free**, so a loading `lambda` is a coefficient
  on the unit-variance factor, **not** in general a correlation; the report
  carries the indicator-factor **correlation**
  (`lambda / sqrt(lambda^2 + sigma^2)` = `sqrt(communality)`) and the
  **communality** (`lambda^2 / (lambda^2 + sigma^2)`) alongside it.
- **Structural.** Word-reading gain (`W` post conditioned on `W` T1 via
  `gamma_own`) regressed on the latent factors (+ non-verbal MA `blocks` + age),
  giving **measurement-error-corrected** factor->gain slopes.
- **Reporting.** `loadings_summary.csv` (loading + communality per indicator),
  `factor_correlation.csv` (the D x D factor correlation), and
  `structural_summary.csv` (factor->gain slopes).

## Domains -> indicators (resolved against the data)

The issue's conceptual domains map to the available standardised `measures.py`
symbols as follows, and the **core set** (per the design decision) keeps the
complete-case n ~ 51:

| Domain     | Core indicators                     | Notes                                                           |
| ---------- | ----------------------------------- | --------------------------------------------------------------- |
| Vocabulary | `R` (ROWPVT), `E` (EOWPVT)          | taught/untaught vocab left out to preserve n                    |
| Code       | `L` (letter sounds), `B` (blending) | `P`/`N` excluded: heavily floored, poor Gaussian CFA indicators |
| Grammar    | `F` (CELF), `T` (TROG)              | exactly two indicators (thin but identified)                    |

**Speech `{SP}`** (DEAP) is **not available** in the statistical `measures.py` and
is dropped (single-indicator anyway). Non-verbal MA (`blocks`) is **not** a
language/literacy domain, so it enters the **structural leg as an observed
covariate**, not a factor indicator.

## DAG compliance (#115) and honesty

This is a **measurement / triangulation** model, **not** causal. Per **ID-2**
every factor->gain slope is a latent-ability-confounded **adjusted association**;
the factor model improves the measurement match, it does not identify effects. The
randomised causal claim continues to live in the ITT suite (`lrpitt10` for word
reading) and the `gain_factors` randomised term - this model does not carry it.
`GA` stays diagnostic/latent (never an implementable adjustment set).

At **n ~ 51** a correlated-factor latent model is fragile and prior-dependent -
report it as triangulation against `gain_factors`, with **wide intervals stated as
the honest result**, exactly as the closed LRP66 did. The reporting-tier re-fit
gives the final magnitudes; the report templates are CSV-driven so a re-render
refreshes them.

## What it adds over merged models

The merged `gain_factors` / `level_factors` families emit a per-coefficient
regression summary only - **no loadings, no communalities**. The loadings /
communality table and the measurement-error-corrected slopes are the unique,
worth-keeping contribution salvaged from LRP66.

## References

- `notes/202606231600-dag-revision-consolidated.md` - locked DAG (the deferred
  "correlated domain factors / bifactor" measurement option; `GA` diagnostic-only;
  ID-2).
- `notes/202606231100-gb-selected-features-tables.md` - the same-construct
  clustering motivating the move.
- `notes/202606261230-gain-level-factors-design.md` - the merged `gain_factors`
  family this complements.
- Closed PR #97 (LRP66) - source of the loadings/communality approach.
