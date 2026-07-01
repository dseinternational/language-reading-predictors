<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# DSE technical report — shared house style

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). A short shared standard for
> Down Syndrome Education International's Bayesian technical reports, derived from a
> comparison of the **language-and-reading** trial report and the **vocabulary-growth**
> trajectory report. Candidate canonical copy — see §Status for relocation.

Date: 2026-06-28

## Principle

**Standardise the backbone; keep the results spine distinct.** Both reports are DSE
Bayesian/PyMC reports (Beta-Binomial-on-logit), for a methods-literate reader, with a
_separate_ lay summary. But a **randomised trial with a causal DAG** earns causal
language and a causal-status structure, while **descriptive trajectory modelling across
populations** earns a population/lineage structure. Do not force either into the other's
mould — each report's architecture should make "what may be read causally" visible at a
glance.

## Shared backbone (make these the same in both)

- **Quarto book**; render to a gitignored `output/report`; `freeze: true`; CSL + `.bib`;
  an **AI-authorship callout** on AI-drafted content; `default-image-extension: png`
  (static figures, no SVG→PDF conversion).
- **Front matter:** Preface + a "how to read this report" router + an **executive
  summary** + an **evidence-at-a-glance** table (one row per outcome / question).
- **Three-way methods split:** `methods-data` (datasets, populations, harmonisation,
  limitations) / `methods-framework` (the canonical Beta-Binomial-on-logit engine,
  priors, sign convention) / `methods-workflow` (fit pipeline, sampling configs,
  convergence gate, model comparison, sensitivity). Appendices are written as _deltas_
  from the framework.
- **Model development:** lineage tables (_Step | added structure | problem addressed |
  LOO/ELPD effect_) and a **model register appendix with a status taxonomy**.
- A **frequentist→Bayesian on-ramp** (bridge) for the p-value/CI reader, plus a
  results-first "how to read the estimates" primer.
- **Reporting:** posterior **median** + an interval (state **HDI vs equal-tailed**
  explicitly); **no p-values**; report direction _and_ uncertainty; a positive-direction
  convention; convergence (R-hat / ESS / divergences) **gated before interpretation**.
- **No hand-entered numbers:** every value pulled from fitted output via a shared
  `_report_data.qmd` include with `show_or_pending` graceful degradation.
- **Repeated caveats stated once and transcluded** (`_caveats-*.qmd`).
- **Standard appendices:** glossary; model register + specs + per-model diagnostics;
  reproducibility & data availability; references.

## Per-report (keep distinct — justified by the study)

| Dimension             | Trial report (language & reading)                              | Trajectory report (vocabulary growth)                                                |
| --------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Results axis          | causal-status badges `[CAUSAL]`/`[ROBUSTNESS]`/`[ASSOCIATION]` | by **population** (DS / signing / TD comparison)                                     |
| Identification figure | adjustment-set **causal DAG**                                  | minimal **total-association** DAG                                                    |
| Evidence summary      | ROPE/δ + **evidence ladder** + 95% **equal-tailed** CrI        | **94% HDI** + _typical-child vs individual_ predictive split + LOO/ELPD supersession |
| Diagnostics           | dedicated **gate part** (few headline claims)                  | per-model **appendix** (many models)                                                 |
| Distinctive part      | gradient-boosting **honest-negative**                          | **practical-reference** + reference tables                                           |

## Shared machinery (`../research` / `dse_research_utils`)

- **Already shared:** `ReportingConfiguration` (the `<id>-<config>` output layout),
  sampling configs, interval helpers, MCMC-diagnostic & predictive plots, console/table
  formatting.
- **Move there next** (genuinely cross-project): the **evidence-ladder + ROPE-summary +
  odds-string** reporting helpers; the **`_report_data` `show_or_pending`** data-access
  pattern; the **convergence-banner** summary. One implementation, both reports.
- **Keep project-local:** model factories, the DAG, the estimands, per-model specs.

## Status

This note is the **candidate canonical copy**. It should move to `../research`
(`dse_research_utils` docs) — or a shared `dse-standards` location — so both report repos
reference a single source rather than drifting copies.
