<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Decision: credible-interval reporting standard — median + 50 % + 89 % equal-tailed, full posterior

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Records a project-wide reporting-convention change signed off 2026-07-17. Supersedes the three-band convention of issue #177 (95 % headline / 50 % central / 90 % sensitivity).

## The standard

Every Bayesian result in this project reports, for each quantity of interest:

- the **posterior median** as the point estimate (transformation-invariant across the logit and probability/items scales — preferred over the mean);
- an **inner 50 %** and an **outer 89 %** _equal-tailed_ credible interval;
- the **full posterior** (density/forest plots), which remains the primary object.

The interval type is **equal-tailed (ETI)**, declared as such. The HPDI is retained only as a per-scale sensitivity companion (it is not transformation-invariant), never the headline. Direction is read from the posterior tail probability (`P(AME > 0)`), never from whether a band excludes zero.

## Why 89 %, not 95 %

95 % is **arbitrary** — a convention imported from frequentist null-hypothesis significance testing, where α = 0.05 was itself a rule of thumb. It carries no special Bayesian meaning, and anchoring on it invites exactly the bright-line "excludes zero / doesn't" reading we are trying to avoid. Two concrete reasons to move off it:

1. **Arbitrariness made visible.** A deliberately non-round coverage (89 %) signals to the reader that the number is a choice, not a threshold (the McElreath convention). We pair it with the full posterior so the interval is a summary, not a verdict.
2. **MCMC stability per effective draw.** The limits of a 95 % interval are the **2.5th / 97.5th** percentiles — the noisiest quantiles to estimate from a finite set of posterior draws, because few draws land in the extreme tails. The 89 % limits (**5.5th / 94.5th**) sit further from the tails and, per effective draw, are estimated markedly more stably (by normal-theory scaling a 95 % limit needs roughly 1.7× the effective sample size of an 89 % limit for the same Monte-Carlo error). Kruschke's Bayesian Analysis Reporting Guidelines make the direction explicit: _"For reasonably stable estimates of limits of highest-density intervals (HDIs), I recommend that ESS ≥ 10,000. For stable estimates of limits of equal-tailed intervals, ESS can be lower."_ (Kruschke 2021, [doi:10.1038/s41562-021-01177-7](https://doi.org/10.1038/s41562-021-01177-7)). This is a per-effective-draw efficiency argument, **not** a claim that the suite is ESS-starved: at reporting config the headline treatment terms attain a Tail-ESS in the low tens of thousands (median ≈ 22,000), comfortably above Kruschke's conservative 10,000, so the 89 % limits are stable well beyond the precision we quote them at. A further convenience: the standard Tail-ESS is calibrated to the 5 %/95 % quantiles (a 90 % interval), so it is the near-exact diagnostic for our 89 % (5.5 %/94.5 %) limits. The full ESS handling and reporting standard — the floor-versus-target distinction, the Bulk-/Tail-ESS mapping, the attained figures, and the derived-estimand diagnostics — is recorded in [`202607181200-ess-reporting-standard.md`](202607181200-ess-reporting-standard.md).

The BARG explicitly treats 95 % as _"conventional but not mandatory"_ and requires only that the chosen mass be **clearly reported** — which we now do (every report and note states "89 % equal-tailed"). The **50 %** inner band (the middle half of the posterior) is kept as a robust, high-ESS visual anchor.

## Why the 90 % "sensitivity" band is retired

The #177 convention carried a third, equal-tailed **90 %** band as a compatibility/sensitivity strip to discourage over-reading the 95 % headline. Under the new standard that role is redundant: an 89 % headline is itself non-round (so it cannot be mistaken for a decision threshold), and the full posterior is always shown. A 90 % band sitting beside an 89 % headline would differ only trivially and add noise, so it is removed everywhere.

## What was reviewed

Three Bayesian methodology / reporting sources were consulted:

- **Kruschke (2021), Bayesian Analysis Reporting Guidelines (BARG)** — the load-bearing source for the interval decision. Beyond the interval-mass/ESS point above, its six-step checklist (model + priors → computation with per-parameter ESS → posterior with predictive checks → decisions/ROPE → prior sensitivity → reproducibility) is already substantially met by this suite's pipeline (priors table, prior-predictive, the R-hat/ESS/BFMI gate, PPC coverage, ROPE summaries, `config.json` reproducibility). The change here aligns the one place we were off-standard: the interval mass and its explicit declaration.
- **Gelman, Vehtari, Simpson, Margossian, Carpenter, Yao, Kennedy, Gabry, Bürkner & Modrák (2020), "Bayesian Workflow"**, [arXiv:2011.01808](https://arxiv.org/abs/2011.01808). The broader **process** framework within which this reporting decision sits. It does **not** address interval width (that is a reporting question, downstream of the workflow); its value here is confirming that the interval standard is the _model-understanding / communication_ step of an iterative workflow this project already follows (see the mapping below).
- **Barons, Hanea, Mascaro & Woodberry (2025), "Reporting Standards for Bayesian Network Modelling"**, _Entropy_ 27(1):69, [doi:10.3390/e27010069](https://doi.org/10.3390/e27010069). Reviewed and found **not to bear on interval choice** — it is a structure/provenance checklist for discrete Bayesian _networks_ (node dictionary, causal-interpretation status, knowledge sources, validation) and makes no credible-interval recommendation. It is relevant only as a general transparency/reproducibility model, which our DAG documentation, adjustment-set reviews and priors tables already follow.

## Situating this in the Gelman et al. Bayesian workflow

The 89 % / 50 % / median / full-posterior standard is the **reporting face** of the _model-understanding_ stage of the Gelman et al. workflow — not a standalone rule. The workflow is explicitly iterative ("in practice we will be fitting many models for any given problem") and stresses transparency about "the decisions made in the process of data analysis". This project's pipeline already realises the workflow's stages, which is why the interval change is a localised reporting tweak rather than a methodology overhaul:

| Gelman et al. workflow stage                               | How this suite realises it                                                                                                                                                      |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model construction from domain knowledge                   | The committed causal **DAG** + per-family factories; adjustment sets derived and CI-guarded against the DAG.                                                                    |
| Prior specification + **prior predictive** checking        | Role-tiered priors (`priors_table.csv`, `PRIORS.md`), prior-predictive simulation and an estimand-scale pushforward per fit (`_prior_predictive.qmd`, `prior_pushforward.csv`). |
| Fitting / inference                                        | NUTS via `nutpie` under shared sampling presets.                                                                                                                                |
| **Validating computation** (R-hat, ESS, divergences, BFMI) | The pass/fail convergence **gate** (`diagnostics_summary.json`) interpreted _before_ any result — and the ESS argument is exactly what drives the 89 % choice here.             |
| **Posterior predictive** checking                          | `ppc_summary.csv` coverage + LOO-PIT / PPC plots; the "90 % prediction range" coverage statistic (a diagnostic, deliberately **not** touched by this interval change).          |
| Model comparison                                           | PSIS-LOO across nested/interaction models (`compare_statistical_models.py`), with gate-exclusion discipline.                                                                    |
| **Model understanding** / communication                    | The findings-first reports and `notes/` — **this is where the median + 50 % + 89 % + full-posterior standard lives.**                                                           |
| Sensitivity analysis                                       | Prior-scale sweeps (`tau_prior_sensitivity.py`), the HPDI companion, and the retained unmeasured-confounding caveats.                                                           |
| Iterative model building / expansion                       | The whole `statistical_models/` history — variants, floor rules, DAG revisions, the reverse-direction and Tier-1 additions — each recorded as a dated note.                     |

The one workflow-level gap worth naming (not closed by this change): several families' unmeasured-confounding (latent general ability) is structural and cannot be resolved by more modelling — a limitation documented in `notes/202607172345-design-lessons-for-future-studies.md`, not a workflow defect.

## Scope of the code change (2026-07-17)

- `ci_prob` default 0.95 → **0.89** (`context.py`), flowing to every headline `lo`/`hi` derived from `lo_q = (1 − ci_prob)/2`. The four `ci_prob = 0.94` growth / correlated-factor overrides and the hardcoded `ci_prob = 0.95` sensitivity refits → 0.89.
- The **90 % band retired**: producers that carried both a 50 % and a 90 % band drop the 90 %; producers that carried a 90 % band but no 50 % band convert it to the 50 % band (`0.05 / 0.95` → `0.25 / 0.75`, `_lo90/_hi90` → `_lo50/_hi50`). Hardcoded `0.025 / 0.975` headline computations → `0.055 / 0.945`. `eti_bands(x, probs=(0.5, 0.9))` → `probs=(0.5,)`.
- A **median** column was added to the summary producers / key-findings builders that previously exposed only the posterior mean.
- Report partials, the key-findings text, `METHODS.md` and the synced `CLAUDE.md` / `AGENTS.md` / `.github/copilot-instructions.md` conventions were updated to the "median + 50 % + 89 % equal-tailed" language.
- **Left unchanged:** posterior-predictive **prediction-range** coverage (the "90 % prediction range" in the PPC/diagnostics), which is a coverage diagnostic, not a credible interval; and the `lo_q = (1 − ci_prob)/2` formula itself.
- The stragglers that used their own scheme were brought onto 89 % too (signed off): the historical-cohort reproduction study (`q2.5/q97.5` → 89 %), the mediation session-calibration bootstrap (95 % → 89 %), and the horseshoe predictor-ranking (94 % HDI → 89 % HDI).

## Rollout

Because the interval limits are written at fit time, the change takes effect on **re-fit**. The full reporting suite is re-fit and re-published under this standard (folded into the pending re-publish). Findings / spec notes carrying specific 95 %-computed intervals are refreshed **after** the re-fit, so their numbers and their "89 %" labels move together — relabelling old 95 % numbers as 89 % without recomputation is explicitly avoided.

## References

- Kruschke, J. K. (2021). Bayesian analysis reporting guidelines. _Nature Human Behaviour_, 5(10), 1282–1291. [doi:10.1038/s41562-021-01177-7](https://doi.org/10.1038/s41562-021-01177-7) (PMID 34400814)
- Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., & Modrák, M. (2020). Bayesian workflow. _arXiv_. [arXiv:2011.01808](https://arxiv.org/abs/2011.01808)
- Barons, M. J., Hanea, A. M., Mascaro, S., & Woodberry, O. (2025). Reporting standards for Bayesian network modelling. _Entropy_, 27(1), 69. [doi:10.3390/e27010069](https://doi.org/10.3390/e27010069)
