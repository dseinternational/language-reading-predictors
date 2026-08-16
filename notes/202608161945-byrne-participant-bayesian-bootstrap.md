> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne paired-growth participant Bayesian bootstrap

## Decision

Retain the instrument-documentation publication gate. The denominator-free participant Bayesian bootstrap passes the five-method robustness rule for WORD reading comprehension (`woco`) and BAS number skills (`basnum`) but gives a strict `no_go` for BAS spelling (`basspel`). The spelling failure is confined to one near-zero contextual contrast: reading-matched minus average-reader total growth over waves 1–4 has a bootstrap median of −0.19 items, versus +0.09 items in the observed-maximum Beta-Binomial fit. All five 89% intervals overlap and span zero. This is not evidence of a substantively reversed developmental pattern, but the sign-only rule was written before the data run and is not relaxed after seeing the result.

## Method

Within each reading group, the analysis assigns the retained participants independent Dirichlet(1, ..., 1) weights, following Rubin's Bayesian bootstrap. One participant-weight vector is shared across all intervals in a draw. Each interval is a weighted mean of child-level raw-score changes among children observed at both endpoints; extension-wave weights are renormalised over that paired subset. Between-group quantities subtract independently weighted group growth draws over the common waves 1–4 horizon. This targets descriptive paired raw-score growth in the retained analysis panel. It introduces neither a score denominator nor an unbounded count likelihood, but it also provides no latent-trajectory smoothing or measurement model.

The full audit uses two independent simulations of 200,000 draws per measure. Numerical stability requires the maximum difference between their 5.5th, 50th and 94.5th percentiles to be no more than 0.5% of the observed extract maximum. Scientific robustness then appends the bootstrap to the four trace-backed likelihood variants from `notes/202608161900-byrne-denominator-likelihood-sensitivity.md` and retains that audit's fail-closed criteria: every median direction must agree, all five 89% intervals must have a common overlap, and the five-method median range must not exceed 10% of the observed maximum. The reference loader verifies the saved hashes of all twelve model traces and the regenerated likelihood-growth table before comparison.

## Results

| Measure | Children | Rows | Decision | Largest five-method median range | Largest bootstrap versus 1× Beta-Binomial median difference | Maximum replicate quantile difference |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| BAS spelling (`basspel`) | 69 | 271 | `no_go` | 0.53 items (3.0% of observed maximum) | 0.33 items (1.8%) | 0.015 items (0.081%) |
| WORD reading comprehension (`woco`) | 77 | 304 | `pass` | 2.63 items (8.5%) | 0.79 items (2.6%) | 0.017 items (0.055%) |
| BAS number skills (`basnum`) | 73 | 272 | `pass` | 1.67 items (2.8%) | 1.01 items (1.7%) | 0.019 items (0.032%) |

Across the 47 reported estimands, all five-method 89% interval intersections are non-empty. Forty-six retain the same median direction. The exception is the spelling reading-matched-minus-average-reader contrast described above: its bootstrap 89% interval is −2.12 to +1.76 items and P(growth contrast > 0) is 0.44, while the four likelihood medians range from +0.09 to +0.34 with probabilities from 0.54 to 0.58. None supports an affirmative directional conclusion.

## Interpretation and limits

The result strengthens a narrow claim: the principal Phase-A raw-growth patterns are not generally created by the provisional score denominators. It does not make the registered fits publishable. The Bayesian bootstrap conditions on the retained participants and assumes that its group-specific empirical distributions are an adequate basis for uncertainty; it does not repair selection into the complete-case core or available-case extension, identify the administered forms, establish score maxima, or explain the `basnum` transformation. It also cannot replace the bounded transforms used by `lrp-rlm-mm-001`, `lrp-rlm-adj-001`, or `lrp-rlm-hs-001`.

The strict spelling `no_go` should be preserved as an audit result, while the scientific prose should state why it is not a meaningful sign reversal: the discrepant estimand is close to zero under every method and its uncertainty spans effects in both directions. A future rule that treats direction as defined only when an interval excludes zero may be methodologically preferable, but adopting it now for this result would be post hoc.

## Reproduction

The implementation is in `src/language_reading_predictors/statistical_models/rlm_growth_bootstrap.py`; the command-line harness is `scripts/rlm_growth_bootstrap.py`; structural and fail-closed decision guards are in `tests/statistical_models/test_rlm_growth_bootstrap.py`. The full command was:

```bash
python scripts/rlm_growth_bootstrap.py --mode study --measure all
```

Derived tables and participant-level reference traces remain under the ignored `output/statistical_models/sensitivity/` tree and are not committed.

## Reference

Rubin, D. B. (1981). The Bayesian Bootstrap. *The Annals of Statistics, 9*(1), 130–134. <https://doi.org/10.1214/aos/1176345338>.
