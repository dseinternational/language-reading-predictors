<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore Byrne MacDonald basread basdig bpvs trog LCSM nutpie -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne reciprocal LCSM feasibility: no-go at the present sample size

**Decision for #338/#409, 2026-08-14: do not register or fit `lrp-rlm-lcsm-001`.** Neither the Down-syndrome-only candidate nor the three-group shared-coupling candidate passed the pre-fit recovery gate. Both models sampled cleanly and their 89% intervals had acceptable coverage, but neither recovered all three pre-specified reading-to-language/auditory-memory paths with adequate directional support. This is a data-information failure, not a sampler failure and not evidence that the true couplings are zero.

## Question and candidate models

The adopted lagged graph pre-specifies three reverse paths over the primary paper-compatible waves 1–3: prior BAS word reading (`basread`) predicting later change in receptive vocabulary (`bpvs`), receptive grammar (`trog`) and auditory digit recall (`basdig`). The study compared:

1. a Down-syndrome-only four-process latent change-score model with transition-specific mean changes and couplings shared over the two annual transitions; and
2. the same four-process system over all three cohorts, with group-by-transition mean changes and couplings shared across groups and transitions.

The second candidate borrows information across cohorts only for the couplings. It does not treat the observational cohorts as interchangeable trajectories, and it does not remove the reading-matched group's selection on reading. Every coupling remains an adjusted predictive association rather than a causal effect.

## Recovery design

The generator and fitted model use bounded Beta-Binomial scores for `basread`/`bpvs`/`trog`/`basdig` with confirmed denominators 90/32/20/34, the real group membership and cell-level missingness, correlated latent initial status, latent process innovations and group-specific empirical trajectory anchors. The fitted model estimates group-specific initial means, group-by-transition change intercepts, outcome-specific self-feedback and age slopes, the adopted forward paths, the three reverse paths, an LKJ-correlated initial-state distribution, process scales and Beta-Binomial concentration. It is deliberately smaller than a free RI-CLPM.

The Down-syndrome design contains 24 children, 21 complete on all four measures over waves 1–3, and 284 observed outcome cells. The three-group design contains 97 children, 68 complete on all four measures, and 1,034 observed cells. Forty independently simulated datasets were fitted for each candidate under a null reverse coupling of 0.00 and a modest positive coupling of 0.10, giving 160 fits. Each used `nutpie`, four chains, 500 warm-up iterations, 400 retained draws per chain and `target_accept=0.95`.

The pre-specified gate required every reverse path to achieve at least 95% fit success, at least 95% zero-divergence fits, absolute mean-median bias no greater than 0.05, 89% interval coverage between 75% and 100%, posterior positive support `Pr(g > 0) >= 0.90` in at least 80% of the positive datasets, and that same support in no more than 15% of null datasets. Requiring all three paths prevents post-hoc retention of only the easiest edge.

## Results

All 160 fits completed and all had zero divergences. The table reports the proportion of replicated datasets reaching `Pr(g > 0) >= 0.90`, followed by mean-median bias and 89% interval coverage under the positive scenario.

| Candidate     | Reverse path        | Null support | Positive support | Bias at 0.10 | 89% coverage | Gate                   |
| ------------- | ------------------- | -----------: | ---------------: | -----------: | -----------: | ---------------------- |
| Down syndrome | `basread -> basdig` |         0.0% |            55.0% |       +0.052 |        97.5% | Fail: support and bias |
| Down syndrome | `basread -> bpvs`   |        15.0% |            50.0% |       +0.027 |        92.5% | Fail: support          |
| Down syndrome | `basread -> trog`   |         5.0% |            30.0% |       -0.005 |       100.0% | Fail: support          |
| Three groups  | `basread -> basdig` |         2.5% |            57.5% |       -0.004 |        95.0% | Fail: support          |
| Three groups  | `basread -> bpvs`   |        12.5% |            70.0% |       +0.001 |        97.5% | Fail: support          |
| Three groups  | `basread -> trog`   |         2.5% |            40.0% |       -0.021 |        90.0% | Fail: support          |

Pooling improved bias and some directional-support rates, but it did not rescue the all-path gate. The pooled vocabulary path reached support in 28 of 40 positive datasets rather than the required 32; the grammar path reached only 16 of 40. The Down-syndrome candidate was weaker and its reading-to-digit-recall mean-median bias also narrowly exceeded the 0.05 ceiling. Null support remained within the pre-specified maximum, so the problem is low sensitivity rather than pervasive false-positive directional claims.

Forty replicates do not estimate each support probability very precisely and should not be used to rank small differences between paths. That Monte Carlo limitation cannot change the decision: the 30–40% support rates for the grammar path are far below the 80% requirement, and the gate was defined over all three paths. The study is also favourable to the proposed model because data are generated from its own functional form, later process innovations are independent across outcomes, and the only tested positive strength is 0.10. Real-data misspecification, time-varying common causes, measurement error beyond the fitted likelihood and reading-matched selection would not make identification easier. Conversely, failure at 0.10 does not establish that a much larger coupling could not be detected and does not demonstrate a null scientific effect.

## Consequence

Retain the adopted lagged DAG as the structural record, but close the current Phase C model proposal as no-go. Do not inspect a real-data posterior from this model and then simplify the edge set around whichever coefficient looks most favourable. Reconsideration needs genuinely new information—additional repeated observations, recovery of the paper's visual-memory measures, or a separately justified and pre-specified narrower question followed by a new recovery study. Waves 4–5 may remain sensitivity data for already registered descriptive growth models; they do not repair the paper-compatible three-wave directional question because the later panel is attrition-selected and wave 5 is Down-syndrome-only.

## Reproduction

The implementation is in `src/language_reading_predictors/statistical_models/rlm_lcsm_feasibility.py`, the command-line harness is `scripts/simulate_rlm_lcsm_feasibility.py`, and structural/unit guards are in `tests/statistical_models/test_rlm_lcsm_feasibility.py`. The exact study command was:

```bash
python scripts/simulate_rlm_lcsm_feasibility.py --mode study --n-sims 40 --draws 400 --tune 500 --chains 4 --output-dir output/statistical_models/feasibility/rlm-lcsm
```

The ignored output directory contains `attempts.csv`, edge-level `replicates.csv`, `summary.csv` and `decision.json`; deterministic simulation and sampler seeds are recorded by the harness. The source decision that defined the graph and required this gate is `notes/202608141700-byrne-lagged-dag-decision.md`.
