<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Dose-response remediation: decisions taken for #587 (2026-08-23)

Preliminary research data and models — all conclusions remain provisional.

- **Tracking issue:** [#587](https://github.com/dseinternational/language-reading-predictors/issues/587)
- **Audit note:** `notes/202608231146-dose-response-model-audit.md` (Codex/GPT-5, PR #589)
- **Scope:** `lrp-rli-dose-077`, `083`, `084`, `177`, `277` plus the shared settings, factory, pipeline, reporting, comparison and report code

This note records the **decisions** taken while implementing the audit's plan, especially where a decision differs from the audit's first recommendation or where a finding was re-scoped. The audit note records the diagnosis; this one records what was done and why.

## Independent verification first

Every finding was re-derived against the code, the five stored reporting-tier fits and the rendered reports before anything was changed. The audit ran in a worktree with no traces, so several claims could only be checked here. Reproduced exactly: 156 rows / 53 children for word reading; a global dose SD of 30.657 sessions; period-1 arm–dose correlation **0.9700** (25 waitlist rows at zero, 28 immediate rows at 45–91); 84/156 shifted rows above their own period's observed maximum and 81/156 above the global maximum; 52/52 period-2 rows and 50/51 period-3 rows whose own baseline is itself a fitted outcome; a mean absolute prior-transform discrepancy of **1.289 items** per draw (the audit said ~1.31); `effective_adjustment: ["attend"]` in all five stored configs; and a rendered report printing "how to read the comparison table" above no table.

Three corrections to the audit:

1. **The ArviZ sign convention was not wrong.** Tested against the installed ArviZ 1.3.0: `elpd_diff` is `0.0` for the best model and negative for worse ones, exactly as the partial said. The _column name_ was wrong (`az.compare` writes `dse`, not `se_diff`); only that half was fixed.
2. **The template-order finding is repo-wide, not a dose defect.** Running `scripts/restructure_statistical_reports.py`'s validator over the real reports fails **254 of 254** statistical templates, including `itt-010` and `gf-001`. Fixing five would leave 249 broken and would tell a future reader the dose family had drifted when it had not. Re-scoped to a follow-up issue.
3. **The `dose-084` guessing-floor finding is also repo-wide.** Thirteen registered models carry `outcome_symbol="B"`; only `itt-008/108` and `lf-006/106` have the paired link sensitivity, and `blending_sensitivity.BLENDING_LINK_MODELS` covers the ITT pair alone. Building a companion for `dose-084` only would apply the policy to one of the nine uncovered fits. Re-scoped, with an explicit qualification added to the fit in the meantime (below).

## The finding the audit understated

The audit says unmodelled arm differences "can load onto the period dose slopes". On the stored fit they demonstrably **did**, and the identity is exact. With `beta_G = -0.07` and `dose_period1 = 0.179` per 30.657 sessions, and treated period-1 children averaging 72.7 sessions:

```
0.179 x (72.7 / 30.657) + (-0.07) = 0.3543
```

`lrp-rli-itt-010`'s randomised word-reading contrast is **0.3544** on the same logit scale. The period-1 "dose" slope — the largest and most confident in the model, and the one driving the published `+1.2 items` headline — was the randomised treatment effect divided by the mean dose. That is why findings 2 and 3 were treated as **one** estimand defect rather than two, as recommended.

## Decisions

### D1 — Separate the extensive and intensive margins by construction

Sessions now enter **centred and standardised over the fitted on-intervention rows only** (treated-row mean 66.2, SD 18.8 for word reading, against the old all-rows SD of 30.7), with every untreated row contributing exactly zero to every dose term. A `theta_treated` indicator carries the extensive margin.

**Why not fit both arm and presence everywhere:** in period 1 `treated` and `G` are the _same column_ — this is a fact about the trial, not a parameterisation choice, and no model can separate them there. Assigned arm therefore enters only from period 2 (`beta_arm_late`), where both arms are on the intervention and arm reads as intervention order. `theta_treated` is then identified from period 1 (where it is randomised) plus the three later zero-session rows. Each coefficient has one meaning, recorded in `config.json` under `coefficient_meanings` and generated from the same resolved plan the factory uses.

This follows the DiD dose family's existing idiom (`treated` + treated-centred dose) rather than inventing a second convention.

### D2 — Mundlak between/within split, on by default

`decompose_between_within=True` splits the treated-centred exposure into each child's study-average attendance (`beta_dose_between`) and their within-child deviation (the period slopes). Checked before adopting: among treated rows the between-child SD of child means is 16.4 sessions and the within-child SD is 11.3 (64% of variance between), with 2–3 treated rows for 50 of 52 children — thin, but real. A lone slope over a child random intercept returns a precision-weighted blend, which is why the catalogue's "adjusted within-child association" was wrong. Mirrors `pooled_levels` (#553).

### D3 — `dose-177` uses verified t1 ability; the old behaviour survives as a labelled comparator

The audit offered "broadcast t1" or "rename the fit". Broadcast was chosen, because the fit's _purpose_ is a pre-randomisation ability sensitivity and renaming would leave the suite without one. Implemented with `_broadcast_phase_zero_optional`, a NaN-tolerant sibling of the DiD family's `_broadcast_phase_zero`: the broadcast runs **before** the outcome mask so a child whose period-1 outcome is missing still contributes their t1 ability, and a child with no verified t1 row is dropped under an attributable mask rather than given a later, treatment-affected substitute. The change moves 98 of 156 rows' ability values, and `dose-177` now fits the same 156 rows / 53 children as `dose-077`, which also makes the pair comparison cleaner.

`ability_baseline_wave="transition_start"` retains the old behaviour, but the resolver rejects it when there are no ability symbols, the plan records it, and the module and report both state it must not be presented as a baseline sensitivity.

**Caveat worth stating plainly:** broadcasting t1 is what makes the fit match its published claim; it is not unambiguously the better _statistical_ choice. A t1 proxy is a weaker measure of ability at the time of a later period's attendance. Both readings leave latent `GA` unblocked, which is now said explicitly rather than assumed away.

### D4 — Support-respecting contrast: within-period treated interquartile

The items-scale headline moves each on-intervention row from **its own period's** observed lower quartile of sessions to that period's upper quartile (period 1: 65→81, period 2: 60→84, period 3: 43→76). Both endpoints are attendance levels observed in the period they are applied to, so the figure is interpolation, not extrapolation. `dose_support.csv` records the per-period quartiles, bounds, contrast width and a `contrast_within_support` flag; `dose_marginal_summary.csv` records the contrast kind, its size in raw sessions, the averaging population and row counts. The key-findings sentence now names the sessions ("about 24 more sessions in a period") instead of an unexplained "1-SD increase".

Chosen over a treated-row SD step because a quartile pair is an _observed_ pair of attendance levels, which is what "inside support" should mean, and because it reads in plain language.

### D5 — Leave-one-child-out PSIS, computed from the same trace

The audit offered leave-future-out or whole-child validation. Whole-child was chosen after testing it on the stored traces: max Pareto-k is 0.59 (077), 0.65 (177), 0.63 (277), 0.71 (084) and 0.78 (083, with 5 children above 0.7) — reliable where it matters, and **free**, because the repo already has the machinery. The factory now persists `loo_child_idx`, which activates `diagnostics._joint_log_likelihood_by_child` and the matching `psense_likelihood_var_names` exclusion built in #572.

Holding out a whole child removes the leak completely: all of that child's rows leave together, so no held-out outcome remains as a retained row's baseline. The plan records `loo_unit="child"` and a `loo_note` stating the target population is a **new child, not a future row** — the acceptance criterion asked for that to be explicit. Leave-future-out was rejected as disproportionate: it needs sequential refits, and a period-varying model cannot predict a period whose slope is prior-only, which would have made the headline comparison uninterpretable.

### D6 — One shared prior/posterior transform

`pipelines.dose_response.dose_marginal_draws` is now the single function both paths call, taking a draws group (prior or posterior), the row phase index, the per-row contrast and the denominator. The generic scalar-term writer is no longer used here. `test_prior_and_posterior_marginals_use_the_same_phase_indexed_transform` constructs slopes of `[2, 0, -2]` across periods and asserts the prior and posterior paths agree **and** that a scalar-slope transform does not reproduce the answer — so the test could actually have caught the original defect.

This is the shared-writer problem #576 raises for the DiD dose path. It is fixed here for `dose_response` only; the DiD path is untouched and still needs its own fix.

### D7 — Reference-coded phase intercepts

`alpha_phase[1] = 0` exactly, with free deviations for the later periods (`alpha_phase_free`), and `alpha_phase` retained as a Deterministic for reporting. The old four-column rank-three design was not an invalid Bayesian model — proper priors keep the posterior proper — but the global/phase split was prior-identified rather than likelihood-identified.

### D8 — Exposure and adjustment set recorded separately

`_effective_model_settings` no longer fills `effective_adjustment` from `prepared.covariates` for this family (which named the exposure and omitted every real adjuster). The family writes a proper record under `extra.effective_adjustment` naming the own baseline, `beta_arm_late`, age and any ability adjusters with their waves, plus `extra.exposure`, `extra.dose_margin` and `extra.dose_standardization`.

### D9 — Say what is enforced, and qualify what is not

The recipe previously told readers to interpret the posterior "only after ... power-scaling sensitivity diagnostics pass". Nothing enforced that: `release.GATED_KINDS` deliberately excludes observational families. Rather than gate an observational family on prior sensitivity, the recipe now says plainly that power scaling is **reported, not enforced**, and tells the reader to read `psense_summary.csv`.

That matters here concretely. On the pre-repair stored fits, `sigma_dose` was flagged prior-sensitive in all four period-varying models, and in `dose-084` the focal `mu_dose` itself was flagged ("potential strong prior / weak likelihood", prior 0.052 / likelihood 0.022). The audit noted the missing enforcement but not that there was something to catch.

`dose-084` additionally carries a standing **qualified-result** sentence in its key findings: blending has ten three-choice items, the ordinary link admits fitted means below the guessing level, and the required companion is not built for this family.

### D10 — Comparison delivery

`dose_response_loo_compare` now copies its result beside **both** paired runs as `dose_loo_compare.csv` — the name the partial reads — with a `comparison_note` naming what the pair differs by. Previously it wrote one file, to a third name, in a directory no report renders from.

## Not done, deliberately

- **Nonlinear-dose comparator.** The audit made it conditional on the new predeclared diagnostics warranting one. `dose_band_calibration.csv` now provides those diagnostics; read them on the refitted fits before registering a comparator.
- **Template order and the B-link companion.** Re-scoped repo-wide, as above.
- **The DiD dose path.** `write_dose_slope_summary`'s new arguments all default to the previous behaviour, and the new presence row is opt-in, so `did-006/007/107` are byte-unchanged. Their audit is #576.

## Refit and expected movement

All five models were refitted at reporting tier; the previous artefacts are kept as `<dir>.pre-587-20260823`. Every published number in this family changes, and the direction is predictable from the repairs: the headline slope is now an intensive-margin quantity per 18.8 treated-row sessions rather than a blended one per 30.7 all-row sessions, and the extensive margin it used to absorb is reported separately as `on_intervention`. **Old and new slopes are not comparable and must not be presented as a revision of the same number.**

## References

- Mundlak, Y. (1978). On the pooling of time series and cross section data. _Econometrica_, 46(1), 69–85. <https://doi.org/10.2307/1913646>
- Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and causal inference in epidemiology. _Epidemiology_, 11(5), 550–560. <https://doi.org/10.1097/00001648-200009000-00011>
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. _Statistics and Computing_, 27(5), 1413–1432. <https://doi.org/10.1007/s11222-016-9696-4>
- Merkle, E. C., Furr, D., & Rabe-Hesketh, S. (2019). Bayesian comparison of latent variable models: conditional versus marginal likelihoods. _Psychometrika_, 84(3), 802–829. <https://doi.org/10.1007/s11336-019-09679-0>
