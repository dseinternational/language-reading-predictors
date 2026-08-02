<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5), recording the divergence-qualification policy agreed by Frank Buckley.

# Divergent-transition qualification policy

## Decision

**Decision (2026-08-02):** zero post-warm-up divergent transitions remain the requirement for an automatic, clean sampling-quality pass. A reporting-tier fit with a genuinely small absolute number of divergences may exceptionally be reported as **qualified, not passed**, but only after a model-, trace- and estimand-specific review establishes that the named reported quantities do not share the divergent geometry. No percentage threshold makes divergences harmless, and qualification is never automatic.

This supersedes the earlier informal “below 1% is usable with a caveat” guidance and the permanent model-spec waiver recorded for `lrp-rlm-mm-001` in `notes/202607241200-mm001-gate-exception.md`. Historical notes retain the evidence available when written; they are not the active policy. No current fit is qualified under this decision.

## Why the policy is not simply zero-or-nothing

A divergent Hamiltonian Monte Carlo trajectory warns that the numerical integrator could not follow part of the posterior geometry reliably. The missed region, rather than the retained divergent draws, is the important risk. Removing divergent draws cannot diagnose missing posterior mass. The Stan diagnostic guidance therefore says that even a small number must be investigated when fully reliable inference is required, while also recognising that a few divergences with otherwise healthy diagnostics can sometimes be adequate after careful model-specific investigation. Betancourt's account explains why varying posterior curvature produces this failure mode and why reparameterisation is usually the substantive remedy (Betancourt 2017, DOI [10.48550/arXiv.1701.02434](https://doi.org/10.48550/arXiv.1701.02434); [Stan diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html)).

The number or percentage of divergences is therefore a screening signal, not a validity test. One divergence located in the geometry of a treatment effect, nonlinear knee, covariance boundary or shrinkage ranking can matter more than several diffuse divergences in a nuisance direction. Conversely, automatically discarding an otherwise stable exploratory fit because of one transition can be disproportionate only after the dedicated geometry-mapping review below provides positive evidence about where the failure occurs; a visually unpatterned handful alone is not reassuring.

## Three statuses

- **PASS:** zero divergences and every implemented automatic gate check passes: R-hat ≤ 1.01, bulk and tail ESS ≥ 400, and BFMI ≥ 0.30. This is the only clean pass; other model and predictive checks remain separate release requirements.
- **QUALIFIED, NOT PASSED:** a small absolute number of divergences remains after reasonable remediation; every other gate check passes; and a trace-bound review covering the exact reported estimands is approved. The ordinary `diagnostics_summary.json` remains `passed: false`.
- **FAIL:** the review is absent or incomplete; any other gate check fails; divergences persist materially, cluster, or implicate a reported estimand or its stochastic ancestors; or the fit is in a category that requires a clean pass. Findings are withheld.

There is deliberately no fixed percentage threshold. “Small” must be justified from the absolute count, its distribution across chains and seeds, and its location in parameter space. A percentage such as 1% cannot substitute for that evidence.

## Qualification requirements

A qualification is eligible only when all of the following are true:

1. The fit is the reporting-tier fit intended for publication, and divergences are its only failed automatic gate check.
2. The reported claim is non-causal and exploratory or secondary. Randomisation-anchored treatment effects, model-of-record causal results, natural- or interventional-effect decompositions, floor/off-floor and survival estimands, nonlinear curve shapes or knees, hierarchical dose heterogeneity, horseshoe rankings, and covariance or latent-structure quantities require zero divergences because their scientific interpretation depends directly on joint or tail geometry.
3. At least one reasonable geometry remedy has been attempted and recorded: a mathematically equivalent reparameterisation, stronger defensible regularisation, longer warm-up, higher `target_accept`, or a simpler scientifically defensible model. A cleaner confirmatory fit replaces the divergent fit rather than qualifying it.
4. An independent-seed confirmatory reporting run has all ordinary checks passing apart from a similarly small absolute number of divergences, and the named headline medians, interval limits, tail probabilities and practical-threshold conclusions are stable within Monte Carlo uncertainty.
5. A dedicated diagnostic run deliberately produces enough transitions from the same problematic geometry to map it—for example, by lowering `target_accept` for diagnosis only—or supplies equivalent positive evidence. Its divergence plots and chain-level counts show no clustering in the named estimands or their stochastic ancestors. The one or few events in the candidate reporting run, and agreement between two seeds that might both miss the same region, are not sufficient. The reviewer must explicitly classify headline geometry as not implicated; “uncertain” fails closed.
6. The review checks the derived reported quantities themselves—such as an average marginal effect—not only their source coefficients. Removing divergent draws may be shown as a diagnostic, but never counts as sufficient evidence.
7. The qualification is bound by hashes to the exact trace, config, data, model source and sampling contract; names the permitted estimands and outputs; records the issue, date, model owner and statistical reviewer; and is invalidated by any refit or input change.
8. The report leads with a visible amber qualification saying the fit did not pass the ordinary zero-divergence gate. Qualified fits are excluded from model-selection comparisons unless the predictive and log-likelihood geometry receives a separate explicit review.

Sensitivity grids, influence refits, exact leave-one-out refits and subsidiary fits remain zero-divergence-only: a qualification of the parent fit does not propagate to another trace.

## Implementation boundary

The automatic gate continues to fail on any divergence. Permanent exceptions declared in `ModelSpec.extra` are retired because they are not bound to one immutable fit and cannot verify the review evidence. A future qualification must arrive in its own reviewed pull request with a content-addressed `divergence_qualification.json`, diagnostic artefacts and a verifier shared by the compact badge, key-findings interlock, integrated report and comparison scripts. Until that verifier exists and the evidence is approved, the fit remains failed and its findings remain withheld.

Statistical-model output directories contain fit-time copies of the report templates and partials. Any copied partial or rendered HTML created before this policy is stale even when the source template is corrected. Do not rely on or republish those files: a subsequent audited refit or report-finalisation refresh must recopy the current partials, regenerate `key_findings.json` and rerender before release.

## Consequence for `lrp-rlm-mm-001`

The earlier `lrp-rlm-mm-001` exception is not active. Its sign-off note required a fresh `target_accept=0.99` reporting fit and probe, while the stored fit also fails R-hat and ESS. More fundamentally, the proposed dozens of divergences occur at the factor-correlation boundary and the correlations are the headline estimands, so that proposal does not meet the new “small and headline geometry not implicated” standard. The next step is a cleaner correlation parameterisation and comparison with a higher-order or general-factor structure, not a standing waiver.

## Publication wording

For an unqualified divergent fit:

> Sampling-quality checks failed because divergent transitions remain. Estimates are withheld: a small divergence percentage and otherwise acceptable R-hat, ESS and BFMI do not establish that the sampler explored all posterior mass relevant to the reported estimand.

For a future properly reviewed qualification:

> **QUALIFIED, NOT PASSED:** a small number of divergent transitions remains. All other sampling checks passed, and a trace-bound independent review found no evidence that the named exploratory estimands share the divergent geometry. These findings are provisional and do not constitute a clean convergence pass.
