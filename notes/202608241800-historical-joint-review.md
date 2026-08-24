<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Historical-joint family review, 2026-08-24

Preliminary research data and models — all conclusions remain provisional.

- **Scope:** `kind="historical_joint"` — `lrp-rlm-jc-001` (between-child correlated growth), `lrp-rlm-jc-002` (within-child coupling) and `lrp-rlm-jc-102` (its registered wider-prior companion) — followed from the typed settings and run plan through the panel loader, the PyMC factory, the pipeline summaries, the diagnostics, the release decision, the key-findings builder and the report templates, and checked against the stored reporting-tier fits.
- **Context:** the family was last reviewed on 2026-08-21 (`notes/202608211820-historical-growth-and-historical-joint-code-review.md`) and again inside the 2026-08-23 joint audit ([#588](https://github.com/dseinternational/language-reading-predictors/issues/588), findings 5, 8 and 11 plus reporting items), remediated in PR [#609](https://github.com/dseinternational/language-reading-predictors/pull/609).

## Verdict

The statistical construction is sound. What this review found is a **release-control gap around the one parameter the within-child headline depends on**, and a general reporting defect that would have hidden the fix: a qualification attached to a publishable fit reached no reader at all. Eight findings, all fixed here; none required a refit.

## What was verified as correct

- **The double centring is exact, and it does not attenuate the reported correlations.** The factory centres the within-child deviations on the subject mean and then on the group-by-wave cell mean — sequentially, which is only exact double centring on a balanced panel. The factory enforces balance (one row per child-wave, one common wave set) before building, and on that design the second sweep provably leaves the subject means at zero. Both projections are applied identically to every measure, so the entrywise cross-measure correlation is preserved exactly; the same holds for the group-centring of the stable offsets. The report's claim that "the correlations themselves are unaffected by the centring" is correct.
- `LKJCorr` is treated throughout as returning the **Cholesky factor**, with `R = L Lᵀ`, matching this PyMC version.
- The three likelihood nodes are distinct observation sets, so summing them for power scaling is not the double count that `y_post_child` produced in the adjusted family (#572).
- `psense_summary.csv` covers every headline parameter in both fits, including `sigma_within` and both correlation-pair vectors.
- Both stored fits pass the convergence gate cleanly (0 divergences; max R̂ 1.0029 and 1.0043).

## Findings

### 1. jc-002 published a prior-dependent classification with nothing saying so

For the within-child fit, the result _is_ a classification: which measures clear the 0.05-logit resolvability threshold, and therefore which correlations may be interpreted at all. On the stored fit only `basread` clears it (P = 1.00 against 0.29 for `bpvs` and 0.44 for `basdig`), so the published headline is "the model did not resolve a within-child correlation".

That classification is decided by `sigma_within` — and the fit's own power scaling makes `sigma_within` the most prior-sensitive quantity in the model (prior sensitivity 0.65, 0.25 and 0.18 against ArviZ's 0.05 flag). The registered sensitivity that would test it, `lrp-rlm-jc-102`, has never been fitted. The family is descriptive, so `gate_applies` excludes it and no robustness verdict is produced at all; `publication_qualification` was empty; and the key-findings box's **unresolved** branch carried no prior caveat, while the resolvable branch did.

**Fixed.** `HISTORICAL_JOINT_PRIOR_COMPANIONS` registers the pairing beside `JOINT_DEPENDENCE_COMPANIONS` and `BLENDING_LINK_MODELS`, with a drift test asserting that the companion differs from its parent in `model_id` and `sigma_within_prior_sigma` and nothing else. `release._historical_joint_prior_companion_qualifications` binds the pair on six fields (measures, window, likelihood, the priors _not_ under test, input checksum, fitted-row identity), rejects a companion fitted under the same prior as varying nothing, fails closed on every unreadable path, and when the pair does bind compares the two fits' resolvability classifications and qualifies on any measure that moves. The qualification quotes the fit's own measured prior sensitivity rather than asserting that the prior matters. A sentence in the unresolved findings branch now says non-resolution is a conclusion under this prior.

Recomputed over the stored fits: `jc-001` is untouched; `jc-002` now carries the qualification naming `jc-102`, the reason (never fitted) and the measured 0.65.

### 2. A qualification on a publishable fit reached no reader

`publication_qualification` was rendered in exactly one place — inside `_gate_badge.qmd`'s `development_only is True` branch. On a reporting-tier fit, which is the only kind that publishes, the field was written to `release_decision.json` and shown to nobody. That silently applied to #610's joint-mechanism new-child coverage floors as well, and would have made finding 1's fix inert.

**Fixed.** The badge now renders a "Published with a qualification" caution for a publishable fit that carries one, as an `elif` so a development fit still shows one banner rather than two.

### 3. The retired "multiple likelihood nodes" reason survived in three places

#609 corrected the run plan's `loo_reason` and the report partial. The claim that one likelihood node per measure makes a pointwise LOO undefined — which the audit showed is wrong, since the nodes share an observation coordinate — remained in `lrp_rlm_jc_001`'s module docstring and in both the module and function docstrings of `pipelines/historical_joint.py`, where it sat a few lines above an inline comment stating the corrected reason. **Fixed**; the run plan is named as the statement of record.

### 4. The within-child scale is pooled across groups and nothing said so

The between-child SD is group-indexed by explicit decision (2026-07-16, taken because the Down syndrome group's spread differs) and the overdispersion is too; the within-child scale `ω_m` is pooled across all three reading groups. The asymmetry is visible in the report's formula and remarked on nowhere, while the shared _correlation matrices_ beside it are flagged as a parsimony assumption twice. It is load-bearing: a pooled scale is driven by whichever group carries most wave-to-wave signal, so a group whose departures differ cannot show it — and the scale is what the classification turns on. **Fixed** by stating it in the run plan's design and in the jc-002 report.

### 5. The resolvability rule is applied to the pre-projection scale

The rule's justification is about what the linear predictor carries ("a 0.05-logit SD moves a probability by at most 1.25 percentage points"), but it is applied to `sigma_within`, which the double sum-to-zero sweep makes about 20% larger than the departures actually carried. The rule is therefore lenient by that factor. No current classification changes either way. **Fixed** by publishing `realised_prob_above_minimum` beside it and saying in the report which scale the rule uses and why.

### 6. The between-child pair table broke the median convention

The table printed at fit time led with `mean` while the within-child table immediately below it led with `median`, and the report partial states the median convention explicitly. **Fixed**; `mean` stays in the CSV for the correlation _matrix_, where entrywise averaging preserves positive semidefiniteness.

### 7. An unread median-child growth deterministic

`growth_first_last_items_{m}` is written on every historical-joint fit and read by nothing. It is the growth for a _mid-group_ child, built from `eta_cell` with the offset at zero, whereas the family's published growth is the matched-children average taken from `fitted_mean_items_obs_{m}` — two different quantities under similar names, one published and one silently present in the trace. Retained for parity with the single-measure `historical_growth` family, which does summarise it, and **documented** as not this family's growth result.

### 8. `jc-102` named a parameter that does not exist

Its docstring told the reader that `omega_m` decides the resolvability rule. `ω_m` is the report's mathematical symbol; the fitted model and every artefact call it `sigma_within`. **Fixed.**

## Observations recorded, not acted on

- **jc-001's headline correlations are themselves power-scaling flagged** — prior sensitivity 0.065 to 0.095 against the 0.05 threshold, with high likelihood sensitivity beside it. That is mild and there is no registered between-scale prior companion to bind, so no qualification is raised. If the family gains one, the machinery added here extends to it directly.
- **The stored reporting artefacts are stale** relative to the code in two known ways: `within_scale_summary.csv` predates the `realised_departure_sd_*` columns, and the directory still holds a `loo_pit.png` written before `include_loo_pit=False`. Both clear on the next refit, which is deferred by decision.

## Verification

- Full suite green; `ruff check src/`, `npm run format:check` and `npm run spellcheck` clean.
- Release decisions recomputed over both stored fits: `jc-001` unchanged, `jc-002` gains the qualification and stays publishable.
- A dev-tier `lrp-rlm-jc-002` fit exercised the changed pipeline paths end to end.
- New tests: the companion drift test, twelve release tests (absent, withheld, same-prior, six tampered binding fields, reclassification, persistence and the out-of-scope between-child fit), the findings-box caveat and the two report-source assertions.
