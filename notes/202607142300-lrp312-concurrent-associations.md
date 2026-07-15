<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).
>
> Substantially edited by a LLM-based AI tool (Codex/GPT-5).

# Concurrent conditional-associations family (LRP-CA, #312)

Date: 2026-07-14

Related: #312, #314 (descriptive-association workstream), #310/#325 (gain-family items-scale marginals; the per-measure-k caveat), #313 (the longitudinal factor-model follow-on), `dag/dag-language-reading.dagitty` (2026-07-10 revision).

## What was added

A new statistical-model family, `kind="concurrent"` / family code `ca`, and its first model `lrp-rli-ca-001` (word reading focal). At each timepoint it fits a between-child Beta-Binomial regression of the focal outcome's _level_ on the standardised same-wave logits of a core skill set (L, B, TR, TE, R, E), plus age and a group nuisance term — the mutually-adjusted concurrent association "at wave t, +n of a predictor is associated with +m of the outcome". Every coefficient is an association; the family makes no causal claim, so conditioning on contemporaneous (post-treatment) skill levels is intentional (unlike the level-factors family, which omits cross-skill terms to protect a causal group×time contrast).

## Design decisions (issue #312 recommendations adopted)

- **Per-wave separate fits, not stacked.** Four cross-sectional fits (one row per child per wave), reported side by side, rather than one stacked model with a child random intercept. The random-intercept version borrows strength but partially converts the coefficients into within-child quantities, changing the estimand; the pure between-child cross-section is the cleaner descriptive object at each wave. Realised as: loop the waves, fit `build_concurrent_model` on each single-wave subset of the `phase_mode="levels"` frame.
- **Diagnostic anchor for the standard artefacts.** The standard pipeline (`trace.nc`, PPC and prior/posterior plots) assumes one model, so the wave with the largest complete-outcome sample (ties → latest) is selected as an operational diagnostic anchor. This rule does not establish that the selected wave has the most statistical power or greater substantive importance. Every adjusted and bivariate fit now has the full R-hat, effective-sample-size, BFMI and divergence gate recorded in `concurrent_fit_diagnostics.csv`; the anchor is also recorded in `config.json` as `diagnostic_anchor_timepoint` and, for backward compatibility, `primary_timepoint`.
- **Group as a flagged nuisance.** Included only to absorb arm composition at each wave (`beta_group_nuisance`, a wide `Normal(0, 1)`), not reported as an association; the report and `config.json` flag it non-interpretable.
- **Regularising priors are load-bearing.** `Normal(0, 0.3)` per-SD slopes: n ≈ 53 with a strongly inter-correlated predictor cluster means the mutually-adjusted coefficients are regularised toward zero. The report carries the Table-2-fallacy caveat (each coefficient answers a different conditional question) and a qualified measurement-error caveat: classical measurement error often attenuates a simple association, but attenuation is not guaranteed in this multivariable nonlinear model. #313 provides a complementary measurement-error-aware analysis.
- **Floored measures (P, N) excluded as predictors.** Off-floor indicators are a possible later extension.
- **Missing predictor values mean-imputed** (0 on the standardised scale — PyMC cannot take NaN inputs). Mean imputation changes the predictor distribution and can bias a conditional coefficient; the direction is not guaranteed when missingness relates to the predictor, outcome or other skills. The output therefore reports the observed and imputed count for each predictor. Rows missing the focal _outcome_ are dropped.

## Natural-scale marginals and output contract

The reader-facing answer ("+n predictor items ↔ +m outcome items") is produced by `reporting.concurrent_marginals` (with a `ConcurrentTerm` descriptor). Because the concurrent model has **no interaction terms**, the perturbation is a scalar shift per posterior draw — `Δη = β · Δz` — so the helper is a small, self-contained pushforward rather than the per-observation interaction machinery of the gain family's `association_marginals`. The predictor enters the model standardised (`β` per SD, with a regularising prior), and the `+k items` row now maps the mean-point item increment with the **same Haldane-corrected logit fitted by the factory**: `h(y, N) = log((y + 0.5) / (N - y + 0.5))` and `Δz = (h(ȳ + k, N) - h(ȳ, N)) / sd_logit`. This corrects the first implementation, which used an ordinary proportion logit and therefore did not exactly match the fitted transformation near the floor or ceiling. `k` is set per measure as `max(1, round(N / 10))` (L→3, B→1, TR/TE→2, R/E→17), so a fixed `+5` does not span 3 %–50 % of scales that differ tenfold.

The issue's requested adjusted/bivariate × logit/probability/items cross-product is now explicit. `concurrent_associations.csv` has one row per wave–predictor pair with adjusted and bivariate logit summaries plus matched `+1 SD` probability- and outcome-items-scale average marginal associations. `concurrent_marginals.csv` is the long-form companion with both fit kinds and both `+1 SD` and predictor-specific `+k items` perturbations. `concurrent_fit_diagnostics.csv` records R-hat, effective sample size, BFMI and divergences for all four adjusted wave fits and all 24 bivariate refits; the selected diagnostic-anchor trace is no longer the only fit with auditable gate quantities.

**Shared-gate scope.** `subfit_convergence` also gates the ITT floor-rule secondary fits, the mediation t3 temporal-ordering sensitivity and ADJ-065's bivariate, prior-sweep and SES sub-fits. Adding BFMI therefore strengthens those future `*_converged` flags too, while `_sample_model` now evaluates its instantiated free variables. Neither change alters posterior estimates, but a published flag may become more conservative when one of those families is next refitted.

## Ids / wiring

`ca` is registered as an **embedded** family (own number space from `ca-001`, legacy `lrpca01`), like the recent `bx` family — added to `FAMILY_BY_KIND` and `_EMBEDDED_FAMILIES` in `model_ids.py`, to `KINDS` and a new `_CA` list in `definitions.py`. There is no central kind→fit dispatch: the spec module calls `fit_concurrent` directly (the family convention). The dev smoke fit used four waves (n = 53/53/53/51) with t3 as the diagnostic anchor; its short chains were adequate for orchestration but not every sub-fit met the full gate, which is why the reporting-tier run below is the acceptance evidence.

## Reporting-tier acceptance run (2026-07-15)

Command: `python scripts/fit_statistical_model.py lrp-rli-ca-001 --config reporting`, writing to `output/statistical_models/models/lrp-rli-ca-001-reporting/`, followed by an HTML render of the copied report. All **28 published fits** passed the complete gate: worst R-hat 1.00059 (threshold ≤ 1.01), minimum bulk/tail effective sample size 16,163 (threshold ≥ 400), minimum per-chain BFMI 0.930 (threshold ≥ 0.3), and zero divergences. The t3 diagnostic-anchor fit separately passed `diagnostics_summary.json` (maximum R-hat 1.00050, minimum effective sample size 25,216, minimum BFMI 0.939, zero divergences). PSIS-LOO classified all 53 anchor-wave observations in the good Pareto-k range (≤ 0.70). The HTML report rendered successfully from these artifacts.

The reporting fit preserves the dev run's direction without turning it into a causal claim. At t3, a +3-item letter-sound difference at the wave's mean letter-sound level was associated with a median **+1.37 word-reading items** after mutual adjustment (95% credible interval +0.26 to +2.52; posterior probability positive 0.992), compared with **+3.04 items** in the predictor-only bivariate fit (95% credible interval +2.06 to +3.95; posterior probability positive 1.000). This adjusted–bivariate difference shows sensitivity to the conditioning set; it is not a decomposition of shared variance and does not show that increasing letter-sound knowledge would cause the word-reading difference.

## Extension: additional focal outcomes (added 2026-07-14, same day)

The extension flagged in issue #312 — letter sounds and taught vocabulary as further focal outcomes — is delivered as `lrp-rli-ca-002` (L), `lrp-rli-ca-003` (TR) and `lrp-rli-ca-004` (TE), each a spec module reusing the unchanged factory/pipeline. The one design decision the extension had to settle is the **predictor set for a non-W focal**: we fixed the family's core skill set as **{W, L, B, TR, TE, R, E}** and each model conditions its focal on the remaining six — i.e. the focal is swapped out of the ca-001 predictor list and word reading swapped in. The alternative (keeping the ca-001 predictor list minus the focal, never admitting W as a predictor) would have made each model's conditional refer to a different joint measure set, losing the family's read as complementary full conditionals of one joint distribution. Two focal-specific notes: the taught pair (TR/TE) stay in each other's predictor sets — strongly correlated, but the regularising priors stabilise the model and the adjusted-versus-bivariate contrast shows sensitivity to the conditioning set rather than decomposing shared variance; and TR approaches its 24-item ceiling at later waves, which the Beta-Binomial respects but which compresses the resolution of `ca-003`'s later-wave associations (noted in its report). Floored P/N stay excluded both as predictors and as focal outcomes. Per-measure `+k items` increments now also cover W as a predictor (`k = max(1, round(79/10)) = 8`).

### Extension reporting sweep (2026-07-15)

Commands: `python scripts/fit_statistical_model.py lrp-rli-ca-00N --config reporting` for N = 2, 3 and 4, followed by direct Quarto renders of the copied reports. The reporting preset used 6 chains with 6,000 tuning and 6,000 posterior draws per chain, `target_accept = 0.95`, and random seed 47. Every model produced the contractually required 24-row `concurrent_associations.csv`, 96-row `concurrent_marginals.csv` and 28-row `concurrent_fit_diagnostics.csv`; every HTML report rendered successfully.

| Model            | Focal outcome                     | Wave n (t1/t2/t3/t4) | Diagnostic anchor | Published fits passing | Worst R-hat | Minimum ESS | Minimum BFMI | Divergences | Anchor gate |
| ---------------- | --------------------------------- | -------------------- | ----------------- | ---------------------: | ----------: | ----------: | -----------: | ----------: | ----------- |
| `lrp-rli-ca-002` | letter sounds (L)                 | 54/54/54/52          | t3                |                  28/28 |    1.000512 |      22,983 |        0.950 |           0 | pass        |
| `lrp-rli-ca-003` | taught receptive vocabulary (TR)  | 54/54/54/53          | t3                |                  28/28 |    1.000848 |      20,052 |        0.917 |           0 | pass        |
| `lrp-rli-ca-004` | taught expressive vocabulary (TE) | 54/54/54/53          | t3                |                  28/28 |    1.000767 |      19,896 |        0.938 |           0 | pass        |

All 84 published extension fits therefore clear the pre-specified thresholds (R-hat ≤ 1.01, effective sample size ≥ 400, BFMI ≥ 0.3 and zero divergences). This is computational acceptance, not evidence that any association is causal or substantively important; interpretation still requires the direction, credible interval and posterior direction probability from the relevant row. Generated fit artefacts remain ignored and are not versioned with this note.

## Extension: standardised-vocabulary focal outcomes (#336, added 2026-07-15)

Issue #336 adds `lrp-rli-ca-005` (standardised receptive vocabulary, `R`) and `lrp-rli-ca-006` (standardised expressive vocabulary, `E`) on the unchanged concurrent factory and pipeline. Each conditions its focal outcome on the other six members of {W, L, B, TR, TE, R, E}, retains age and group as nuisance terms, and inherits the four separate wave fits, `Normal(0, 0.3)` slopes, mean-imputed predictors, adjusted-versus-bivariate outputs and predictor-specific `+k items` marginals. Both focal measures have 170-item denominators and do not carry `ca-003`'s late-wave ceiling-compression warning for the 24-item `TR` outcome.

The `lrp-rli-lcf-001` directed comparison is extended from 32 to 48 rows: in addition to `L` as target versus all four vocabulary indicators and `TR`/`TE` as targets versus `L`/`B`, it now includes `R`/`E` as targets versus `L`/`B`. This preserves the distinction between the latent-domain mean-operating-point translation and the concurrent models' observed-test average marginals; the comparison remains triangulation, not a correction factor or causal test.

The optional blending-focal full conditional is deferred. Its 10-item scale is the coarsest side of the seven-measure system, and it is not needed to answer #336's vocabulary-focused question. The two new reporting-tier fits and the resulting `lrp-rli-lcf-001` comparison refresh belong in the next refit sweep; no numerical result is claimed before those artefacts pass the convergence gate.

## Follow-on

#313 (longitudinal latent factor model) is the complementary measurement-error-aware analysis; it avoids treating observed test scores as error-free predictors but rests on its own measurement-model assumptions.

## Known doc drift (pre-existing, not fixed here)

`docs/models/README.md`'s at-a-glance family counts remain stale relative to `definitions.MODEL_REGISTRY` (now 144), and the `block_exposure` family is missing from the at-a-glance table entirely. The concurrent row and detailed section are present; re-auditing every other family remains a separate docs-sync pass, also flagged on #327.
