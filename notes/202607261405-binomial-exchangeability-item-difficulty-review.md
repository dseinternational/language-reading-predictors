> [!NOTE]
> Clarity and currency edits by a LLM-based AI tool (Codex/GPT-6).

<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Available-case modified ITT terminology updated by a LLM-based AI tool (Codex/GPT-5).

# Review: item-difficulty non-exchangeability, the Beta-Binomial working likelihood, and the negative age association

**Historical analysis, with interpretation corrections added on 21 October 2026.** The numerical probes describe July fits. Current likelihood checks and phoneme-blending pairing requirements are in [METHODS.md](../METHODS.md). Agreement in these probes does not prove that likelihood or measurement assumptions cannot affect a later result.

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5). Records the 2026-07-26 review of whether ordered item difficulty (later test items are harder) undermines the Binomial/Beta-Binomial working likelihoods; the decision to keep the Beta-Binomial with targeted diagnostics and local sensitivities; and exploratory checks of the negative word-reading age association.

## The question

Frank asked (2026-07-26): we use Binomial/Beta-Binomial likelihoods for many outcomes, but the items on these measures are likely not truly exchangeable — later words are more difficult than earlier words on word-reading and vocabulary tests. (1) Is this a concern? (2) Should we adjust the likelihood? (3) Might it explain the negative association with age for word reading — older children are more likely to be tackling more difficult words?

## 1. What ordered item difficulty does and does not threaten

**Sum scores and measurement assumptions.** Under the Rasch model's assumptions, a total score is sufficient for the person's ability given the item calibration (Andersen 1977). This does not establish that these instruments satisfy that model, nor preserve the information needed to diagnose item-specific fit, differential functioning or stopping rules. The available data support a bounded-total working likelihood. Its adequacy still requires predictive checks and targeted sensitivities, as specified in `METHODS.md`.

**Variance: two opposing violations, and the Beta-Binomial can express only one.** Given a child's ability, a sum of unequal-difficulty items is Poisson-binomial, whose variance is _below_ the binomial with the same mean — a steep, near-Guttman ladder makes the score nearly deterministic given ability. Between-child heterogeneity beyond the linear predictor pushes the marginal variance _above_ binomial. The Beta-Binomial (and the logistic-normal-Binomial used by the within-wave levels designs) can only represent at-or-above-binomial variance: it absorbs the second violation and cannot express the first. The observable symptom of unexpressed conditional underdispersion is posterior-predictive **overcoverage** — 50 % / 90 % prediction bands covering more than 50 % / 90 % of observations. The stored `lrp-rli-itt-010` (word reading) reporting fit already shows the signature: its 50 % band covers **62.3 %** of observations (33/53). Follow-up A below pools this statistic across the whole fitted suite from the `ppc_summary.csv` every fit writes.

**Mean shape: the true score-versus-ability curve is not one logistic.** The test characteristic curve — the sum of the item curves — is flatter than a single logistic and kinked where an item bank changes (the `EWRSWR` composite's early-word-reading → single-word-reading boundary; the documented distributional kink above ~25 words, `notes/202607241600-findings-word-reading-bands.md`). For the 170-item vocabulary scales the observed maxima are 82 (`R`) and 77 (`E`), so the logit link's ceiling curvature sits at a ceiling no child approaches: the effective ceiling is ladder steepness and discontinue rules, not `n_trials`. Stopping rules (documented for `P` in `measures.py`; the ROWPVT/EOWPVT/TROG manuals' basal/discontinue rules) additionally make the count a _curtailed_ sum — items beyond the stopping point are scored as failures without being attempted — not `n` attempted trials.

**Interpretation: items are not equal-interval ability units.** Three items at the hard end of a ladder represent more latent progress than three at the easy end. Items-scale effects _within_ a measure remain well defined (both arms take the same instrument, and the reported marginal effect averages over the analysis rows), but items-scale magnitudes are not comparable _across_ measures, and this should remain a stated reporting caveat.

**Interpretation correction, 21 October 2026.** Randomisation supports the assigned-arm comparison under the stated selection and missing-data assumptions. It does not guarantee that a misspecified likelihood or link preserves the sign, magnitude or uncertainty of a model-based marginal effect. The probes below describe sensitivity in their recorded fits; they do not establish invariance for other likelihoods or fits. The floor rule for `P`/`N` (`bernoulli_offfloor`) uses a separate off-floor estimand and requires its own checks.

## 2. Decision: keep the Beta-Binomial; diagnose, then patch locally

A wholesale move to ordinal or item-response likelihoods across the suite would cost a great deal and mostly relabel conclusions. The adopted policy is a ladder:

1. **Diagnose from artefacts we already produce.** Pool the `ppc_summary.csv` 50 % / 90 % coverage across reporting fits by outcome (follow-up A; `scripts/ppc_coverage_sweep.py`). Systematic overcoverage on the laddered tests is the conditional-underdispersion signature; undercoverage is ordinary shape misfit. Small-denominator caveat: central intervals of a discrete count distribution overcover mechanically (a "50 %" interval on a 10-item score covers at least 50 %), so measures of similar length are compared with each other, not against an absolute 0.50.
2. **One principled link fix regardless of diagnostics: blending.** `B` is ten three-alternative forced-choice items, so the expected score cannot fall below chance (≈ 3.3 of 10), yet the standard logit link happily models sub-chance means. A guessing-floor link, mean = 1/3 + (2/3) · logit⁻¹(η), is mechanically justified (follow-up B prototypes it).
3. **If conditional underdispersion is material: the Conway–Maxwell-binomial.** The CMB (Kadane 2016) is the natural drop-in that can express _under_- as well as over-dispersion relative to the Binomial (its extra parameter ν: ν = 1 Binomial, ν > 1 underdispersed, ν < 1 overdispersed), and its normalising constant is a finite sum over 0…n, so a custom PyMC log-probability is cheap even at n = 170 (follow-up C prototypes it for `W`).
4. **If mean-shape misfit is material: a flexible-monotone own-baseline term.** The HSGP machinery and the mechanism-family knee tests are the in-house precedent. Section 3 below suggests low urgency for the age question specifically.
5. **Ordinal (continuation-ratio — the natural generative model for a ladder with discontinue rules) or full item-response modelling only as a last resort.** Item-level responses are absent from `rli_data_long.csv` (block totals only). Settled same day: the original scoring sheets are not readily available, closing this route — see §7.

## 3. Sensitivity of the negative word-reading age association

The suite-wide finding under test: conditional on own baseline, linear age is credibly negative for word reading — **−0.12** on the logit scale in the stacked gain-factors models (P(< 0) = 0.99; [202607082140-statistical-models-full-reporting-fit.md](https://github.com/dseinternational/language-reading-predictors/blob/b62fd6ce093c2cb81e5c842cf96ff11ffa6e4d95/notes/202607082140-statistical-models-full-reporting-fit.md), [202607111100-replite-full-statistical-fit.md](https://github.com/dseinternational/language-reading-predictors/blob/b62fd6ce093c2cb81e5c842cf96ff11ffa6e4d95/notes/202607111100-replite-full-statistical-fit.md)), pointing the same way in the single-window ITT (`lrp-rli-itt-010` stored reporting trace: `gamma_A` median **−0.08**, 89 % [−0.21, +0.05], P(< 0) = 0.83 — one transition, so wider), **−0.26** in the adjusted-association model `adj-065`, and **−0.34** in the errors-in-variables mechanism fit (`notes/202607141700-lrp228-eiv-mechanism.md`). The hypothesis: older children sit higher on the item ladder, where words are harder, so the association is a measurement artefact.

**Measurement question.** Equal observed baseline totals do not establish equal latent ability or equal measurement error. The recorded age–baseline correlation is +0.37 in the 53-child randomised-window sample. Baseline curvature is one possible route by which measurement assumptions can affect the conditional age slope; these checks do not exhaust the possibilities.

**Real-data check (t1 → t2, n = 53).** Regressing W post-score on baseline + age + arm, then adding hinge terms at 25 and 30 items (the kink region), the age slope barely moves: **−1.06 → −1.02 items per SD of age** (SE ≈ 0.55) on the raw scale; **−0.066 → −0.028** (SE ≈ 0.14) on the empirical-logit scale; the hinge estimates are small relative to their standard errors (±0.4, SE ≈ 1.0). Only three children start above 25 items, so power above the kink is minimal — but the gradient-boosting step, which fits the baseline fully flexibly with trees, reaches the same negative age ranking for word-reading gain ([historical record](https://github.com/dseinternational/language-reading-predictors/blob/043c72db2211b9cf067fd8d1b076fda8fe597dc9/notes/202607021100-word-reading-reporting-tier-ranking.md)), which provides a separate predictive check and does not rule out measurement error or confounding.

**Simulation under the null.** `scripts/age_artefact_check.py` generates Rasch-type data from three 79-item ladders — kinked (30 easy early-word-reading items, then a single-word-reading difficulty jump), smoothly graded, and homogeneous — calibrated to the observed baseline mean/SD, corr(age, ability) = 0.37, mean gain ≈ 3.4 items, and **zero true age effect on growth**, then fits the ITT-style regression (1,000 replicates, n = 53, seed 20260726):

| ladder            | age coefficient (empirical-logit scale), mean [2.5, 97.5 %] | P(coef ≤ −0.12) |
| ----------------- | ----------------------------------------------------------- | --------------- |
| kinked (EWR/SWR)  | **+0.175** [−0.090, +0.459]                                 | 0.02            |
| smoothly graded   | **+0.184** [−0.073, +0.452]                                 | 0.01            |
| homogeneous items | **+0.218** [−0.074, +0.524]                                 | 0.01            |

Every ladder produces a **positive** mean age coefficient under the null (≈ +0.4 to +0.5 items per SD of age on the raw scale). The dominant channel runs the other way from the hypothesis: with a noisy baseline, age is a proxy for true ability, and higher true ability means a higher post-score — an errors-in-baseline channel that biases the age coefficient _upwards_, which the difficulty ladder does nothing to reverse. The chance that this data-generating process yields the observed −0.12 is 1–2 %.

**Cross-model comparison.** The recorded slopes point in the same negative direction, but the families use different fitted rows, scales, adjustments and measurement assumptions. Their numerical ordering does not diagnose the direction or amount of measurement-error bias. Agreement is descriptive sensitivity evidence, not independent replication.

**Conclusion.** The simulated ladders did not reproduce the observed negative age association on average under their chosen data-generating assumptions. This narrows that particular explanation; it does not rule out likelihood misspecification, other measurement mechanisms or residual confounding, and it does not establish a more negative true age effect. The observed relationship remains conditional on baseline and the fitted population. These checks do not distinguish developmental timing from selection of children with different prior trajectories.

**Caveats.** The quick regressions use empirical-logit ordinary least squares with naive standard errors on a single randomised window; the simulation's ability distribution is moment-matched on a coarse grid. These are direction-and-magnitude checks, not replacement estimates; the registered models remain the authoritative quantities.

## 4. Follow-up A — posterior-predictive coverage sweep (results)

`scripts/ppc_coverage_sweep.py` pooled 380 `ppc_summary.csv` coverage rows from **190 stored reporting fits** (2026-07-26; three output directories lacked the files). Headline: 50 % prediction bands **overcover everywhere** — pooled coverage by outcome runs **0.65 (`T`) to 0.92** (the historical-cohort measures), with `W` at **0.76** across 66 models, `L` 0.71, `R` 0.68, `E` 0.75, `B` 0.74, `TE` 0.80, `N` 0.82. By family the cross-sectional ITT suite is mildest (**0.66**; `aligned` 0.63, `corr_factor` 0.58 closest to nominal) and the longitudinal random-intercept families largest (`did` 0.82, `lcsm` 0.85, `historical_growth` 0.85). The 90 % bands sit much closer to nominal (0.95–1.00). The off-floor mode has 2–31 group cells per level and is uninformative about dispersion.

Interpretation requires the two mechanical inflators named in §2: equal-tailed intervals on a _discrete_ count overcover by construction (worst for short scales and for the floor-concentrated score distributions, where the modal low counts sit inside any central band), and the longitudinal families' in-sample checks condition on fitted child effects. The pointed question — is the remaining overcoverage evidence that the Beta-Binomial's variance floor binds (conditional underdispersion)? — is answered by follow-up C: **no**. A dispersion family free to go below binomial variance chooses overdispersion for `W`, and its coverage barely improves (0.62 → 0.60). The 50 %-band overcoverage is therefore substantially mechanical, not a likelihood defect; the sweep stays useful as a _relative_ monitor (a measure drifting away from its peers) after future full runs.

## 5. Follow-up B — blending guessing-floor link prototype (results)

`scripts/likelihood_sensitivity_prototypes.py blending-chance-floor` refits the `lrp-rli-itt-008` structure (n = 54, 10 items) twice with shared priors and sampler (4 chains × 2,000 draws, `nutpie`, `target_accept` 0.92, seed 20260726; both fits: R-hat ≤ 1.002, min ESS ≥ 4,137, 0 divergences — comfortably inside the suite gate), differing only in the link. Validation: the standard arm reproduces the stored reporting fit (τ **+0.44** here vs **+0.45** stored; items-scale effect **+0.96** vs **+0.99**).

| model                                       | τ (logit), median [89 %] | treatment effect (items), median [89 %] | P(> 0) |
| ------------------------------------------- | ------------------------ | --------------------------------------- | ------ |
| standard Beta-Binomial                      | +0.44 [+0.09, +0.81]     | **+0.96** [+0.20, +1.78]                | 0.98   |
| chance-floor (mean = 1/3 + 2/3 · logit⁻¹ η) | +0.40 [−0.14, +0.92]     | **+0.49** [−0.16, +1.10]                | 0.90   |

Respecting the guessing floor roughly **halves the items-scale treatment effect and weakens the evidence from strong to moderate** — under the chance-floor link, movement in the sub-chance region is attributed to guessing noise rather than ability, and what remains is the contrast on the above-chance share. Against the adopted ROPE δ(B) = 1 item, the standard headline sits at δ while the chance-floor estimate sits at about δ/2. PSIS-LOO cannot separate the two links (elpd difference 1.0 with difference-SE 2.7 — a tie), so the data do not adjudicate; the argument for the chance-floor link is mechanistic (the instrument is three-alternative forced choice), not predictive. Side-observations: `gamma_own` rises to ≈ 1.0 under the chance-floor link (closer to the autoregressive prior's centre), and predictive coverage is identical between links (0.63 / 0.93), so this is purely an estimand-scale sensitivity, not a fit improvement.

## 6. Follow-up C — word-reading Conway–Maxwell-binomial prototype (results)

`scripts/likelihood_sensitivity_prototypes.py word-reading-cmb` refits the `lrp-rli-itt-010` structure (n = 53, 79 items; same shared priors and sampler; both fits: R-hat ≤ 1.002, min ESS ≥ 1,494, 0 divergences), swapping the Beta-Binomial for the Conway–Maxwell-binomial with ν ~ LogNormal(0, 0.5). Validation: the standard arm reproduces the stored reporting fit to three decimals (τ +0.354 both).

| model                   | τ (logit), median [89 %] | treatment effect (items), median [89 %] | P(> 0) | dispersion                | 50 % / 90 % coverage |
| ----------------------- | ------------------------ | --------------------------------------- | ------ | ------------------------- | -------------------- |
| standard Beta-Binomial  | +0.35 [+0.12, +0.60]     | **+2.38** [+0.77, +4.01]                | 0.99   | κ = 48 [30, 76]           | 0.62 / 0.94          |
| Conway–Maxwell-binomial | +0.20 [+0.07, +0.34]     | **+2.49** [+0.96, +4.02]                | 0.99   | **ν = 0.50 [0.37, 0.67]** | 0.60 / 0.89          |

In this probe, ν is below 1, supporting overdispersion relative to the Binomial under the fitted Conway–Maxwell-binomial model. This does not exclude conditional underdispersion under a different model or in another family. The two items-scale medians, +2.38 and +2.49, and their intervals are similar; they are not identical. The comparison supports stability for these two fitted specifications, not invariance to likelihood choice in general. The logit coefficients describe different mean parameterisations, so their magnitudes do not provide a common effect scale.

## 7. Would a Rasch model help? (assessed and closed)

An item-response model could support checks of the test characteristic curve, measurement error and differential item functioning if suitable item records and defensible measurement assumptions were available. Rasch sufficiency under that model does not show that a new measurement model would preserve the headline treatment results. Nor does one count-family comparison establish how a Rasch model with guessing or stopping rules would behave. Calibration would also need adequate support across item difficulty; the historical sample and discontinue rules limit that support. No such calibration was performed here.

**Closed 2026-07-26: Frank confirmed the original scoring sheets are not readily available**, so no item-level response matrix can be assembled without a substantial archival effort that the expected gains do not justify. The end state for this reanalysis is therefore the Beta-Binomial working likelihood, the items-scale reporting caveat, and the blending chance-floor sensitivity. If item-level records ever do surface, the value-ordered agenda recorded here is: (1) a one-off Rasch calibration of `W` to pin the test characteristic curve and re-express the ≈ 2.4-item effect in ability logits; (2) Rasch-based measurement error feeding the mechanism/mediation couplings; (3) arm-DIF screening on `R`/`E` — and nothing suite-wide.

## Decisions

Recorded 2026-07-26; owners in brackets.

1. **Keep the Beta-Binomial as the default working likelihood** for bounded-count outcomes. The CMB probe found overdispersion (ν = 0.50) and similar items-scale treatment summaries under the two fitted specifications. No suite-wide likelihood change.
2. **Adopt `scripts/ppc_coverage_sweep.py` as a standing diagnostic** after full reporting sweeps, read _relatively_ (a measure drifting from its peers), with the discreteness and in-sample-random-effect caveats stated in §4. [rerun with future sweeps]
3. **Blending (`B`) headline is link-sensitive and needs a stated sensitivity.** The ≈ 1-item, strong-evidence available-case modified ITT estimate halves to ≈ 0.5 items, moderate evidence, under the mechanically-motivated guessing-floor link, and LOO cannot separate the links. Whether to promote the chance-floor variant to a registered model (a `link=` option in `factories.py` plus guard tests) — or at minimum to caveat `B`'s items-scale estimate and its ROPE reading in reports — is a substantive call. [Frank / education lead]
4. **No Conway–Maxwell-binomial adoption; the ordinal / item-response route is closed.** Settled 2026-07-26: Frank confirmed the original scoring sheets are not readily available, so no item-level data can be assembled for a Rasch or ordinal measurement model (§7). Revisit only if item-level records surface, per the value-ordered agenda in §7. [closed]
5. **Propose one reporting-guardrail sentence for `METHODS.md`**: items-scale effects are averages over the realised score distribution; items are not equal-interval ability units and are not comparable across measures. Not made in this change. [future docs change]
6. **The negative word-reading age association stands as substantive** (§3): not a difficulty-ladder or likelihood artefact; developmental-timing versus trajectory-selection remains unresolvable cross-sectionally. Attribution corrected: −0.12 (P = 0.99) is the stacked gain-factors figure; the single-window ITT posterior is −0.08 (P(< 0) = 0.83).

## References

- Andersen, E. B. (1977). Sufficient statistics and latent trait models. _Psychometrika_, 42(1), 69–81. DOI [10.1007/BF02293746](https://doi.org/10.1007/BF02293746)
- Kadane, J. B. (2016). Sums of possibly associated Bernoulli variables: the Conway–Maxwell-binomial distribution. _Bayesian Analysis_, 11(2), 403–420. DOI [10.1214/15-BA955](https://doi.org/10.1214/15-BA955)
- Burgoyne, K., Duff, F. J., Clarke, P. J., Buckley, S., Snowling, M. J., & Hulme, C. (2012). Efficacy of a reading and language intervention for children with Down syndrome: a randomized controlled trial. _Journal of Child Psychology and Psychiatry_, 53(10), 1044–1053. DOI [10.1111/j.1469-7610.2012.02557.x](https://doi.org/10.1111/j.1469-7610.2012.02557.x)
