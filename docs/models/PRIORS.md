> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Prior Rationale and Inventory

This is a working prior-audit document for issue
[#141](https://github.com/dseinternational/language-reading-predictors/issues/141).
It is intentionally plain Markdown rather than Quarto-optimised report prose. The
technical report can later lift the stable parts into narrative form.

The immediate purpose is to make the current prior surface auditable:

- identify the priors currently used by the statistical-model factories;
- classify each prior by modelling role;
- separate shared, reportable constructors from inline or family-specific priors;
- record the first-pass rationale and the scale checks still needed.

## Source Of Truth

The main source of truth is
`src/language_reading_predictors/statistical_models/priors.py`, especially
`ALL_PRIORS`, `_ROLE_BY_CTOR`, `_RV_TO_CTOR`, and `_INLINE_PRIORS`. Per-fit
`priors_table.csv` files are generated from the actual PyMC model by
`priors.priors_table()` and emitted by `pipeline._emit_priors()`.

This inventory was generated on 2026-06-29 from:

- the named prior registry in `priors.py`;
- direct `pm.Normal`, `pm.HalfNormal`, and `pm.LKJCholeskyCov` priors found in
  `statistical_models/factories.py`;
- HSGP helper defaults in `statistical_models/hsgp.py`;
- bounded-count denominators in `statistical_models/measures.py`;
- the requested review scope in GitHub issue #141.

Rows marked `coverage gap` are used in model factories but are not yet fully
represented by a named constructor or clean `priors_table.csv` mapping.

## Prior-Setting Protocol

These models should use weakly informative, scale-calibrated priors, not
"uninformative" priors. In bounded-count logit models, very broad coefficient
priors can imply implausibly large probability- or item-scale effects.

Allowed inputs for prior choice:

- measurement ceilings, score bounds, and test-scale properties;
- external literature and previous intervention evidence;
- expert judgement from the education and research team;
- pre-specified substantive constraints, such as monotone baseline continuity;
- computational diagnostics, when clearly labelled as a sampling-stability
  constraint.

Not allowed as primary prior information:

- posterior results from the same fitted model;
- tightening a prior only because a posterior estimate is inconveniently
  uncertain;
- outcome-specific tuning that is not documented as sensitivity or a modelling
  decision.

## Role Definitions

| Role               | Meaning                                                                                                                          |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `causal`           | Prior on a term whose estimand is identified by randomisation or the locked design.                                              |
| `precision`        | Prior on a covariate included to sharpen a causal estimate without changing the causal claim.                                    |
| `association`      | Prior on an adjusted, non-randomised coupling. These coefficients must not be reported as "drives" or causal effects.            |
| `nuisance`         | Prior on intercepts, dispersion, random-effect scales, non-centred offsets, latent-state scales, or other supporting parameters. |
| `gp`               | Prior on a flexible-function component, usually an HSGP amplitude or lengthscale.                                                |
| `sensitivity-only` | Prior used only in an explicitly labelled sensitivity model or optional branch.                                                  |

## Generated Inventory Table

The named-constructor rows below were generated from `ALL_PRIORS` and
`_ROLE_BY_CTOR` in `priors.py`. Inline and family-specific rows were generated
from the factory search described above.

| Prior key                 | Distribution                                     | Role                                                     | Representative terms                                                                                                                             | Current rationale                                                                                                                                                                                                  | Review status                                                                                                                                               |
| ------------------------- | ------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `alpha`                   | `Normal(0, 1.5)`                                 | `nuisance`                                               | `alpha`; also mediation intercept aliases such as `a0`, `b0`, `aL0`, `aE0`                                                                       | Intercept on the logit scale when standardised predictors are at their reference values. A 95% Normal range is about -2.9 to +2.9 logits, or baseline probabilities near 0.05 to 0.95 before covariate effects.    | Check prior-predictive baseline score distributions. Alias mapping is a coverage gap for some mediation names.                                              |
| `tau`                     | `Normal(0, 0.5)`                                 | `causal` by default; sometimes `association` by override | `tau`, `beta_G`, `beta_period`, `delta`, `beta_trt`, `b_grp_time`, `beta_grp`, `a_G`, `b_G`; `beta_cohort` and dose terms by contextual override | Centred at no effect. A 95% Normal range is about +/-0.98 logits, or odds ratios about 0.38 to 2.66. Intended to be weakly informative on the logit scale while avoiding very large item-scale effects by default. | Highest-impact review item. Run sensitivity with SD in `{0.25, 0.5, 0.75}` and push forward to probability and item scales.                                 |
| `gamma_own`               | `Normal(1, 0.5)`                                 | `precision`                                              | `gamma_own`, `a_L`, `a_comp`, `b_W`                                                                                                              | Encodes baseline continuity: a child's own pre-score is expected to predict the post-score on the same logit scale, but with meaningful uncertainty.                                                               | Compare `Normal(1, 0.25)` and `Normal(1, 0.5)` in representative models. Check whether the prior is too strong for noisy or floored baselines.              |
| `gamma_cross`             | `Normal(0, 0.3)`                                 | `association`                                            | `gamma_*`, `b_*`, `a_*`, interaction terms, ability terms, dose-stage terms                                                                      | Shrinks cross-skill, covariate, interaction, and adjusted-association terms toward zero. A 95% Normal range is about +/-0.59 logits, or odds ratios about 0.55 to 1.80.                                            | Run sensitivity with SD in `{0.2, 0.3, 0.5}` for high-impact adjusted couplings and interactions.                                                           |
| `gamma_age`               | `Normal(0, 0.3)`                                 | `precision`                                              | `gamma_A`                                                                                                                                        | Linear age term on standardised age. In the ITT suite, age is a precision covariate because the randomised effect is identified by the empty adjustment set.                                                       | Confirm whether all non-ITT uses should remain labelled precision or be reported as adjusted association.                                                   |
| `kappa`                   | `HalfNormal(50)`                                 | `nuisance`                                               | `kappa`, `kappa_M`, `kappa_Y`; LCSM `kappa` uses the same sigma by default                                                                       | Beta-Binomial concentration. Larger values reduce overdispersion; smaller values allow extra-binomial variability.                                                                                                 | Review if posterior predictive checks show dispersion, floor, or ceiling mismatch. Consider alternatives only if dispersion materially affects conclusions. |
| `predictor_slope`         | `Normal(0, sigma)`, default `sigma = 0.5`        | `association`                                            | LRP65 `beta_{predictor}` terms                                                                                                                   | Per-SD coefficient on mutually adjusted baseline predictors in the between-child gain model. Default matches the treatment-effect scale but is explicitly associational.                                           | LRP65 already includes sensitivity at `sigma = 0.3` and `0.7`; document stability of clear-zero conclusions.                                                |
| `eta_main`                | `HalfNormal(0.3)`                                | `gp`                                                     | `f_A__eta`, `f_ypre__eta`, other main-effect HSGP amplitudes                                                                                     | Shrinks optional nonlinear main effects toward zero after earlier GP fits showed weak identification and divergent-transition risk.                                                                                | For any active nonlinear fit, inspect prior-predictive functions and sensitivity to amplitude.                                                              |
| `eta_tau`                 | `HalfNormal(0.3)`                                | `gp`                                                     | `g_tauA__eta` and other treatment-effect modifier amplitudes                                                                                     | Tight prior on nonlinear treatment-effect heterogeneity, keeping the default close to the constant-effect model.                                                                                                   | Use only in explicit heterogeneity sensitivity models unless substantively justified.                                                                       |
| `ell`                     | `InverseGamma(3, 1)`                             | `gp`                                                     | `f_*__ell`, `g_*__ell`                                                                                                                           | HSGP lengthscale on standardised inputs. Works with the basis-size and boundary-factor calibration in `hsgp.py`.                                                                                                   | Review jointly with `eta_*`; lengthscale checks should use function-scale prior draws, not only parameter plots.                                            |
| `eta_partial_pool`        | `HalfNormal(0.3)`                                | `gp`                                                     | joint-model outcome-specific age-GP deviation amplitudes                                                                                         | Allows outcome-specific nonlinear deviations in the joint age-GP branch while shrinking toward the shared smooth effect.                                                                                           | Sensitivity-only unless age-GP branches are reactivated.                                                                                                    |
| `beta_mech`               | `Normal(0, 1)`                                   | `association`                                            | `beta_mech`, dose-response `beta_dose`, `mu_dose` by override                                                                                    | Per-SD mechanism or dose slope. A 95% Normal range is about +/-1.96 logits, or odds ratios about 0.14 to 7.10, so this is materially wider than `gamma_cross`.                                                     | Review whether the unit-scale prior is too broad for bounded high-denominator outcomes.                                                                     |
| `b_path`                  | `Normal(0, 1)`                                   | `association`                                            | `b_M`, `b_L`, `b_E`                                                                                                                              | Mediator-to-outcome slope in mediation models, on a standardised mediator scale. Wide enough to let the decomposition be data-informed but still regularising.                                                     | Translate to outcome-scale mediated effects before final reporting.                                                                                         |
| `sigma_mediator`          | `HalfNormal(1.0)`                                | `nuisance`                                               | `sigma_M`                                                                                                                                        | Residual SD for the Gaussian mediator in the reading-route mediation model. The mediator is standardised, so 1.0 is weakly informative.                                                                            | Check prior predictive mediator distribution.                                                                                                               |
| `sigma_dose`              | `HalfNormal(0.5)`                                | `nuisance`                                               | `sigma_dose`                                                                                                                                     | Between-period SD for partial-pooled dose slopes; shrinks period-specific dose effects toward the pooled dose effect.                                                                                              | Review in period-varying dose models with the `gamma_cross` / dose-slope sensitivity grid.                                                                  |
| `alpha_phase`             | `Normal(0, 0.5)`                                 | `nuisance`                                               | `alpha_phase`                                                                                                                                    | Per-phase intercept offset in repeated-period models.                                                                                                                                                              | Covered by `_INLINE_PRIORS`; check prior predictive period-to-period score shifts.                                                                          |
| `alpha_time`              | `Normal(0, 0.5)`                                 | `nuisance`                                               | `alpha_time`                                                                                                                                     | Per-timepoint intercept offset in level-factor models.                                                                                                                                                             | Covered by `_INLINE_PRIORS`; check implied trajectories over four waves.                                                                                    |
| `sigma_child`             | `HalfNormal(0.5)`                                | `nuisance`                                               | `sigma_child`                                                                                                                                    | Child random-intercept SD for repeated-measure models.                                                                                                                                                             | Covered by `_INLINE_PRIORS`; inspect whether random-intercept variation dominates treatment or association terms.                                           |
| `beta_dose_phase_raw`     | `Normal(0, 1)`                                   | `nuisance`                                               | `beta_dose_phase_raw`                                                                                                                            | Standard-normal non-centred offset, scaled by `sigma_dose`, for period-specific dose slopes.                                                                                                                       | Covered by `_INLINE_PRIORS`; no substantive interpretation without `mu_dose` and `sigma_dose`.                                                              |
| `joint_residual_corr`     | `LKJCholeskyCov(eta=4, sd_dist=HalfNormal(0.5))` | `nuisance`, `sensitivity-only`                           | `chol`, `corr`, `sigmas`, `z_raw` in the optional joint residual-correlation branch                                                              | Optional residual-correlation block in the joint model. It is off by default after earlier fits suggested weak identification and prior dominance.                                                                 | Coverage gap. If reactivated, add named documentation for the LKJ shape, residual SD, and non-centred residual offsets.                                     |
| `lcsm_mu1`                | `Normal(wave_1_anchor, 1.0)`                     | `nuisance`                                               | LRP67 `mu1`                                                                                                                                      | Initial latent logit true-score mean, anchored to the observed wave-1 mean by outcome.                                                                                                                             | Coverage gap. Document anchor construction and prior predictive initial score distributions.                                                                |
| `lcsm_sigma1`             | `HalfNormal(1.0)`                                | `nuisance`                                               | LRP67 `sigma1`                                                                                                                                   | Initial latent between-child SD on the logit scale.                                                                                                                                                                | Coverage gap. Check whether it is too wide for low-denominator outcomes.                                                                                    |
| `lcsm_a_change`           | `Normal(0, 1.5)`                                 | `nuisance`                                               | LRP67 `a_change`                                                                                                                                 | Latent change intercept by outcome. Wide on the logit-change scale.                                                                                                                                                | Coverage gap. Translate to expected wave-to-wave count changes.                                                                                             |
| `lcsm_b_self`             | `Normal(0, 0.5)`                                 | `association`                                            | LRP67 `b_self`                                                                                                                                   | Self-feedback/proportional-change term in the latent change recursion.                                                                                                                                             | Coverage gap. Review sign and magnitude implications for stability versus regression to the mean.                                                           |
| `lcsm_d_age`              | `Normal(0, 0.5)`                                 | `association`                                            | LRP67 `d_age`                                                                                                                                    | Age effect on latent change.                                                                                                                                                                                       | Coverage gap. Check against maturation expectations.                                                                                                        |
| `lcsm_g_cross`            | `Normal(0, 0.5)` by default                      | `association`                                            | LRP67 `g_L`, `g_E`, and other cross-measure couplings into reading change                                                                        | Within-trajectory coupling from prior skill level to later reading change. This is the LCSM analogue of adjusted predictor slopes.                                                                                 | Coverage gap and high-priority review item for LRP67. Consider sensitivity around `0.3`, `0.5`, and `0.7`.                                                  |
| `lcsm_sigma_proc`         | `HalfNormal(0.5)`                                | `nuisance`                                               | LRP67 `sigma_proc`                                                                                                                               | Process-noise SD for latent changes, optionally shared across outcomes.                                                                                                                                            | Coverage gap. Review together with posterior predictive trajectory variability.                                                                             |
| `lcsm_noncentred_offsets` | `Normal(0, 1)`                                   | `nuisance`                                               | LRP67 `z1_*`, `zproc_*`                                                                                                                          | Standard-normal non-centred offsets for initial latent states and process noise.                                                                                                                                   | Coverage gap. No direct substantive interpretation.                                                                                                         |

## Outcome Denominators For Prior Pushforwards

Item-scale prior implications must use the relevant bounded-count denominator in
`MEASURES[symbol].n_trials`.

| Symbol | Denominator | Measure                                                                  |
| ------ | ----------: | ------------------------------------------------------------------------ |
| `W`    |          79 | Word reading (EWRSWR)                                                    |
| `R`    |         170 | Receptive vocabulary (ROWPVT)                                            |
| `E`    |         170 | Expressive vocabulary (EOWPVT)                                           |
| `L`    |          32 | Letter-sound knowledge (YARC-LSK)                                        |
| `P`    |          92 | Phonetic spelling (SPPHON); heavily floored in the ITT/factor floor rule |
| `B`    |          10 | Phoneme blending                                                         |
| `F`    |          18 | Basic concept knowledge (CELF)                                           |
| `T`    |          32 | Receptive grammar (TROG-2)                                               |
| `N`    |           6 | Nonword reading; post-only and floored                                   |
| `TE`   |          24 | Taught expressive vocabulary, block 1                                    |
| `TR`   |          24 | Taught receptive vocabulary, block 1                                     |
| `UE`   |          12 | Not-taught expressive vocabulary, block 1; denominator unconfirmed       |
| `UR`   |          12 | Not-taught receptive vocabulary, block 1; denominator unconfirmed        |

High-denominator outcomes (`R` and `E`, 170 items) need special care: a logit
prior that looks modest can still imply very large item-count differences at
some baseline probabilities.

## First Model-Scale Calibration Checks

The table below is only a quick logit-scale orientation. The full review still
needs probability- and item-scale pushforwards at representative baselines, for
example 0.2, 0.5, and 0.8.

| Prior family     | Approximate 95% logit range | Approximate odds-ratio range | Immediate concern                                                                                   |
| ---------------- | --------------------------: | ---------------------------: | --------------------------------------------------------------------------------------------------- |
| `Normal(0, 0.3)` |              -0.59 to +0.59 |                 0.55 to 1.80 | Moderate shrinkage for adjusted couplings and interactions.                                         |
| `Normal(0, 0.5)` |              -0.98 to +0.98 |                 0.38 to 2.66 | Plausible default for randomised effects, but item-scale implications vary strongly by denominator. |
| `Normal(0, 1.0)` |              -1.96 to +1.96 |                 0.14 to 7.10 | Wide for mechanism and mediation slopes; needs outcome-scale checking.                              |
| `Normal(0, 1.5)` |              -2.94 to +2.94 |                 0.05 to 18.9 | Used for intercepts and some latent-change intercepts; check prior predictive score distributions.  |
| `Normal(1, 0.5)` |              +0.02 to +1.98 |                 1.02 to 7.24 | Own-baseline continuity prior; not an effect-size prior. Needs baseline-noise sensitivity.          |

## Representative Prior-Predictive Review Set

The first full audit should inspect at least one fitted/prior-predictive run from
each family:

| Family        | Suggested representative model                                 | What to inspect                                                                                         |
| ------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| ITT           | `lrpitt10` (`W`) plus a floored model such as `lrpitt09` (`P`) | Treatment prior pushforward, off-floor risk for floored outcomes, baseline score plausibility.          |
| Joint         | `lrpitt12`                                                     | Outcome-specific tau prior, covariance branch if reactivated, high-denominator vocabulary item effects. |
| Mechanism     | `lrp56` or `lrp72`                                             | `beta_mech`, HSGP or linear mechanism priors, adjusted-association interpretation.                      |
| Mediation     | `lrp59`, `lrp62`, and `lrp64`                                  | `a` and `b` paths, mediator distribution, mediated-effect pushforward.                                  |
| DiD           | `lrpdid01` plus `lrpdid07`                                     | Period effect, treatment-period contrast, dose-slope priors.                                            |
| Gain factors  | `lrpgf01` plus floored `lrpgf05`                               | On-intervention term, child random intercept, floor-rule off-floor probability.                         |
| Level factors | `lrplf01`                                                      | Clean t2 group contrast versus later post-crossover associations.                                       |
| Aligned       | `lrpal01` plus `lrpal01d`                                      | Cohort and dose terms as associations, not randomised effects.                                          |
| Dynamic LCSM  | `lrp67`                                                        | Latent trajectory variability, process noise, cross-skill couplings into reading change.                |

## Open Inventory Gaps

- Mediation intercept aliases (`a0`, `b0`, `aL0`, `aE0`) use `alpha_prior()` but
  are not all mapped cleanly to the `alpha` panel/rationale row.
- LCSM priors are direct `pm.*` calls in `factories.py`; they need either named
  constructors or explicit `prior_info_for_rv()` mappings so `priors_table.csv`
  does not report them as generic model priors.
- The optional joint residual-correlation branch has an LKJ prior and
  non-centred residual offsets that are currently off by default but should be
  documented if any sensitivity fit reactivates them.
- HSGP basis weights are implied by the GP covariance construction rather than
  represented as separate substantive priors. The report should explain that
  the meaningful review is at the function scale: amplitude, lengthscale, and
  prior draws of `f(x)`.
- Contextual role overrides matter. For example, the same `tau_prior()` backs a
  randomised treatment effect in ITT models, but `beta_cohort` in aligned models
  is an association because the per-protocol cohort contrast is not randomised.

## Next Work

1. Generate probability-scale and item-scale implications for the main logit
   priors at baseline probabilities 0.2, 0.5, and 0.8, using the denominators
   above.
2. Inspect prior-predictive output for the representative model set.
3. Run sensitivity grids for `tau`, `gamma_cross`, `gamma_own`, high-impact
   mechanism slopes, `kappa` when dispersion is influential, and active HSGP
   amplitude/lengthscale priors.
4. Decide whether coverage gaps should be fixed by adding shared constructors,
   `_RV_TO_CTOR` mappings, or family-specific documentation rows.
5. Once decisions are stable, update `METHODS.md`, the Quarto report prior
   partials, and dated `notes/` entries for any prior changes.
