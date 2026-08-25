<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# The phoneme-blending guessing-floor policy covers every `B` fit, under one evidence tier

- **Date:** 2026-08-24
- **Status:** scientific decision
- **Issue:** #608 (subsumes #596; also closes acceptance item 7 of #575 and the open half of #584 finding 4)

## Decision

1. **Scope.** The mandatory guessing-floor link sensitivity `mu = 1/3 + (2/3) * inverse_logit(eta)` applies to **every registered model with `outcome_symbol="B"`, in every family**, whether the published quantity is a randomised contrast or an adjusted association. The release gate is keyed on the outcome symbol, not on `kind`, so a new `B` model in any family fails closed until it is paired or carries a recorded, dated exemption.
2. **Evidence tier.** **One tier.** Every pair is bound by the content-addressed archive apparatus, not only the ITT pair. The local two-directory pair check is retired as a permanent arrangement.
   > [!IMPORTANT]
   > **Decision 2 was superseded on 2026-08-25** by `notes/202608252100-blending-pair-binding-608-decision-2.md` (signed off by Frank on 2026-08-25). Pairs bind their **resolved run plan** and a staleness check instead; the content-addressed archive stays ITT-only. The text above is left as the record of what was decided on 2026-08-24 — do not act on it. Decision 1 stands and is implemented.

`METHODS.md` has been updated to say this. The wording that prompted the question — "any headline `B` interpretation" inside the ITT bullet — was correct as written; what was missing was that it applied to families with no pair built.

## Why: the defect is realised everywhere, not merely permitted

The policy's stated rationale is that the ordinary link "permits fitted means below the one-third guessing level". Permitting is not the same as doing, so the first question is empirical: does the fitted posterior actually use that room? For a Beta-Binomial with a logit score mean this is directly measurable — `expit(eta) < 1/3` is `eta < -log 2` — and `eta` is stored as a Deterministic in every fit's trace.

Measured over the stored reporting-tier fits:

| Model     | Family          | Rows with posterior-mean expected proportion < ⅓ | Posterior mass below ⅓ | Worst single row |
| --------- | --------------- | ------------------------------------------------ | ---------------------- | ---------------- |
| LRPLF06   | `level_factors` | 24/215                                           | 13.7 %                 | 99.5 %           |
| LRPDID03  | `did`           | 15/162                                           | 13.1 %                 | 99.6 %           |
| LRPMED87  | `mediation`     | 5/53                                             | 11.6 %                 | 99.1 %           |
| LRPGF06   | `gain_factors`  | 15/161                                           | 10.7 %                 | 99.8 %           |
| LRPGF206  | `gain_factors`  | 13/161                                           | 10.5 %                 | 99.9 %           |
| LRPCA07   | `concurrent`    | 4/54                                             | 9.7 %                  | 80.4 %           |
| LRPITT08  | `itt`           | 5/54                                             | 8.9 %                  | 100 %            |
| LRPGF106  | `gain_factors`  | 4/135                                            | 8.4 %                  | 91.5 %           |
| LRPDOSE84 | `dose_response` | 8/160                                            | 7.0 %                  | 100 %            |
| LRPAL06   | `aligned`       | 2/54                                             | 4.9 %                  | 98.0 %           |

The LRPLF06 row reproduces the figure already recorded in `lrp_rli_lf_106.py`'s docstring (24 of 215), which validates the method against an independently computed number.

**No fit is clean.** Every one contains rows where the model is 80–100 % certain that a child performs below chance on a three-alternative forced-choice test. The mildest case still carries 55 % of the below-chance mass of LRPITT08, the fit whose companion the policy was written for. A threshold rule — require the pair only where the floor demonstrably binds — would therefore exempt nothing on current data. It remains worth implementing as the _fail-closed gate mechanism_ for future models, but it is not a scope rule.

## Why: the link changes the finding, not the scale

LRPITT08 / LRPITT108 is the only pair with both links fitted at reporting tier. Both converged with zero divergences on identical rows (`n = 54`, 28 intervention, 26 control) and identical data and sampling contracts.

|                           | LRPITT08 (logit) | LRPITT108 (guessing floor) |
| ------------------------- | ---------------- | -------------------------- |
| Effect, items             | 0.99             | **0.49**                   |
| 89 % CI, items            | 0.22 to 1.74     | **−0.14 to 1.09**          |
| P(effect > 0)             | 0.980            | 0.893                      |
| Evidence label            | **strong**       | **suggestive**             |
| P(meaningful benefit)     | 0.49             | **0.088**                  |
| P(practically negligible) | 0.51             | **0.91**                   |

The estimate halves, the interval crosses zero, the evidence label drops a rung, and the practical verdict inverts. Meanwhile `guessing_floor_minus_logit_elpd = +1.09` (SE 2.68): the floor link fits marginally **better** by PSIS-LOO, comfortably inside noise but certainly not worse. The constraint costs nothing in fit and is mechanically motivated by the test's own design, so it is the ordinary-link number that requires justification, not the floored one.

This is what makes the uneven application the serious failure rather than a tidiness problem. A reader comparing LRPITT08's blending result with LRPGF06's sees two similar-looking numbers, one of which has been shown to halve under a better-fitting link and one of which has never been checked.

## Why the causal / observational split does not rescue an exemption

The argument for exempting observational families was that the guessing floor might bear differently on a slope than on a headline contrast. It does not survive its own named example. `METHODS.md` defines every dose fit's focal estimand as _the natural-scale treated-row dose marginal_ — a natural-scale quantity, so link-dependent in exactly the way LRPITT08's items-scale effect is. The link determines the mapping from the latent scale to the reported one; any quantity reported on the natural scale inherits that dependence regardless of what identifies it.

A latent-logit coefficient's sign is more robust to the link than a natural-scale marginal is. That is an argument for reporting latent-scale quantities where they answer the question, not an argument for leaving a misspecified link in place under natural-scale headlines.

## Why one evidence tier

Two tiers already exist by accident: the ITT pair is byte-bound to a content-addressed archive covering both traces, the data, the environment lock, the source commit, the config, the row map and the scientific artefacts; the level pair reads two stored artefact directories and is documented as "one rung down in evidence strength ... still binding".

The second is genuinely weaker — it cannot detect that one half of the pair was fitted from different source, and it is not visible in the rendered report which kind of evidence backs a given card. Since the issue's own complaint is that a reader cannot tell which blending results carry validated evidence, tiering the apparatus by family would preserve exactly the ambiguity the policy exists to remove. The archive machinery already exists and works; generalising it is bounded work.

## Reproduction

```bash
uv run python -c "
import arviz as az, numpy as np, math
idata = az.from_netcdf('output/statistical_models/models/lrp-rli-gf-006-reporting/trace.nc')
eta = idata.posterior['eta'].values.reshape(-1, idata.posterior['eta'].shape[-1])
mu = 1/(1+np.exp(-eta))
print('rows with mean below chance:', int((mu.mean(axis=0) < 1/3).sum()), 'of', mu.shape[1])
print('posterior mass below chance: {:.1%}'.format((eta < -math.log(2)).mean()))
print('worst row P(mu < 1/3): {:.1%}'.format((eta < -math.log(2)).mean(axis=0).max()))
"
```

The ITT comparison is read straight from `blending_link_sensitivity.csv` in either ITT fit's output directory.

## What this implies

Eight models need companions built: LRPAL06, LRPCA07, LRPDOSE84, LRPGF06, LRPGF106, LRPGF206, LRPLF206 and LRPMED87. Five families need `score_mean_link` added to their settings, run plan and factory (`gain_factors`, `aligned`, `concurrent`, `dose_response`, `mediation`); the link functions themselves (`apply_score_mean_link`, `invert_score_mean_link`, `beta_binomial_from_score_mean_link`) are already shared and family-generic.

Two registered companions have never been fitted at reporting tier — LRPLF106 and LRPDID103 — so their pairs do not currently bind despite the code being in place. LRPLF206 has no stored fit at all. Those are prerequisites, not extras: until they exist, three of the pairs the policy names are nominal.

The archive needs parameterising by pair and by per-family focal columns; it is currently hardcoded to one global pair with ITT-shaped free variables (`alpha`, `tau`, `gamma_own`, `gamma_A`, `kappa`) and an ITT-shaped summary-column map.

Two further items follow from the decision rather than being discretionary: the gain family's treatment marginal must be computed as a **score mean** rather than a raw inverse-logit, or a floor-link posterior will publish an ordinary-link number; and each family with an empirical-Bayes intercept anchor must map it back through the link, as the level family's did (it moved 1.1 logits).

Sequencing note: because the pairing requires both halves to share a source commit and run plan, adding `score_mean_link` to a family's settings may invalidate the stored primary's provenance even when the fitted model is unchanged. Check the run-plan digest before assuming the eight existing primaries can stand as-is; if they cannot, the batch is sixteen reporting fits rather than eight.

## Not decided here

The threshold and reporting format for the per-fit below-chance diagnostic, which should accompany every `B` card so a reader can see how hard the floor binds in that fit. Recommended, but it is a reporting design question, not a scope one.
