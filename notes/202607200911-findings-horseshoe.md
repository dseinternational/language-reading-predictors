# Horseshoe (sparse predictor selection) findings (2026-07-20)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is family note 11 in the series introduced by the [findings index and reading guide](202607200900-findings-00-index-and-reading-guide.md); read that note first for the study, the outcome measures, the house reporting standard and the causal-versus-association rule. All five models here were re-fit at the production `reporting` configuration (6000 draws × 6000 tune × 6 chains) on 2026-07-19. Everything below is preliminary.

## What this family does and the question it answers

A **horseshoe** is a Bayesian variable-selection prior. Fit an ordinary regression with many candidate predictors and each gets a coefficient; the horseshoe adds a rule that pulls (shrinks) every coefficient hard toward zero **unless** the data insist otherwise, in which case a heavy-tailed component lets a genuinely strong predictor "escape" and keep a large value. The practical effect is that noise predictors collapse to ≈0 while real signals stand out — a sparse, automatic short-list. (The `posterior` is the model's post-data probability distribution for each coefficient; we summarise it below.) These fits use the _regularised_ ("Finnish") horseshoe, which additionally caps how large an escaped coefficient can grow so the sampler stays well-behaved.

The family answers one question the others do not: **which measured predictors survive a sparse selection, and does that short-list agree with the gradient-boosting (Step-1, LightGBM/SHAP) importance ranking?** It is a second opinion on "which features track the outcome", reached by a completely different route. Crucially, every coefficient here is an **adjusted association**, never a causal lever — a predictor that survives tells you _who progresses or scores higher_, not _what to change_. The randomised causal claims live in the ITT suite ([note 01](202607200901-findings-itt-suite.md)), not here.

Each coefficient is a **standardised logit slope**: the change in the log-odds of the bounded outcome (a proportion-correct score modelled on the logit scale) per one-standard-deviation increase in the predictor, with predictor and outcome on a common scale. In the RLI models, block-design general ability (`blocks`) and a behaviour rating (`behav`) are entered as fixed pre-covariates, so the other predictors' slopes are already adjusted for those two.

## How to read these numbers

A compact recap (full version in the [reading guide](202607200900-findings-00-index-and-reading-guide.md)):

- **Point estimate = the posterior median** (`beta_median`), not the mean. Where the mean is noticeably larger than the median I flag it — under a horseshoe that gap is the signature of a coefficient sitting on a spike near zero with a long tail, i.e. a predictor that was mostly shrunk.
- **Uncertainty = the 89% credible interval.** These fits report the 89% **highest-density interval (HDI)** — the narrowest band holding 89% of the posterior mass — not the equal-tailed interval used in most other family notes. Read it as "given the data and priors, an 89% probability the slope lies here". The CSVs carry no inner-50% column, so none is quoted.
- **The ranking key is `p_abs_gt_delta` = P(|β| > 0.1)**: the posterior probability the coefficient's _magnitude_ exceeds a "worth-noticing" threshold of δ = 0.1 logit. This is a **magnitude / ROPE-exceedance probability, not a direction probability.** It says "the association, whichever way it points, is probably bigger than a trivial ±0.1-logit band around zero" — _not_ "probably positive". **The project's evidence ladder (inconclusive/suggestive/…/very strong) is a direction grading and therefore does NOT apply to this column** — I do not attach those labels to `p_abs_gt_delta`. Direction is carried instead by the `sign` and by whether the 89% HDI excludes zero.
- **"Survives selection"** below means the 89% HDI excludes zero (direction established); a predictor whose HDI spans zero has been effectively shrunk to noise. `lambda_mean` (local shrinkage) is a supporting read: small = squashed toward zero, large = escaped.
- **Original units.** A standardised _predictor_ slope on the logit scale has no single fixed item translation, because the same log-odds change moves the proportion-correct by different amounts at different points on the scale. For the two clean level models I give a rough "near the middle of the scale" items sense (multiply the logit slope by 0.25 — the logit-to-probability conversion at p = 0.5 — then by the test's item maximum), clearly flagged as approximate. For the flat gain models an items figure would be false precision, so none is given.

## Per-model results

| Model            | Outcome (framing)                     | N               | Survivors (89% HDI excludes 0) | Top predictor: median slope [89% HDI], P(\|β\|>0.1) | Gate                           |
| ---------------- | ------------------------------------- | --------------- | ------------------------------ | --------------------------------------------------- | ------------------------------ |
| `lrp-rli-hs-002` | Word reading **W**, level             | 210 obs / 53 ch | **L, E, T**                    | L +0.33 [+0.19, +0.47], 0.995                       | pass                           |
| `lrp-rli-hs-004` | Letter sounds **L**, level            | 214 obs / 54 ch | **W, E**                       | W +0.70 [+0.52, +0.87], 1.00                        | pass                           |
| `lrp-rli-hs-001` | Word reading **W**, gain              | 51 ch           | none                           | age −0.13 [−0.34, +0.02], 0.582                     | **divergence-only** (2/36 000) |
| `lrp-rli-hs-003` | Letter sounds **L**, gain             | 52 ch           | none                           | T +0.07 [−0.04, +0.37], 0.453                       | pass                           |
| `lrp-rlm-hs-001` | BAS word reading (Byrne cohort), gain | 69 ch           | **age (−)**                    | age −0.35 [−0.61, −0.07], 0.912                     | pass                           |

Predictor keys (RLI): **W** word reading, **L** letter-sound knowledge, **R** receptive vocabulary (ROWPVT), **E** expressive vocabulary (EOWPVT), **B** phoneme blending, **F** basic concepts (CELF), **T** receptive grammar (TROG-2), **age**, **blocks** block-design general ability, **behav** behaviour rating.

### `lrp-rli-hs-002` — word-reading _level_ · association · gate PASS

The score at each timepoint (stacked across 210 observations from 53 children; no own-baseline term). The horseshoe cleanly picks out the concurrent reading-system partners. Three predictors survive with 89% HDIs wholly above zero: **letter sounds L** (median +0.33 logit, 89% HDI +0.19 to +0.47, `P(|β|>0.1)` = 0.995), **expressive vocabulary E** (+0.41, +0.22 to +0.60, 0.992) and **receptive grammar T** (+0.19, +0.07 to +0.30, 0.886). All three are positive and their means match their medians, i.e. these coefficients escaped the shrinkage spike. Note that E has the _larger_ slope but ranks second: the ordering is by `P(|β|>0.1)`, a magnitude-exceedance probability, not by effect size. Below the survivors, age (+0.19, HDI −0.01 to +0.38, 0.745), R (+0.15, −0.01 to +0.31, 0.670) and B (+0.12, 0.628) are plausible but their HDIs graze or cross zero; F is shrunk (+0.05, 0.261). Rough near-midpoint items sense for the leaders (× 0.25 × 79 words): L ≈ +6.5, E ≈ +8, T ≈ +3.7 words of 79 per predictor-SD — approximate only.

### `lrp-rli-hs-004` — letter-sound _level_ · association · gate PASS

The mirror image (214 observations, 54 children). Two predictors survive decisively: **word reading W** (median +0.70 logit, 89% HDI +0.52 to +0.87, `P(|β|>0.1)` = 1.00, `lambda_mean` 28.4 — the strongest escape in the family) and **expressive vocabulary E** (+0.47, +0.29 to +0.66, 0.998). Everything else is shrunk to the floor: block-design B (+0.05, 0.285), F (+0.04, 0.259), age (+0.01, 0.176), R (+0.004, 0.113) and T (−0.004, 0.071) all have HDIs spanning zero. Near-midpoint items sense (× 0.25 × 32 letter sounds): W ≈ +5.6, E ≈ +3.8 of 32 per predictor-SD — approximate. So word-reading level and letter-sound level top _each other's_ rankings, exactly the concurrent coupling the mechanism and correlation families report.

### `lrp-rli-hs-001` — word-reading _gain_ · association · gate DIVERGENCE-ONLY

The change in word reading over the span (51 children, one row each; gb reference `lrp-rli-gbg-012`). **No predictor survives** — every 89% HDI spans zero. The nominal leader is age (median −0.13 logit, HDI −0.34 to +0.02, `P(|β|>0.1)` = 0.582, hinting older children gain a little less but well short of established), then L (median +0.07 but mean +0.11, HDI −0.04 to +0.33, 0.429), behav (−0.06, 0.380) and F (+0.04, 0.331). The median-below-mean gaps (L, F) are the horseshoe telling us these coefficients sit on the near-zero spike. Honest reading: **word-reading gain is essentially unpredictable from baseline features at this sample size.** Gate: 2 divergences of 36 000 draws (0.006%, far under the ≤1% guidance); R-hat ≤ 1.001, min ESS ≈ 9 400, BFMI healthy — usable with the divergence caveat noted.

### `lrp-rli-hs-003` — letter-sound _gain_ · association · gate PASS

Like the word-reading gain model, near-flat (52 children). **No predictor survives**; the top row, receptive grammar T, has median +0.07 (mean +0.13, HDI −0.04 to +0.37, `P(|β|>0.1)` = 0.453) with an HDI crossing zero, then F (+0.02, 0.287) and W (+0.02, 0.280). All the rest are shrunk to ≈0. Same message: gain in letter-sound knowledge is hard to predict from baseline features here.

### `lrp-rlm-hs-001` — Byrne-cohort word-reading gain · association · gate PASS

A separate **historical cohort** (Byrne/RLM reading-language-memory data; 69 children, 28 rows dropped), ranking wave-1 predictors of BAS word-reading gain over waves 1→3. The predictor set is that study's battery: bpvs (British Picture Vocabulary Scale, receptive vocabulary), trog (grammar), basdig (BAS digit span, verbal short-term memory), bassim (BAS similarities, verbal reasoning), basnum (BAS number skills) and age. **Only age survives**, negatively: median −0.35 logit, 89% HDI −0.61 to −0.07 (wholly below zero), `P(|β|>0.1)` = 0.912 — older children in this cohort gaining less. Every cognitive/language predictor is shrunk to near-zero: basdig (+0.04, 0.354), bassim (+0.01, 0.211), basnum (+0.001, 0.199), trog (−0.003, 0.172), bpvs (−0.000, 0.139). The negative age association echoes the [adjusted](202607161800-findings-adjusted.md) family. This is a historical, non-randomised cohort — associational only.

## What the family concludes

Two clean signals and three flat results, and both patterns are informative. For **levels**, the horseshoe independently recovers the reading system's obvious concurrent structure: **word reading and letter sounds top each other's rankings** (hs-002, hs-004), with **expressive vocabulary** a strong second in both and **receptive grammar** a third contributor to word-reading level. This is the "letter-sounds/word-reading dominate reading outcomes" pattern one expects from the gradient-boosting SHAP ranking, arrived at by a completely different method — the convergence is reassurance that the ranking reflects real signal in the data, not a tree-method artefact. For **gains**, both RLI models (hs-001 word reading, hs-003 letter sounds) find **nothing** that clears selection — the honest small-sample result that _how much a child improves_ is far harder to predict from baseline features than _where a child stands_. The one gain predictor that does survive is **age in the Byrne cohort** (negative), consistent with older children gaining less.

Triangulation: this family is the associational cross-check that sits alongside the causal families. The level-model survivors (L↔W↔E) mirror the [mechanism](202607161800-findings-mechanism.md) and correlation/measurement descriptions of skill coupling; the flat gain models are consistent with the [gain-factors](202607161800-findings-gain_factors.md) covariates being weak associations; and none of it competes with the randomised ITT estimates in [note 01](202607200901-findings-itt-suite.md) — a ranking describes, it does not identify a cause.

## Caveats and convergence

- **Gate.** 4 of 5 pass cleanly (0 divergences, max R-hat ≤ 1.0018, min ESS ≥ 4 221, BFMI ≥ 0.58). `lrp-rli-hs-001` is the sole flag — divergence-only (2/36 000, 0.006%), everything else passing — so its ranking is reported as usable-with-a-caveat, which does not change the conclusion (it found no survivors regardless).
- **Nothing here is causal.** All coefficients are adjusted associations confounded by latent general ability; the block-design and behaviour covariates adjust the RLI slopes for measured ability but not fully. Do not read a surviving predictor as a lever (the Table-2 fallacy). Adjustment/covariate sets are DAG- and study-fixed.
- **The ranking metric is a magnitude probability, not a direction grade** — see the reading recap. Closely-ranked predictors should not be treated as meaningfully ordered.
- **Mean-imputation.** PyMC cannot take missing inputs (unlike LightGBM, which handles them natively), so any missing standardised predictor is set to its mean (0 on the standardised scale) before fitting. A construct with more missingness is therefore pulled toward the null — read its rank as a _lower bound_ on its importance. This is why the family is framed as a sensitivity read rather than a calibrated fit.
- **Small N and floors.** With ~51–69 children the gain models have little power to distinguish predictors, and word-reading/letter-sound scores are partly floored early, both of which suppress detectable gain associations.
- **GB comparison table.** The `horseshoe_vs_gb.csv` construct-level rank comparison was not generated for this run (absent from all five output directories), so the agreement claims above are qualitative, against the expected boosting pattern, rather than a computed rank-correlation/top-k overlap.
