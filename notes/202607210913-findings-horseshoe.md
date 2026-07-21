# Horseshoe (sparse predictor selection) findings (2026-07-21)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is one of a set of per-family findings notes from the full 2026-07-21 re-fit of every Bayesian statistical model in the study (production `reporting` configuration, 6 chains × 6000 draws, 89% credible intervals). Read the [findings index and reading guide](202607210900-findings-00-index-and-reading-guide.md) first for the study background, the outcome measures, the house reporting standard and — importantly — the rule that separates causal effects from adjusted associations. This family has **five models**; **four pass the convergence gate cleanly** and the fifth (`lrp-rli-hs-001`) is a divergence-only review case that is usable with a note. Everything below is preliminary: roughly 51–69 children contribute per model, so estimates are uncertain and should be read interval-first.

## What this family probes

A **horseshoe** is a Bayesian variable-selection prior. Fit an ordinary regression with many candidate predictors and each earns a coefficient; the horseshoe adds a rule that shrinks (pulls) every coefficient hard toward zero **unless** the data insist otherwise, in which case a heavy-tailed component lets a genuinely strong predictor "escape" the shrinkage and keep a sizeable value. The practical effect is that noise predictors collapse to approximately zero while real signals stand out, producing a sparse, automatic short-list. These fits use the _regularised_ ("Finnish") horseshoe, which additionally caps how large an escaped coefficient can grow so the sampler stays well-behaved. The **posterior** is the model's post-data probability distribution for each coefficient; we summarise it below.

The family answers one question the others do not: **which measured predictors survive a sparse selection, and does that short-list agree with the Step-1 gradient-boosting (LightGBM/SHAP) importance ranking?** It is a second opinion on "which features track the outcome", reached by a completely different statistical route, and so it acts as a cross-check on the boosting layer rather than a new claim. Crucially, **every coefficient here is an adjusted association, never a causal lever.** A predictor that survives tells you _who progresses or scores higher_, not _what to change_. The randomised causal claims live in the ITT suite (note 01), not here; residual confounding by latent general ability remains for all these associations.

Four models rank predictors of RLI (Reading and Language Intervention) outcomes; the fifth, `lrp-rlm-hs-001`, ranks predictors in a separate historical **Byrne cohort** (the reading-language-memory battery). The RLI candidate set entered in the ranking includes: age, letter-sound knowledge (LS), a behaviour rating (behav), basic concepts (LF, from CELF), receptive vocabulary (RV, ROWPVT), receptive grammar (RG, TROG-2), expressive vocabulary (EV, EOWPVT), block-design general ability (blocks) and phoneme blending (PA). Which predictors are eligible for each model is DAG- and study-fixed, not chosen after seeing the data.

## How to read these numbers

A compact recap (full version in the [reading guide](202607210900-findings-00-index-and-reading-guide.md)):

- **Point estimate = the posterior median** (`beta_median`), not the mean. Where the mean sits noticeably above the median, that gap is the horseshoe's signature of a coefficient parked on a spike near zero with a long tail — i.e. a predictor that was mostly shrunk.
- **Uncertainty = the 89% credible interval.** These fits report the 89% **highest-density interval (HDI)** — the narrowest band holding 89% of the posterior mass — read as "given the data and priors, an 89% probability the slope lies here". Each coefficient is a **standardised logit slope**: the change in the log-odds of the bounded (proportion-correct) outcome per one-standard-deviation increase in the predictor, with predictor and outcome on a common scale.
- **The ranking key is `p_abs_gt_delta`** — the posterior probability that the coefficient's _magnitude_ exceeds a small "worth-noticing" threshold δ. This is a **magnitude probability, not a direction probability.** It says "the association, whichever way it points, is probably bigger than a trivial band around zero" — _not_ "probably positive". For that reason the project's **evidence ladder (inconclusive/suggestive/moderate/strong/very strong) does NOT apply to this column** and I attach no such label to it; direction is instead carried by the `sign` and by whether the 89% HDI excludes zero.
- **"Survives selection"** below means the 89% HDI excludes zero (direction established); a predictor whose HDI spans zero has been effectively shrunk to noise. Because a standardised _predictor_ slope on the logit scale has no single fixed item translation (the same log-odds change moves the proportion-correct by different amounts at different points on the scale), and the ground-truth key findings supply none, I report the standardised logit slopes and HDIs directly and do not invent an "items" figure.
- **Closely-ranked predictors should not be treated as meaningfully ordered** — the ranking is an adjusted predictive sensitivity check, a point the models' own key-findings summaries repeat.

## Per-model findings

| Model            | Outcome (framing)                     | NW              | Top predictor: median slope [89% HDI], P(\|β\|>δ) | Survives?        | Gate                         |
| ---------------- | ------------------------------------- | --------------- | ------------------------------------------------- | ---------------- | ---------------------------- |
| `lrp-rli-hs-002` | Word reading **WR**, level            | 210 obs / 53 ch | LS +0.33 [+0.19, +0.47], 0.99                     | **yes (LS)**     | pass                         |
| `lrp-rli-hs-004` | Letter sounds **LS**, level           | 214 obs / 54 ch | WR +0.70 [+0.52, +0.87], 0.999                    | **yes (WR)**     | pass                         |
| `lrp-rli-hs-001` | Word reading **WR**, gain             | 51 obs / 51 ch  | age −0.13 [−0.34, +0.02], 0.582                   | no               | **review (divergence-only)** |
| `lrp-rli-hs-003` | Letter sounds **LS**, gain            | 52 obs / 52 ch  | RG +0.07 [−0.04, +0.37], 0.45                     | no               | pass                         |
| `lrp-rlm-hs-001` | BAS word reading (Byrne cohort), gain | 69 obs / 69 ch  | age −0.35 [−0.61, −0.07], 0.91                    | **yes (age, −)** | pass                         |

### The two clean level models — the signals

**`lrp-rli-hs-002` — word-reading _level_ · association · gate PASS.** Ranking predictors of the word-reading score at each timepoint (210 observations from 53 children; no own-baseline term). The top-ranked predictor is **letter-sound knowledge (LS)**, with a standardised _positive_ association of **+0.33 logit units** (89% HDI +0.19 to +0.47, whole interval above zero, so it survives selection); its probability of exceeding the worth-noticing magnitude threshold, `P(|β|>δ)`, is **99%**. This is exactly the concurrent reading-system coupling one expects: where children stand on letter sounds tracks where they stand on word reading. The ground truth surfaces only the top predictor's figures for this note; the ranking's own caveat — closely-ranked predictors are not meaningfully ordered — applies to whatever sits below LS.

**`lrp-rli-hs-004` — letter-sound _level_ · association · gate PASS.** The mirror image (214 observations, 54 children). The top-ranked predictor is **word reading (WR)**, with a standardised _positive_ association of **+0.70 logit units** (89% HDI +0.52 to +0.87, wholly above zero → survives), and `P(|β|>δ)` = **99.9%** — the strongest, most decisively-escaped coefficient in the family. So word-reading level and letter-sound level top _each other's_ rankings, the two-way concurrent coupling the mechanism and correlation/measurement families also report.

### The two flat RLI gain models — nothing survives

**`lrp-rli-hs-001` — word-reading _gain_ · association · gate REVIEW (divergence-only).** Ranking predictors of the _change_ in word reading (51 children, one row each). **No predictor survives** — every top HDI spans zero. The full top rows of the surfaced `predictor_ranking.csv` are:

| Rank | Predictor            | P(\|β\|>δ) | Median slope | Sign | 89% HDI        |
| ---- | -------------------- | ---------- | ------------ | ---- | -------------- |
| 1    | age                  | 0.582      | −0.13        | −    | −0.34 to +0.02 |
| 2    | LS (letter sounds)   | 0.429      | +0.07        | +    | −0.04 to +0.33 |
| 3    | behav (behaviour)    | 0.380      | −0.06        | −    | −0.27 to +0.04 |
| 4    | LF (basic concepts)  | 0.331      | +0.04        | +    | −0.04 to +0.24 |
| 5    | RV (receptive vocab) | 0.224      | +0.01        | +    | −0.09 to +0.22 |

The nominal leader, age, hints that older children gain a little less, but its HDI still crosses zero and its magnitude probability is only 0.582, so nothing is established. LS is a telling case: its median is +0.07 but its mean is +0.11 — the median-below-mean gap is the horseshoe reporting a coefficient sitting on the near-zero spike with a thin tail. Honest reading: **word-reading gain is essentially unpredictable from baseline features at this sample size.** Gate: 2 divergences of 36,000 draws (0.006%, far under the ≤1% guidance), with max R-hat ≤ 1.001 and minimum effective sample size ≈ 9,350 — a healthy fit flagged only on the divergence count, and usable with that caveat (which changes nothing, as it found no survivors regardless).

**`lrp-rli-hs-003` — letter-sound _gain_ · association · gate PASS.** The companion for letter-sound gain (52 children), and equally flat. **No predictor survives.** The top-ranked predictor is receptive grammar (RG), median **+0.07 logit units** (89% HDI −0.04 to +0.37, crossing zero), with a magnitude probability of only **45%**. Same message as the word-reading gain model: how much a child improves in letter-sound knowledge is far harder to predict from baseline features than where the child stands.

### The historical cohort

**`lrp-rlm-hs-001` — Byrne-cohort word-reading gain · association · gate PASS.** A separate historical cohort (Byrne reading-language-memory data; 69 children), ranking wave-1 predictors of BAS word-reading gain over waves 1→3. **Only age survives, and negatively**: median **−0.35 logit units** (89% HDI −0.61 to −0.07, whole interval below zero), with `P(|β|>δ)` = **91%** — older children in this cohort gain less. Every cognitive and language predictor in the battery is shrunk toward zero and none clears selection alongside age. This is a historical, non-randomised cohort, so the age association is descriptive only.

## What to take away

Two clean signals and three flat results, and both patterns are informative. For **levels**, the horseshoe independently recovers the reading system's obvious concurrent structure: **word reading and letter sounds top each other's rankings** (hs-004 puts WR first for LS; hs-002 puts LS first for WR), the +0.70 and +0.33 standardised slopes being the strongest surviving associations in the family. That is the "letter-sounds and word-reading dominate reading outcomes" pattern one expects from the gradient-boosting SHAP ranking, arrived at by a wholly different method — the convergence is reassurance that the boosting ranking reflects real signal in the data, not a tree-method artefact. For **gains**, both RLI models (hs-001 word reading, hs-003 letter sounds) find **nothing** that clears selection — the honest small-sample result that _how much_ a child improves is much harder to predict from baseline features than _where_ a child stands. The one gain predictor that does survive anywhere is **age in the Byrne cohort** (negative), consistent with older children gaining less.

Two standing caveats frame all of it. First, **nothing here is causal**: every surviving predictor is an adjusted association confounded by latent general ability, so reading a survivor as a lever would be the Table-2 fallacy. This family is the associational cross-check that sits _alongside_ the randomised ITT estimates (note 01), never a substitute for them — a ranking describes who progresses; it does not identify a cause. Second, **PyMC mean-imputes missing predictors** (unlike LightGBM, which handles them natively), which pulls a more-missing construct toward the null; a low rank should therefore be read as a _lower bound_ on a predictor's importance, which is why the family is framed as a sensitivity read.
