# Findings — growth family (multivariate growth curves; does baseline ability shape trajectories?)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Preliminary.

## What these models ask

Fit each child's **trajectory** — their starting level plus their growth rate — across the four RLI waves for five verbal/reading outcomes jointly, and ask one question: **does a child's baseline non-verbal ability predict how fast their verbal and reading skills grow?** Baseline non-verbal ability here is the WPPSI Block Design score measured once at wave 1 (n = 54), standardised so that "+1 SD ability" means comparing two children who sit roughly one standard deviation apart on Block Design (a sizeable gap — about the distance between the middle of the group and its upper third). The five outcomes are receptive vocabulary (R, ROWPVT), expressive vocabulary (E, EOWPVT), receptive grammar (T, TROG-2), word reading (W, EWRSWR) and letter-sound knowledge (L, YARC-LSK). Every outcome is modelled on a masked Beta-Binomial logit scale, so all the coefficients below are changes in **log-odds** of the trajectory, not raw test points.

Crucially, these trajectory coefficients are **between-child adjusted associations**: they describe _which_ children grow faster or sit higher, not what would happen if you could change a child's ability. Baseline non-verbal ability is itself confounded by latent general ability, so nothing here is a lever — see [What is causal](#what-is-causal).

Three models share this design and differ only in structure:

- **gc-069 (independent-core)** — five separate growth curves fitted together but with no shared machinery linking them; the plainest version.
- **gc-070 (shared growth-tempo factor)** — the genuinely _joint_ version: it adds one latent "growth-tempo" factor that all five outcomes load onto, so a child who runs fast on one measure is expected to run fast on the others. This is the only model that estimates cross-measure loadings and a tempo correlation.
- **gc-085 (age × ability interaction)** — adds a baseline-age main effect on growth rate (`gamma_age`) plus an **age × ability interaction** (`gamma_int`): it was built to test whether the ability→growth-rate association is larger for older or younger children at baseline, mirroring the age × ability interaction the gain-factor models reported for grammar and concepts.

The three main quantities each model reports, per outcome, are: `gamma` = ability → **growth rate** (the headline: do abler children grow faster?); `delta` = ability → **level at mid-study age** (do abler children sit higher?); and `beta` = the **mean growth rate** itself per SD of age (how fast does the average child grow, ignoring ability?).

Companion families: the causal intervention effects live in the [ITT](202607161800-findings-itt.md) and [gain-factor](202607161800-findings-gain_factors.md) notes; this family only characterises trajectory heterogeneity.

## How to read the numbers

For readers new to Bayesian output: a **95% credible interval** is the range within which the parameter lies with 95% probability _given the data and priors_ — a direct probability statement about the parameter, unlike a frequentist confidence interval. **P(effect > 0)** ("prob positive") is the posterior probability the true effect is positive. The **evidence label** grades that direction probability, not the effect's size, on the project ladder (#179): inconclusive < 0.75 ≤ suggestive < 0.91 ≤ moderate < 0.97 ≤ strong < 0.99 ≤ very strong. The growth CSVs report no formal ROPE (region of practical equivalence — a band around zero deemed too small to matter), so "big enough to matter" is read here from whether the credible interval excludes negligible values, not from a pre-registered threshold.

## Convergence gate

All 3 models **passed** (thresholds R̂ ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.3, 0 divergences). Minimum ESS ≈ 2335–3244 across the three; max R̂ ≤ 1.003; BFMI ≈ 0.61–0.70. Posterior-predictive coverage is well calibrated: about 76% of observations fall inside the model's 50% prediction range and about 99% inside the 90% range (all three models).

## Results — all models

### Headline: ability → growth rate for receptive grammar (T)

The clearest single ability→growth-rate association, and the one previously reported, is for receptive grammar. All three models agree on roughly the same size but differ in how confidently the interval clears zero:

| Model  | Structure                                       | γ for receptive grammar (T)                                                            | Evidence    | Causal?     |
| ------ | ----------------------------------------------- | -------------------------------------------------------------------------------------- | ----------- | ----------- |
| gc-085 | age × ability interaction (ability main effect) | +0.155 logit growth-rate change per +1 SD ability, 95% [+0.034, +0.275], P(>0) = 0.993 | very strong | association |
| gc-069 | independent-core growth curves                  | +0.114 logit, 95% [+0.002, +0.228], P(>0) = 0.977                                      | strong      | association |
| gc-070 | shared growth-tempo factor                      | +0.111 logit, 95% [−0.008, +0.228], P(>0) = 0.966                                      | moderate    | association |

Note the gc-070 row: P(>0) = 0.966 displays as "97%" if rounded but is below the 0.97 strong cut, so it is correctly graded **moderate**, not strong. Note also that the gc-085 figure is the ability **main effect** (`gamma`); the interaction the model was named for is a _different, much weaker_ coefficient (`gamma_int`, below).

### γ (ability → growth rate) for **all five outcomes**

This is where the single-outcome headline is misleading: a positive ability→growth-rate association is clear **only** for receptive grammar. For receptive vocabulary and letter-sound it leans _negative_ (abler children grow slightly slower), and for expressive vocabulary and word reading it is inconclusive.

| Outcome                 | gc-069 median [95%], P(>0), evidence                        | gc-070 median [95%], P(>0), evidence                      | gc-085 median [95%], P(>0), evidence                      |
| ----------------------- | ----------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| **T** receptive grammar | +0.114 [+0.002, +0.228], 0.977 **strong +**                 | +0.111 [−0.008, +0.228], 0.966 **moderate +**             | +0.155 [+0.034, +0.275], 0.993 **very strong +**          |
| **R** receptive vocab   | −0.035 [−0.106, +0.037], 0.831 for **negative**, suggestive | −0.041 [−0.118, +0.034], 0.862 for negative, suggestive   | −0.012 [−0.086, +0.062], 0.630 for negative, inconclusive |
| **E** expressive vocab  | −0.017 [−0.094, +0.058], 0.672 for negative, inconclusive   | −0.024 [−0.107, +0.057], 0.719 for negative, inconclusive | +0.011 [−0.068, +0.090], 0.612, inconclusive              |
| **W** word reading      | −0.047 [−0.272, +0.169], 0.667 for negative, inconclusive   | −0.074 [−0.315, +0.153], 0.739 for negative, inconclusive | +0.095 [−0.110, +0.292], 0.826 for positive, suggestive   |
| **L** letter-sound      | −0.148 [−0.373, +0.087], 0.893 for **negative**, suggestive | −0.149 [−0.380, +0.085], 0.896 for negative, suggestive   | −0.066 [−0.310, +0.177], 0.701 for negative, inconclusive |

Reading: the direction of the ability→growth-rate association is **outcome-specific**, not uniform. Grammar (T) is the one measure where abler children reliably grow faster. For letter-sound (L) and receptive vocabulary (R) the association is suggestively the other way. All five are adjusted associations, none causal.

### δ (ability → level at mid-study age) for all five outcomes — the strongest, most consistent signal

Where the growth-rate story is mixed, the _level_ story is not: on every outcome, in every model, children with higher baseline ability sit **higher** at mid-study age, and for the three language measures the evidence is very strong. This is the clearest ability signal in the family and was previously omitted.

| Outcome                 | gc-069 δ [95%], P(>0), evidence                         | gc-070 δ [95%], P(>0), evidence                         | gc-085 δ [95%], P(>0), evidence                           |
| ----------------------- | ------------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------- |
| **R** receptive vocab   | +0.187 [+0.091, +0.279], 1.00 **very strong +**         | +0.179 [+0.092, +0.267], 1.00 very strong +             | +0.171 [+0.072, +0.272], 1.00 very strong +               |
| **E** expressive vocab  | +0.223 [+0.119, +0.322], 1.00 **very strong +**         | +0.217 [+0.122, +0.312], 1.00 very strong +             | +0.198 [+0.089, +0.306], 1.00 very strong +               |
| **T** receptive grammar | +0.228 [+0.099, +0.352], 1.00 **very strong +**         | +0.217 [+0.095, +0.336], 0.999 very strong +            | +0.232 [+0.089, +0.373], 0.999 very strong +              |
| **W** word reading      | +0.224 [−0.152, +0.586], 0.883 for positive, suggestive | +0.212 [−0.158, +0.578], 0.871 for positive, suggestive | +0.224 [−0.147, +0.592], 0.880 for positive, suggestive   |
| **L** letter-sound      | +0.136 [−0.198, +0.458], 0.793 for positive, suggestive | +0.150 [−0.171, +0.462], 0.825 for positive, suggestive | +0.087 [−0.256, +0.415], 0.695 for positive, inconclusive |

### β (mean growth rate per SD of age) — trajectory shape

`beta` is the average child's growth per SD of age, ignoring ability. It shows that the two reading measures grow far faster over age than the three language measures: word reading (W) and letter-sound (L) run at roughly +1.0 to +1.15 logit per SD of age, versus +0.22–0.30 for R/E/T. All are very strong positive (P(>0) = 1.00 or 0.9997). So over the study window the reading skills are climbing steeply while the vocabulary/grammar measures climb gently.

| Outcome                 | gc-069 β | gc-070 β | gc-085 β |
| ----------------------- | -------- | -------- | -------- |
| **W** word reading      | +1.148   | +1.153   | +1.130   |
| **L** letter-sound      | +1.053   | +1.020   | +0.969   |
| **E** expressive vocab  | +0.294   | +0.296   | +0.286   |
| **R** receptive vocab   | +0.252   | +0.258   | +0.252   |
| **T** receptive grammar | +0.221   | +0.228   | +0.247   |

(All β very strong positive, 95% intervals well clear of zero — e.g. gc-069 W [+0.945, +1.367], L [+0.798, +1.304].)

### gc-070 only: shared growth-tempo factor

The joint layer's payoff is one latent "growth-tempo" factor. All five outcomes load **very strongly and positively** on it (P(load > 0) = 1.00), so children who grow fast on one measure do tend to grow fast on the others. Word reading loads hardest:

| Outcome                 | Loading [95%], evidence               |
| ----------------------- | ------------------------------------- |
| **W** word reading      | +0.378 [+0.081, +0.692] very strong + |
| **L** letter-sound      | +0.349 [+0.051, +0.679] very strong + |
| **T** receptive grammar | +0.170 [+0.036, +0.324] very strong + |
| **E** expressive vocab  | +0.130 [+0.044, +0.225] very strong + |
| **R** receptive vocab   | +0.107 [+0.028, +0.196] very strong + |

But the **growth-tempo correlation** — whether that shared tempo co-moves with anything — is essentially null: median −0.016, 95% [−0.283, +0.253], P(>0) = 0.453 (inconclusive). And the extra structure does not earn its keep in fit (see model comparison below).

### gc-085 only: the age × ability interaction and the baseline-age effect

This is the model's stated reason to exist, and the result is a genuine null-ish finding worth surfacing. The **age × ability interaction** (`gamma_int`, does the ability→growth-rate slope depend on baseline age?) is mostly inconclusive — and for grammar, the very measure the gain-factor models flagged, it is essentially zero:

| Outcome                 | γ_int median [95% HDI]  | Reading                                         |
| ----------------------- | ----------------------- | ----------------------------------------------- |
| **L** letter-sound      | +0.268 [−0.008, +0.549] | suggestively positive, the only non-trivial one |
| **W** word reading      | +0.100 [−0.114, +0.322] | inconclusive                                    |
| **E** expressive vocab  | +0.073 [−0.014, +0.162] | inconclusive                                    |
| **R** receptive vocab   | +0.045 [−0.036, +0.127] | inconclusive                                    |
| **T** receptive grammar | +0.008 [−0.125, +0.140] | ≈ zero                                          |

So this family does **not** reproduce the strong positive age × ability interaction the gain-factor models reported for grammar/concepts. The separate **baseline-age main effect** (`gamma_age`, older-at-baseline → faster/slower growth) is negative across the board and clearly so for word reading: W −0.396 [−0.577, −0.226] (interval entirely negative), with R −0.073 [−0.139, −0.007], E −0.094 [−0.165, −0.024], T −0.104 [−0.212, +0.001], L −0.143 [−0.369, +0.092]. Older-at-baseline children grow more slowly, most markedly on word reading.

### Which model to trust — PSIS-LOO comparison

The three are **not** interchangeable. Leave-one-out cross-validation (higher elpd = better out-of-sample fit) prefers the interaction model and slightly _disfavours_ the shared-factor model:

| Model  | LOO elpd | Verdict                                                       |
| ------ | -------- | ------------------------------------------------------------- |
| gc-085 | −3129.06 | **best** — the age terms improve fit                          |
| gc-069 | −3139.17 | middle — the plain independent-core baseline                  |
| gc-070 | −3141.53 | worst — the shared growth-tempo factor does not earn its keep |

## The one-paragraph story

Higher **baseline non-verbal ability** is most clearly associated with a **higher level** at mid-study age on every measure — abler children sit ahead, robustly so for the three language skills. Whether they also **grow faster** is outcome-specific: the ability→growth-rate association is reliably positive only for **receptive grammar** (about +0.11 to +0.16 logit per +1 SD ability, evidence moderate-to-very-strong across the three models). For receptive vocabulary and letter-sound it leans the other way (abler children grow a touch slower), and for expressive vocabulary and word reading it is inconclusive — so the earlier blanket "abler children grow faster on the verbal/reading measures" was an over-generalisation of the grammar result. Word reading and letter-sound are the fastest-growing trajectories over age regardless of ability. The model built to test whether this ability effect varies with baseline age (gc-085) finds essentially no such interaction for grammar; cross-validation nonetheless prefers gc-085 (for its negative baseline-age effect) over the plain gc-069, and slightly disfavours the shared-factor gc-070. All of this describes heterogeneity between children; none of it is a claim that raising ability would speed growth.

## Word reading specifically

Word-reading **growth** is _not_ well explained by baseline non-verbal ability: the ability→growth-rate association (γ_W) is inconclusive and even trends slightly negative in gc-069 (−0.047) and gc-070 (−0.074), turning only suggestively positive in gc-085 (+0.095, P(>0) = 0.826). What _is_ clear for word reading: it is one of the two fastest-growing trajectories over age (β_W ≈ +1.13–1.15, very strong); higher ability is suggestively associated with a higher word-reading _level_ at mid-study age (δ_W ≈ +0.22, P(>0) ≈ 0.88); it loads hardest on the shared growth-tempo factor (loading +0.378); and older-at-baseline children grow more slowly on word reading (γ_age_W = −0.396, interval entirely negative). All descriptive/associational — none causal.

## What is causal

**Nothing.** These are between-child associations between baseline ability (and baseline age) and trajectory shape, all confounded by latent general ability (`estimand_type = descriptive`, `causal_status = adjusted`). They characterise _who_ grows faster and _who_ sits higher — useful context for the causal intervention findings in the [ITT](202607161800-findings-itt.md) and [gain-factor](202607161800-findings-gain_factors.md) notes — but they are not levers.
