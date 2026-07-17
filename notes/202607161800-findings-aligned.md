# Findings — aligned family (onset-aligned per-protocol comparison)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Preliminary.

## What these models ask

Instead of comparing arms _as randomised_, these models **align both arms by when they actually started** the intervention (immediate arm t1→t3, waiting-list arm t2→t4) and compare the two cohorts' ~40-week gains in one cross-sectional Beta-Binomial ANCOVA per child (no random intercept — each child appears once, at their own aligned onset). This "per-protocol" framing answers "what did roughly 40 weeks of programme look like?" — but the cohort comparison is **not randomised**: age-at-onset and cohort/timing differences can confound it. So **no coefficient here is causal** — every one is an adjusted association. There is one model per outcome (`al-001`–`008`, covering word reading W, receptive vocabulary R, expressive vocabulary E, letter sounds L, phonetic spelling P, phoneme blending B, basic concepts F and receptive grammar T), plus `al-101`, a dose-sensitivity variant of word reading that adds cumulative sessions (a partial collider, sensitivity-only). Samples are small — 52 to 54 children, each contributing a single aligned gain — which is why the credible ranges below are wide.

A quick orientation for readers new to the Bayesian vocabulary. A **95% credible range** is the interval the parameter falls in with 95% probability _given the data and priors_ — a direct probability statement about the parameter, unlike a frequentist confidence interval. **P(effect>0)** (or **P(<0)**) is the posterior probability the true effect is positive (negative) — we read the size of that probability off the project's evidence ladder (inconclusive < 0.75 ≤ suggestive < 0.91 ≤ moderate < 0.97 ≤ strong < 0.99 ≤ very strong), applied to the raw, unrounded probability. This family reports no ROPE ("region of practical equivalence") band, so there is no "big enough to matter" verdict to add — just direction and uncertainty.

## Convergence gate

All 9 models (`al-001`–`008` + the dose-sensitivity variant `al-101`) **passed**, with 0 divergences each.

## Results — all models (cohort differences: immediate minus waiting-list, onset-aligned)

Positive = the immediate cohort finishes higher. Each row gives the difference on the natural items scale and, in parentheses, the same contrast expressed as a risk difference on the probability scale (the change in the expected proportion of items scored). Every row is an **association**, not a causal effect.

| Model    | Outcome                        |   n | Difference (items) | 95% range      | Prob-scale (risk diff) | 95% range        | P(favoured dir) | Evidence     | Causal? |
| -------- | ------------------------------ | --: | ------------------ | -------------- | ---------------------- | ---------------- | --------------- | ------------ | ------- |
| `al-004` | **Letter sounds (L)**          |  53 | +2.23              | −0.26 to +4.67 | +0.070                 | −0.008 to +0.146 | P(>0)=0.961     | moderate     | assoc.  |
| `al-002` | Receptive vocab, std (R)       |  54 | +2.73              | −2.87 to +8.18 | +0.016                 | −0.017 to +0.048 | P(>0)=0.832     | suggestive   | assoc.  |
| `al-001` | Word reading (W)               |  52 | +2.14              | −1.09 to +5.35 | +0.027                 | −0.014 to +0.068 | P(>0)=0.906 †   | suggestive   | assoc.  |
| `al-101` | Word reading (W), dose variant |  52 | +2.15              | −1.03 to +5.41 | +0.027                 | −0.013 to +0.068 | P(>0)=0.909 †   | suggestive   | assoc.  |
| `al-006` | Phoneme blending (B)           |  54 | +0.30              | −0.79 to +1.40 | +0.030                 | −0.079 to +0.140 | P(>0)=0.706     | inconclusive | assoc.  |
| `al-005` | Phonetic spelling (P)          |  53 | +0.03              | −0.10 to +0.16 | (scales coincide) ‡    | —                | P(>0)=0.698     | inconclusive | assoc.  |
| `al-007` | Basic concepts (F)             |  53 | −0.61              | −1.92 to +0.71 | −0.034                 | −0.107 to +0.039 | P(<0)=0.820     | suggestive   | assoc.  |
| `al-008` | Receptive grammar (T)          |  54 | −1.43              | −3.52 to +0.70 | −0.045                 | −0.110 to +0.022 | P(<0)=0.909 †   | suggestive   | assoc.  |
| `al-003` | Expressive vocab, std (E)      |  54 | −3.06              | −7.76 to +1.61 | −0.018                 | −0.046 to +0.009 | P(<0)=0.900     | suggestive   | assoc.  |

† These three read as "91%" when rounded, but are labelled **suggestive** not moderate: the raw probabilities are 0.906 (W), 0.909 (W-dose) and 0.9093 (T), all just below the 0.91 moderate threshold, and the ladder is applied to the unrounded value. ‡ Phonetic spelling is a compressed, near-floor measure, so its items and probability scales effectively coincide over this narrow range; a phonetic-spelling "item" is not a natural everyday unit — read it as "essentially no cohort difference".

Letter sounds (`al-004`) is the strongest positive-leaning outcome (moderate), word reading (`al-001`/`al-101`) and receptive vocabulary (`al-002`) lean positive but only suggestively, phoneme blending and phonetic spelling are flat and inconclusive, and receptive grammar (`al-008`), expressive vocabulary (`al-003`) and basic concepts (`al-007`) lean _negative_ (suggestive). Every interval spans zero and the samples are tiny, which is exactly the behaviour expected of a non-randomised cohort contrast that timing and age can distort in either direction.

## The adjusted associations (what the models condition on)

Each ANCOVA adjusts the cohort contrast for the child's own aligned baseline (`gamma_own`), age-at-onset (`gamma_A`) and general cognitive ability measured by block design (`gamma_ability`); the dose variant `al-101` adds cumulative sessions (`gamma_dose`). All coefficients below are on the logit (log-odds) scale and are **latent-ability-confounded adjusted associations — never causal**.

| Model    | Outcome | `gamma_own` (own baseline) | Evidence    | `gamma_A` (age-at-onset) | Evidence     | `gamma_ability` (block design) | Evidence     |
| -------- | ------- | -------------------------- | ----------- | ------------------------ | ------------ | ------------------------------ | ------------ |
| `al-001` | W       | +0.741 [+0.624, +0.856]    | very strong | −0.179 [−0.354, −0.000]  | strong       | +0.107 [−0.084, +0.294]        | suggestive   |
| `al-002` | R       | +0.772 [+0.547, +1.003]    | very strong | −0.036 [−0.129, +0.058]  | suggestive   | +0.043 [−0.059, +0.148]        | suggestive   |
| `al-003` | E       | +0.806 [+0.609, +1.006]    | very strong | −0.093 [−0.179, −0.006]  | strong       | +0.136 [+0.035, +0.237]        | very strong  |
| `al-004` | L       | +0.645 [+0.468, +0.829]    | very strong | +0.030 [−0.218, +0.287]  | inconclusive | −0.021 [−0.280, +0.240]        | inconclusive |
| `al-005` | P       | +0.806 [+0.503, +1.132]    | very strong | +0.031 [−0.440, +0.492]  | inconclusive | +0.457 [−0.008, +0.935]        | strong       |
| `al-006` | B       | +0.678 [+0.388, +0.988]    | very strong | −0.062 [−0.347, +0.219]  | inconclusive | +0.036 [−0.262, +0.330]        | inconclusive |
| `al-007` | F       | +0.563 [+0.380, +0.758]    | very strong | −0.002 [−0.212, +0.209]  | inconclusive | −0.015 [−0.269, +0.230]        | inconclusive |
| `al-008` | T       | +0.470 [+0.242, +0.714]    | very strong | −0.059 [−0.201, +0.085]  | suggestive   | +0.273 [+0.117, +0.425]        | very strong  |
| `al-101` | W-dose  | +0.742 [+0.626, +0.857]    | very strong | −0.171 [−0.354, +0.008]  | moderate     | +0.093 [−0.114, +0.296]        | suggestive   |

Three things stand out. First, the child's **own aligned baseline is the dominant term everywhere**: `gamma_own` runs +0.47 to +0.81 with the posterior probability of a positive sign equal to 1.00 (very strong) in all nine models — "higher start → higher finish", the most strongly resolved association in the family. Second, **age-at-onset is the family's operative confounder**, the concrete quantity the "timing and age can distort it" caveat points at: it trends negative for most outcomes (higher age-at-onset → lower aligned finish) and is best resolved for word reading W (−0.179, neg 0.975, strong), expressive vocabulary E (−0.093, neg 0.982, strong) and the W-dose variant (−0.171, neg 0.969, moderate); elsewhere it is weak and inconclusive. Third, **block-design ability is positive and well-resolved for a subset** — expressive vocabulary E (+0.136, very strong), receptive grammar T (+0.273, very strong) and phonetic spelling P (+0.457, strong) — but near-null and inconclusive for word reading, letter sounds, blending and basic concepts.

The dose variant `al-101` adds one further association, cumulative sessions: `gamma_dose` = +0.034 [−0.146, +0.227], P(>0)=0.641, **inconclusive** — negligible. This is the concrete evidence that dose adds nothing to the cohort headline: with dose in the model the word-reading difference is +2.15 items (vs +2.14 without), essentially unchanged. Dose is in any case a partial collider here, so it enters only as a sensitivity check and is never interpreted causally.

## The one-paragraph story

Aligning the two cohorts by intervention onset gives a positive-leaning read for **letter sounds** (moderate) and **word reading** (suggestive), broadly consistent with the randomised analyses — but the intervals are wide, the samples tiny (52–54 children, one aligned gain each), and several outcomes (expressive vocabulary, receptive grammar, basic concepts) wander negative, exactly as you would expect from a **non-randomised** cohort contrast that timing and age can distort. The child's own baseline dominates every model (very strong), age-at-onset is a real negative confounder for the reading outcomes, and adding cumulative-session dose changes nothing. This family is a triangulation / sensitivity view, not evidence of record.

## What is causal

**Nothing.** The cohort contrast (`beta_cohort`) is confounded by design — age-at-onset and cohort/timing are not balanced the way randomisation would balance them — and dose (a partial collider) enters only the `al-101` sensitivity variant. Every coefficient reported above is an adjusted association: the child's own baseline is, as everywhere, the most strongly resolved (posterior probability of a positive sign = 1.00, "very strong", in all nine models, for "higher start → higher finish"), and age-at-onset and block-design ability are the confounders the model conditions on, not effects it estimates. Read the [ITT](202607161800-findings-itt.md) / [gain_factors](202607161800-findings-gain_factors.md) / [did](202607161800-findings-did.md) families for the causal claims; this one only triangulates them.
