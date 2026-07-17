# Findings — concurrent family (per-wave conditional associations between skills)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. This family was added in issue #312; the 2026-07-16 refit is its first gate-passing `reporting` fit. Preliminary.

## What these models ask

At a single wave, "among children alike on age and the other skills, does a child who is higher on skill X also tend to be higher on the focal skill Y?" This is the regression-style description of the **joint distribution of skills at each timepoint** — a modern version of the paper's own correlation tables. Each coefficient is reported two ways: **adjusted** (holding every other same-wave skill fixed) and **bivariate** (that predictor alone). Both are put on the items scale as an average marginal effect — the expected change in the focal skill's raw item count per **+1 SD** (one standard deviation, i.e. one typical between-child step) of the predictor — computed separately at each of the four timepoints (t1–t4).

The family covers six focal skills, one model each: **ca-001** word reading (W), **ca-002** letter sounds (L), **ca-003** taught receptive vocabulary (TR), **ca-004** taught expressive vocabulary (TE), **ca-005** receptive vocabulary (R), **ca-006** expressive vocabulary (E). All use a Beta-Binomial likelihood on bounded post-score counts through a logit-linear predictor, with no interaction, spline, or nonlinearity term — so there is no knee or threshold estimand to report, only per-wave slopes.

**This family makes no causal claim at all — by design.** All children are on the programme from t2, so every one of these coefficients conditions on post-treatment skills; that would wreck a causal reading, but it is perfectly fine for pure description. Every term is flagged `Status.ASSOCIATION`. The report carries the **Table-2-fallacy** caveat — each adjusted coefficient answers a _different_ conditional question (its own "holding-the-rest-fixed" world), so the column should not be read as a single ranking of "importance" — and a **regression-dilution** note: because the predictors are noisy observed scores rather than the latent skills themselves, the adjusted associations are attenuated (pulled toward zero) relative to the underlying truth, and the shared measurement error is part of why some adjusted terms flip sign once collinear partners are held fixed.

Because this family is purely descriptive, no region of practical equivalence (ROPE — a band around zero deemed too small to matter) is defined for it, and no "big enough to matter" verdict is issued. Evidence is graded only on **direction**: the posterior probability that the true association is positive, P(>0), read against the project evidence ladder (inconclusive < 0.75 ≤ suggestive < 0.91 ≤ moderate < 0.97 ≤ strong < 0.99 ≤ very strong). A "95% credible range" below is the interval the coefficient lies in with 95% probability given the data and priors — a direct probability statement about the parameter, unlike a frequentist confidence interval.

## Convergence gate

All 6 models **passed** cleanly. Zero divergences everywhere; maximum R-hat ≤ 1.0005 across the family (range 1.0002–1.0005, all well under the 1.01 ceiling); minimum effective sample size (ESS — the number of effectively independent posterior draws) ran from 23,896 (ca-005) to 26,195 (ca-004), i.e. roughly 24k–26k, far above the 400 floor; per-chain BFMI (an energy-based check that the sampler mixed) all sat around 0.90–1.02, comfortably over 0.3.

| Model  | Focal skill                  | Max R-hat | Min ESS | Divergences | Gate |
| ------ | ---------------------------- | --------- | ------- | ----------- | ---- |
| ca-001 | Word reading (W)             | 1.0005    | 25,216  | 0           | pass |
| ca-002 | Letter sounds (L)            | 1.0003    | 25,144  | 0           | pass |
| ca-003 | Taught receptive vocab (TR)  | 1.0004    | 25,057  | 0           | pass |
| ca-004 | Taught expressive vocab (TE) | 1.0004    | 26,195  | 0           | pass |
| ca-005 | Receptive vocab (R)          | 1.0002    | 23,896  | 0           | pass |
| ca-006 | Expressive vocab (E)         | 1.0005    | 26,033  | 0           | pass |

## Results — the single clearest same-wave adjusted link per focal skill

The table below picks, for each model, the adjusted partner with the **highest posterior probability of being positive** at its clearest wave — the fit's own "clearest" headline, chosen by certainty of direction, **not** by size of coefficient. (For TR this diverges from the largest coefficient: see the footnote.) All six clear ≥0.99, so all are graded very strong on direction. Every entry is an **association**, never a causal effect.

| Model  | Focal skill                  | Clearest adjusted same-wave partner (highest P(>0)) | Association (per +1 SD)                   | P(>0)  | Evidence    |
| ------ | ---------------------------- | --------------------------------------------------- | ----------------------------------------- | ------ | ----------- |
| ca-001 | Word reading (W)             | Letter sounds, at t2                                | +8.1 outcome items (95% cr +4.9 to +11.3) | 1.0000 | very strong |
| ca-002 | Letter sounds (L)            | Word reading, at t2                                 | +4.3 items (+2.9 to +5.4)                 | 1.0000 | very strong |
| ca-006 | Expressive vocab (E)         | Taught expressive vocab, at t3                      | +6.4 items (+2.3 to +10.7)                | 0.9993 | very strong |
| ca-005 | Receptive vocab (R)          | Expressive vocab, at t4                             | +6.2 items (+1.2 to +11.5)                | 0.9919 | very strong |
| ca-004 | Taught expressive vocab (TE) | Expressive vocab, at t3                             | +2.5 items (+1.2 to +3.7)                 | 0.9999 | very strong |
| ca-003 | Taught receptive vocab (TR)  | Receptive vocab, at t4                              | +1.1 items (+0.2 to +1.9)                 | 0.9908 | very strong |

_Footnote (TR, ca-003)._ The +1.1-item Receptive-vocab link at t4 is the **highest-probability** ("clearest") adjusted partner, not the **largest**. The biggest-magnitude adjusted coefficient in this model is actually **Taught expressive vocab at t3, +1.3 items (+0.2 to +2.3), P(>0)=0.9898 → strong**. TR is the weakest-associated focal skill in the family overall (every adjusted term has |effect| ≤ 1.34 items). The earlier column heading called this "strongest", which was inaccurate — selection is by direction-certainty, so a smaller coefficient can win the row.

## Results — all six models, per wave, with adjusted vs bivariate

The concurrent story is best seen **across the four waves** and against the **bivariate** (single-predictor) numbers, because the gap between the two is the whole regression-dilution / collinearity point: the bivariate associations are large and near-certain-positive throughout (P(>0) ≈ 0.99–1.00), while the adjusted associations — after every other same-wave skill is held fixed — are smaller, and a handful flip sign. For each model the strongest partner is traced at all four waves; second-tier partners that still exclude zero, and any notably negative adjusted terms, follow.

### ca-001 — word reading (W)

**Letter sounds is the strongest adjusted partner at every wave, and each wave excludes zero** — this is the tightest coupling in the family. Per +1 SD of letter-sound knowledge, adjusted: **t1 +2.7 items (+0.4 to +5.7), P(>0)=0.992 → very strong; t2 +8.1 (+4.9 to +11.3), 1.0000 → very strong; t3 +4.2 (+0.8 to +8.0), 0.992 → very strong; t4 +5.7 (+2.4 to +8.9), 0.9996 → very strong.** The coupling is present at t1, peaks around t2, dips at t3 and firms up again at t4 — a non-monotonic temporal pattern (descriptive only; the family models no change). The bivariate W~L is much larger at every wave (t1 +4.8, **t2 +10.5**, t3 +9.9, t4 +11.6; all P(>0) ≈ 1.0000), so adjustment roughly halves the letter-sound coefficient — the regression-dilution / shared-ability story in one line. **Blending** is a positive adjusted partner only at t1 (+1.8, +0.1 to +3.8, 0.983 → strong) and collapses to ≈0 by t2 (+0.05, 0.52 → inconclusive), even though bivariate Blending stays large (≈ +5 to +9 items). Vocabulary partners are essentially 0 once L and Blending are held fixed.

### ca-002 — letter sounds (L)

Reciprocally, **word reading is the only adjusted partner that excludes zero at all four waves: t1 +2.9 (+1.0 to +4.7), 0.998 → very strong; t2 +4.3 (+2.9 to +5.4), 1.0000 → very strong; t3 +2.1 (+0.3 to +3.6), 0.991 → very strong; t4 +1.5 (+0.2 to +2.8), 0.985 → strong.** So W↔L is a mutually strong within-wave pair. **Blending** excludes zero at t3 (+1.6, +0.1 to +3.0, 0.982 → strong). Suggestive/moderate second-tier terms: Receptive vocab at t1 (+2.2, 0.959 → moderate) and Expressive vocab at t2 (+2.0, 0.967 → moderate). A few small negative adjusted terms — t1 Taught receptive vocab −0.7 (P(>0)=0.243) and t2 Taught expressive vocab −0.8 (0.193) — are directionally negative but their intervals span zero, so they are **inconclusive**. Bivariate L~W is +4.4/+5.2/+4.0/+3.5 across waves (all ≈ 1.0000), again larger than adjusted.

### ca-003 — taught receptive vocab (TR)

The weakest-associated focal skill; only two adjusted coefficients exclude zero. **Taught expressive vocab at t3 is the largest, +1.3 items (+0.2 to +2.3), 0.9898 → strong**, and **Receptive vocab at t4 is the clearest, +1.1 (+0.2 to +1.9), 0.9908 → very strong** (the headline). Receptive vocab is otherwise suggestive/moderate — t1 +1.2 (0.969 → moderate), t3 +0.9 (0.968 → moderate). **Letter sounds at t1 is negative, −0.6 (−1.6 to +0.4), P(>0)=0.120** — directionally negative but interval crosses zero, so inconclusive. Bivariate TR associations are uniformly larger (≈ +1.2 to +2.5 items, all ≈ 1.0000).

### ca-004 — taught expressive vocab (TE)

**Expressive vocab is the strongest adjusted partner and excludes zero at every wave: t1 +1.3 (+0.3 to +2.5), 0.994 → very strong; t2 +1.9 (+0.6 to +3.2), 0.998 → very strong; t3 +2.5 (+1.2 to +3.7), 0.9999 → very strong (headline); t4 +1.7 (+0.4 to +3.0), 0.993 → very strong.** Second-tier: **Taught receptive vocab** excludes zero at t3 (+1.4, +0.4 to +2.4, 0.997 → very strong) and t4 (+1.6, +0.3 to +2.8, 0.993 → very strong). **Notable negative:** at t3, **Blending −0.9 (−1.9 to +0.05), P(>0)=0.032 → probability negative 0.968 = moderate evidence of a negative adjusted association** — a suppression signal once vocabulary is held fixed, not an effect. Bivariate TE~Blending is positive (+1.4 to +2.5), so the sign flip is created by adjustment.

### ca-005 — receptive vocab (R)

**Expressive vocab is the strongest adjusted partner, excluding zero at t2 +5.8 (+0.3 to +11.7), 0.982 → strong; t3 +6.0 (+0.4 to +12.0), 0.982 → strong; t4 +6.3 (+1.2 to +11.5), 0.9919 → very strong (headline).** **Taught receptive vocab** excludes zero at t4 (+5.1, +0.7 to +9.8, 0.987 → strong) and is suggestive/moderate earlier (t2 +3.9, 0.934; t3 +3.7, 0.966). **Notable negative:** at t1, **Word reading −1.8 (−5.0 to +1.6), P(>0)=0.146** — directionally negative, interval crosses zero, so **inconclusive** — despite a bivariate W~R of **+7.8** (≈1.0000). Intervals are wide throughout (small n, heavily collinear vocabulary measures), which is why even +6-item effects only reach "strong" rather than "very strong".

### ca-006 — expressive vocab (E)

**Taught expressive vocab is the strongest adjusted partner: t1 +4.3 (+0.6 to +8.5), 0.988 → strong; t2 +4.8 (+0.9 to +9.0), 0.993 → very strong; t3 +6.4 (+2.3 to +10.7), 0.9993 → very strong (headline); t4 +3.7, 0.970 → strong (suggestive/moderate, interval near zero).** **Receptive vocab** excludes zero at t4 (+4.7, +0.5 to +9.1, 0.987 → strong). **Notable negative:** at t2, **Word reading −1.8 (−5.3 to +2.1), P(>0)=0.180** — directionally negative but interval crosses zero, so **inconclusive** — despite a bivariate W~E of **+8.0** (≈1.0000). As in ca-005, the vocabulary measures are mutually collinear, so adjusted intervals are wide.

## The one-paragraph story

Within any given wave the skills cluster exactly as reading science would predict. **Word reading and letter-sound knowledge are the most tightly linked pair**: adjusted W~~L peaks at +8.1 items/SD at t2 and excludes zero at every wave, and the reciprocal L~~W excludes zero at every wave too. **The vocabulary measures cluster among themselves** — receptive with expressive (R~~E ≈ +6 items/SD), expressive with taught-expressive (E~~taught-E ≈ +5–6), and the smaller taught pairs (TE~~E ≈ +2.5, TR~~R ≈ +1.1). A handful of adjusted coefficients turn **negative** once collinear partners are held fixed — word reading inside the two vocabulary models, blending inside the taught-expressive model — but only the TE-Blending term (moderate) even reaches non-trivial evidence; the rest have intervals that span zero. These are **suppression/collinearity artefacts of the observed-score measures, not effects**. Bivariately, every skill pair is large and near-certain positive, so the adjusted-vs-bivariate gap is the whole point: these are clean _descriptions_ of how skills co-occur in the cohort, useful for understanding structure, saying nothing about what drives what.

## What is causal

**Nothing — intentionally.** Every coefficient is a per-wave adjusted (or bivariate) association describing how skills co-occur within a timepoint among children alike on age and the other same-wave skills. The models condition on post-treatment skills, and the predictors are latent-ability-confounded noisy observed scores, so no causal pathway is licensed and no term may be read as "X drives Y". The family is also strictly **cross-sectional per wave**: it models no growth or change, so it cannot speak to word-reading _growth_ across timepoints — only to the concurrent co-occurrence structure at each wave. For the causal questions (the randomised intervention effects) see [ITT](202607161800-findings-itt.md); for the assumption-heavy "which skill carries the effect" decomposition see [mediation](202607161800-findings-mediation.md).
