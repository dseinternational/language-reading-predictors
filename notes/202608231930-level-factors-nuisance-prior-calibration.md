> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Level-factors #584 decision 4: calibrating the dispersion and child-SD priors, 2026-08-23

Decision 4 changes two nuisance priors for the level family. Both scales are **calibration exercises, not constants copied from another family**, and this note records how each was derived and what it does and does not buy. The decision itself is in `notes/202608231800-level-factors-584-decisions.md`.

## The dispersion prior: `1/sqrt(kappa) ~ HalfNormal(0.25)`

### What the re-derivation found

The RLM historical families adopted `inv_sqrt_kappa_prior(0.25)` with a scale chosen to be calibration-preserving at **their** denominators (n = 18–90). Its docstring explicitly scoped itself there and deferred the high-denominator RLI outcomes. The question for this family was whether 0.25 still holds at n up to 170.

It does, and for a reason worth stating: the Beta-Binomial variance inflation over the Binomial is `(n + kappa) / (1 + kappa)`, which is **monotone in kappa at fixed n**. Matching the median inflation is therefore matching the median `kappa`, and one scale does every denominator at once. Matching `HalfNormal(50)`'s median kappa of 33.7 exactly gives a scale of **0.2553**; the registered 0.25 gives 35.2, within 3% on median inflation at every level denominator:

| n   | median inflation, HN(50) on `kappa` | median inflation, HN(0.25) on `1/sqrt(kappa)` | ratio | P(within 10% of Binomial): old → new |
| --- | ----------------------------------: | --------------------------------------------: | ----: | ------------------------------------ |
| 10  |                                1.26 |                                          1.25 | 0.992 | 0.0753 → 0.329                       |
| 18  |                                1.49 |                                          1.47 | 0.988 | 0.0007 → 0.242                       |
| 24  |                                1.66 |                                          1.64 | 0.985 | 0.0000 → 0.209                       |
| 32  |                                1.89 |                                          1.86 | 0.982 | 0.0000 → 0.180                       |
| 79  |                                3.24 |                                          3.16 | 0.974 | 0.0000 → 0.114                       |
| 170 |                                5.85 |                                          5.67 | 0.969 | 0.0000 → 0.078                       |

So **0.25 is kept**, not re-derived to a family-specific value: the exercise confirmed the RLM constant rather than replacing it, and using the same number across families is worth more than the third-decimal improvement 0.2553 would buy.

### What actually changes

The last column. The old prior gave the near-Binomial region — "no extra-Binomial dispersion beyond the child random intercept", an ordinary hypothesis for a bounded count — essentially **zero** mass at every denominator above 10. The new one gives it 8–33%. The fitted posteriors show the cost of forbidding it:

| Outcome | `kappa` median | prior mass above it, HN(50) | prior mass above it, new |
| ------- | -------------: | --------------------------: | -----------------------: |
| R       |            170 |                      0.0007 |                    0.242 |
| E       |            198 |                      0.0001 |                    0.225 |
| TE      |             98 |                      0.0503 |                    0.316 |
| W       |             83 |                      0.0970 |                    0.341 |

R and E sat past the old prior's 99th percentile. A dev-preset refit of R under the new prior moves its `kappa` median from 170 to **411** — the data wanted the region the prior had excluded.

### Where this sits relative to the ITT family

The 2026-08-22 ITT audit (finding 5) reached the same conclusion for **its** high-denominator outcomes and switched `lrp-rli-itt-006` and `lrp-rli-itt-022` (both E) to the dispersion parameterisation, leaving the rest of the ITT suite on the concentration prior. The level family adopts it **family-wide**, on this family's own evidence: power scaling flagged the dispersion parameter in eight of the nine graded level fits, not only the high-denominator ones, and the calibration table above shows the switch is median-neutral at every denominator including n = 10.

That is a deliberate difference in scope between the two families, and it is worth a decision if the ITT suite is revisited: the argument for family-wide adoption is not denominator-specific.

The ITT note also records a caution worth carrying: R and E have a 170-item ceiling against observed maxima of 82 and 77, so substantial genuine overdispersion is expected there and a large fitted `kappa` should be read as "little residual dispersion **after** the child intercept", not as "the test behaves Binomially".

## The child-SD prior: `sigma_child ~ HalfNormal(1.0)`

Chosen on the **prior predictive**, not fitted to the posteriors. `sigma_child` is the SD of child deviations on the level logit, so a scale is defensible if the range of child-level scores it implies is one a reader of this cohort would call plausible, and indefensible if it forbids the spread the tests were built to measure.

Middle 95% of children implied at each candidate's **median** `sigma_child`:

| prior           | on a low-scoring measure (p = 0.15) | on a mid measure (p = 0.30) | on blending (p = 0.49) |
| --------------- | ----------------------------------- | --------------------------- | ---------------------- |
| HalfNormal(0.5) | 0.08 – 0.25                         | 0.18 – 0.45                 | 0.33 – 0.65            |
| HalfNormal(1.0) | 0.04 – 0.40                         | 0.10 – 0.62                 | 0.20 – 0.78            |
| HalfNormal(1.5) | 0.02 – 0.56                         | 0.06 – 0.76                 | 0.12 – 0.87            |

And at each candidate's **95th percentile**, which is the check against asserting absurd spreads:

| prior           | low-scoring measure | mid measure | blending    |
| --------------- | ------------------- | ----------- | ----------- |
| HalfNormal(0.5) | 0.03 – 0.55         | 0.06 – 0.75 | 0.12 – 0.87 |
| HalfNormal(1.0) | 0.00 – 0.89         | 0.01 – 0.95 | 0.02 – 0.98 |
| HalfNormal(1.5) | 0.00 – 0.98         | 0.00 – 0.99 | 0.00 – 1.00 |

**HalfNormal(1.0) is the smallest scale whose median admits the cohort's plausible spread and whose upper tail does not assert impossible ones.** At 0.5 the median says the middle 95% of children span 0.18 to 0.45 of a mid-difficulty measure — narrower than these tests resolve, in a cohort whose word-reading scores run from 0 to most of 79 words. At 1.5 and above the 95th percentile asserts children spanning essentially 0 to 1, which is more than the measures distinguish and would let the child intercept absorb variation belonging to the wave and arm terms.

The fitted posteriors are a **check, not the criterion** — a prior fitted to its own posterior is no prior — but a prior that puts every fitted value past its 99th percentile has been refuted by the data it met:

| prior           | fitted medians past its 99th pct | past its 95th pct | smallest P(prior > fitted) |
| --------------- | -------------------------------: | ----------------: | -------------------------: |
| HalfNormal(0.5) |                                2 |                 3 |                     0.0008 |
| HalfNormal(1.0) |                                0 |                 0 |                     0.0949 |
| HalfNormal(1.5) |                                0 |                 0 |                     0.2656 |

## What this does _not_ establish

Dev-preset refits confirm the machinery and show the direction of travel — `kappa` rises where the old prior was binding (R 170 → 411, W 83 → 101), `sigma_child` rises where it was pinned (P 1.67 → 2.46, W 1.39 → 1.50) — but they say nothing reliable about whether the **power-scaling flags clear**. Dev fits are short chains, and power scaling on a short chain is noisy; the stored flags were measured on reporting traces, so the two are not comparable.

Indeed the dev numbers hint the answer may be mixed: `sigma_child`'s prior sensitivity does not obviously improve for W or R, and for an outcome whose true child SD is genuinely small (R, around 0.24) a wider prior is if anything _more_ prior-sensitive, because the parameter is weakly identified there either way. That is a property of the parameter, not a defect in the scale, and it is exactly what the two new registered sweep axes exist to measure:

- `--axis kappa` sweeps the dispersion scale over (0.125, **0.25**, 0.5);
- `--axis sigma_child` sweeps the child SD over (**0.5**, **1.0**, 1.5) — including the pre-decision 0.5, so the sweep answers "did changing this prior move the answer?" directly rather than by comparison with a differently-run fit.

Both write their own CSV and are never gate evidence; the gate's contract remains the treatment-prior sweep.

The real before/after is the reporting refit, with the current stored fits backed up as the comparator.
