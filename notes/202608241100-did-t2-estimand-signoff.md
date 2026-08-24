> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Sign-off: the `did` family's t2 estimand is the arm-gap **level**, not a gap change

**Date:** 2026-08-24. **Decision owner:** Frank Buckley (DSE International). **Raised by:** the 2026-08-20 DiD family code review (finding 2, `notes/202608201406-did-family-code-review.md`) and re-raised as an open sign-off item by the 2026-08-24 review, issue #576 finding 4.

## The question left open

Despite carrying "difference-in-differences" in its name, the `did` family's binary models estimate a **free arm gap at each wave**. The parameter named `tau_t2` is therefore the covariate-adjusted t2 arm-gap _level_ — the immediate-minus-waitlist difference at t2 — and not the differenced quantity `tau_t2 - arm_gap_t1`. The 2026-08-20 review corrected the prose that had promised a differencing which does not happen, and recorded the choice itself as an open question:

> whether the DiD family should also be re-parameterised on the t1-referenced gap-change, mirroring #552, or whether keeping the two families on deliberately different randomised estimands is the more informative triangulation.

The question matters because after #552 the level-factor family's causal headline **is** a gap change (`d_grp_time[t2]`, a difference-in-differences of adjusted levels), so the two families now report different randomised estimands for the same trial. They coincide exactly when the realised t1 gap is zero and diverge otherwise.

## Decision

**Keep the gap level.** The `did` family continues to publish `tau_t2` as the covariate-adjusted t2 arm-gap level, and the level-factor family continues to publish the t1-referenced gap change. The two are deliberately different randomised estimands of the same trial, and reporting both is more informative than making them agree by construction.

Three reasons.

1. **Both are validly identified.** Randomisation makes the t2 arm gap a causal contrast without any baseline adjustment at all; baseline terms in a randomised comparison are for precision, not identification. Neither parameterisation is more "correct" — they answer "how far apart are the arms at t2?" and "how much further apart did they get?" respectively.

2. **Two estimands are a real triangulation; one is a duplicate.** The `did` family exists as a longitudinal sensitivity analysis beside the ITT suite. Its value comes from differing from ITT in specification, not from agreeing with it. Making it agree with the level family's parameterisation as well would remove the one place the design's sensitivity to baseline handling is visible in the published results.

3. **The disagreement is diagnostic, not a defect.** Where the two families disagree materially on an outcome, that is information about the realised t1 imbalance for that outcome — precisely what `arm_gap_t1` exists to surface — rather than a bug in either.

## What the decision requires, and what was added

The decision is only defensible if the _soft_ baseline adjustment the level parameterisation carries is documented and tested rather than assumed. Both were missing, and both are supplied by the #576 remediation.

**The mechanism, stated.** With free per-wave gaps, a realised t1 imbalance has exactly two places to go: the tightly regularised `arm_gap_t1` (`Normal(0, 0.3)`) and the arm-mean of the shared child random intercepts (`sigma_child ~ HalfNormal(0.5)`). Whatever the intercepts absorb is netted out of all three wave gaps, ANCOVA-style. `tau_t2` therefore sits between the unadjusted t2 gap and a fully baseline-corrected one, with the mix set by those two prior widths and nothing else. This is now stated in the resolved run plan's estimand text, in `METHODS.md`, in `docs/models/README.md` and in the family docstring.

**The allocation, made variable and swept.** `DiDModelSettings` gains `arm_gap_t1_prior_sigma` and `sigma_child_prior_sigma`, and **LRPDID104** is the registered estimand-matched sensitivity for word reading: `arm_gap_t1` widened 0.3 → 1.0 and `sigma_child` 0.5 → 1.0, with `tau_t2` held at its tier prior. This is a different question from the registered treatment-prior sweep, which varies `tau_t2`'s own prior and leaves the allocation untouched.

**The estimand, tested under material imbalance.** The production recovery test simulated a **zero** baseline gap, so it could not distinguish the level from the change. Two new tests in `tests/statistical_models/test_did_statistical_validation.py` simulate a material +0.45-logit t1 gap and require that (a) `tau_t2` recovers the level (0.60) and _not_ the change (0.15 must fall outside its interval), (b) `arm_gap_t1` recovers the realised imbalance rather than being shrunk away by its tight prior, and (c) the derived `tau_t2 - arm_gap_t1` recovers the true change. A third test fits the same panel under both prior settings and requires both posteriors to cover the truth, so the LRPDID104 companion is a sensitivity rather than a second, differently-defined model.

## Consequences

- A reader comparing a `did` t2 number with a `lf` t2 number is comparing a **level** with a **change**; where the realised t1 gap for that outcome is non-negligible they should not be expected to match, and `arm_gap_t1` is the quantity that says by how much.
- LRPDID104 must be fitted at the reporting tier before the sign-off's empirical half is complete. Until then this note records the decision and the tests that pin it, not evidence that the published word-reading contrast is insensitive to the allocation.
- If LRPDID104 shows a material shift, this decision should be revisited: that would mean part of the published number is a function of the `arm_gap_t1` prior, which is the case for re-parameterising onto the t1-referenced change after all.
