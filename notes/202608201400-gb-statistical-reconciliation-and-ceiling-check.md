> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

<!-- cspell:ignore erbto deapp gbg gbl -->

# Closing issue #554: the step-1/step-2 reconciliation, and the letter-sound ceiling check

**Date:** 2026-08-20. Two items completed after the full refit (`notes/202608200800-full-refit-both-layers-2026-08.md`): the part of item 5 that compares the gradient-boosting rankings against the `gain_factors` and `pooled_levels` coefficients rather than only against the `horseshoe` ranking, and item 6, the optional one-dimensional ceiling check. Both are diagnostic; neither changes a release decision, a stored fit or a finding.

## Item 5, completed: gradient boosting against the structural families

The horseshoe comparison was reported with the refit; this adds the two structural comparisons the issue also names (`scripts/compare_gb_vs_statistical.py`, written to `output/statistical_models/comparison/gb_vs_statistical.csv`). Both sides are reduced to construct symbols; the gradient-boosting side takes the maximum permutation importance among a construct's columns, which is also how the horseshoe comparison aggregates.

### Gains: `gbg-012` against `gf-001` (7 shared constructs, Spearman rho = +0.36)

| Construct                 | `gain_factors` coefficient | its rank | GB permutation importance | its rank |
| ------------------------- | -------------------------: | -------: | ------------------------: | -------: |
| Own word-reading baseline |                     +0.752 |        1 |                    −0.005 |        3 |
| Age                       |                     −0.124 |        2 |                    +0.038 |        1 |
| Letter sounds             |                     +0.068 |        3 |                    −0.046 |        7 |
| Receptive vocabulary      |                     +0.060 |        4 |                    −0.010 |        5 |
| Nonword decoding          |                     +0.048 |        5 |                    −0.002 |        2 |
| Phoneme blending          |                     +0.030 |        6 |                    −0.007 |        4 |
| Expressive vocabulary     |                     +0.014 |        7 |                    −0.020 |        6 |

The rank correlation is +0.36, but the more useful observation is in the third column: **only age has a positive permutation importance.** For every other construct, shuffling the column _improved_ the model's out-of-sample accuracy — which is what a variable carrying no usable signal looks like in a model whose pooled out-of-sample R² is 0.08. The tree ensemble is not ranking these predictors weakly; on this target it is not ranking them at all. Its one positive signal, age, is also the coefficient `gain_factors` resolves most clearly after the child's own baseline, and the two agree on its direction.

So the step-1 layer corroborates the negative half of question 3's answer — gains are barely predictable from anything measured here — and contributes nothing to the positive half. The `gain_factors` ordering below age is not confirmed by an independent method; it rests on that family's own adjustment.

### Levels: `gbl-012` against the `pooled_levels` between-child coefficients (5 shared constructs, rho = +0.40)

| Construct             | Between-child coefficient | its rank | GB permutation importance | its rank |
| --------------------- | ------------------------: | -------: | ------------------------: | -------: |
| Letter sounds         |                    +1.607 |        1 |                    +0.238 |        1 |
| Receptive vocabulary  |                    +1.014 |        2 |                    −0.016 |        5 |
| Expressive vocabulary |                    +0.973 |        3 |                    +0.075 |        2 |
| Phonological memory   |                    +0.880 |        4 |                    +0.012 |        3 |
| Speech production     |                    +0.459 |        5 |                    +0.003 |        4 |

Both layers put letter sounds first by a clear margin, and four of the five constructs have a positive permutation importance here against one of seven in the gain model. The single disagreement is receptive vocabulary, which the pooled model ranks second and the tree ensemble ranks last — expected when two collinear vocabulary measures are offered to a tree, since it can take either and the permutation test then finds the other one redundant.

**Five constructs is too few for the rank correlation to carry weight**, and it is reported for completeness rather than as evidence. The agreement worth quoting is the ordering at the top, not the coefficient.

### What the whole reconciliation says

Across all three comparisons — horseshoe, gain factors, pooled levels — the pattern is the same one the refit found from the models' own accuracy: **the two layers agree about levels and disagree about change.** That is not a defect in either. It is what two honest methods do when asked to rank predictors of a target that carries little predictable signal, and it is the reason this project's headline claims about progress rest on the randomised contrast rather than on any predictor ranking.

## Item 6, completed: the letter-sound ceiling check

`notes/202608191600-moderation-items-scale-not-2d-surface.md` raised a competing explanation for a pattern in the moderated `mechanism` fits: every moderated letter-sound → word-reading fit returns a negative `gamma_int` of about the same size whatever the moderator, including age. Because the Haldane-corrected logit stretches the top of the letter-sound scale, the interaction column `z_L·z_M` correlates about 0.7 with `z_L²`, so a negative interaction might be curvature at the exposure ceiling rather than moderation. The note called for one refit with the interaction built on a basis that does not stretch the ceiling.

`scripts/mech_ceiling_check.py` runs it on `mech-061` (letter sounds × blending → word reading). The check is exact rather than approximate: in a curve-mechanism moderated fit `z_mech_logit` enters **only** the interaction — the mechanism main effect is an HSGP on the raw logit — so replacing that one data vector changes the interaction basis and nothing else. Rows, priors and sampling settings are the registered fit's own, including its declared `target_accept` of 0.999.

| Interaction basis                   | `gamma_int`             | P(negative) | corr(`z_L·z_M`, `z_L²`) | Divergences |
| ----------------------------------- | ----------------------- | ----------: | ----------------------: | ----------: |
| Standardised logit (as registered)  | −0.106 [−0.201, −0.014] |       0.967 |                   0.709 |           0 |
| Count-standardised                  | −0.177 [−0.298, −0.061] |       0.993 |                   0.569 |           0 |
| Top-clipped logit (90th percentile) | −0.127 [−0.230, −0.026] |       0.978 |                   0.590 |           0 |

**The premise measures as the note predicted and its consequence does not follow.** The correlation between the interaction column and the squared exposure is 0.709 on the registered basis, matching the note's "about 0.7", and both alternatives reduce it (to 0.57 and 0.59). If the negative interaction were ceiling curvature, `gamma_int` should move toward zero as that correlation falls. It does the opposite: the count-standardised basis gives a _more_ negative interaction, and the clipped basis is essentially unchanged.

Two limits on how far to take this. The coefficient is per standard deviation of whichever basis is used, and the three bases have different relationships to the underlying count, so **the magnitudes are not on a common scale** — what the comparison establishes is the sign and its evidential strength, not that the interaction is larger under one basis. And this is one fit; the other eight moderated fits were not re-run, on the reasoning that the mechanism proposed was a property of the shared exposure axis, so refuting it on the axis is enough.

The conclusion for the family note stands as written: for blending the substitution survives translation to words, and it now also survives removal of the ceiling stretch. The hypothesis recorded in the August note as "some of the uniform negativity may be curvature at the letter-sound ceiling" is **not supported**.

## What this closes

With these two items, every acceptance criterion on issue #554 is met. One deviation is recorded rather than hidden: the criterion asked for the August run record to be updated, and the refit wrote a new run record instead, because the run replaces that baseline rather than extending it. The optional `exposure_lag` setting from #553 was not implemented and is not carried forward.
