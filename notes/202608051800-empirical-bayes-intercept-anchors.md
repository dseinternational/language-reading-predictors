<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Empirical-Bayes intercept anchors: labelling the three families that use one (2026-08-05)

Records the implementation of conditions 2 and 3 of the ruling on #390's empirical-Bayes intercept prior, and what surveying the suite for the same pattern turned up. Condition 1 is not implemented and is stated as outstanding at the end.

## The ruling

#390 P1 reported that the DiD family builds its logit-scale intercept from the pooled t1 outcome, and that those same t1 observations then enter the likelihood. That is an empirical-Bayes prior, not one independent of the data, and it means the reported prior-predictive distribution is partly informed by the outcomes it is supposed to be checked against.

Frank ruled option B on 2026-07-24 — keep the anchor, label it — with three conditions:

1. the sensitivity fit must use a **genuinely independent** prior, not a wider σ around the same anchor, since widening σ leaves the mean data-dependent, which is the actual defect;
2. the report must state the **prior-predictive limitation** explicitly, not just carry the EB label — "that's the cost you're accepting";
3. apply the same label to **growth and LCSM**, or record why not — "growth's anchor is the weaker case and will be found".

## Three families anchor, and they anchor on different things

Condition 3 asked for growth and LCSM specifically. Surveying the factories for the pattern confirms exactly three, and the differences between them matter more than the shared label:

| Family   | Parameter                    | Anchored on                                     | Comment                                                            |
| -------- | ---------------------------- | ----------------------------------------------- | ------------------------------------------------------------------ |
| `did`    | `alpha` (via `alpha_offset`) | pooled observed **t1** logit                    | Baseline only; the t2 contrast the family reports is a later wave. |
| `lcsm`   | `mu1`                        | observed **wave-1** mean logit per outcome      | Baseline only, same shape as DiD.                                  |
| `growth` | `alpha`                      | **grand mean observed logit across every wave** | The weak case.                                                     |

**Frank's prediction was right.** The growth anchor is not a baseline anchor at all: it averages every wave of the outcome, so the prior mean is computed from the same observations the model's growth trajectory is fitted to, not merely from a pre-period. Where DiD and LCSM use a statistic of the first wave to locate a level the model then evolves away from, growth uses a statistic of the whole series to locate the series' own centre. The label is the same; the exposure is not.

**The DiD dose variants are not anchored at all** — they build an ordinary free `alpha ~ Normal(0, 1.5)`. Worth stating because "the DiD family uses an empirical anchor" is the natural shorthand and it is wrong for three of the fourteen fits.

## What was found while labelling it, and it is a defect in its own right

The growth family's priors table described its intercept as `Normal(0, 1.5)` — quoting `alpha_prior`'s docstring — while the distribution column beside it read `Normal(<constant>, 1.5)`. The prior described was not the prior fitted. The cause is routing: growth registers its anchored intercept inline, but the name `alpha` resolves to the shared constructor, so the table inherited a rationale for a zero-centred prior the model does not use. LCSM's `mu1` reached no rationale at all and shipped an empty cell.

So the labelling work was not only additive. Two of the three anchored families were, in different ways, **describing their own intercept prior wrongly in the published report**, and neither would have been caught by a check on the distribution column, which was correct throughout.

## How it is implemented

Detection is by **distribution, not by parameter name**. `alpha` is anchored in `growth` and a free zero-centred deviation everywhere else, so a name-keyed rule would mislabel the whole ANCOVA suite. A rendered `Normal(<constant>, ...)` mean is the signature of a location computed from data; across all 194 fits it matches growth's `alpha` and LCSM's `mu1` and nothing else. The other `<constant>` renderings in the suite are LKJ dimensions and a ZeroSumNormal shape — structural arguments, not locations — and are excluded by the `Normal(` prefix. A regression test pins both the positives and those negatives.

DiD is matched by name instead, because its anchor is applied downstream in a `Deterministic` and its free `alpha_offset` therefore renders zero-centred. Its family-specific prose stays in the DiD pipeline, but the empirical-Bayes sentence itself is imported from `priors.EMPIRICAL_BAYES_SENTENCE`, so the family prose and the suite-wide label cannot drift apart.

Condition 2 is met in `_prior_predictive.qmd`, which now prints a per-family note above the check saying which statistic the prior is centred on, that those same observations enter the likelihood, and — the substantive part — that **the check is correspondingly weaker: it cannot detect a prior that is well-centred only because it was told where the data are**, so it should be read as a check on spread rather than on location.

## Outstanding

**Condition 1 is not implemented.** It needs an independent-prior sensitivity fit — a genuinely external or weakly-informative scale-aware location, not a wider σ around the same anchor — and therefore a compute cycle rather than a labelling pass. It stays open on #390.

The growth case should take priority within that work when it happens. Its anchor is the one that uses the whole outcome series, so it is both the most exposed and the case where an independent-prior refit would be most informative about whether the anchor is doing any real work.
