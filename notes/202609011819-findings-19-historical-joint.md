> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `historical_joint` family — do skills travel together in the historical cohort?

**Read `findings-00-overview` first.** This note covers the 3 models in the `historical_joint` family. **Nothing here is causal.** All 3 pass the convergence gate with zero divergences and are publishable (2026-09-01 rebuild, refitted at the #626 branch commit `b18ea944`). The within-scale prior sensitivity `rlm-jc-102`, registered by the #588 review, is reported here for the first time.

## The data

**The Reading, Language and Memory cohort only.** 71 children. `jc-001` uses waves 1 to 3 as a complete-case core plus the available-case extension waves (284 rows); `jc-002` and `jc-102` use the balanced waves 1 to 3 only (213 rows), so each child contributes exactly three waves. Three measures are modelled together: BAS word reading, BPVS receptive vocabulary and BAS recall of digits.

## What the model is for

The `historical_growth` family fits each measure separately; this family fits three at once, keeping each measure's group-by-wave means and adding a correlation structure between children's stable deviations across the measures. `jc-001` asks about **between-child** correlation: do children who are persistently stronger on one measure tend to be stronger on another? `jc-002` asks about **within-child** correlation: when a child has an unusually good wave on one measure, do they also have one on another? `jc-102` repeats `jc-002` with a wider prior on the within-child scale, because that prior decides which measures clear the resolvability threshold.

## What was found

**Between children, the measures are strongly related.** In `jc-001` the stable-level correlations are word reading ↔ vocabulary **+0.69** (89% +0.53 to +0.80), word reading ↔ digit recall +0.66 (+0.51 to +0.77) and vocabulary ↔ digit recall +0.54 (+0.33 to +0.70). The balanced `jc-002` gives +0.77, +0.70 and +0.59. Children who read better also have larger vocabularies and longer digit spans, consistently across the study.

**Within children, no correlation pair met the model's resolution rule, under either prior.** The wave-specific residual scale is resolvable for word reading (median 0.32 logits, 89% 0.26 to 0.38) but not for vocabulary (0.03, probability 0.29 of exceeding the 0.05 threshold) or digit recall (0.04, probability 0.44). Because every pair contains at least one unresolved scale, all three within-child correlations are withheld from interpretation. Their posteriors are very wide — reading ↔ vocabulary +0.17 (89% −0.62 to +0.78), reading ↔ digits +0.32 (−0.42 to +0.79), vocabulary ↔ digits +0.04 (−0.63 to +0.67) — and effects in either direction remain compatible with the fit. Widening the within-scale prior (`jc-102`) leaves the per-measure scales and the verdict unchanged (word reading 0.316 against 0.316; the other two identical), so the non-resolution is not an artefact of the tighter prior.

**Non-resolution is a limit on identification, not a null result.** The between-child +0.69 says stable differences are shared across skills and is compatible with shared stable factors including general ability. The within-child result says the data do not identify those correlations at the chosen resolution; it is not evidence that within-child couplings are zero.

## The new-child prediction target

This family is the one where importance sampling cannot serve the declared prediction target — a child contributing four waves across three correlated measures puts the leave-one-child-out posterior too far from the full one — so it takes grouped child-level K-fold refits. All fifteen fold refits converged. The new-child expected log predictive densities are −2414 (SE 47) for `jc-001` and −1840 (SE 27) and −1838 (SE 26) for `jc-002` and `jc-102`; the first is not comparable with the other two (284 against 213 rows). Paired over the 71 children, `jc-102` minus `jc-002` is +1.1 (paired SE 0.8), inconclusive under the standing rule, which is the predictive half of the sensitivity the report promised: the correlation conclusions do not detectably depend on the within-scale regularisation. Held-out child-level probability-integral-transform medians sit between 0.50 and 0.54 for every measure.

## What these models cannot tell you

**Correlation says nothing about direction or cause.** **These are not the trial children.** **One correlation structure is shared across the three reading groups**, so the family cannot say whether the relationship differs for children with Down syndrome. **A resolvable within-child correlation might exist with more waves, more precise measurement or a larger sample.**

## Model inventory

All 3 pass the convergence gate with zero divergences and are publishable: `rlm-jc-001` (between-child correlated growth), `rlm-jc-002` (within-child joint coupling) and `rlm-jc-102` (its wider within-scale prior sensitivity). The `.pre-626-20260901` directories beside them are the pre-#626 fits.
