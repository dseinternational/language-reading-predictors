> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne stacked-transition estimand decision

Date: 2026-08-16. Issue: #409 D2.

## Decision

Register `lrp-rlm-adj-006` as a pooled annual-transition association model for BAS word-reading progress over waves 1→2, 2→3, 3→4 and 4→5. Condition each post score on its own pre-wave score, use transition-specific intercepts and a child random intercept, retain observational reading-group indicators as nuisance terms, and share the predictor slopes across transitions. Transform bounded predictors to Haldane logits and standardise every predictor separately within transition so the pooled coefficient represents a one-standard-deviation contrast among children at the same developmental wave.

Use only the confirmed-input battery: BPVS receptive vocabulary, TROG receptive grammar, BAS digit recall, BAS similarities and age. Exclude BAS number skills because its ceiling remains provisional, despite its confirmed instrument identity.

## Scope condition and sensitivities

The complete-case analysis contains 225 transition rows from 84 children: 76 for waves 1→2, 76 for 2→3, 56 for 3→4 and 17 for 4→5. The first three transitions contain all three observational reading groups; the final transition contains only children with Down syndrome because wave 5 is blank for both comparison cohorts. The four-transition fit remains the declared primary panel extension, but it must not be described as common late-age evidence across cohorts.

Two sensitivities are therefore required. First, a common-horizon refit excludes the waves-4→5 tail and estimates the same pooled slopes through wave 4. Second, an independent transition-specific-slope refit shows whether the common coefficient masks materially different associations; its final-transition coefficients are expected to be weakly identified and are not promoted to headline estimands. The pooled model uses child-level PSIS-LOO, because leaving out one transition row while retaining the same child's other rows would overstate out-of-child predictive performance.

## Interpretation

All slopes are adjusted longitudinal associations. Temporal ordering, baseline conditioning and the child random intercept do not identify the effect of changing a predictor. The random intercept partially pools stable heterogeneity but does not remove time-stable confounding as a child fixed effect would. Complete-case selection and the reading-matched cohort's selection on word-reading level remain substantive limitations.

## Rep-lite validation

The four-chain, 4,000-draw rep-lite fit passed the automatic sampling gate with zero divergences, maximum R-hat 1.004, minimum effective sample size 1,363 and minimum chain BFMI 0.415. All five bivariate, two prior-width, common-horizon and transition-specific-slope refits also passed their convergence checks with zero divergences.

Child-level PSIS-LOO was less reassuring: 9 of 84 children had Pareto $k>0.7$, although none exceeded 1. The approximate predictive score is therefore not fully reliable without exact leave-one-child-out refits or an integrated new-child likelihood. This limitation concerns predictive validation rather than the sampler's representation of the fitted coefficient posterior, but it must remain visible and the LOO estimate must not be used for fine model ranking. Publication remains blocked independently by the unresolved 96-versus-97 participant source-lineage discrepancy.
