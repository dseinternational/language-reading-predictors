> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Findings: the `historical_joint` family — do skills travel together in the historical cohort?

**Read `findings-00-overview` first.** This note covers the 2 models in the `historical_joint` family. **Nothing here is causal.**

## The data

**The Reading, Language and Memory cohort only** — the separate observational study, not the RLI trial. 71 children: `jc-001` uses waves 1 to 3 as a complete-case core plus wave 4 for all groups and wave 5 for the Down syndrome group as an available-case extension (284 rows); `jc-002` uses the balanced waves 1 to 3 only (213 rows). Data are stacked by wave.

Three measures are modelled together: BAS word reading, BPVS receptive vocabulary, and BAS recall of digits (a verbal short-term memory task).

## What the model is for

The `historical_growth` family fits each measure separately. This family fits three **at once**, so it can ask how they relate.

Each measure keeps its own group-by-wave means and its own group-specific variability, exactly as in the separate models. What is added is a correlation structure linking a child's stable deviations across the three measures: if a child is persistently above average on reading, are they also above average on vocabulary and memory?

The two models ask different versions of this. `rlm-jc-001` asks about **between-child** correlation — do children who are generally stronger on one measure tend to be stronger on another? `rlm-jc-002` asks about **within-child** correlation — when a particular child has an unusually good wave on one measure, do they also have an unusually good wave on another?

## What was found

**Between children, the measures are strongly related.** The clearest coupling is between BAS word reading and BPVS receptive vocabulary: a stable-level correlation of **+0.71** (89% +0.55 to +0.83). Children who read better also have larger vocabularies, consistently across the study.

**Within children, no correlation pair met the model's resolution rule.** The wave-specific standard deviation was resolvable for BAS word reading (median 0.315 on the log-odds scale), but not for BPVS vocabulary (0.032) or BAS digit recall (0.044) against the pre-specified 0.05 threshold. Because every pair contains at least one unresolved scale, `rlm-jc-002` withholds all three within-child correlations.

## Non-resolution is a limit on identification, not a null result

This pair illustrates a distinction worth understanding.

The between-child correlation of +0.71 says that **stable differences between children are shared across skills**. It is compatible with shared stable factors, including general ability, but this symmetric correlation does not identify which factor or combination of factors produces the relationship.

The within-child result does **not** show that the correlations are zero, that almost all covariance is between children, or that the measures are stable year to year. It says the data do not identify those correlations at the chosen resolution threshold. The posterior intervals remain very wide: for example, the reading–vocabulary within-child correlation has an 89% interval from −0.62 to +0.78. Effects in either direction remain compatible with the fit.

The model reporting "not resolvable" rather than promoting a posterior median is the correct behaviour. A correlation is weakly identified when either underlying within-child scale is too small for this design to resolve. That is an uncertainty statement, not affirmative evidence that no within-child coupling exists.

## What these models cannot tell you

**Correlation says nothing about direction or cause.** Reading and vocabulary moving together does not mean either produces the other. The trial-cohort `mechanism` models describe baseline-conditional adjusted associations, and the `mediation` models provide an assumption-dependent decomposition that is not causally identified; neither resolves direction or cause here.

**These are not the trial children.**

**The between-child correlation is shared across the three reading groups.** The model estimates one correlation structure, not a separate one per group, so it cannot say whether the relationship differs between children with Down syndrome and their peers.

**A resolvable within-child correlation might exist with more waves, more precise measurement or a larger sample.** Non-resolution here reflects the information available for the within-child scale parameters; it is not proof of absence.

## Model inventory

Both models pass the convergence gate with zero divergences and are publishable: `rlm-jc-001` (between-child correlated growth) and `rlm-jc-002` (within-child joint coupling).
