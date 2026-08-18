> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `historical_joint` family — do skills travel together in the historical cohort?

**Read `findings-00-overview` first.** This note covers the 2 models in the `historical_joint` family. **Nothing here is causal.**

## The data

**The Reading, Language and Memory cohort only** — the separate observational study, not the RLI trial. 71 children across **waves 1 to 4**, contributing 284 rows in one model and 213 in the other. Data are stacked by wave.

Three measures are modelled together: BAS word reading, BPVS receptive vocabulary, and BAS recall of digits (a verbal short-term memory task).

## What the model is for

The `historical_growth` family fits each measure separately. This family fits three **at once**, so it can ask how they relate.

Each measure keeps its own group-by-wave means and its own group-specific variability, exactly as in the separate models. What is added is a correlation structure linking a child's stable deviations across the three measures: if a child is persistently above average on reading, are they also above average on vocabulary and memory?

The two models ask different versions of this. `rlm-jc-001` asks about **between-child** correlation — do children who are generally stronger on one measure tend to be stronger on another? `rlm-jc-002` asks about **within-child** correlation — when a particular child has an unusually good wave on one measure, do they also have an unusually good wave on another?

## What was found

**Between children, the measures are strongly related.** The clearest coupling is between BAS word reading and BPVS receptive vocabulary: a stable-level correlation of **+0.71** (89% +0.55 to +0.83). Children who read better also have larger vocabularies, consistently across the study.

**Within children, the model found nothing to report.** `rlm-jc-002` returned no resolvable within-child correlation: no pair of measures had wave-specific fluctuations large enough — above 0.05 on the log-odds scale — for a correlation between them to be estimated at all.

## The null result is informative, and it is not a failure

This pair is a nice illustration of a distinction worth understanding.

The between-child correlation of +0.71 says that **stable differences between children are shared across skills**. That is the general-ability picture that recurs throughout this project: children who are doing well are doing well across the board.

The within-child result says that once you remove each child's stable level, **the wave-to-wave wobble is too small to correlate**. Practically, this means almost all the covariation between these measures is the stable between-child part, and there is little left over. It also means the measures are reasonably stable year to year — a child's position does not bounce around much.

The model reporting "not resolvable" rather than producing a number is the correct behaviour. Estimating a correlation between two quantities that are themselves near zero produces a number with no meaning attached, and the fit declined to publish one.

## What these models cannot tell you

**Correlation says nothing about direction or cause.** Reading and vocabulary moving together does not mean either produces the other; the `mechanism` and `mediation` families take that question up in the trial cohort, where an intervention was actually delivered.

**These are not the trial children.**

**The between-child correlation is shared across the three reading groups.** The model estimates one correlation structure, not a separate one per group, so it cannot say whether the relationship differs between children with Down syndrome and their peers.

**A resolvable within-child correlation might exist with better measurement.** The absence here reflects small wave-specific variation in these instruments, which is a property of the measures as much as of the children.

## Model inventory

Both models pass the convergence gate with zero divergences and are publishable: `rlm-jc-001` (between-child correlated growth) and `rlm-jc-002` (within-child joint coupling).
