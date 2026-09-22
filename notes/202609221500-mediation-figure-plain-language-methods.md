<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# How the "words routed through each skill" figure is calculated

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

Date: 2026-09-22 — **Status: INFORMATIONAL** (methods explainer; no decision recorded).

## What this note is for

A plain-language figure shows the intervention's whole effect on word reading (+2.3 words) as a dashed line, with four bars beneath it labelled letter sounds (+2.1), taught words: understanding them (+0.5), taught words: saying them (+0.3) and nonword reading (+0.1). The caption says the bars are "words of the reading gain that the model routes through each skill". This note explains, without the statistical shorthand, how those numbers are produced, what the dots and bars mean, why the bars do not add up to the whole effect, and what the numbers can and cannot claim. It is written for a reader comfortable with ordinary regression and confidence intervals but not necessarily with Bayesian or causal-mediation methods.

The figure itself is not stored in this repository. The mapping from bars to models below is inferred from the registered model catalogue rather than read from a saved figure script, and this note was written without fitted mediation outputs to hand, so it describes the method rather than re-checking the plotted values.

## Four separate models, not one

Each bar comes from its own fitted model. The repository registers one single-mediator mediation model per skill:

| Bar                              | Mediator                     | Model             |
| -------------------------------- | ---------------------------- | ----------------- |
| Letter sounds                    | Letter-sound knowledge       | `lrp-rli-med-059` |
| Taught words: understanding them | Taught receptive vocabulary  | `lrp-rli-med-080` |
| Taught words: saying them        | Taught expressive vocabulary | `lrp-rli-med-068` |
| Nonword reading                  | Nonword decoding             | `lrp-rli-med-074` |

Each model asks the same question about its own skill: of the intervention's whole effect on word reading, how much is carried by the intervention's effect on this skill, and how much arrives by every other route? The "through this skill" part is what the bar shows. Its statistical name is the natural indirect effect. The remainder is the natural direct effect, and within one model the two add up exactly to the whole effect.

## The data each model uses

The intervention was a waiting-list trial. Between the first and second assessments (t1 to t2) one group received the teaching and the other did not, and which group a child joined was decided at random. After t2 both groups were taught, so t1 to t2 is the only window in which "taught versus not yet taught" is a randomised comparison. Every mediation model uses that window only, with one row per child.

Each row holds the child's group, their word-reading score at t1 and t2, their score on the mediating skill at t1 and t2, and a fixed set of baseline (t1) measures used as adjustment terms. In the letter-sound model these are age, expressive and receptive vocabulary, hearing status and speech production (with indicator variables for missing values so no child is dropped for a missing covariate), plus the t1 scores of both the mediator and word reading. The other three models each include their own mediator's t1 score and phonological memory (word and nonword repetition), and each drops one or two of the letter-sound model's terms; the exact set is declared in each model module and recorded in its `config.json`. About 53 children contribute to the letter-sound model; the count differs slightly by model because each needs complete data on its own mediator.

Scores are counts of items correct out of a fixed test maximum. Word reading has 79 items. The models treat each score as a count out of its maximum rather than as a continuous measure, which respects the floor at zero and the ceiling at the maximum.

## The fitted model

Two regressions are fitted together on the logit (log-odds) scale using a Beta-Binomial likelihood. The Beta-Binomial is a binomial that allows more spread between children than a coin-flip model would, which these test scores show.

```text
Skill at t2:        logit(skill_t2)   = a0 + a_G·group + a_M·logit(skill_t1) + a_A·age + Σ a_c·baseline_c + a_W·logit(reading_t1)
Word reading at t2: logit(reading_t2) = b0 + b_G·group + b_M·z(logit skill_t2) + b_GM·group·z(logit skill_t2) + b_W·logit(reading_t1) + b_A·age + Σ b_c·baseline_c + b_L·logit(skill_t1)
```

In words: the first equation says how the teaching and the baseline measures move the skill. The second says how the teaching, the skill at t2 and the same baseline measures move word reading. Both equations condition on the same list of baseline variables, including both t1 scores; the counterfactual step below only makes sense if the two legs share one adjustment set. The skill enters the second equation as its standardised logit (`z`), and there is a group-by-skill product term so the model can express a skill that matters more, or less, for taught children than for untaught ones.

Weakly informative priors from the shared constructors keep the estimates sensible in a sample of this size. The treatment terms are centred on zero with a standard deviation of 0.5 on the logit scale, the skill-to-reading slope has a standard deviation of 1 per standard deviation of the skill, the product term has 0.3, and the extra-spread (dispersion) parameter has a half-normal prior with scale 50. The posterior is sampled with the No-U-Turn Sampler through `nutpie`. Every fit must pass the usual computation gate (R-hat at most 1.01, effective sample sizes at least 400, energy diagnostic at least 0.3, no divergent transitions) before any number is read.

## Splitting the effect: what the model imagines

The split is not "coefficient a_G times coefficient b_M". That shortcut is only valid for a linear model on an unbounded scale, and these are logit models on bounded counts. Instead the fitted model is used to imagine three situations for every child in the sample and every posterior draw of the parameters:

1. **Taught, with the skill level teaching produces.** Run the first equation with `group = taught` to get the child's predicted skill distribution, then run the second equation with `group = taught` at each possible skill score. Call the resulting expected word-reading score A.
2. **Not taught, with the skill level no teaching produces.** The same, with `group = not taught` in both equations. Call it B.
3. **Taught, but with the skill level no teaching produces.** `group = taught` in the second equation, but the skill drawn from the first equation with `group = not taught`. Call it C.

Then, per draw and averaged over children:

- Whole effect = A − B. The dashed line.
- Direct effect (every other route) = C − B.
- Indirect effect (through this skill) = A − C. The bar.

Because A − B equals (C − B) plus (A − C), the two parts add exactly to the whole within one model.

Two practical details matter for trusting the numbers. First, "the skill level teaching produces" is not a single number but a distribution over the possible scores. The code walks through every score from zero to the mediator test's maximum, weights each by its fitted Beta-Binomial probability and sums, so this integration is exact rather than simulated. Second, the averaging is over the actual children's covariate profiles, not a constructed average child, so the reported effect is an average over the fitted sample.

## From proportions to words, and what the dots and bars mean

The three quantities come out as proportions of the word-reading test. Multiplying by the 79 items gives the "words" scale used in the figure. A whole effect of +2.3 words therefore means: averaging over the children in the fit, the model's expected t2 word-reading score is 2.3 items higher when taught than when not yet taught.

Repeating the calculation for every posterior draw gives a full distribution for each quantity. Under the house convention the figure summarises that distribution as follows:

- The dot is the posterior median.
- The thick bar is the central 50% interval (25th to 75th percentiles).
- The thin bar is the central 89% interval (5.5th to 94.5th percentiles).

Read the intervals as ranges of values consistent with the data under the model, not as significance tests. The fit also records the posterior probability that each quantity is above zero, which is the direction statement to quote alongside a bar. For the taught-word and nonword bars the 89% interval includes zero, so the honest reading is "the model routes little of the gain through this skill, and the data cannot pin the amount down", not "zero".

## Why the bars do not add up to the whole effect

The four bars sum to about 3.0 words against a whole effect of 2.3. That is expected, and it is not an error. Each bar is the indirect effect from its own model, in which that skill is the only mediator and everything else, including the other three skills, sits in the direct effect. Letter sounds and taught vocabulary move together in taught children, so a letter-sound model attributes some shared change to letter sounds and a vocabulary model attributes the same shared change to vocabulary. Only a model with several mediators fitted together can hand out non-overlapping shares. The repository registers two-mediator versions (`lrp-rli-med-064`, letter sounds with expressive vocabulary, and `lrp-rli-med-066`, letter sounds with phoneme blending) and a composite reading-route model (`lrp-rli-med-062`) for that purpose. When presenting the four bars together, say that they are four separate views of the same gain and are not shares of a pie.

The dashed whole-effect line is one model's total. Each model has its own total, and they should agree with each other and with the trial's intention-to-treat word-reading contrast in sign and rough size, but not to the decimal, because each fit uses a slightly different adjustment set and complete-case sample.

## What these numbers are not

The methods document classes every mediation result as a model-based decomposition, not an identified causal pathway. Three limits bind, and a public version of the figure needs them beside it.

- **Unmeasured confounding of the skill-to-reading link.** Randomisation guarantees that group is unconfounded, but it does nothing for the link between a child's skill and their reading. A child who is generally more able will tend to gain on both, and the adjustment terms cannot capture that fully. The "through this skill" share therefore rests on an assumption of no unmeasured skill-to-reading confounding that cannot be checked in these data.
- **Dose is caused by the treatment.** The number of teaching sessions a child received depends on their group and affects both the skill and reading. Natural direct and indirect effects are not identified in that structure even under randomisation, and adjusting for sessions does not repair it, because sessions are downstream of the treatment (VanderWeele, Vansteelandt and Robins, 2014). Interventional versions of the estimand (`lrp-rli-med-078` for the letter-sound route) avoid the cross-world quantity but still need the no-unmeasured-confounding assumption above (Hejazi et al., 2022), so they are weaker-assumption targets, not identified effects.
- **Skill and reading are measured at the same visit.** The mediator and the outcome are both t2 scores, so the model decomposes a within-visit association and assumes the skill came first. Where it converged, a sensitivity fit with reading measured at t3 is reported on the model page as a check on direction.

The nonword-reading model carries an extra caution: many children score zero on nonword reading at t1 and a large share still do at t2, so the mediator has little room to vary and its bar is detection-limited rather than informative about a small effect.

None of this makes the figure wrong. It makes the figure a description of how a fitted model apportions the gain under stated assumptions, which is what the caption's "the model routes" wording is trying to say.

## Where to find it

- Model specifications: `src/language_reading_predictors/statistical_models/lrp_rli_med_059.py` and the three companions named above. Each module's docstring records its adjustment set and the reasoning behind it.
- Model construction: `build_mediation_model` in `src/language_reading_predictors/statistical_models/factories/mediation.py`.
- The three-situation calculation: `decompose` in `src/language_reading_predictors/statistical_models/mediation.py`, with the exact count integration in `count_cells` in `mediation_integration.py` beside it.
- Outputs: `output/statistical_models/models/{model_id}-{config}/mediation_summary.csv`, whose rows are `total`, `NDE`, `NIE` and `proportion_mediated`, and whose `words_*` columns hold the medians and intervals plotted in the figure. `key_findings.json` in the same folder carries the sentence-form summary.
- Report pages: `docs/models/lrp-rli-med-059/index.qmd` and companions, which render the shared `docs/models/_partials/_results_mediation.qmd` partial.
- Method background: the mediation entries in `METHODS.md` and the adjustment-set decision in `notes/202607142340-lrp264-mediation-adjustment-dsep.md`.

## References

- Hejazi NS, Rudolph KE, van der Laan MJ, Díaz I (2022). Nonparametric causal mediation analysis for stochastic interventional (in)direct effects. _Biostatistics_. DOI [10.1093/biostatistics/kxac002](https://doi.org/10.1093/biostatistics/kxac002).
- Imai K, Keele L, Tingley D (2010). A general approach to causal mediation analysis. _Psychological Methods_, 15(4), 309–334. DOI [10.1037/a0020761](https://doi.org/10.1037/a0020761).
- VanderWeele TJ, Vansteelandt S, Robins JM (2014). Effect decomposition in the presence of an exposure-induced mediator-outcome confounder. _Epidemiology_, 25(2), 300–306. DOI [10.1097/EDE.0000000000000034](https://doi.org/10.1097/EDE.0000000000000034).
