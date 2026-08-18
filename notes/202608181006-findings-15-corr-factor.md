> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `corr_factor` and `long_corr_factor` families — skill domains behind the tests

**Read `findings-00-overview` first.** This note covers the 5 `corr_factor` models and the single `long_corr_factor` model. **Nothing here is causal.**

## The data

**Mostly the RLI trial.** Four `corr_factor` models use 51 children at baseline, one row each. The `long_corr_factor` model uses all four waves — 54 children, 216 rows. One further model (`rlm-mm-001`, 75 children from the historical cohort) is **withheld**; see below.

## What the model is for

Every test score contains measurement error, and no single test perfectly captures the skill it targets. If you correlate two noisy tests directly, the correlation is dragged towards zero by that noise — a well-known attenuation.

These are **measurement models**. They treat the underlying skill domains — vocabulary, code/decoding, grammar — as unobserved quantities that several tests each measure imperfectly, then estimate the correlations _between the domains_ with measurement error removed. The result answers "how related are these abilities?" rather than "how related are these test scores?".

The `long_corr_factor` model extends this across all four waves with loadings held constant over time, so the same construct is being measured the same way at each wave, and separates the stable trait part of a skill from the wave-specific state part.

## What was found

| Relationship         | Latent correlation | 89% range      |
| -------------------- | ------------------ | -------------- |
| Vocabulary ↔ grammar | **+0.83**          | +0.67 to +0.93 |
| Vocabulary ↔ code    | **+0.80**          | +0.60 to +0.94 |

These correlations are **very high** — high enough to raise a question the models cannot settle: whether these are genuinely distinct domains at all in this sample, or facets of one broadly-varying ability. A latent correlation of 0.83 between vocabulary and grammar means the two domains share about two-thirds of their variance once measurement error is removed.

Two prior-sensitivity variants give +0.76 and +0.80 for the vocabulary–code correlation against the main model's +0.80, so these are not artefacts of the prior.

The longitudinal model translates its couplings into item terms: at wave 1, one additional letter-sound item corresponds to **+0.60 receptive vocabulary items** (89% +0.50 to +0.69).

## Why the high correlations matter for the rest of the project

This is the most useful thing this family contributes, and it is not really about the correlations themselves.

If vocabulary, code and grammar are as tightly bound as these estimates suggest, then any model that adjusts for one while estimating the effect of another is adjusting for something largely overlapping. That makes the individual coefficients in the `mechanism` and `concurrent` families harder to interpret, not easier — you cannot cleanly separate the contribution of two things that move together this closely.

It also gives a concrete measurement to set beside the `mechanism` family's failed negative controls. Both point the same way: a large shared component runs through these skills, and models that do not measure it directly will distribute it across whatever coefficients they have.

## A note on the structural coefficients

Two models (`mm-002`, `mm-102`) add a structural leg: a latent code factor predicting word reading, with measurement error accounted for. In principle this is the better-measured version of the letter-sound-to-reading relationship.

The **correlations from these models are robust**; the **structural coefficients should be read more cautiously**. Latent-variable models of this kind have a known difficult geometry — the factor scale and the coefficients trade off against each other — and this family has historically been the hardest in the project to fit. In this run all four trial models converge cleanly with zero divergences, which is an improvement on their history, but the structural estimates remain sensitive to specification in a way the correlations are not.

## The withheld model

`rlm-mm-001`, the historical-cohort version, is **withheld at the inputs stage**: three of its measures have no confirmed maximum score, so the bounded-count likelihood is resting on a guess about the instrument. See the overview and the run note. It is withheld because a measurement fact is missing, not because it produced an unwelcome answer.

## What these models cannot tell you

**Correlation between domains says nothing about direction or cause.**

**High correlations do not prove the domains are one thing.** They are consistent with that, and also with genuinely distinct abilities that develop together.

**The measurement model is an assumption.** Which tests load on which domain was specified in advance; a different assignment would give different correlations.

**These are between-child correlations at a moment.** They do not describe how a single child's skills move together over time — that is the `lcsm` and `long_corr_factor` question.

## Model inventory

Five of six pass the convergence gate with zero divergences and are publishable: `mm-001` (three-domain measurement model), `mm-002` (errors-in-variables code → word reading), `mm-101` and `mm-102` (prior sensitivity), and `lcf-001` (four-wave longitudinal). `rlm-mm-001` is withheld at the inputs stage.
