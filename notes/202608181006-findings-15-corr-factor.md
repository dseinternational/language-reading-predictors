> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Findings: the `corr_factor` and `long_corr_factor` families — skill domains behind the tests

**Read `findings-00-overview` first.** This note covers the 5 `corr_factor` models and the single `long_corr_factor` model. **Nothing here is causal.**

## The data

**Mostly the RLI trial.** Four `corr_factor` models use 51 children at baseline, one row each. The `long_corr_factor` model uses all four waves — 54 children, 216 rows. One further model (`rlm-mm-001`, 75 children from the historical cohort) is **withheld**; see below.

## What the model is for

Every test score contains measurement error, and no single test perfectly captures the skill it targets. If you correlate two noisy tests directly, the correlation is dragged towards zero by that noise — a well-known attenuation.

These are **measurement models**. They treat the underlying skill domains — vocabulary, code/decoding, grammar — as unobserved quantities that several tests each measure imperfectly, then estimate the correlations _between the domains_ with measurement error removed. The result answers "how related are these abilities?" rather than "how related are these test scores?".

The `long_corr_factor` model extends this across all four waves with loadings held constant over time, so the same construct is being measured the same way at each wave, and separates the stable trait part of a skill from the wave-specific state part.

## What was found

| Model    | Relationship         | Latent correlation | 89% range      |
| -------- | -------------------- | ------------------ | -------------- |
| `mm-001` | Vocabulary ↔ grammar | **+0.83**          | +0.67 to +0.93 |
| `mm-001` | Vocabulary ↔ code    | **+0.78**          | +0.57 to +0.93 |
| `mm-002` | Vocabulary ↔ code    | **+0.80**          | +0.60 to +0.94 |

These correlations are **very high** — high enough to raise a question the models cannot settle: whether these are genuinely distinct domains at all in this sample, or facets of one broadly-varying ability. A latent correlation of 0.83 between vocabulary and grammar means the two domains share about two-thirds of their variance once measurement error is removed.

The direct loading-prior sensitivity `mm-101` gives +0.76 for vocabulary–code against `mm-001`'s +0.78, so that correlation is broadly stable to the tested loading geometry. `mm-102` also gives +0.80, but it changes only the structural code-slope prior and therefore is not a correlation-prior sensitivity. Moreover, power-scaling flags several cross-sectional parameters for potential prior–data conflict. The high correlations recur across specifications, but “not an artefact of the prior” would be stronger than the checks establish.

The longitudinal model adds two important results. Its per-wave latent correlations are stable rather than identical to the cross-sectional fits: vocabulary–code is about +0.70 at every wave, vocabulary–grammar about +0.88, and code–grammar about +0.61. It attributes about 94–96% of each domain's latent variance to the stable child trait rather than wave-specific state. Its item-scale conditional translation at wave 1 is that one additional letter-sound item corresponds to **+0.60 receptive-vocabulary items** (89% +0.50 to +0.69); that is an association, not a longitudinal causal coupling.

## Why the high correlations matter for the rest of the project

This is the most useful thing this family contributes, and it is not really about the correlations themselves.

If vocabulary, code and grammar are as tightly bound as these estimates suggest, then any model that adjusts for one while estimating the effect of another is adjusting for something largely overlapping. That makes the individual coefficients in the `mechanism` and `concurrent` families harder to interpret, not easier — you cannot cleanly separate the contribution of two things that move together this closely.

It also gives a concrete measurement to set beside the non-specific component the `mechanism` family's negative controls expose. Both are compatible with a substantial shared component across skills, but neither identifies its source or determines how another model distributes it across coefficients.

## A note on the structural coefficients

All four cross-sectional trial models contain a structural leg, not only `mm-002` and `mm-102`. `mm-001` and its loading-prior companion `mm-101` regress word-reading gain on all three latent factors; their code slopes are +0.20 [−0.15, +0.56] and +0.19 [−0.15, +0.55], with P(> 0) = 0.82 and 0.81 — suggestive on the project's ladder, with intervals well inside both signs. `mm-002` uses a different, code-focused adjustment specification and gives +0.35 [+0.09, +0.60]. Widening only that focal slope prior in `mm-102` moves it to +0.46 [+0.16, +0.76], showing substantive prior sensitivity in the structural estimate.

The **correlations are consistently high across these specifications but still carry prior-sensitivity qualifications**; the **structural coefficients are more visibly specification- and prior-sensitive**. Latent-variable models of this kind have a difficult geometry — the factor scale and the coefficients trade off against each other — and this family has historically been the hardest in the project to fit. In this run all four cross-sectional trial models converge cleanly with zero divergences, which is an improvement on their history. Every structural coefficient remains an adjusted association confounded by latent ability, not a causal effect.

## The withheld model

`rlm-mm-001`, the historical-cohort version, is **withheld at the inputs stage**: three of its measures have no confirmed maximum score, so the bounded-count likelihood is resting on a guess about the instrument. See the overview and the run note. It is withheld because a measurement fact is missing, not because it produced an unwelcome answer.

## What these models cannot tell you

**Correlation between domains says nothing about direction or cause.**

**High correlations do not prove the domains are one thing.** They are consistent with that, and also with genuinely distinct abilities that develop together.

**The measurement model is an assumption.** Which tests load on which domain was specified in advance; a different assignment would give different correlations.

**The cross-sectional fits describe between-child latent correlations at baseline.** `long_corr_factor` extends those correlations across waves and separates stable trait from wave-specific state; it does not estimate whether change in one skill causes or precedes change in another. The cross-lagged change question belongs to `lcsm`.

## Model inventory

All six pass the convergence gate with zero divergences. Five are publishable: `mm-001` (three-domain measurement model), `mm-002` (errors-in-variables code → word reading), `mm-101` and `mm-102` (prior sensitivity), and `lcf-001` (four-wave longitudinal). `rlm-mm-001` is computationally clean but withheld at the inputs stage for unresolved denominator provenance.
