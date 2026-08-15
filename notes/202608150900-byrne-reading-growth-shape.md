> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne baseline verbal reasoning and reading-trajectory shape (#409 D4)

## Decision

Register one bounded growth-family port, `lrp-rlm-gc-001`, rather than mechanically copying the multivariate RLI pair. The model asks whether wave-1 BAS Similarities is associated with the shape of the BAS word-reading trajectory over the three annual waves reported by Byrne, MacDonald and Buckley (2002). Both measures have confirmed denominators and confirmed instrument identities, so the model does not inherit the `basspel`, `woco`, `basnum` or `basmat` measurement blockers.

## Analysis contract

The outcome is `basread`; the baseline ability proxy is `bassim`, transformed with the registered 21-item denominator and standardised. A child must have wave-1 similarities and word reading observed at two or more of waves 1–3. This retains 87 of the prepared extract's 97 children: four are excluded for missing wave-1 similarities and six for fewer than two observed reading waves. Remaining reading-score gaps are masked rather than imputed.

The three historical cohorts have separate population intercepts, age slopes, child-level intercept scales and Beta-Binomial concentrations. The model uses child random intercepts but no child random slopes: one outcome over three waves does not support 87 additional latent slope residuals reliably. The `gamma` and `delta` associations are shared across reading groups because the available sample does not support three separate ability coefficients reliably. `gamma` is the association between a 1-SD higher baseline similarities score and the word-reading logit growth rate; `delta` is the association with word-reading level at the pooled mean age.

## Interpretation boundaries

The coefficients are descriptive conditional associations, not causal effects. Reading group was not randomised, BAS Similarities is a noisy verbal-reasoning proxy rather than latent general ability itself, and the reading-matched cohort was selected on the outcome. The shared `gamma` also assumes that the ability–growth association is sufficiently similar across the three groups. Waves 4–5 are excluded: wave 4 is attrition-sensitive and wave 5 is Down-syndrome-only, so neither belongs in the primary paper-compatible trajectory question.

## Computational gate

The first test-tier candidate included child random slopes. Although its sampler passed, observation-level PSIS-LOO was unreliable for 23 of 249 cells, including 3 with Pareto-k above 1. This motivated the pre-interpretation simplification to random intercepts only. The reduced `rep-lite` fit passed the computational gate with zero divergences, maximum R-hat 1.0044 and minimum effective sample size 1,952. Its PSIS-LOO warning improved but did not disappear: 8 of 249 cells exceeded 0.70 and 1 exceeded 1. The score must not be used for model comparison; if the source gate is later cleared, an observation-level influence or exact-LOO check is still required before interpreting the association. Release remains `inputs_unresolved` because the prepared 97-participant extract has not been reconciled with the separate 96-participant raw export.

Reference: Byrne, A., MacDonald, J., & Buckley, S. (2002). Reading, language and memory skills: a comparative longitudinal study of children with Down syndrome and their typically developing peers. _British Journal of Educational Psychology, 72_(4), 513–529. https://doi.org/10.1348/00070990260377497
