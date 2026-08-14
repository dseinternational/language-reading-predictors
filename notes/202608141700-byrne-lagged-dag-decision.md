<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore Byrne MacDonald basread basdig bpvs trog readgrp dagitty LCSM -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne lagged-DAG decision: adopt a narrow reciprocal template

**Decision for #338/#409, 2026-08-14.** Adopt `dag/dag-reading-language-memory-lagged.dagitty` as the working two-slice companion to the contemporaneous Byrne graph. This clears the graph-design gate; it does not register or authorise a fitted cross-lagged model. A pre-fit simulation and explicit model specification remain required because the usable repeated-measures sample is small.

## Question anchored to the primary source

Byrne, MacDonald and Buckley (2002) asked whether learning to read enhances language and memory development in children with Down syndrome and reported no supporting evidence over the two-year study. The paper measured three annual occasions and treated vocabulary, receptive grammar, auditory digit recall and visual recall as distinct outcomes. The prepared repository extract adds later waves, but it does not include either visual-recall variable.

The implementable source-compatible reverse hypotheses are therefore deliberately narrow:

- prior BAS word reading (`basread_t`) → later receptive vocabulary (`bpvs_t1`);
- prior BAS word reading (`basread_t`) → later receptive grammar (`trog_t1`);
- prior BAS word reading (`basread_t`) → later auditory short-term memory (`basdig_t1`).

This is not a test of “memory” in full: the paper's visual-memory part cannot be reproduced until those variables are recovered. It is also not a general search over every later score.

## Structural decisions

1. **Use the copied-per-wave structure.** Each slice contains the contemporaneous Byrne cascade; age and all observed measures carry forward. This is the RLM analogue of the adopted RLI Option A graph. A pure-lagged graph would assert that vocabulary, memory, decoding and comprehension have no within-year relationships, which is implausible at an annual interval.
2. **Pre-specify exactly three reverse edges.** The three edges above operationalise the paper's named language-and-memory hypothesis. Do not add reverse edges from word reading into spelling, reading comprehension, number skills or the ability indicators. The paper's discussion suggests a possible reading threshold for spelling, but that is a separate exploratory question and `basspel` still has an unconfirmed denominator. Reading comprehension is already downstream of word reading within each slice.
3. **Retain autoregressive carry-over.** Every observed measure has `X_t → X_t1`; age has `age_t → age_t1`. The lagged arrows therefore ask whether prior reading adds information about later change beyond each target's prior level, rather than confusing stable level differences with development.
4. **Promote hearing to an explicit latent common cause.** The primary paper notes that hearing impairment is common in Down syndrome, but the prepared extract has no hearing measure. `HS` is therefore latent and points to digit recall, vocabulary, grammar, word reading and spelling in both slices. General ability (`GA`) remains latent with noisy `bassim`/`basmat` indicators. These unmeasured common causes prevent a causal interpretation of any reverse coupling.
5. **Keep cohort and selection limitations structural.** `readgrp` is an observational population marker, never a treatment. The reading-matched group was selected on `basread`; pooled analyses that include it inherit that selection. A model may use those groups for contextual precision, but it must not reinterpret the association as a Down-syndrome-specific causal effect.

## Data and model gate

The four pre-specified measures have confirmed bounded-score denominators, so the `basspel`/`woco`/`basnum` manuals do not block this narrow question. The dataset-lineage gate still blocks publication.

The source-compatible waves 1–3 have 21 Down-syndrome children and 68 children across all three groups with all four measures observed. The corresponding balanced counts are 18/52 through wave 4 and 15 Down-syndrome children through wave 5. The extra waves improve transition count but are not part of the published two-year design, and wave 5 is Down-syndrome-only. A future model must therefore distinguish the published waves 1–3 analysis from later-wave sensitivity analyses rather than pool them without qualification.

Before `lrp-rlm-lcsm-001` is registered, run a simulation-based feasibility gate over the actual missingness patterns. Compare at least a Down-syndrome-only pooled-coupling LCSM with a three-group model using group-specific trajectory intercepts and shared couplings. Reject any specification that cannot recover plausible coupling signs with calibrated uncertainty. A free RI-CLPM remains out of scope: the RLI feasibility study had effectively no directional power at approximately 54 children over four waves, and this cohort's scientifically central group is smaller.

## Interpretation contract

Every cross-lagged coefficient is a temporally ordered, adjusted predictive association. It may show that earlier reading contains information about later language or auditory-memory change; it cannot show that teaching reading caused that change. Stable general ability, unmeasured hearing, measurement error, reading-matched selection and attrition all remain alternative explanations. No threshold or tail-probability rule can turn those coefficients into causal effects.

Primary source: Byrne, A., MacDonald, J., & Buckley, S. (2002). Reading, language and memory skills: A comparative longitudinal study of children with Down syndrome and their mainstream peers. _British Journal of Educational Psychology, 72_(4), 513–529. <https://doi.org/10.1348/00070990260377497>.
