> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Findings: the `concurrent` family — which skills go together at each timepoint

**Read `findings-00-overview` first.** This note covers the 13 models in the `concurrent` family. **Nothing here is causal**, and the reason is unusually stark.

## The data

**Both studies.** Eleven models use the RLI trial and two use the separate Reading, Language and Memory cohort. Sample size is wave-specific rather than fixed: the full RLI word-reading fit has 53 children at timepoints 1–3 and 51 at timepoint 4; `rlm-ca-001` has 96, 88, 78 and 61, while `rlm-ca-002` has 88, 88, 78 and 61. The historical timepoint-4 result is explicitly an available-case, attrition-sensitive extension beyond the source paper's audited timepoints 1–3, so cross-wave changes can partly reflect a changing analysis sample.

These models are fitted **separately at each timepoint** — waves 1 to 4 for the trial, waves 1 to 4 for the historical cohort. Each fit uses **one row per child**: everything measured at the same moment. Nothing is stacked or collapsed across waves; instead the same model is run once per wave and the results compared across waves.

## What the model is for

The question is descriptive: **at a single moment, which skills go with which?** The outcome is regressed on model-specific sets of skills measured at the _same_ time, with age and a group nuisance term. Some RLI models also adjust for pre-declared background covariates including block-design ability, hearing, speech and phonological memory, with missingness indicators where required.

Effects are reported per standard deviation of each predictor, translated into outcome items.

## The interpretive problem, stated plainly

Everything here is measured **simultaneously**, which removes even the weak temporal ordering the other families have. If letter-sound knowledge and word reading are measured on the same day and move together, the data are equally consistent with letter sounds supporting reading, reading practice teaching letter sounds, both being driven by general ability, or all three at once.

The models also adjust for other skills measured at the same time, which introduces a second hazard: conditioning on something that sits on the path between two variables, or on a common effect of both, can shrink a real relationship or manufacture a spurious one. These are **conditional** associations, and they change when the adjustment set changes — as this family itself demonstrates below.

## What was found

The strongest relationships in the trial cohort, at the wave where each was clearest:

| Outcome                    | Strongest same-wave correlate | Association per +1 SD    |
| -------------------------- | ----------------------------- | ------------------------ |
| Word reading (t2)          | Letter sounds                 | +9.0 items [+6.3, +11.8] |
| Letter sounds (t2)         | Word reading                  | +4.2 items [+3.3, +5.1]  |
| Expressive vocabulary (t3) | Taught expressive vocabulary  | +6.5 items [+2.6, +10.5] |
| Receptive vocabulary (t3)  | Word reading                  | +5.8 items [+1.9, +9.9]  |
| Receptive grammar (t3)     | Taught receptive vocabulary   | +2.0 items [+0.8, +3.1]  |
| Basic concepts (t1)        | Taught receptive vocabulary   | +1.5 items [+0.7, +2.3]  |
| Phoneme blending (t4)      | Word reading                  | +0.8 items [+0.3, +1.2]  |

**Letter sounds and word reading are each other's strongest correlate**, which is the clearest illustration of why these cannot be read directionally. The same relationship appears from both ends.

**Word reading appears as the strongest correlate of receptive vocabulary** (+6.2 items). That coefficient alone does not show that word reading causes vocabulary: reverse direction, general progress and other common causes remain compatible with it. The result is a useful reminder of what these coefficients are.

**Adjustment changes the answer substantially.** Three models estimate the same letter-sound-to-word-reading relationship at t2 with different covariates:

| Model    | Adjustment                             | Association per +1 SD     |
| -------- | -------------------------------------- | ------------------------- |
| `ca-010` | Letter sounds + minimal background set | +11.0 items [+8.2, +13.5] |
| `ca-001` | Six-skill + broader background set     | +9.0 items [+6.3, +11.6]  |
| `ca-011` | Letter sounds + nonword + minimal set  | +7.2 items [+4.4, +10.3]  |

These are not three sequentially nested adjustments. `ca-011` adds nonword decoding to the same minimal specification as `ca-010`, reducing the letter-sound estimate from +11.0 to +7.2 items; `ca-001` instead uses a substantially broader six-skill and background adjustment set and should not be read as the middle step. All estimate conditional associations, but what is held constant differs. No single number is _the_ answer, and quoting one without its adjustment set would be misleading.

In the **historical cohort**, verbal reasoning is the strongest same-wave correlate of BAS word reading at wave 1 (+7.5 items [+3.8, +11.3]; 2026-08-22 refit under the dispersion-scale concentration prior, unchanged at one decimal) — consistent with the general-ability reading of these associations, since verbal reasoning is about as close to a general-ability marker as this battery has.

## What these models cannot tell you

**No direction at all.** Simultaneous measurement provides no temporal ordering.

**No causal claim, and no mechanism.** For the mechanism question with at least a temporal ordering see the `mechanism` family; for the randomised question see `itt`.

**The coefficients depend on the adjustment set**, as shown above.

**These are not predictions.** An association at one wave does not establish that measuring the predictor forecasts later outcomes; that is what `lcsm` and `growth` attempt.

## Model inventory

All 13 pass the convergence gate with zero divergences and are publishable. Trial cohort: `ca-001` (word reading), `002` (letter sounds), `003`/`004` (taught vocabulary), `005`/`006` (broad vocabulary), `007` (blending), `008` (basic concepts), `009` (grammar), `010`/`011` (letter-sound to word-reading adjustment variants). Historical cohort: `rlm-ca-001` (BAS word reading), `rlm-ca-002` (BPVS receptive vocabulary).
