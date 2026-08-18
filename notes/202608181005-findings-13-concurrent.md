> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `concurrent` family — which skills go together at each timepoint

**Read `findings-00-overview` first.** This note covers the 13 models in the `concurrent` family. **Nothing here is causal**, and the reason is unusually stark.

## The data

**Both studies.** Eleven models use the RLI trial (53–54 children); two use the separate Reading, Language and Memory cohort (`rlm-ca-001` with 96 children, `rlm-ca-002` with 88).

These models are fitted **separately at each timepoint** — waves 1 to 4 for the trial, waves 1 to 4 for the historical cohort. Each fit uses **one row per child**: everything measured at the same moment. Nothing is stacked or collapsed across waves; instead the same model is run once per wave and the results compared across waves.

## What the model is for

The question is descriptive: **at a single moment, which skills go with which?** The outcome is regressed on the other skills measured at the _same_ time, adjusting for age and a group nuisance term.

Effects are reported per standard deviation of each predictor, translated into outcome items.

## The interpretive problem, stated plainly

Everything here is measured **simultaneously**, which removes even the weak temporal ordering the other families have. If letter-sound knowledge and word reading are measured on the same day and move together, the data are equally consistent with letter sounds supporting reading, reading practice teaching letter sounds, both being driven by general ability, or all three at once.

The models also adjust for other skills measured at the same time, which introduces a second hazard: conditioning on something that sits on the path between two variables, or on a common effect of both, can shrink a real relationship or manufacture a spurious one. These are **conditional** associations, and they change when the adjustment set changes — as this family itself demonstrates below.

## What was found

The strongest relationships in the trial cohort, at the wave where each was clearest:

| Outcome                    | Strongest same-wave correlate | Association per +1 SD    |
| -------------------------- | ----------------------------- | ------------------------ |
| Word reading (t2)          | Letter sounds                 | +9.0 items [+6.3, +11.6] |
| Letter sounds (t2)         | Word reading                  | +4.2 items [+3.2, +5.1]  |
| Expressive vocabulary (t3) | Taught expressive vocabulary  | +6.4 items [+2.5, +10.3] |
| Receptive vocabulary (t3)  | Word reading                  | +6.2 items [+2.3, +10.2] |
| Receptive grammar (t3)     | Taught receptive vocabulary   | +2.0 items [+0.9, +3.1]  |
| Basic concepts (t1)        | Taught receptive vocabulary   | +1.6 items [+0.7, +2.4]  |
| Phoneme blending (t4)      | Word reading                  | +0.8 items [+0.3, +1.2]  |

**Letter sounds and word reading are each other's strongest correlate**, which is the clearest illustration of why these cannot be read directionally. The same relationship appears from both ends.

**Word reading appears as the strongest correlate of receptive vocabulary** (+6.2 items). Nobody thinks word reading causes vocabulary at this age and over this window; it is a marker of general progress. That result is a useful reminder of what these coefficients are.

**Adjustment changes the answer substantially.** Three models estimate the same letter-sound-to-word-reading relationship at t2 with different covariates:

| Model    | Adjustment            | Association per +1 SD     |
| -------- | --------------------- | ------------------------- |
| `ca-010` | minimal               | +11.0 items [+8.2, +13.5] |
| `ca-001` | full skill set        | +9.0 items [+6.3, +11.6]  |
| `ca-011` | plus nonword decoding | +7.2 items [+4.4, +10.3]  |

The estimate falls by about a third across the three. All are "the association between letter sounds and word reading"; they differ only in what is held constant. No single number is _the_ answer, and quoting one without its adjustment set would be misleading.

In the **historical cohort**, verbal reasoning is the strongest same-wave correlate of BAS word reading at wave 1 (+7.5 items [+3.9, +11.3]) — consistent with the general-ability reading of these associations, since verbal reasoning is about as close to a general-ability marker as this battery has.

## What these models cannot tell you

**No direction at all.** Simultaneous measurement provides no temporal ordering.

**No causal claim, and no mechanism.** For the mechanism question with at least a temporal ordering see the `mechanism` family; for the randomised question see `itt`.

**The coefficients depend on the adjustment set**, as shown above.

**These are not predictions.** An association at one wave does not establish that measuring the predictor forecasts later outcomes; that is what `lcsm` and `growth` attempt.

## Model inventory

All 13 pass the convergence gate with zero divergences and are publishable. Trial cohort: `ca-001` (word reading), `002` (letter sounds), `003`/`004` (taught vocabulary), `005`/`006` (broad vocabulary), `007` (blending), `008` (basic concepts), `009` (grammar), `010`/`011` (letter-sound to word-reading adjustment variants). Historical cohort: `rlm-ca-001` (BAS word reading), `rlm-ca-002` (BPVS receptive vocabulary).
