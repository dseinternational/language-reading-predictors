> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `pooled_levels` family — one association across all four waves, split between and within children

**Read `findings-00-overview` first.** This note covers the 7 models in the `pooled_levels` family. **Nothing here is causal.** All 7 pass the convergence gate with zero divergences and all are publishable (2026-09-01 rebuild). Three of them (`pl-003`–`005`) are among the fits whose rows changed under the #631 ERB quarantine.

## The data

**RLI trial only**, all four timepoints stacked: 53–54 children contributing 201–215 child-wave rows, with rows dropped where the exposure, the outcome or a same-wave skill adjuster is unobserved.

## What the model is for

`concurrent` asks the levels question at each wave separately; `mechanism` asks a change question conditional on each outcome's starting score. Neither answers the plain pooled question: across all four waves, how does one skill's level go with another's? Stacking waves creates a trap: a model with one exposure coefficient and a child random intercept returns a precision-weighted blend of the **between-child** association (do children who sit higher on the exposure across the study also sit higher on the outcome?) and the **within-child** association (at the waves where a child is above their own exposure average, are they above their own outcome average?). These models split the exposure into each child's mean and their deviation from it and report the two coefficients separately, with per-wave intercepts, the assigned arm, hearing, speech, the t1 ability proxy and age as adjusters.

## What was found

Both coefficients on the outcome's log-odds scale per 1 SD of the exposure (the pooled row-level SD, shared by both terms):

| Model    | Exposure              | Outcome          | Between children         | Within a child                     | Between, in items per SD |
| -------- | --------------------- | ---------------- | ------------------------ | ---------------------------------- | ------------------------ |
| `pl-002` | Letter sounds         | Nonword decoding | **+1.78** [+1.41, +2.21] | **+0.33** [−0.02, +0.69], P = 0.93 | +2.0 of 6 [+1.5, +2.5]   |
| `pl-001` | Letter sounds         | Word reading     | **+1.61** [+1.35, +1.88] | +0.04 [−0.06, +0.15], P = 0.746    | +17 of 79 [+13, +21]     |
| `pl-004` | Receptive vocabulary  | Word reading     | **+1.02** [+0.63, +1.42] | +0.11 [−0.06, +0.30], P = 0.84     | +8.7 [+4.6, +13.9]       |
| `pl-003` | Expressive vocabulary | Word reading     | **+0.98** [+0.56, +1.39] | **+0.19** [+0.01, +0.38], P = 0.95 | +8.1 [+3.9, +13.5]       |
| `pl-005` | Phonological memory   | Word reading     | **+0.88** [+0.56, +1.21] | +0.14 [−0.06, +0.35], P = 0.86     | +7.0 [+3.9, +10.9]       |
| `pl-006` | Speech production     | Word reading     | **+0.47** [+0.11, +0.82] | **+0.30** [+0.10, +0.50], P = 0.99 | +3.2 [+0.7, +6.5]        |

**The between-child association is large and the within-child one is not.** Children who know more letter sounds across the study read far more across the study — about 17 words out of 79 per SD. But for word reading, knowing more letter sounds than usual at a particular wave carries essentially no signal about reading more than usual at that wave: +0.04, inconclusive. That dissociation is what a shared-cause account predicts and a direct-influence account does not; if letter-sound knowledge were driving word reading within a child, the within-child coefficient is where it should show. The `mechanism` family's between/within split of its own letter-sound slope (`mech-301`: between +0.44 per SD, within +0.03) reaches the same conclusion from a transition design.

**Nonword decoding behaves differently.** Its within-child letter-sound coefficient is +0.33 with moderate evidence — roughly eight times word reading's, though its interval grazes zero and the two outcomes sit on very different scales (6 floored items against 79). The two come from separate models with no fitted contrast, so the gap is not an estimated difference; but its direction agrees with the decoding-specificity result the `mechanism` and `joint_mechanism` families reached on a different decomposition. Two unrelated designs pointing the same way is worth more than either alone.

**The four other predictors of word reading repeat the pattern, with one exception.** Expressive vocabulary, receptive vocabulary and phonological memory all show a large between-child association (+7 to +9 words per SD) and a within-child one between +0.11 and +0.19 — the same dissociation as letter sounds, if less extreme, and for expressive vocabulary the within-child term now clears zero. **Speech production is the exception**: its between-child association is the smallest of the set (+0.47) and its within-child association the largest (+0.30, P = 0.99), so it is the one exposure whose within-child signal is a substantial fraction of its between-child one — the profile of something tracking a child over time rather than merely separating children. It remains an association from a model that adjusts for neither reverse causation nor a shared cause, and speech production is also a resolved covariate in the letter-sound models (+0.26 and +0.50 logits per SD in `pl-001` and `pl-002`).

## Why the wave intercepts matter

`pl-101` is `pl-001` without per-wave intercepts and is a warning rather than a result: its within-child coefficient is **+0.19** (89% +0.08 to +0.29), very strong, against +0.04 and inconclusive once waves are accounted for. Both measures rise across the study, so within any child the later waves have both higher letter sounds and higher reading; without a wave term that shared maturation is counted as a within-child association. The comparator exists to show how much apparent within-child effect that alone can manufacture.

## What these models cannot tell you

**Nothing here is ordered in time.** **The between-child coefficient absorbs every stable difference** — measured ability is adjusted for, the latent construct is not. **A small within-child coefficient is not evidence of no influence**; within-child variation is small relative to between-child variation here. **The arm term is not a treatment effect**: it pools a pre-treatment wave, the randomised wave and two post-crossover waves and conditions on a same-wave skill the intervention changed, so its negative values in the letter-sound models (−0.31 and −0.52 logits) carry no causal reading.

## What changed since the August notes

The three letter-sound fits reproduce the 2026-08-20 values exactly; speech production (`pl-006`), which adjusts for phonological memory, moved in the second decimal (between-child +0.47 against +0.46, within-child +0.30 against +0.29); the three fits that read the quarantined ERB cell moved in the second decimal (expressive vocabulary within-child +0.19 against +0.19; receptive vocabulary +0.11 against +0.12; phonological memory +0.14 against +0.14). No direction or label changed.

## Model inventory

All 7 pass the convergence gate with zero divergences and are publishable: `pl-001` (letter sounds → word reading), `002` (letter sounds → nonword decoding), `003` (expressive vocabulary → word reading), `004` (receptive vocabulary → word reading), `005` (phonological memory → word reading), `006` (speech production → word reading) and `101` (the no-wave-intercept comparator for `001`).
