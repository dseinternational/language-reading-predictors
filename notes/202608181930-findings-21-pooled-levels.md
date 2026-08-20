> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `pooled_levels` family — one association across all four waves

**Read `findings-00-overview` first.** This note covers the 3 models in the `pooled_levels` family, added after the main run. **Nothing here is causal.**

## The data

**RLI trial only**, all four timepoints stacked: 210 child-wave rows from 53 children for word reading, with 4 rows dropped where a child had the exposure but not the outcome. A child contributes as many rows as they have complete waves. Data are pooled across waves, not collapsed to one number per child.

## What the model is for

Two families already sit either side of this one. `concurrent` asks the levels question **at each wave separately** — it fits a different model at each timepoint. `mechanism` asks a **change** question: given where a child started a period, does a higher exposure go with ending higher. Neither answers the obvious middle question: pooled across all four waves, how does one skill's level go with another's?

That gap could not be filled by setting a flag on either. `concurrent` is per-wave by construction, so pooling is a different likelihood rather than an option. `mechanism` conditions on each outcome's own starting score, which removes exactly the stable between-child variation a levels question is about.

The one thing that did pool — the `horseshoe` ranking — is unsuitable as an estimate on three counts: it is shrinkage-regularised, it is framed as a ranking, and it carries **no child random intercept** despite stacking about four rows per child, so it treats repeated measures on one child as independent.

## The decomposition, which is the whole point

Stacking waves creates a trap. A model with one exposure coefficient and a child random intercept does **not** return "the pooled association" — it returns a precision-weighted blend of two quite different things:

- the **between-child** association: do children who sit higher on letter sounds across the study also read more across the study?
- the **within-child** association: at the waves where one child is above their own letter-sound average, are they above their own reading average?

On these data those two correlations are 0.81 and 0.45 for the letter-sound and word-reading scores on the model's log-odds scale (0.70 and 0.51 on the raw counts), so a blend of them is a number that answers neither question. These models therefore split the exposure into each child's mean and their deviation from it, and report the two coefficients separately.

**Refreshed from the 2026-08-20 full refit** (`notes/202608200800-full-refit-both-layers-2026-08.md`), which added the four exposures from #553 and refitted the family under the corrected hearing coding.

## What was found

| Model    | Exposure              | Outcome          | Between children         | Within a child                    |
| -------- | --------------------- | ---------------- | ------------------------ | --------------------------------- |
| `pl-001` | Letter sounds         | Word reading     | **+1.61** [+1.35, +1.88] | +0.04 [−0.06, +0.15] P = 0.75     |
| `pl-002` | Letter sounds         | Nonword decoding | **+1.78** [+1.41, +2.21] | **+0.33** [−0.02, +0.69] P = 0.93 |
| `pl-004` | Receptive vocabulary  | Word reading     | **+1.01** [+0.62, +1.41] | +0.12 [−0.06, +0.30] P = 0.85     |
| `pl-003` | Expressive vocabulary | Word reading     | **+0.97** [+0.56, +1.38] | +0.19 [+0.01, +0.37] P = 0.95     |
| `pl-005` | Phonological memory   | Word reading     | **+0.88** [+0.55, +1.21] | +0.14 [−0.04, +0.32] P = 0.89     |
| `pl-006` | Speech production     | Word reading     | **+0.46** [+0.11, +0.82] | **+0.29** [+0.09, +0.50] P = 0.99 |

Both coefficients are on the outcome's log-odds scale per **1 standard deviation of the letter-sound score's log-odds** (the pooled row-level SD, so the between and within terms share a unit and match the `mechanism` family's per-SD scale). Translated into items at the average fitted level, the between-child association is about **+17 words out of 79** per SD (89% +13 to +21) for word reading and about **+2.0 nonwords out of 6** (89% +1.5 to +2.5) for decoding. The four exposures added in #553 translate to +8.6, +8.0, +7.1 and +3.1 words per SD for receptive vocabulary, expressive vocabulary, phonological memory and speech production.

**The between-child association is large and the within-child one is not.** Children who know more letter sounds across the study read far more across the study — the posterior puts essentially all its mass on a positive between-child association. But for **word reading**, knowing more letter sounds than usual at a particular wave carries essentially no signal about reading more than usual at that wave: +0.04, inconclusive.

That dissociation is what a shared-cause account predicts and a direct-influence account does not. If letter-sound knowledge were driving word reading within a child, the within-child coefficient should be where it shows up.

**The four exposures added in #553 (`pl-003`–`006`) repeat the pattern, with one exception.** Expressive vocabulary, receptive vocabulary and phonological memory all show a large between-child association and a within-child one between +0.12 and +0.19 — the same dissociation as letter sounds, if less extreme. **Speech production is the exception**: its between-child association is the smallest of the set (+0.46) and its within-child association the largest (+0.29, P = 0.99), so it is the one exposure here whose within-child signal is an appreciable fraction of its between-child one. That is the profile of something tracking a child over time rather than merely separating children, which makes it the more interesting candidate for a design that could test it — and it remains an association from a model that adjusts for neither reverse causation nor a shared cause.

**Nonword decoding behaves differently.** Its within-child coefficient is +0.33 with moderate evidence — roughly eight times word reading's, though its interval still grazes zero, and the two outcomes sit on very different scales (6 items with a floor against 79). The two come from separate models with no fitted contrast between them, so the gap should not be quoted as an estimated difference. But the direction of it agrees with the `mechanism` family's decoding-specificity result, which was reached on a completely different decomposition. Two unrelated designs pointing the same way is worth more than either alone.

## Why the wave intercepts matter

`pl-101` is the same model without per-wave intercepts, and it is a warning rather than a result. Its within-child coefficient is **+0.19 [+0.08, +0.29]**, very strong — against +0.04 and inconclusive once waves are accounted for.

The reason is simple: both measures rise across the study, so within any child the later waves have both higher letter sounds and higher reading. Without a wave term, that shared maturation is counted as a within-child association. The comparator exists to show how much of an apparent within-child effect that alone can manufacture.

## What these models cannot tell you

**Nothing here is ordered in time.** Exposure and outcome are measured at the same wave, so this family carries _less_ temporal structure than `mechanism`, not more.

**The between-child coefficient absorbs every stable difference.** General ability, home environment, schooling, and anything else that makes a child do well on both measures is inside it. Measured ability (block design) is adjusted for; the latent construct is not.

**A small within-child coefficient is not evidence of no influence.** Within-child variation is small relative to between-child variation here, and levels are a blunt instrument for it — the `mechanism` family's transition models are the better-powered within-child view.

**The arm term is not a treatment effect.** Each model carries the assigned arm as an adjuster, but that coefficient pools all four waves — one before either arm was taught, one at the randomised contrast, and two after both arms had been taught — and conditions on a same-wave skill the intervention itself changed, so it is neither the randomised timepoint-2 contrast nor any other estimate of the intervention's effect. It is there to keep the skill association clean, and its priors-table role is recorded as an association for that reason.

## Model inventory

All 3 pass the convergence gate with zero divergences and are publishable: `pl-001` (letter sounds → word reading), `pl-002` (letter sounds → nonword decoding), `pl-101` (the no-wave-intercept comparator for `pl-001`).
