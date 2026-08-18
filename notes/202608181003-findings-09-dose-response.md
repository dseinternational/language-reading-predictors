> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Findings: the `dose_response` family — does more intervention mean more progress?

**Read `findings-00-overview` first.** This note covers the 5 models in the `dose_response` family. **Nothing here is causal**, and the reason is more interesting than usual.

## The data

**RLI trial only.** All transitions are stacked — 53–54 children contributing about 156–160 rows, one per child per period. Each row carries the outcome at the end of the period, the child's own starting score, and the **number of intervention sessions** that child actually received during that period, standardised.

## What the model is for

The treatment families ask whether being _assigned_ to the intervention helped. This family asks the natural follow-up: across the full sample, were periods with **more intervention sessions** associated with more progress? The fit includes the wait-list arm's zero-dose period-1 rows, which anchor the slope at dose zero; it is not restricted to children who received intervention in every fitted period.

The appeal is obvious — a dose-response gradient is classic supporting evidence for a real effect. The problem is equally fundamental.

## Why this cannot be causal, however tempting

**Children were randomised to the intervention. They were not randomised to attend more of it.** How many sessions a child received depends on health, family circumstances, travel, engagement and how well the sessions were going. Every one of those also plausibly affects progress directly.

Worse, attendance is what is called a **collider** on some paths: it is influenced both by the child's characteristics and by how the intervention is going, so conditioning on it can _create_ associations that do not exist. This is why the project treats dose as a sensitivity view rather than as evidence, and why the difference-in-differences dose companions report no causal headline at all.

## What was found

| Outcome                | Per 1 SD more sessions | 89% range    |
| ---------------------- | ---------------------- | ------------ |
| Word reading           | **+1.2 items**         | +0.4 to +2.0 |
| Letter-sound knowledge | **+0.8 items**         | +0.2 to +1.4 |
| Phoneme blending       | +0.3 items             | −0.0 to +0.6 |

The gradient is positive and reasonably well estimated for word reading and letter sounds, and it lines up with the outcomes the treatment families found effects on — which is at least internally coherent.

Two sensitivity variants for word reading agree closely: adjusting additionally for baseline skills gives **+1.3** [+0.5, +2.1], and a pooled version that does not let the slope vary by period gives **+1.3** [+0.6, +2.0]. A formal predictive comparison between the period-varying and pooled versions found no meaningful difference, so there is no evidence the dose relationship changes across periods.

## How to read this honestly

The right summary is: **periods with more recorded sessions were associated with better outcomes, and we cannot tell how much of that association is due to the sessions.** A plausible share of this gradient is children who were doing well attending more, and children with complicating circumstances attending less; the zero-dose wait-list rows also contribute to the fitted comparison.

Notice the magnitudes are compatible with the randomised estimates without adding to them: the randomised word-reading effect is about +2.4 items overall, and here one standard deviation of extra attendance is associated with about +1.2. If anything, that consistency is what you would expect whether or not the dose relationship is causal, so it does not discriminate between the explanations.

## What these models cannot tell you

**They cannot support "more sessions would produce more progress."** That is a claim about intervening on attendance, which this design cannot address.

**They cannot be used to recommend a dose.** No optimal number of sessions can be read off these curves.

**The baseline and covariate coefficients are associations**, as everywhere else in the project.

## Model inventory

All 5 pass the convergence gate with zero divergences and are publishable: `dose-077` (W, period-resolved), `083` (L), `084` (B), `177` (W, ability-adjusted sensitivity), `277` (W, pooled comparator).
