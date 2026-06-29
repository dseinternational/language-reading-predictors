<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Longitudinal-modelling stance: why our trial estimates are not cross-lagged panel models, and guardrails for "does reading drive language?" work

> [!WARNING]
> This note was prepared by an AI tool and may contain mistakes. Treat the
> statistical claims as a starting point for review, not as settled fact.

Date: 2026-06-20

## Introduction

This note explains a well-known problem with a popular longitudinal model — the
**cross-lagged panel model (CLPM)**, set out by Lucas (2023) — and then works
through which parts of _our_ modelling are affected and which are not. The
trigger is the project roadmap (`notes/2026-05-12-project-review.md`, Option F),
which floats fitting cross-lagged models to ask whether reading progress drives
later language progress (and vice versa). Writing it down now makes that decision
deliberate rather than something we discover in peer review.

## Background: what a CLPM is trying to do

Suppose you measure two things — say reading (X) and language (Y) — on the same
children at several time points. A natural question is: does being good at
reading _now_ cause better language _later_, or the other way round, or both?

The CLPM tries to answer this with a regression. To test "reading drives later
language," it predicts language at time 2 from **reading at time 1**, while
**controlling for language at time 1** (the child's earlier language level). The
hope is intuitive: if earlier reading still helps predict later language _after
we have already accounted for earlier language_, then maybe reading is doing
something — adding a push to language over and above where the child already was.
The reverse path (language at time 1 -> reading at time 2, controlling for
reading at time 1) is tested at the same time. Those two "controlling-for-the-
past-outcome" paths are the **cross-lagged** paths, and people read them as
reciprocal causal effects.

It is a popular model because it needs little data and looks like it is doing
something careful (it "controls for the baseline"). The problem is that the
careful-looking control does not do what people think it does.

## The problem

### People have stable differences, and that breaks the model

Think of any human characteristic as having two ingredients:

- a **stable-trait** part — a baseline level that barely changes over time (think
  of it as a person's "set point", driven by genetics, home environment, general
  ability, and so on); and
- a **changing** part — the ups and downs around that set point from one occasion
  to the next.

The CLPM assumes there is _no_ stable set-point ingredient — it assumes everything
is the slowly-drifting "changing" part (the technical name is an **autoregressive**
process: today's value is last time's value nudged up or down). That assumption
is almost always wrong for real psychological and educational measures.

Here is why it matters. Imagine reading and language are _both_ partly driven by a
shared stable background — general cognitive ability, say, or a language-rich
home. Then children who are high on reading tend to be high on language _at every
time point_, not because one is causing the other over time, but because the same
stable background is pushing both up. The CLPM has no stable-background ingredient
to absorb this, so it has to explain the leftover correlation _somehow_ — and the
only knob it has left is the cross-lagged path. So it reports a reading -> language
"effect" (and a language -> reading "effect") that is really just the footprint of
the shared, stable background.

> **Everyday analogy.** Tall parents tend to have tall children, and a child's
> height at age 5 and age 10 are both high if the family is tall. If you ran a
> CLPM-style analysis — "predict height at 10 from _shoe size_ at 5, controlling
> for height at 5" — you could easily get a significant "shoe size drives later
> height" path. Nothing is causing anything; there is just a stable family-size
> factor making everything about that child large at once. The model mistakes
> "these go together because of a stable common cause" for "this one drives that
> one over time."

Lucas's simulations make this concrete: when realistic amounts of stable-trait
variation exist, the CLPM reports a _spurious_ (false) cross-lagged effect a very
large fraction of the time — often essentially 100% in plausible scenarios.

### Measurement error makes it worse, on its own

"Controlling for earlier language" sounds like it should remove the shared
background. But you can only control for the _measured_ earlier language, and
every test is noisy — a child's score is their true level plus measurement error.
Controlling for a **noisy proxy** does not fully remove what you wanted to remove.
The leftover, uncontrolled bit of the shared background then shows up as — again —
a spurious cross-lagged path. The slogan is: **imperfect control is incomplete
control.** Crucially, this happens _even with no stable trait at all_ — ordinary
measurement noise is enough to manufacture a fake effect.

### "Good" study features make it worse, not better

You might hope a bigger sample or more measurement waves would rescue you. They do
the opposite: more data gives you more statistical power, and power applied to a
_biased_ estimate just makes you more confident in the wrong answer. So larger,
longer studies are _more_ likely to report the spurious effect as "significant."

### The model is not even safely conservative

It would be some comfort if the CLPM only ever erred toward "no effect." It does
not: when a real cross-lagged effect _does_ exist, the same stable-trait problem
can make the CLPM _under_-state it. So the bias runs in both directions — invent
effects that are not there, shrink ones that are.

## The fix

The standard repair is the **random-intercept CLPM (RI-CLPM)**. In plain terms, it
gives every person their own **personal baseline** (a "random intercept") that
captures their stable set point, and then studies only the _within-person_
wobble around that baseline. A cross-lagged path now means something much more
defensible: _"when a child is higher than their own usual reading, do they later
rise above their own usual language?"_ — an over-time, within-child question that
the stable background can no longer fake.

A still-fuller model, **STARTS**, additionally carves out the per-occasion
measurement noise. It is the most honest model, but it is data-hungry (needs more
waves) and often will not estimate at all unless you fit it carefully (Bayesian
methods help). The practical ladder is: plain CLPM (worst) -> RI-CLPM (much
better, needs >= 3 waves) -> STARTS (best, needs ~4+ waves and is fragile).

Two escape hatches people reach for, and why they fail: calling the analysis
"just descriptive" does not help, because the moment you _control for_ the earlier
outcome you have made a causal modelling choice that needs justifying (Wysocki et
al., 2022); and calling it "just prediction" only works if you genuinely abandon
the reciprocal-cause interpretation and use prediction-first tools — at which
point it is no longer a CLPM.

## How this maps onto our project

### Why our main intervention results are safe

Our headline models — the **intention-to-treat (ITT)** models and the joint
model (`LRP52`-`55`, `LRP60`/`60a`) — predict a child's outcome at the end of the
trial phase from their own starting score plus the **treatment group** they were
assigned to. On paper the equation looks like a CLPM (an outcome regressed on its
own baseline). But the number we actually interpret is the **treatment effect**
(called `tau` / τ in the code): the difference between the two randomly-assigned
arms.

This is the key point: **random assignment is what makes that comparison causal —
not the baseline adjustment.** When children are randomised to groups, the two
groups are, on average, balanced on _everything_, including all the hidden stable
traits that wreck the CLPM. The coin flip does the de-confounding; the baseline
score is in the model only to sharpen the estimate (tighten the interval), not to
rescue it from confounding. So the entire Lucas problem — stable common causes
masquerading as effects — does not threaten our treatment effects on word reading,
letter-sound knowledge, and the rest.

_Action:_ say this in one sentence in the ITT write-up — _"the treatment effect is
identified by randomisation during the trial phase; the baseline adjustment is for
precision, not identification"_ — so a reviewer holding the Lucas paper sees
immediately that it does not apply.

### Where we are still exposed (and what we have already done about it)

The vulnerable parts are the places where we look at associations that were _not_
randomised. These have CLPM-like structure and the critique applies.

1. **"Baseline X predicts later Y" couplings (the `gamma_cross` terms in the ITT
   models).** This is literally a cross-lagged path, so it can be inflated by a
   shared stable trait or by measurement noise. _What protects us:_ we never
   interpret these as causal, and we report that none is credibly different from
   zero (`notes/202604181600-lrp52-58-findings.md`). Keep that discipline — never
   let one of these become a "X drives Y" sentence.

2. **The mechanism slope (`f_mech` in `LRP56`-`58`): does letter-sound knowledge
   relate to word reading?** This _is_ an observational association, so it is
   exposed. _What protects us, partly:_ these models already include a **per-child
   baseline (a subject random intercept)** — which is exactly the RI-CLPM repair
   above. So we are running the recommended better model, not the naive one. _What
   still leaks:_ (a) the mechanism is measured at the _same wave_ as the outcome,
   so it is a "go-together" association more than a clean "earlier -> later" one;
   and (b) we do not separately model measurement noise in the skill, which Lucas
   shows can still inflate the path. _Stance:_ report `f_mech` as "a credible
   positive association, after adjusting for each child's stable level," not as a
   clean causal dose-response curve.

3. **Mediation: how much of the treatment effect flows _through_ a mediator
   (`LRP59`, `LRP62`).** Because the treatment is randomised, the treatment ->
   mediator and treatment -> outcome arrows are clean. The exposed arrow is
   **mediator -> outcome**: a stable trait shared by the mediator and the outcome
   can inflate it, and measurement noise in the mediator biases the "how much
   flows through" number. We already flag these assumptions; keep them prominent
   and treat the _percentage mediated_ as a soft, fragile number rather than a
   headline.

### The discovery models (LightGBM, LRP01-22) are not affected

These are **prediction/screening** models — their job is to find _which_ baseline
skills are worth putting into the careful Bayesian models, not to make causal
claims. Lucas explicitly exempts genuine prediction work from his critique. The
only rule: do not quietly turn a predictive importance (e.g. on a `_GAIN` or
`_NEXT` variable) into a causal "drives" statement.

## The two things that limit us most

- **Measurement error is the underrated half of the problem.** Our tests are
  short, bounded-count measures with floor effects, measured on only 54 children —
  i.e. noisy. Per Lucas, noise _alone_ can manufacture fake "earlier -> later"
  paths. So this is not just a worry for a future CLPM; it gently inflates the
  mechanism and mediation associations too.
- **Our sample (n = 54) caps which repair we can afford.** Four waves is enough to
  fit the random-intercept repair, but **not** enough to reliably fit the fuller
  STARTS model that would also strip out the measurement noise. So we can adjust
  for stable traits but largely cannot also model the noise — which means any
  reciprocal-effects estimate would stay fragile even if we did everything
  "properly."

## Link to the earlier "should we model gains?" question

This connects to the earlier discussion about whether to model change/gains at
all. Lucas notes that the alternative of modelling raw **gain scores**
(later minus earlier) assumes the starting level does _not_ affect the amount of
change. In our data it clearly _does_: the "own-baseline" coefficient in the
Bayesian models is well below 1 (`gamma_own` ≈ 0.44-0.66), which is the signature
of **regression to the mean** (children who start low tend to gain more, partly as
a statistical artefact). That is exactly why our baseline-adjusted form is the
right choice **given that we have randomisation to lean on**, and why a raw
gain-score CLPM would be doubly wrong here.

## Guardrails for any "does reading drive language?" model (roadmap Option F)

This reciprocal-effects question is the direct target of the Lucas paper. If we
pursue it at all:

1. **Never fit a plain CLPM.** Use the random-intercept version (a Bayesian
   RI-CLPM / random-intercept growth model in PyMC). We already have the
   random-intercept and Bayesian machinery, which is the route the methods
   literature recommends for these otherwise hard-to-estimate models.
2. **Be explicit that we cannot separate measurement noise** at n = 54 / four
   waves, so any estimate is fragile by construction.
3. **Frame it as describing _correlates of change_, not proving reciprocal
   causation.** There is a sensible order of questions for longitudinal data
   (Rogosa, 1995, endorsed by Lucas): first describe how each thing changes, then
   study what predicts change, and only _then_ ask about reciprocal feedback. We
   are still at the first two steps; the reciprocal question is premature.
4. **If a reviewer insists, fit the plain CLPM and the random-intercept version
   side by side** and show the cross-lagged paths shrink or vanish once each
   child's stable level is included — that contrast is itself the most persuasive
   evidence the paths were artefacts.
5. **Default recommendation: defer it.** The payoff is small and the risk of a
   misleading result is high at this sample size.

## Practical checklist

- [ ] ITT / joint write-ups: add the "identified by randomisation, not by baseline
      control" sentence.
- [ ] Never interpret a `gamma_cross` ("baseline X predicts later Y") term as
      causal.
- [ ] Mechanism / mediation reports: add a short caveat that the association could
      be inflated by a shared stable trait and by measurement noise; mark the
      percentage-mediated as fragile.
- [ ] Any Option-F model: random-intercept model at minimum, framed as correlates
      of change, shown side by side with the naive version, with explicit
      caveats — or deferred.

## Glossary

- **Cross-lagged panel model (CLPM):** a regression-based longitudinal model that
  estimates "X earlier -> Y later" while controlling for "Y earlier" (and the
  mirror path), and interprets those paths as reciprocal causal effects.
- **Stable trait:** the part of a measure that is a person's roughly-fixed set
  point, the same across all waves.
- **Autoregressive:** a process where each wave's value is the previous wave's
  value adjusted up or down; the CLPM assumes the data are purely this, with no
  stable set point.
- **Random intercept:** a person-specific baseline added to a model so it studies
  only each person's wobble around their own average; the core of the RI-CLPM fix.
- **RI-CLPM / STARTS:** the random-intercept repair, and a fuller version that also
  separates per-occasion measurement noise.
- **Spurious effect:** an estimated effect the model reports as causal that is
  really an artefact (here, of an unmodelled stable common cause or of noise).
- **ITT (intention-to-treat):** analysing children by the group they were
  randomised to; the basis of our causal treatment-effect claims.
- **Mediator / mediation:** a variable on the pathway from cause to outcome;
  mediation analysis asks how much of an effect flows through it.
- **Regression to the mean:** the tendency of extreme scores to move toward the
  average on retest, which can look like (but is not) real change.

## References

- Lucas, R. E. (2023). Why the cross-lagged panel model is almost never the right
  choice. _Advances in Methods and Practices in Psychological Science, 6_(1),
  1-22. https://doi.org/10.1177/25152459231158378
- Hamaker, E. L., Kuiper, R. M., & Grasman, R. P. P. P. (2015). A critique of the
  cross-lagged panel model. _Psychological Methods, 20_(1), 102-116.
  https://doi.org/10.1037/a0038889
- Westfall, J., & Yarkoni, T. (2016). Statistically controlling for confounding
  constructs is harder than you think. _PLOS ONE, 11_(3), Article e0152719.
  https://doi.org/10.1371/journal.pone.0152719
- Wysocki, A. C., Lawson, K. M., & Rhemtulla, M. (2022). Statistical control
  requires causal justification. _Advances in Methods and Practices in
  Psychological Science, 5_(2). https://doi.org/10.1177/25152459221095823
- Rogosa, D. R. (1995). Myths and methods: "Myths about longitudinal research,"
  plus supplemental questions. In J. M. Gottman (Ed.), _The analysis of change_
  (pp. 3-65). Lawrence Erlbaum Associates. (Print book chapter; no DOI.)
