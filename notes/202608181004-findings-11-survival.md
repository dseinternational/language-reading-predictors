> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Findings: the `survival` family — how quickly children first move off the floor

**Read `findings-00-overview` first.** This note covers the 2 models in the `survival` family.

## The data

**RLI trial only**, restricted to the children who were **at the floor at timepoint 1** — scoring zero on the measure. Phonetic spelling has 41 such children contributing 100 person-period rows; nonword reading has 36 children contributing 74 rows.

The data are reshaped into **person-period** form: each child contributes one row per interval during which they were still at the floor and still being followed. Once a child moves off the floor they stop contributing rows, because the event being modelled has happened.

## What the model is for

Two of the outcomes in this study are so heavily floored that a conventional analysis has almost nothing to work with — most children score zero, so there is very little variation to explain.

Rather than give up, this family changes the question: **for a child sitting at zero, how likely are they to move above zero during the next interval, and how is that interval hazard associated with intervention-aligned treated status?**

This is a discrete-time survival model, the same tool used for time-to-event data in medicine. The "event" is scoring above zero for the first time. Each interval has its own baseline probability, so the model does not assume the chance is constant over time.

The result is reported as a **hazard ratio** under a complementary-log-log link: how much treated status multiplies the underlying interval hazard of coming off the floor. It is not a fixed multiplier of the event probability; the probability change depends on the baseline hazard. A ratio of 1 means no hazard difference; above 1 means faster and below 1 slower.

## What was found

| Outcome           | Hazard ratio | 89% range    | P(favoured direction) | Evidence     |
| ----------------- | ------------ | ------------ | --------------------- | ------------ |
| Nonword reading   | 1.35         | 0.75 to 2.44 | 0.80 (faster)         | suggestive   |
| Phonetic spelling | 0.84         | 0.46 to 1.56 | 0.67 (slower)         | inconclusive |

**Neither supports a firm claim, but they are not equivalent.** Nonword reading reaches _suggestive_ evidence for a positive treated-status association — P(positive) = 0.798, about 4:1 posterior odds on the direction. Phonetic spelling is _inconclusive_, and note that its favoured direction is **slower**, not faster; an evidence label attaches to whichever direction the data lean towards, however weakly. Power-scaling flags `tau` in both models for potential prior–data conflict, so prior robustness has not been established even for the directional reading.

Taking the nonword point estimate at face value gives a **35% higher underlying interval hazard** when treated status is on, not a 35% higher event probability and not an identified causal effect. Its 89% interval runs from a 25% lower hazard to a 144% higher hazard, so the point estimate carries little weight.

## How to read this, and why it is still worth having

It would be easy to present these as disappointing. They are better understood as **the most that could honestly be extracted from very thin measurements**.

Consider the arithmetic. Nonword reading has 6 items; 36 children were at zero; and from 74 person-period rows the model must estimate a separate baseline hazard for every interval _and_ a treatment shift. The fitted baseline probability of first coming off the floor runs about 22–29% per interval for nonword reading and 16–22% for phonetic spelling, so events are not especially rare — the binding constraint is the number of children and the number of quantities estimated from them, not a shortage of events.

This family's real contribution is methodological: it shows how far a floored outcome can be pushed and where it stops. For nonword reading three approaches now lean the same way — +10 percentage points off-floor in the `itt` family, +2 points in `gain_factors`, and a hazard ratio of 1.35 here. The survival estimate is a pooled prognostic association rather than the same causal estimand, so this is descriptive triangulation only. None is strong, and a chain of weak agreement is worth no more than its weakest link; the phonetic-spelling picture is still less coherent, with the three approaches straddling zero.

## What these models cannot tell you

**They cannot show the intervention did not help these skills.** Inconclusive is not null.

**They describe only the floored subgroup** — children at zero at timepoint 1 — not the whole sample.

**Moving off the floor is a low bar.** Scoring 1 out of 6 is a different achievement from fluent decoding, and this model treats any non-zero score as the event.

**Later intervals are post-crossover**, so the pooled treatment quantity blends the randomised interval with intervals in which both arms had been treated.

**The pooled hazard ratio is prognostic, not causal.** Only the first interval is randomisation-anchored; the fitted `tau` pools that interval with post-crossover periods, so “the intervention made children move off the floor faster” is not licensed.

## Model inventory

Both models pass the convergence gate with zero divergences and are publishable: `surv-009` (phonetic spelling) and `surv-011` (nonword reading).
