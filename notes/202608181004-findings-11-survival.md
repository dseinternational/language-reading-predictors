> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `survival` family — how quickly children first move off the floor

**Read `findings-00-overview` first.** This note covers the 2 models in the `survival` family.

## The data

**RLI trial only**, restricted to the children who were **at the floor at timepoint 1** — scoring zero on the measure. Phonetic spelling has 41 such children contributing 100 person-period rows; nonword reading has 36 children contributing 74 rows.

The data are reshaped into **person-period** form: each child contributes one row per interval during which they were still at the floor and still being followed. Once a child moves off the floor they stop contributing rows, because the event being modelled has happened.

## What the model is for

Two of the outcomes in this study are so heavily floored that a conventional analysis has almost nothing to work with — most children score zero, so there is very little variation to explain.

Rather than give up, this family changes the question to one the data can answer: **for a child sitting at zero, how likely are they to move above zero during the next interval, and does that depend on the intervention?**

This is a discrete-time survival model, the same tool used for time-to-event data in medicine. The "event" is scoring above zero for the first time. Each interval has its own baseline probability, so the model does not assume the chance is constant over time.

The result is reported as a **hazard ratio**: how much the intervention multiplies the chance of coming off the floor in a given interval. A ratio of 1 means no difference; above 1 means faster; below 1 means slower.

## What was found

| Outcome           | Hazard ratio | 89% range    | P(favoured direction) | Evidence     |
| ----------------- | ------------ | ------------ | --------------------- | ------------ |
| Nonword reading   | 1.35         | 0.75 to 2.44 | 0.80 (faster)         | suggestive   |
| Phonetic spelling | 0.84         | 0.46 to 1.56 | 0.67 (slower)         | inconclusive |

**Neither supports a firm claim, but they are not equivalent.** Nonword reading reaches _suggestive_ evidence that treated children came off the floor faster — roughly 3:1 odds on the direction. Phonetic spelling is genuinely _inconclusive_, and note that its favoured direction is **slower**, not faster; an evidence label attaches to whichever direction the data lean towards, however weakly.

Taking the nonword ratio at face value would mean the intervention made children about 35% more likely to move off the floor in any interval — but the interval runs from a 25% _reduction_ to a near-tripling, so the point estimate carries little weight.

## How to read this, and why it is still worth having

It would be easy to present these as disappointing. They are better understood as **the most that could honestly be extracted from very thin measurements**.

Consider the arithmetic. Nonword reading has 6 items; 36 children were at zero; and from 74 person-period rows the model must estimate a separate baseline hazard for every interval _and_ a treatment shift. The fitted baseline probability of first coming off the floor runs about 22–29% per interval for nonword reading and 16–22% for phonetic spelling, so events are not especially rare — the binding constraint is the number of children and the number of quantities estimated from them, not a shortage of events.

This family's real contribution is methodological: it shows how far a floored outcome can be pushed and where it stops. For nonword reading three approaches now lean the same way — +10 percentage points off-floor in the `itt` family, +2 points in `gain_factors`, and a hazard ratio of 1.35 here. None is strong, and a chain of weak agreement is worth no more than its weakest link, but it is more coherent than the phonetic-spelling picture, where the three approaches straddle zero.

## What these models cannot tell you

**They cannot show the intervention did not help these skills.** Inconclusive is not null.

**They describe only the floored subgroup** — children at zero at timepoint 1 — not the whole sample.

**Moving off the floor is a low bar.** Scoring 1 out of 6 is a different achievement from fluent decoding, and this model treats any non-zero score as the event.

**Later intervals are post-crossover**, so the pooled treatment quantity blends the randomised interval with intervals in which both arms had been treated.

## Model inventory

Both models pass the convergence gate with zero divergences and are publishable: `surv-009` (phonetic spelling) and `surv-011` (nonword reading).
