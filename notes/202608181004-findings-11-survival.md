> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).
>
> Revised for the 2026-08-21 reparameterised refits by a LLM-based AI tool (Claude Code/Fable 5): the earlier pooled `tau` had no likelihood contrast beyond the randomised first interval, and its direction — notably phonetic spelling's "slower" lean — was set by the zero-centred baseline-hazard priors, not by the data (see `notes/202608211300-survival-aligned-concurrent-code-review.md`, finding 1). The numbers below are from the refits, in which `tau` enters the randomised first interval only.

# Findings: the `survival` family — how quickly children first move off the floor

**Read `findings-00-overview` first.** This note covers the 2 models in the `survival` family.

## The data

**RLI trial only**, restricted to the children who were **at the floor at timepoint 1** — scoring zero on the measure. Phonetic spelling has 42 such children, of whom 41 contribute the 100 person-period rows (one has no observed timepoint-2 score, so no interval); nonword reading has 36 children contributing 74 rows.

The data are reshaped into **person-period** form: each child contributes one row per interval during which they were still at the floor and still being followed. Once a child moves off the floor they stop contributing rows, because the event being modelled has happened.

## What the model is for

Two of the outcomes in this study are so heavily floored that a conventional analysis has almost nothing to work with — most children score zero, so there is very little variation to explain.

Rather than give up, this family changes the question: **for a child sitting at zero, how likely are they to move above zero during the next interval, and did the randomised arms differ in that chance over the one interval where they can be compared?**

This is a discrete-time survival model, the same tool used for time-to-event data in medicine. The "event" is scoring above zero for the first time. Each interval has its own baseline probability, so the model does not assume the chance is constant over time.

One structural fact governs the treatment term: after the wait-list crossover **every child in the risk set is on the intervention**, so only the first interval (t1→t2) contains a treated-versus-untreated comparison. Since the 2026-08-21 refits `tau` is therefore fitted **in the randomised first interval only** — it is the immediate-vs-wait-list contrast among children at the floor at t1 — and the later intervals fit their own both-arms-treated baseline hazards. (The previous pooled `tau` spread one coefficient across all intervals; because the data carry no contrast beyond interval 1, its post-crossover share was set by the zero-centred baseline-hazard priors, which dragged it downward and, for phonetic spelling, flipped its sign. The pooled form survives only as an explicit `treatment_window="pooled"` comparator.)

The result is reported as a **hazard ratio** under a complementary-log-log link: how much the intervention arm multiplies the underlying first-interval hazard of coming off the floor. It is not a fixed multiplier of the event probability; the probability change depends on the baseline hazard. A ratio of 1 means no hazard difference; above 1 means faster and below 1 slower.

## What was found

| Outcome           | Hazard ratio (interval 1) | 89% range    | P(favoured direction) | Evidence     |
| ----------------- | ------------------------- | ------------ | --------------------- | ------------ |
| Nonword reading   | 1.60                      | 0.85 to 3.00 | 0.88 (faster)         | suggestive   |
| Phonetic spelling | 1.08                      | 0.56 to 2.09 | 0.57 (faster)         | inconclusive |

**Neither supports a firm claim, but they are not equivalent.** Nonword reading reaches _suggestive_ evidence for a positive first-interval association — P(positive) = 0.88, about 7:1 posterior odds on the direction. Phonetic spelling is _inconclusive_ and close to even (P(positive) = 0.57): the randomised-window information leans weakly faster, consistent with the `itt-009` off-floor headline, and the previous edition's "slower" lean is gone — it was an artefact of the pooled parameterisation, not something the data supported. Power-scaling still flags `tau` in both models (the deliberately regularising Normal(0, 0.5) prior is strong relative to a first-interval contrast estimated from ~40 children), so prior robustness has not been established even for the directional reading.

Taking the nonword point estimate at face value gives a **60% higher underlying first-interval hazard** for the immediate arm, not a 60% higher event probability and not an identified causal effect. Its 89% interval runs from a 15% lower hazard to a 200% higher hazard, so the point estimate carries little weight.

## How to read this, and why it is still worth having

It would be easy to present these as disappointing. They are better understood as **the most that could honestly be extracted from very thin measurements**.

Consider the arithmetic. Nonword reading has 6 items; 36 children were at zero; and from 74 person-period rows the model must estimate a separate hazard for every interval _and_ the first-interval treatment contrast. The fitted per-interval off-floor probabilities run about 25% (untreated first interval) to 29–37% (later, both-arms-treated intervals) for nonword reading, and about 19% down to 14–17% for phonetic spelling, so events are not especially rare — the binding constraint is the number of children and the number of quantities estimated from them, not a shortage of events.

This family's real contribution is methodological: it shows how far a floored outcome can be pushed and where it stops. For nonword reading three approaches now lean the same way — +10 percentage points off-floor in the `itt` family, +2 points in `gain_factors`, and a first-interval hazard ratio of 1.60 here. The survival estimate shares its randomised anchor with the `itt` floor rule but adds covariate adjustment and the hazard framing, so this is descriptive triangulation, not three independent confirmations. None is strong, and a chain of weak agreement is worth no more than its weakest link. The phonetic-spelling picture is now weakly coherent rather than contradictory: all three approaches lean faintly positive (`itt-009` +4 points at P = 0.72; this model's hazard ratio 1.08 at P = 0.57), with every interval straddling no-difference.

## What these models cannot tell you

**They cannot show the intervention did not help these skills.** Inconclusive is not null.

**They describe only the floored subgroup** — children at zero at timepoint 1 — not the whole sample.

**Moving off the floor is a low bar.** Scoring 1 out of 6 is a different achievement from fluent decoding, and this model treats any non-zero score as the event.

**Later intervals carry no arm comparison.** After the wait-list crossover both arms are on the intervention, so the model can describe the later intervals' hazards but cannot compare arms there; the treatment contrast lives entirely in the first interval.

**The hazard ratio is prognostic, not causal.** The contrast is randomisation-anchored (the at-floor subgroup is defined pre-randomisation) but it is adjusted for baseline covariates, conditioned on available-case follow-up, and released without the causal families' robustness gating, so "the intervention made children move off the floor faster" is not licensed.

## Model inventory

Both models pass the convergence gate with zero divergences and are publishable: `surv-009` (phonetic spelling) and `surv-011` (nonword reading).
