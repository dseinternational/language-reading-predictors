> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `mechanism` family — which skills track which

**Read `findings-00-overview` first.** This note covers the 34 models in the `mechanism` family, the largest in the project. **Nothing in this family is causal.** Every number is an adjusted association.

## The data

**RLI trial only.** These models stack **all period transitions** — timepoint 1→2, 2→3 and 3→4 — into one dataset, so a typical fit has 53 children contributing about 150 rows. Data are pooled across periods, not collapsed: each row is one child in one period.

The structure of a row is: the outcome at the end of the period, the same outcome at the start (an autoregressive baseline), the _exposure_ skill measured in that same period, and adjustment terms. Each child has a random intercept for their repeated rows.

## What the model is for

The question is: **as one skill changes, does another change with it, once we account for where the child started and for the other things we can measure?**

Some models fit a straight line (a single slope). Others fit a flexible curve, which allows the relationship to bend — to flatten at high levels, or to have a threshold below which nothing happens. Where a curve is fitted, the reported number is the average steepness across the observed range.

Adjustment terms include age, hearing, speech production, phonological memory and attendance, chosen from the causal diagram.

## How to read these results — and why the negative controls matter most

It is tempting to read "letter-sound knowledge → word reading, strongly positive" as "teaching letter sounds raises word reading". **These models cannot support that**, and this family contains its own built-in demonstration of why.

The suite includes **negative-control outcomes**: relationships that the causal diagram says should _not_ exist. Letter-sound knowledge should predict written-code outcomes (word reading, nonword reading) but should not predict oral-language outcomes (vocabulary, grammar, basic concepts) except through shared general ability. If the negative controls come out at zero, the specific-channel story survives. If they come out positive, something common to all outcomes — general ability, maturation, engagement — is driving the associations.

On a common scale (log-odds per standard deviation of the exposure), here is what happened:

| Route                                     | Role               | Slope     | 89% range        | P(>0)  |
| ----------------------------------------- | ------------------ | --------- | ---------------- | ------ |
| Letter sounds → **nonword reading**       | positive control   | **1.030** | +0.739 to +1.336 | 1.000  |
| Letter sounds → **basic concepts**        | _negative control_ | **0.291** | +0.163 to +0.423 | 0.9998 |
| Letter sounds → **word reading**          | positive control   | 0.251     | +0.153 to +0.346 | 1.000  |
| Letter sounds → **grammar**               | _negative control_ | 0.124     | +0.045 to +0.203 | 0.993  |
| Letter sounds → **receptive vocabulary**  | _negative control_ | 0.109     | +0.061 to +0.157 | 0.9999 |
| Letter sounds → **expressive vocabulary** | _negative control_ | 0.103     | +0.056 to +0.151 | 0.9997 |

**The negative controls are not zero. Every one of them is clearly positive.** And the association between letter sounds and _basic concept knowledge_ (0.291) is **larger** than the association between letter sounds and _word reading_ (0.251) — the very relationship the family exists to characterise.

That is the single most important finding in this family, and it points one way: **a substantial part of every mechanism slope here reflects a common cause rather than a specific channel.** Children who know more letter sounds are, on average, children who are doing better generally; and children doing better generally score higher on vocabulary, grammar and concepts too. The models adjust for what was measured, and latent general ability was not measured.

What does survive is the **contrast between outcomes**. Letter sounds track nonword reading far more strongly than they track word reading: the difference is **+0.78 log-odds per SD** (89% +0.475 to +1.099). Nonword reading can only be done by decoding — the words are invented, so they cannot be recognised by sight — and it is exactly the outcome where letter-sound knowledge stands furthest clear of everything else. A general-ability confound should lift all outcomes together; it does not obviously predict that one specific outcome would separate by this margin.

So the defensible summary is: **the pattern is consistent with a decoding-specific channel from letter sounds to nonword reading, sitting on top of a broad general-ability association that inflates every slope in this family, including the letter-sound-to-word-reading slope.**

## Other results

**Word reading's other candidate routes.** On the comparable per-SD scale used in the cross-model forest: letter sounds 0.238 [0.082, 0.410], expressive vocabulary 0.122 [−0.013, 0.257], receptive vocabulary 0.064 [−0.057, 0.185]. Letter sounds lead, with the two vocabulary routes weaker and their intervals including zero.

**Curve tests.** Several models fit flexible curves to test for a threshold — a level below which a skill does not yet help. The curve models for the vocabulary routes (`mech-156`, `157`) return small average slopes with intervals spanning zero, so no reliable threshold shape was found. For letter sounds the curve and linear versions agree (`mech-058` +6.78 items, `mech-071` +5.29 items on the outcome scale).

**Moderation.** Whether the letter-sound route varies with age (`mech-073`) or phonological memory (`mech-104`, `204`) was tested, and the two answers differ. The age interaction is −0.057 with P(negative) = 0.93 — **moderate evidence that the association is weaker for older children**, though its 89% interval still includes zero and the estimate is an adjusted association, not a moderated effect. The phonological-memory interaction is −0.008 with P(negative) = 0.55, which is **inconclusive**; a formal predictive comparison of the memory-moderation pair likewise returned "inconclusive (|elpd_diff| < 4)". Neither result licenses a claim that the intervention or the decoding route works differently for different children.

**Other exposures.** Phonological memory → word reading +3.07 items [+0.16, +5.88]; phonological memory → nonword reading and speech production → nonword reading are both clearly positive. Given the negative-control result above, read all of these as associations carrying the same general-ability component.

## What these models cannot tell you

**No slope here is a lever.** The negative controls demonstrate this within the family itself — not as a theoretical caution but as a measured result.

**Adjusting for measured covariates does not remove the confound.** Latent general ability is not in the data, and the models say so in their own recorded assumptions.

**Direction is not established.** These are contemporaneous-period associations with an autoregressive baseline. A child whose reading improves may attend more to letters, as easily as the reverse.

**The exposure was not manipulated.** For the one relationship that _was_ manipulated — assignment to the intervention — see the `itt` and `mediation` families.

## Model inventory

All 34 pass the convergence gate with zero divergences and are publishable. Three (`mech-073`, `104`, `204`) initially failed on a single divergence each and were refitted at a higher acceptance target; their estimates moved by about 1%, confirming the original values. Key models: `056`/`057`/`058` (R/E/L → W), `096`/`101` (Tier-1 decoding contrast), `097`–`100` (negative controls), `088`/`089` (taught vocabulary → W), `090`/`102` (phonological memory), `103` (speech production), `061`/`063`/`093`–`095`/`161`/`163` (joint-readiness interactions), `156`–`158`/`188`–`191` (curve tests), `072`/`172` (code route).
