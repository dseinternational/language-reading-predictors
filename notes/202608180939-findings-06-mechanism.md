> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Findings: the `mechanism` family — which skills track which

**Read `findings-00-overview` first.** This note covers the 34 models in the `mechanism` family, the largest in the project. **Nothing in this family is causal.** Every number is an adjusted association.

## The data

**RLI trial only.** These models stack **all period transitions** — timepoint 1→2, 2→3 and 3→4 — into one dataset, so a typical fit has 53 children contributing about 150 rows. Data are pooled across periods, not collapsed: each row is one child in one period.

For the score-exposure models, a row contains the outcome at the end of the period, the same outcome at the start (an autoregressive baseline), the exposure skill's **post-period level**, and adjustment terms. These models do not regress the outcome on a change score for the exposure. Some variants instead use a standardised raw covariate, such as phonological memory, rather than a period-specific score exposure. Each child has a random intercept for their repeated rows.

## What the model is for

For the score-exposure fits, the question is: **is a higher post-period exposure level associated with a higher post-period outcome after conditioning on the outcome's own starting score and the measured adjustment terms?** The raw-covariate variants ask the analogous conditional association for their standardised covariate. Neither design estimates exposure change.

Some models fit a straight line (a single slope). Others fit a flexible curve, which allows the relationship to bend — to flatten at high levels, or to have a threshold below which nothing happens. Where a curve is fitted, the reported number is the average steepness across the observed range.

Adjustment terms include age, hearing, speech production, phonological memory and attendance, chosen from the causal diagram.

## How to read these results — and why the negative controls matter most

It is tempting to read "letter-sound knowledge → word reading, strongly positive" as "teaching letter sounds raises word reading". **These models cannot support that**, and this family contains its own built-in demonstration of why.

The suite includes **negative-control outcomes**: relationships that the causal diagram says should _not_ represent the proposed decoding channel. Letter-sound knowledge should predict written-code outcomes (word reading, nonword reading) but should not predict oral-language outcomes (vocabulary, grammar, basic concepts) through that channel. Positive negative controls would show that the fitted slopes are not specific; they would be compatible with a shared cause such as general ability, maturation or engagement, but would not identify which explanation is responsible.

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

That is the single most important finding in this family: **the slopes are not specific to the proposed decoding channel.** A shared underlying ability is one plausible explanation, but the negative controls cannot identify the source or quantify what fraction of each slope it contributes; residual confounding, measurement differences and model misspecification can produce the same pattern. The models adjust for what was measured, and latent general ability was not measured.

The **contrast between outcomes** is nevertheless informative as a pattern. Separate single-outcome fits put the nonword-minus-word slope difference at **+0.78 log-odds per SD** (89% +0.475 to +1.099), but that comparison pairs independent marginal draws and is explicitly **not an identified posterior contrast**. The identified within-model contrast from `jm-002` is +0.81 [+0.50, +1.13] and agrees numerically. Nonword reading requires decoding rather than sight-word recognition, so the larger association is compatible with a decoding-specific channel, but differential measurement and residual confounding remain alternative explanations.

So the defensible summary is: **the pattern is consistent with a decoding-specific association between letter sounds and nonword reading, against a background of broad non-specific association across outcomes.** The data do not identify that background uniquely as general ability or establish either slope as a causal channel.

## Other results

**Word reading's other candidate routes.** On the comparable per-SD scale used in the cross-model forest: letter sounds 0.238 [0.082, 0.410], expressive vocabulary 0.122 [−0.013, 0.257], receptive vocabulary 0.064 [−0.057, 0.185]. Letter sounds lead, with the two vocabulary routes weaker and their intervals including zero.

**Curve tests.** Several models fit flexible curves to investigate nonlinearity or a threshold. The vocabulary-route curves (`mech-156`, `157`) have small average slopes with intervals spanning zero; those averages do not by themselves prove that a threshold is absent. `mech-058` and `mech-071` are both flexible letter-sound curves, with endpoint contrasts of +6.78 and +5.29 items respectively; `mech-071` additionally includes expressive-vocabulary moderation. The actual linear anchor is `mech-101`, at +9.88 items [+6.10, +13.51]. The positive endpoint contrasts agree in direction, but they are not a formal test that the curve is linear.

**Moderation.** Whether the letter-sound association varies with age (`mech-073`) or phonological memory (`mech-104`, `204`) was tested, and the two answers differ. The age interaction is −0.057 with P(negative) = 0.93 — **moderate evidence that the adjusted association is weaker for older children**, though its 89% interval still includes zero; this is associational moderation, not causal effect modification. The phonological-memory interaction is −0.008 with P(negative) = 0.55, which is **inconclusive**; a formal predictive comparison of the memory-moderation pair likewise returned "inconclusive (|elpd_diff| < 4)". Neither result licenses a claim that the intervention or a causal decoding route works differently for different children.

**Other exposures.** Phonological memory → word reading +3.07 items [+0.16, +5.88]; phonological memory → nonword reading and speech production → nonword reading are both clearly positive. Given the negative-control result above, read all of these as adjusted associations subject to the same non-specificity and residual-confounding concern.

## What these models cannot tell you

**No slope here is identified as a lever.** That follows from the observational design; the positive negative controls add measured evidence that the fitted associations are not specific to the proposed channel.

**Adjusting for measured covariates does not remove residual confounding.** Latent general ability is one plausible omitted cause, but the negative controls do not establish that it is the only explanation.

**Direction is not established.** These are contemporaneous-period associations with an autoregressive baseline. A child whose reading improves may attend more to letters, as easily as the reverse.

**The exposure was not manipulated.** For the one relationship that _was_ manipulated — assignment to the intervention — see the `itt` and `mediation` families.

## Model inventory

All 34 pass the convergence gate with zero divergences and are publishable. Three (`mech-073`, `104`, `204`) initially failed on a single divergence each and were refitted at a higher acceptance target; their headline slopes moved little relative to their posterior uncertainty. Key models: `056`/`057`/`058` (R/E/L → W), `096`/`101` (Tier-1 decoding contrast), `097`–`100` (negative controls), `088`/`089` (taught vocabulary → W), `090`/`102` (phonological memory), `103` (speech production), `061`/`063`/`093`–`095`/`161`/`163` (joint-readiness interactions), `156`–`158`/`188`–`191` (curve tests), `072`/`172` (code route).
