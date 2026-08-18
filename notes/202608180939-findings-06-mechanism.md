> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).
>
> Negative-control reading corrected and the ability-adjusted panel added by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `mechanism` family — which skills track which

**Read `findings-00-overview` first.** This note covers the 41 models in the `mechanism` family, the largest in the project. **Nothing in this family is causal.** Every number is an adjusted association.

## The data

**RLI trial only.** These models stack **all period transitions** — timepoint 1→2, 2→3 and 3→4 — into one dataset, so a typical fit has 53 children contributing about 150 rows. Data are pooled across periods, not collapsed: each row is one child in one period.

For the score-exposure models, a row contains the outcome at the end of the period, the same outcome at the start (an autoregressive baseline), the exposure skill's **post-period level**, and adjustment terms. These models do not regress the outcome on a change score for the exposure. Some variants instead use a standardised raw covariate, such as phonological memory, rather than a period-specific score exposure. Each child has a random intercept for their repeated rows.

## What the model is for

For the score-exposure fits, the question is: **is a higher post-period exposure level associated with a higher post-period outcome after conditioning on the outcome's own starting score and the measured adjustment terms?** The raw-covariate variants ask the analogous conditional association for their standardised covariate. Neither design estimates exposure change.

Some models fit a straight line (a single slope). Others fit a flexible curve, which allows the relationship to bend — to flatten at high levels, or to have a threshold below which nothing happens. Where a curve is fitted, the reported number is the average steepness across the observed range.

Adjustment terms include age, hearing, speech production, phonological memory and attendance, chosen from the causal diagram.

## How to read these results — the specificity test

It is tempting to read "letter-sound knowledge → word reading, strongly positive" as "teaching letter sounds raises word reading". **These models cannot support that**, and this family contains its own built-in demonstration of why.

The suite includes **negative-control outcomes**: relationships that the causal diagram says should _not_ represent the proposed decoding channel. Letter-sound knowledge should predict written-code outcomes (word reading, nonword reading) but should not predict oral-language outcomes (vocabulary, grammar, basic concepts) through that channel. Negative controls that come out well above zero would show a shared non-specific component, compatible with a common cause such as general ability, maturation or engagement — though they could not identify which. The pre-specified design treats the panel as one of three strands and makes the nonword-versus-word contrast, not the panel, the primary decoding signature.

On a common scale (log-odds per standard deviation of the exposure), here is what happened:

| Route                                     | Role               | Slope     | 89% range        | P(>0)  |
| ----------------------------------------- | ------------------ | --------- | ---------------- | ------ |
| Letter sounds → **nonword reading**       | positive control   | **1.030** | +0.739 to +1.336 | 1.000  |
| Letter sounds → **basic concepts**        | _negative control_ | **0.291** | +0.163 to +0.423 | 0.9998 |
| Letter sounds → **word reading**          | positive control   | 0.251     | +0.153 to +0.346 | 1.000  |
| Letter sounds → **grammar**               | _negative control_ | 0.124     | +0.045 to +0.203 | 0.993  |
| Letter sounds → **receptive vocabulary**  | _negative control_ | 0.109     | +0.061 to +0.157 | 0.9999 |
| Letter sounds → **expressive vocabulary** | _negative control_ | 0.103     | +0.056 to +0.151 | 0.9997 |

**The negative controls are not zero.** All four are clearly positive, so letter-sound knowledge carries a broad association that reaches outcomes the diagram gives it no causal path to. Something shared runs through every skill here, and the models cannot say what.

But the panel has a gradient, and the gradient is the point. The three load-bearing oral-language controls sit at 0.103 to 0.124; word reading sits at 0.251 and nonword reading at 1.030 — roughly two and nine times higher. Basic concepts (0.291) is the exception, and it is the row that should carry the least weight: it is an 18-item measure, the model fitting it describes it as "the weakest of the four negative controls — a supporting, not load-bearing, panel row", and the pre-specified design listed it as optional. Its interval (+0.163 to +0.423) also overlaps word reading's (+0.153 to +0.346) across almost its whole length, and no contrast between the two was ever fitted. "Basic concepts exceeds word reading" is therefore a comparison of two point estimates from separate models, not a result, and it should not be quoted as one.

So the panel does not show that the letter-sound slopes are indistinguishable from general association. It shows a broad non-specific component **plus** a substantial excess on the written-code outcomes — and the size of that excess is what the suite was built to test.

That test is the **contrast between outcomes**, and it is the pre-specified primary signature rather than a supporting observation. The design note (`notes/202607172330-tier1-decoding-specificity-spec.md`) sets it out in advance: a pure general-ability account gives no reason for letter sounds to predict nonwords _more_ than words, since general ability should if anything favour the broader word-reading skill. Separate single-outcome fits put the nonword-minus-word difference at **+0.78 log-odds per SD** (89% +0.475 to +1.099), but that pairs independent marginal draws and the project's own artefact flags it as **not an identified posterior contrast**. The identified within-model version, from the bivariate `jm-002` fit with a shared child intercept, is **+0.81 [+0.50, +1.13]** with essentially all posterior mass above zero. On its own terms the decoding-specificity test passed, and passed clearly.

Two things stop that from settling the question. Nonword reading is a 6-item measure with most children at the floor, so its logit slope is estimated on a scale that is not directly commensurate with a 79-item word-reading scale — differential measurement is a live alternative to a decoding mechanism. And general ability remains unblockable by construction, so neither slope is causal.

So the defensible summary is: **letter-sound knowledge shows a broad, low-level association with every skill measured, and on top of that a large and well-identified excess for nonword decoding specifically.** The broad component is consistent with a shared cause such as general ability, though the panel cannot identify it as such. The excess is the pattern a decoding route predicts and a pure-confounding account does not, but it is an adjusted association, not a demonstrated channel.

## Testing the ability explanation directly

The panel above says a shared cause is doing some of the work but cannot name it. Since a
measured general-ability score (WPPSI Block Design) exists for every child, is recorded
before the intervention and never changes, the whole panel was refitted with it partialled
out (`mech-196`–`201`, one per outcome, identical to their parents in every other respect
and fitted on exactly the same rows).

| Route                      | Role               | Without ability | With ability adjusted | Change |
| -------------------------- | ------------------ | --------------- | --------------------- | ------ |
| Letter sounds → nonwords   | positive control   | 1.030           | **1.027**             | −0.3%  |
| Letter sounds → word read. | positive control   | 0.251           | **0.246**             | −2.0%  |
| Letter sounds → concepts   | _negative control_ | 0.291           | 0.249                 | −14.4% |
| Letter sounds → grammar    | _negative control_ | 0.124           | 0.072                 | −41.6% |
| Letter sounds → rec. vocab | _negative control_ | 0.109           | 0.092                 | −15.3% |
| Letter sounds → exp. vocab | _negative control_ | 0.103           | 0.103                 | −0.6%  |

Three things come out of this, and they do not all point the same way.

**The written-code slopes are untouched.** Nonword reading moves by 0.3% and word reading by
2%, both well inside their own uncertainty. The decoding contrast is unchanged: +0.78
[+0.48, +1.10] before, +0.78 [+0.47, +1.11] after. Whatever the letter-sound-to-decoding
excess is, it is not measured general ability.

**Measured ability really does predict the oral-language outcomes** — its own coefficient is
clearly positive for all four (basic concepts +0.23, grammar +0.19, receptive vocabulary
+0.10, expressive vocabulary +0.08, every one with a probability of at least 0.995) — and
clearly _not_ for the written-code outcomes (nonwords +0.03, P = 0.57; word reading +0.04,
P = 0.77). So the adjustment is doing real work exactly where the panel predicted a
general-ability path would run.

**But the negative controls mostly survive it.** Only grammar attenuates substantially, from
0.124 with an interval clear of zero to 0.072 with an interval that now includes it.
Receptive vocabulary and basic concepts lose about a seventh of their slope and expressive
vocabulary none at all; three of the four remain positive with very strong directional
evidence. Adjusting for measured ability does not dissolve the non-specific component.

The same holds for the fitted **shape**, not just the slope. `mech-258` repeats the headline `mech-058` curve with ability partialled out and nothing else changed: the two curves lie almost on top of each other, the endpoint contrast moving from +6.78 items [+2.52, +11.11] to +6.51 [+2.22, +10.90], a 4% shift, with a maximum pointwise gap of 0.15 items. The overlay is `mechanism_curve_ability_overlay.png` in the comparison directory.

The natural conclusion is that the shared cause is not, or not only, the general ability this
battery measures. It could be an ability dimension Block Design does not capture, shared
teaching dose, maturation over the study, or common method variance across tests
administered together. It is also possible that ability _is_ the explanation and a single
subtest is simply too noisy a stand-in to absorb it — adjusting for a mismeasured confounder
removes only part of its influence, so a small attenuation is not evidence of a small
confound. These fits narrow the field; they do not close it.

## Other results

**Word reading's other candidate routes.** On the comparable per-SD scale used in the cross-model forest: letter sounds 0.238 [0.082, 0.410], expressive vocabulary 0.122 [−0.013, 0.257], receptive vocabulary 0.064 [−0.057, 0.185]. Letter sounds lead, with the two vocabulary routes weaker and their intervals including zero.

**Curve tests.** Several models fit flexible curves to investigate nonlinearity or a threshold. The vocabulary-route curves (`mech-156`, `157`) have small average slopes with intervals spanning zero; those averages do not by themselves prove that a threshold is absent. `mech-058` and `mech-071` are both flexible letter-sound curves, with endpoint contrasts of +6.78 and +5.29 items respectively; `mech-071` additionally includes expressive-vocabulary moderation. The actual linear anchor is `mech-101`, at +9.88 items [+6.10, +13.51]. The positive endpoint contrasts agree in direction, but they are not a formal test that the curve is linear.

**Moderation.** Whether the letter-sound association varies with age (`mech-073`) or phonological memory (`mech-104`, `204`) was tested, and the two answers differ. The age interaction is −0.057 with P(negative) = 0.93 — **moderate evidence that the adjusted association is weaker for older children**, though its 89% interval still includes zero; this is associational moderation, not causal effect modification. The phonological-memory interaction is −0.008 with P(negative) = 0.55, which is **inconclusive**; a formal predictive comparison of the memory-moderation pair likewise returned "inconclusive (|elpd_diff| < 4)". Neither result licenses a claim that the intervention or a causal decoding route works differently for different children.

**Other exposures.** Phonological memory → word reading +3.07 items [+0.16, +5.88]; phonological memory → nonword reading and speech production → nonword reading are both clearly positive. Given the negative-control result above, read all of these as adjusted associations subject to the same non-specificity and residual-confounding concern.

## What these models cannot tell you

**No slope here is identified as a lever.** That follows from the observational design. The non-zero negative controls add measured evidence that part of every slope is non-specific, and the nonword-versus-word excess is still an adjusted association rather than a demonstrated channel.

**Adjusting for measured covariates does not remove residual confounding.** General ability is a latent node in the project's causal diagram and, in the design note's own words, "structurally unblockable": no adjustment set closes that path, which is why it is absent from the diagram-derived conditioning set the primary panel uses. The ability-adjusted panel above partials out the measured block-design proxy and leaves most of the non-specific component standing, so the residual confounding these models carry is not reducible to the ability this battery measures — and a single subtest cannot rule out the latent node behind it.

**Direction is not established.** These are contemporaneous-period associations with an autoregressive baseline. A child whose reading improves may attend more to letters, as easily as the reverse.

**The exposure was not manipulated.** For the one relationship that _was_ manipulated — assignment to the intervention — see the `itt` and `mediation` families.

## Model inventory

All 41 pass the convergence gate with zero divergences and are publishable — 34 original models, the six-model ability-adjusted panel `mech-196`–`201` added for the test above, and `mech-258`, the ability-adjusted counterpart of the `mech-058` curve. Three (`mech-073`, `104`, `204`) initially failed on a single divergence each and were refitted at a higher acceptance target; their headline slopes moved little relative to their posterior uncertainty. Key models: `056`/`057`/`058` (R/E/L → W), `096`/`101` (Tier-1 decoding contrast), `097`–`100` (negative controls), `088`/`089` (taught vocabulary → W), `090`/`102` (phonological memory), `103` (speech production), `061`/`063`/`093`–`095`/`161`/`163` (joint-readiness interactions), `156`–`158`/`188`–`191` (curve tests), `072`/`172` (code route), `196`–`201` (ability-adjusted Tier-1 panel), `258` (ability-adjusted curve).
