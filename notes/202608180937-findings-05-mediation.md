> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `mediation` and `mediation_multi` families — how the reading gain happened

**Read `findings-00-overview` first.** This note covers 15 `mediation` models and 4 `mediation_multi` models, which ask _through what route_ the intervention improved word reading.

## The data

**RLI trial only.** Most of these models use the **randomised timepoint 1 to timepoint 2 window**, one row per child, 50–53 children depending on which measures a child has. One model (`med-092`) stacks all periods (157 rows) as a sensitivity check.

The chain being modelled is: assigned arm → the mediator measured at timepoint 2 → word reading at timepoint 2, with baselines for both.

## What the model is for

The treatment families establish _that_ word reading improved. This family asks _how_. If the intervention taught letter sounds, and letter-sound knowledge is what lets a child decode a word, then the reading gain should mostly travel through letter-sound knowledge rather than appear independently of it.

The method splits the total effect into two parts:

- **The indirect effect (NIE)** — the part that runs _through_ the mediator. This is the "it worked by teaching letter sounds" component.
- **The direct effect (NDE)** — everything else, the part that would remain if the mediator had somehow not changed.

These are computed by **counterfactual simulation** (a g-formula): the fitted model is used to simulate each child twice — once as assigned, once under the alternative assignment — and the difference is averaged over children. That is why these fits are the slowest in the project, and why they report no predictive score: they are not a single regression whose fit can be scored pointwise.

**The proportion mediated** is the indirect effect as a share of the total. It is intuitive but statistically badly behaved: when the total effect is small, the ratio's denominator approaches zero and the interval explodes. Several intervals below run outside 0–1 for exactly this reason, which is a property of the ratio, not a sign the model failed.

## The interpretive caution that governs everything here

**Mediation is not randomised, even inside a randomised trial.** Children were randomly assigned to the intervention; they were _not_ randomly assigned to end up with more letter-sound knowledge. Splitting a total effect into direct and indirect parts requires the assumption that nothing unmeasured causes both the mediator and the outcome — an assumption the data cannot check.

So the honest reading is: _if_ the causal diagram is right, this is how the effect decomposes. The decomposition is much more assumption-dependent than the treatment effect it decomposes.

## What was found

### Word reading through letter-sound knowledge (`med-059`)

| Quantity                        | Estimate (words) | 89% range          | P(>0)     |
| ------------------------------- | ---------------- | ------------------ | --------- |
| Total                           | +1.95            | −0.21 to +4.07     | 0.926     |
| **Indirect, via letter sounds** | **+1.70**        | **+0.58 to +3.24** | **0.996** |
| Direct                          | +0.15            | −1.72 to +2.08     | 0.549     |

**Almost the entire word-reading gain runs through letter-sound knowledge.** The indirect path is well supported (very strong evidence for a positive route); the direct path is indistinguishable from zero — a point estimate of +0.15 words with a probability of 0.55, which is as close to "no information" as a result gets. The proportion mediated is about 0.83.

Note the pattern: the _total_ effect here is less certain than the _indirect_ component. That is not a contradiction. The total absorbs the noise in both paths, while the indirect path is estimated from a chain of two well-measured relationships (the intervention clearly raised letter-sound knowledge, and letter-sound knowledge clearly tracks word reading).

### Two mediators at once

Running letter sounds against a rival mediator sharpens the picture:

| Model     | Rival mediator        | Via letter sounds                   | Via the rival                  |
| --------- | --------------------- | ----------------------------------- | ------------------------------ |
| `med-064` | Expressive vocabulary | **+1.87** [+0.61, +3.64], P = 0.996 | +0.03 [−0.47, +0.74], P = 0.58 |
| `med-066` | Phoneme blending      | **+1.62** [+0.52, +3.18], P = 0.995 | −0.03 [−0.64, +0.42], P = 0.42 |

In both, letter-sound knowledge carries essentially the whole indirect effect and the rival carries nothing detectable. The blending result is worth dwelling on, because blending _did_ improve under the intervention — but improving alongside the outcome is not the same as being the route to it.

### The negative control — the most reassuring result here

`med-079` deliberately runs the same machinery through **grammar**, a mediator the causal diagram says should _not_ carry a reading effect. If the method were simply manufacturing indirect effects, this would produce one.

| Quantity              | Estimate (words) | 89% range      | P(>0) |
| --------------------- | ---------------- | -------------- | ----- |
| Total                 | +2.23            | +0.41 to +4.00 | 0.976 |
| Indirect, via grammar | **+0.08**        | −0.19 to +0.63 | 0.711 |
| Direct                | +2.08            | +0.31 to +3.83 | 0.970 |

The indirect path through grammar is essentially zero (proportion mediated 0.04), and the effect stays in the direct component. The method finds a route where theory predicts one and finds nothing where it does not. That is real evidence the letter-sound result is not an artefact.

`med-074` provides a second check through nonword decoding, and also finds nothing (indirect +0.02 words, P = 0.56) — but that measure is severely floored, so an absence there is weak evidence either way.

### Sensitivity checks

`med-062` uses a broader composite code-based route: indirect +0.93 [+0.08, +2.16], direct +0.60 — same direction, less concentrated.

`med-092` stacks all periods rather than the randomised window alone: total +3.03, indirect +0.75 [+0.29, +1.39], direct +2.24 [+0.28, +4.03], proportion mediated 0.26. **This is the one that disagrees**, putting most of the effect in the direct path. It is also the model whose later periods are post-crossover and therefore not randomised, so its total is not the same estimand. It is reported for completeness; the randomised-window models are the ones to quote.

## What these models cannot tell you

**They cannot prove letter sounds are the mechanism.** They show the data are consistent with that route, under a diagram assumed rather than tested. An unmeasured factor driving both letter-sound gains and reading gains would produce the same numbers.

**The proportion mediated should not be quoted as a percentage with confidence.** Its intervals here run well outside 0–1.

**"Direct effect" does not mean "unexplained by anything".** It means "not through _this_ mediator" — it may run through routes not modelled.

**Nothing here licenses a teaching claim.** That teaching letter sounds _more_ would raise reading _further_ is an extrapolation this design cannot support.

## Model inventory

All 19 models pass the convergence gate with zero divergences and are publishable. These are the slowest fits in the project (`med-064` took 42 minutes) because the counterfactual simulation is expensive. Single-mediator: `med-059` (via L), `062` (code route), `068`, `074` (via N), `076`, `078`, `079` (negative control, grammar), `080`, `086`, `087`, `092` (period-stacked), `176`, `186`, `187`, `276`. Two-mediator: `med-060`, `064` (L vs E), `066` (L vs B), `075`.
