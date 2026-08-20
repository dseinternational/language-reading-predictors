> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Findings: the `mediation` and `mediation_multi` families — how the reading gain happened

**Refreshed from the 2026-08-20 full refit** (`notes/202608200800-full-refit-both-layers-2026-08.md`). The g-formula fits carry the hearing adjuster, so every decomposition below moved slightly; the direction and the reading are unchanged.

**Read `findings-00-overview` first.** This note covers 15 `mediation` models and 4 `mediation_multi` models, which ask _through what route_ the intervention improved word reading.

## The data

**RLI trial only.** Most of these models use the **randomised timepoint 1 to timepoint 2 window**, one row per child, 50–53 children depending on which measures a child has. One model (`med-092`) stacks all periods (157 rows) as a sensitivity check.

The main chain, exemplified by `med-059`, is: assigned arm → the mediator measured at timepoint 2 → word reading at timepoint 2, with baselines for both. Other fits change the mediator, outcome, timing or estimand, so this description does not apply literally to every model in the inventory.

## What the model is for

The treatment families establish _that_ word reading improved. This family asks _how_. If the intervention taught letter sounds, and letter-sound knowledge is what lets a child decode a word, then the reading gain should mostly travel through letter-sound knowledge rather than appear independently of it.

The method splits the total effect into two parts:

- **The indirect effect (NIE)** — the part that runs _through_ the mediator. This is the "it worked by teaching letter sounds" component.
- **The direct effect (NDE)** — everything else, the part that would remain if the mediator had somehow not changed.

These are computed by **counterfactual simulation** (a g-formula): the fitted model is used to simulate each child twice — once as assigned, once under the alternative assignment — and the difference is averaged over children. That is why these fits are the slowest in the project, and why they report no predictive score: they are not a single regression whose fit can be scored pointwise.

**The proportion mediated** is the indirect effect as a share of the total. It is intuitive but statistically badly behaved: when the total effect is small, the ratio's denominator approaches zero and the interval explodes. Several intervals below run outside 0–1 for exactly this reason, which is a property of the ratio, not a sign the model failed.

## The interpretive caution that governs everything here

**Mediation is not randomised, even inside a randomised trial.** Children were randomly assigned to the intervention; they were _not_ randomly assigned to end up with more letter-sound knowledge. Splitting a total effect into direct and indirect parts requires no unmeasured mediator–outcome confounding, yet latent general ability plausibly affects both and is not measured here.

There is a second, structural problem. Intervention sessions are caused by assignment and plausibly affect both letter-sound knowledge and word reading, making dose a **treatment-induced mediator–outcome confounder**. That means the natural direct and indirect effects are not point-identified under the project's own causal diagram; measuring and adjusting for dose would not repair the cross-world assumption. In addition, letter sounds and word reading are both measured at timepoint 2, so the main fit does not establish that the mediator came first.

The honest reading is therefore: these are **model-based g-formula decompositions under strong, partly violated identification assumptions**, not identified natural effects. They describe how the fitted model allocates the association; they do not establish the route by which the intervention worked.

## What was found

### Word reading through letter-sound knowledge (`med-059`)

| Quantity                        | Estimate (words) | 89% range          | P(>0)     |
| ------------------------------- | ---------------- | ------------------ | --------- |
| Total                           | +2.06            | −0.18 to +4.23     | 0.931     |
| **Indirect, via letter sounds** | **+1.79**        | **+0.66 to +3.37** | **0.997** |
| Direct                          | +0.17            | −1.80 to +2.15     | 0.555     |

**Within this fitted decomposition, almost the entire word-reading gain is allocated to letter-sound knowledge.** The model-based indirect component has very strong directional evidence; the direct component is poorly determined — a point estimate of +0.17 words with P(>0) = 0.56. The proportion mediated is about 0.83, conditional on the assumptions above.

Note the pattern: the _total_ effect here is less certain than the _indirect_ component. That is not a contradiction: the quantities are different functions of correlated posterior draws and need not have intervals of the same width. It is not evidence that the indirect component is better identified causally.

### Two mediators at once

The two-mediator fits compare how the fitted decomposition is allocated between letter sounds and a rival mediator:

| Model     | Rival mediator        | Via letter sounds                   | Via the rival                  |
| --------- | --------------------- | ----------------------------------- | ------------------------------ |
| `med-064` | Expressive vocabulary | **+2.01** [+0.70, +3.85], P = 0.997 | +0.03 [−0.50, +0.76], P = 0.57 |
| `med-066` | Phoneme blending      | **+1.77** [+0.60, +3.41], P = 0.996 | −0.06 [−0.68, +0.32], P = 0.36 |

In both fitted decompositions, almost all of the indirect component is allocated to letter-sound knowledge. The rival-mediator posteriors are centred near zero but remain imprecise. The blending result is worth dwelling on, because blending _did_ improve under the intervention — but improving alongside the outcome is not the same as being the route to it.

### The negative-control check — limited reassurance

`med-079` deliberately runs the same machinery through **grammar**, a mediator the causal diagram says should _not_ carry a reading effect. A comparably large indirect component here would be a warning that the decomposition lacked route specificity; a small estimate can provide only limited reassurance because the negative control need not share the main mediator's confounding and measurement structure.

| Quantity              | Estimate (words) | 89% range      | P(>0) |
| --------------------- | ---------------- | -------------- | ----- |
| Total                 | +2.23            | +0.41 to +4.00 | 0.976 |
| Indirect, via grammar | **+0.08**        | −0.19 to +0.63 | 0.711 |
| Direct                | +2.08            | +0.31 to +3.83 | 0.970 |

The point estimate through grammar is small, but this is **not evidence of equivalence to zero**. Its 89% interval still runs from −0.19 to +0.63 words, P(>0) = 0.711 is below the project's threshold for even suggestive directional evidence, and no negligible-effect threshold was tested. The check did not reveal a strong grammar route, but it cannot validate the letter-sound decomposition or rule out artefact.

`med-074` provides a second check through nonword decoding. Its indirect-component median is +0.02 words with P(>0) = 0.56, an unresolved direction; because that measure is severely floored, the check has little power to distinguish a genuinely small route from poor measurement.

### Sensitivity checks

`med-062` uses a broader composite code-based route: indirect +0.86 [+0.01, +2.14], direct +0.84 — same direction, less concentrated.

The named dose-confounding calibration is more consequential. At its point calibration, session-related confounding maps to an NIE of +1.63 words [+0.56, +3.14], close to the primary +1.79. The broad endpoint scenario — an envelope of separate 89% slope endpoints, not a joint credible interval — reaches the tipping point at which the fitted NIE's 89% interval first includes zero (its median is still slightly positive there); that tipping point is only 52.5% of the fitted mediator–outcome slope. The stored verdict is therefore that intervention-session confounding **could plausibly account for the estimated NIE**, not that the decomposition is robust.

A temporal-ordering sensitivity uses letter sounds at timepoint 2 and word reading at timepoint 3. It preserves a positive model-based indirect component (+3.14 [+1.31, +5.38]) but changes the total (+1.78 [−1.45, +4.87]) and direct (−1.46 [−4.01, +1.13]) components materially. Because timepoint 3 is post-crossover, this is a temporal check rather than an identified randomised mediation estimand.

`med-092` stacks all periods rather than the randomised window alone: total +3.10, indirect +0.75 [+0.28, +1.39], direct +2.32 [+0.33, +4.18], proportion mediated 0.25. **This is the one that disagrees**, putting most of the effect in the direct path. It is also the model whose later periods are post-crossover and therefore not randomised, so its total is not the same estimand. It is reported for completeness; if the fitted decompositions are described, the randomised-window versions are the closer match to the treatment-effect window, but they retain all the mediation-identification failures above.

## What these models cannot tell you

**They cannot identify letter sounds as the mechanism.** The natural effects are structurally non-identified because dose is treatment-induced, latent general ability can confound mediator and outcome, and the primary mediator and outcome are contemporaneous. An unmeasured factor driving both letter-sound gains and reading gains could produce the same numbers.

**The proportion mediated should not be quoted as a percentage with confidence.** Its intervals here run well outside 0–1.

**"Direct effect" does not mean "unexplained by anything".** It means "not through _this_ mediator" — it may run through routes not modelled.

**Nothing here licenses a teaching claim.** That teaching letter sounds _more_ would raise reading _further_ is an extrapolation this design cannot support.

## Model inventory

All 19 models pass the convergence gate with zero divergences and are publishable. These are the slowest fits in the project (`med-064` took 42 minutes) because the counterfactual simulation is expensive. Single-mediator: `med-059` (via L), `062` (code route), `068`, `074` (via N), `076`, `078`, `079` (negative control, grammar), `080`, `086`, `087`, `092` (period-stacked), `176`, `186`, `187`, `276`. Two-mediator: `med-060`, `064` (L vs E), `066` (L vs B), `075`.
