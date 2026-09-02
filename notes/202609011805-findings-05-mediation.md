> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `mediation` and `mediation_multi` families — how the reading gain happened

**Read `findings-00-overview` first.** This note covers 16 `mediation` models and 4 `mediation_multi` models, which ask _through what route_ the intervention improved word reading (and, in three fits, nonword decoding and phoneme blending). All 20 pass the convergence gate with zero divergences and all are publishable (2026-09-01 rebuild). `med-059`'s total effect of 2.3195 words is the batch's reproduction check: it matches the 2026-08-27 value exactly.

## The data

**RLI trial only.** Most models use the **randomised timepoint 1 to timepoint 2 window**, one row per child, 50–53 children depending on which measures a child has. `med-092` stacks all periods (157 rows) and headlines the period-1 window; `med-076`, `176` and `276` use a later outcome wave to impose temporal ordering.

The main chain, exemplified by `med-059`, is assigned arm → the mediator at timepoint 2 → word reading at timepoint 2, with baselines for both. Since #585 every leg conditions on one common pre-exposure vector — the outcome baseline, the mediator baseline(s) and every bounded-measure confounder — so the g-formula composes coherent legs; a floored measure's baseline enters as the binary off-floor indicator.

## What the model is for

The treatment families establish _that_ word reading improved. This family asks _how_, by counterfactual simulation: the fitted model simulates each child under both assignments and splits the total contrast into an **indirect component** through the mediator (NIE) and a **direct component** (NDE), everything else. The **proportion mediated** is the ratio of the two and is badly behaved when the total is small — several of its intervals run outside 0–1, which is a property of the ratio.

## The interpretive caution that governs everything here

**Mediation is not randomised, even inside a randomised trial.** Children were not randomly assigned to letter-sound knowledge, and splitting a total effect requires no unmeasured mediator–outcome confounding, which latent general ability plausibly violates. Intervention sessions are a **treatment-induced mediator–outcome confounder**, so the natural effects are not point-identified under the project's own causal diagram. And in the main fits the mediator and outcome are measured at the same wave. Every decomposition below is a **model-based g-formula allocation under strong, partly violated assumptions**. It describes how the fitted model distributes the association, not the route by which the intervention worked.

## What was found

### Word reading through letter-sound knowledge (`med-059`)

| Quantity                        | Estimate (words) | 89% range        | P(>0)     |
| ------------------------------- | ---------------- | ---------------- | --------- |
| Total                           | +2.3             | +0.1 to +4.5     | 0.954     |
| **Indirect, via letter sounds** | **+2.1**         | **+0.9 to +3.7** | **0.999** |
| Direct                          | +0.1             | −1.8 to +2.2     | 0.544     |

Within this fitted decomposition almost the whole word-reading gain is allocated to letter-sound knowledge; the direct component is centred on zero and poorly determined. The proportion mediated is about 0.9 (89% 0.2 to 3.0). The interventional relabelling `med-078` is numerically identical, as it must be: intervention dose is in neither leg, so the label changes the target, not the identification.

### Two mediators at once and the sequential routes

| Model     | Rival or second mediator          | Via letter sounds                | Via the rival               | Total             |
| --------- | --------------------------------- | -------------------------------- | --------------------------- | ----------------- |
| `med-064` | Expressive vocabulary (parallel)  | **+2.5** [+1.0, +4.4], P = 0.998 | +0.0 [−0.6, +0.7], P = 0.51 | +2.5 [+0.1, +5.0] |
| `med-066` | Phoneme blending (parallel)       | **+2.1** [+0.8, +3.8], P = 0.996 | −0.1 [−0.7, +0.3], P = 0.37 | +2.2 [−0.1, +4.5] |
| `med-075` | L → blending → reading (sequence) | **+2.1** [+0.8, +3.8], P = 0.997 | −0.1 [−0.7, +0.4], P = 0.37 | +2.2 [−0.1, +4.5] |
| `med-060` | L → nonword off-floor → reading   | **+2.8** [+1.1, +4.9], P = 0.998 | −0.1 [−0.6, +0.2], P = 0.30 | +2.3 [−0.2, +4.9] |

In every fitted decomposition the indirect component is allocated to letter-sound knowledge and essentially none to the rival — including phoneme blending and nonword decoding, both of which the intervention improved. Improving alongside the outcome is not the same as being the route to it.

### Single mediators other than letter sounds

| Model     | Mediator                         | Indirect (words)            | Direct                      | Total             |
| --------- | -------------------------------- | --------------------------- | --------------------------- | ----------------- |
| `med-068` | Taught expressive vocabulary     | +0.3 [−0.3, +1.4], P = 0.77 | +2.0 [−0.1, +4.1], P = 0.94 | +2.4 [+0.2, +4.6] |
| `med-080` | Taught receptive vocabulary      | +0.5 [−0.3, +2.0], P = 0.85 | +1.9 [−0.2, +4.0], P = 0.93 | +2.6 [+0.2, +5.1] |
| `med-074` | Nonword decoding (floored)       | +0.1 [−0.3, +0.6], P = 0.65 | +2.8 [+0.8, +4.7], P = 0.99 | +2.9 [+0.8, +4.9] |
| `med-079` | Receptive grammar (neg. control) | +0.1 [−0.2, +0.7], P = 0.70 | +2.1 [+0.3, +3.9], P = 0.97 | +2.2 [+0.4, +4.0] |
| `med-062` | Code-route composite (L + B)     | +0.8 [−0.1, +2.2], P = 0.92 | +1.1 [−1.1, +3.3], P = 0.78 | +2.0 [−0.2, +4.1] |

The taught-vocabulary routes carry little of the allocation (proportion mediated about 0.1–0.2), and the nonword route almost none — partly mechanical, since the intervention barely moved most children off the nonword floor. **The negative control** through grammar, a mediator the causal diagram gives no reading route, allocates +0.1 words (89% −0.2 to +0.7). That is not evidence of a zero grammar route — P(>0) = 0.70 is below even suggestive, and no negligible-effect threshold was tested — but it did not reveal a large spurious channel, which is the limited reassurance a negative control can give.

### Temporal ordering and the reverse direction

`med-076` uses letter sounds at timepoint 2 and word reading at timepoint 4: the indirect component through letter sounds is +3.6 words (89% +1.7 to +6.0) while the total is +1.7 (−1.6 to +5.1) and the direct component −2.0 (−5.0 to +1.1). The temporal-ordering sensitivity attached to `med-059` (reading at timepoint 3) gives the same shape: indirect +3.6 (+1.8 to +5.8), direct −1.4, total +2.3 (−0.8 to +5.3). Ordering the mediator before the outcome preserves and if anything enlarges the letter-sound allocation, at the cost of a direct component that turns negative and a total that no longer clears zero; both later waves are post-crossover, so these are ordering checks rather than randomised mediation estimands.

The **reverse** models are new to this series' reading. `med-176` runs word reading at timepoint 2 as the mediator of the intervention's effect on letter sounds at timepoint 4: indirect +0.5 items (89% +0.0 to +1.2, P = 0.95), direct +1.3, total +1.8. `med-276` (letter sounds at timepoint 3) gives indirect +0.6 (+0.0 to +1.6, P = 0.96), total +0.9. A modest but resolved reverse allocation exists: the model is as willing to route a later letter-sound gain through earlier reading as it is to route reading through letter sounds, at a smaller size. That is exactly what a shared reading-development component would produce, and it is a reason not to read the forward allocation as a one-way channel.

### Decoding and blending as outcomes

`med-086` decomposes the nonword off-floor risk difference: total +10.7 percentage points (89% +0.6 to +21.1), indirect via letter sounds +10.0 (+3.8 to +17.9, P = 0.996), direct +0.4. `med-087`/`387` decompose phoneme blending under the two links: total +0.7 items (−0.2 to +1.5) or +0.5 (−0.2 to +1.1), indirect +0.3 either way (P = 0.82 and 0.88). The blending pair is released together; the letter-sound share of the blending gain is suggestive at best.

### The dose-confounding calibration and the stacked companion

At its point calibration, session-related confounding of the mediator–outcome leg maps to an indirect effect of +2.0 words (89% +0.8 to +3.7), close to the primary +2.1. The tipping point at which the fitted indirect effect's interval first includes zero is only 52.5% of the fitted mediator–outcome slope, and the broad envelope of separate 89% slope endpoints reaches it. The stored verdict is that intervention-session confounding **could plausibly account** for the estimated indirect effect; the decomposition is not robust to it.

`med-092` stacks all periods and now headlines the period-1 window (the only one with untreated children: 28 treated, 25 untreated): total +2.6 words (89% +1.0 to +4.3), indirect +0.6 (+0.2 to +1.2), direct +2.0 (+0.3 to +3.5), proportion mediated 0.25. Its all-period average (total +3.2, indirect +0.8) is written separately as an explicit extrapolation. **This is the one that disagrees** with the randomised-window fits, putting most of the effect in the direct path. It conditions on the period-start letter-sound score in every leg, which the single-window fits do not, so it answers a different question; it is reported for completeness and is not the closer match to the treatment-effect window.

## What changed since the August notes

The August series was written before the #585 leg-contract fix. Six headline totals (`med-059`/`078`, `064`, `080`, `086`/`186`) now exclude zero where they crossed it before, driven by larger letter-sound indirect components (+2.1 against +1.8 words for `med-059`); the code-route composite's indirect component weakened to include zero; `med-092` re-scoped its headline to period 1. The direct components barely moved. The 2026-09-01 rebuild reproduces those 2026-08-26 values exactly.

## What these models cannot tell you

**They cannot identify letter sounds as the mechanism.** The natural effects are non-identified because dose is treatment-induced, latent ability can confound mediator and outcome, and the primary mediator and outcome are contemporaneous; the reverse-direction fits show the machinery will allocate in either direction. **The proportion mediated should not be quoted as a percentage with confidence.** **"Direct effect" means "not through this mediator"**, not "unexplained". **Nothing here licenses a teaching claim** that more letter-sound teaching would raise reading further.

## Model inventory

All 20 pass the convergence gate with zero divergences and are publishable. Single-mediator: `med-059` (via L), `062` (code route), `068` (TE), `074` (N), `076` (L at t2 → W at t4), `078` (interventional relabel of 059), `079` (negative control, T), `080` (TR), `086` (N off-floor via L), `087`/`387` (B via L, ordinary and guessing-floor links), `092` (period-stacked), `176`/`276` (reverse: W at t2 → L at t4 / t3), `186`/`187` (interventional relabels of 086/087). Two-mediator (`mediation_multi`): `med-060` (L → N → W), `064` (L vs E), `066` (L vs B), `075` (L → B → W). `med-092` was one of eight fits blocked by undeclared inline priors until PR #650 and was fitted at the batch's second commit.
