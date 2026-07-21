# Joint (multivariate ITT) findings (2026-07-21)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is one of a set of per-family notes from the full 2026-07-21 re-fit of every Bayesian statistical model in the study (production `reporting` configuration, 6 chains × 6000 draws, 89% credible intervals). For the study background, the outcome measures and their maxima, and — importantly — how to read Bayesian numbers (medians, credible intervals, tail probabilities, the evidence ladder, and the causal-versus-association distinction), see the index and reading guide, `notes/202607210900-findings-00-index-and-reading-guide.md`. The **joint** family has **4** models, and **all 4 pass the convergence gate**.

## What this family probes

The single-outcome [ITT models](202607210901-findings-itt.md) fit one skill at a time. The joint family instead fits several of those outcomes **inside one multivariate model** — optionally with an LKJ prior on the residual correlation across outcomes, so the model can describe how the skills co-vary after adjustment and borrow a little strength across them. The pay-off is a single, internally consistent picture across outcomes, and the ability to form principled **contrasts between outcomes** (for example: did the programme help _taught_ vocabulary more than _not-taught_ vocabulary?).

The four models split into one broad joint fit and three focused two-outcome contrasts:

- **LRP-RLI-ITT-012** — the main joint graph over the ten core suite outcomes together: taught receptive/expressive vocabulary (TR, TE), not-taught receptive/expressive vocabulary (UR, UE), broad receptive and expressive vocabulary (RV, EV), letter-sound knowledge (LS), phoneme blending (PA), phonetic spelling (PS) and word reading (WR).
- **LRP-RLI-ITT-015** — the _expressive_ generalisation contrast: taught versus not-taught expressive vocabulary, block 1.
- **LRP-RLI-ITT-016** — the _modality_ contrast: taught expressive (TE) versus taught receptive (TR) vocabulary, block 1.
- **LRP-RLI-ITT-115** — the _receptive_ generalisation contrast: taught versus not-taught receptive vocabulary, block 1 (a companion fit, parent `lrpitt115`).

The adjustment set is DAG-pre-specified and deliberately minimal. Each outcome carries its **own baseline score and a linear age term as precision covariates** — they sharpen the estimate but are not causal effects — and there are **no cross-baselines and no skill-to-skill couplings** in this family. That is by design: the intention-to-treat effect is identified by random assignment with an _empty_ confounder adjustment set, so nothing further needs to be (or should be) conditioned on. Socio-economic status is excluded (it is not a node on the diagram and is statistically redundant here). About 53–54 children contribute to each fit, so the samples are small and the estimates correspondingly uncertain.

## How to read these numbers

Effects are modelled on a proportion-correct scale and then translated back into **items** — how many more items, out of that test's maximum, a treated child gets right on average — because "+3 of 32 letter sounds" is far easier to grasp than a logit coefficient. Read each outcome's effect as a **median** best estimate with an **89% equal-tailed credible interval** (the range that holds the true value with 89% probability given the data and priors). **Direction** comes from the **tail probability** P(effect > 0), graded on the fixed evidence ladder — inconclusive (< 0.75), suggestive (≥ 0.75), moderate (≥ 0.91), strong (≥ 0.97), very strong (≥ 0.99) — never from whether an interval excludes zero and never as a p-value. **Direction is not size:** a high tail probability says an effect is _probably positive_, not _large_. The separate size claim is the probability the benefit clears a pre-set **smallest-important difference** (a "big enough to matter" bar, agreed post hoc and outcome-specific).

On causal status: because the outcomes here are estimated under random assignment, **each outcome's treatment effect (τ_k) is causal** — it has exactly the same identification as the single-outcome ITT. The taught-minus-not-taught (generalisation) and taught-expressive-minus-taught-receptive (modality) comparisons are **derived contrasts built from two causal randomised effects**, and so also support a cause-and-effect reading. Residual confounding by latent general ability is not an issue for the τ_k themselves (randomisation handles it), but as always applies to any purely descriptive between-outcome ordering.

## Per-model findings

### LRP-RLI-ITT-012 — the ten-outcome suite, fitted jointly (causal per outcome)

Every outcome's effect here is a **causal** intention-to-treat estimate. Across the ten outcomes, the best (median) estimates ranged from **−0.1 to +3.5 items**, and the individual 89% credible ranges extended from **−3.4 to +5.4 items** overall — that is, the clearest outcomes sit well clear of zero while the most uncertain ones straddle it by several items in each direction. The clearest single directional result is **letter-sound knowledge (YARC-LSK, LS): a 99.9% probability that the true effect is positive — very strong evidence that the intervention helps.** On the size question, of the nine outcomes that have a smallest-important-difference bar (phonetic spelling, PS, has none), **four were more likely than not to reach it**, with the outcome-specific "big enough to matter" probabilities ranging from **9% to 90%**. Gate: **PASS**.

This reproduces the single-outcome ITT story from inside one model: the effect is largest and clearest on the code-related skill (letter sounds), the overall spread of medians tops out around +3.5 items, and a minority of outcomes are confident enough to clear a practical bar while the rest are genuinely uncertain (wide intervals spanning zero). The joint and one-at-a-time analyses therefore agree — a reassuring consistency check rather than two competing answers.

### The three two-outcome contrasts (each marginal causal; the contrast a comparison of two randomised effects)

These are separate two-outcome fits. Each reports both outcomes' causal treatment effects (their marginals) and targets the contrast between them. The surfaced key findings quote the two marginals as items ranges and name the clearest directional result; read the contrast itself qualitatively from which side's marginal is larger and clears its bar.

| Fit         | What it contrasts                                                   | Marginals: best est. range (items) | Marginals: 89% range (items) | Clearest directional result                             | Cleared the "matters" bar      |
| ----------- | ------------------------------------------------------------------- | ---------------------------------- | ---------------------------- | ------------------------------------------------------- | ------------------------------ |
| **ITT-016** | Modality: taught expressive (TE) vs taught receptive (TR), block 1  | +1.4 to +1.5                       | +0.2 to +2.7                 | Taught expressive (b1extau): 99% positive — **strong**  | 2 of 2 (probabilities 68%–78%) |
| **ITT-015** | Expressive generalisation: taught vs not-taught expressive, block 1 | +0.4 to +1.5                       | −0.4 to +2.7                 | Taught expressive (b1extau): 98% positive — **strong**  | 1 of 2 (probabilities 8%–79%)  |
| **ITT-115** | Receptive generalisation: taught vs not-taught receptive, block 1   | +0.7 to +1.4                       | +0.0 to +2.5                 | Taught receptive (b1retau): 97% positive — **moderate** | 1 of 2 (probabilities 26%–70%) |

All three fits **PASS** the gate. Reading them:

- **Modality (ITT-016).** Taught expressive and taught receptive vocabulary benefit by almost the same amount — the two medians sit at +1.4 to +1.5 items with 89% ranges from +0.2 to +2.7, both pointed away from zero. Taught expressive is the cleaner of the two (99% positive, strong), and **both** outcomes were more likely than not to clear their smallest-important-difference bar (68%–78%). The two taught modalities move together; there is little to separate them.
- **Expressive generalisation (ITT-015).** This is the contrast that leans towards the taught side: the two expressive marginals span +0.4 to +1.5 items, with 89% ranges from −0.4 to +2.7. Taught expressive is 98% positive (strong), and only **one of the two** outcomes was more likely than not to clear its bar, with probabilities running from 8% (the not-taught side) up to 79% (the taught side). So the taught words carry a confident, practically-meaningful benefit while the not-taught words do not — consistent with a teaching effect concentrated on the specific vocabulary taught.
- **Receptive generalisation (ITT-115).** Both receptive marginals are positive (best estimates +0.7 to +1.4 items; 89% ranges from +0.0 to +2.5). Taught receptive is 97% positive (moderate — just onto the "strong" boundary at 0.97 but reported as moderate here). Again **one of the two** outcomes was more likely than not to clear its bar, probabilities from 26% (not-taught) to 70% (taught). The pattern echoes the expressive case, though a touch softer.

## What to take away

Fitting the suite together tells the same story as the single-outcome ITT models, with the reassurance that it comes from one internally consistent model. The intervention's clearest, largest-and-tightest effect is on **letter-sound knowledge** (99.9% positive, very strong), best estimates across the ten outcomes top out around +3.5 items, and roughly four of the nine bar-holding outcomes are more likely than not to clear a practical threshold — while several others (the broad standardised vocabulary measures in particular) remain genuinely uncertain, with intervals spanning several items either side of zero. On the targeted-teaching question, the **taught** vocabulary outcomes carry confident benefits (taught expressive 98%–99% positive, strong; taught receptive 97%, moderate) and clear their practical bars, whereas the **not-taught** counterparts are weaker and generally fall short of theirs. That pattern is consistent with a benefit concentrated on the specific words taught, with limited spill-over to untaught vocabulary. As always in this small study, lead with the intervals rather than the point estimates, and treat all figures as preliminary.
