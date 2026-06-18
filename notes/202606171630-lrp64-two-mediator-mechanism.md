# LRP64 — letter-sounds vs vocabulary: a two-mediator decomposition of word reading

> [!WARNING]
> AI-assisted analysis. Headline numbers are produced by
> `python scripts/fit_statistical_model.py lrp64 --config reporting` (writes
> `mediation_summary.csv` and renders `docs/models/lrp64`); the interpretation should
> be reviewed by the study team. Language is deliberately **associational / under
> stated assumptions** — this is a decomposition under assumptions, not proof of a
> causal route.

Date: 2026-06-17

## Context

Meeting action item: do **vocabulary** and **letter-sounds** act as *independent*
paths to word reading, or does vocabulary act partly *through* letter-sounds? The
single-mediator LRP59 (letter-sound only, ~62%) and the route-composite LRP62
(letter-sound + blending, ~38%) each reduce the question to one pathway and cannot
separate the two. LRP64 fits both mediators together and decomposes the word-reading
effect into a path via letter-sound knowledge ($L$), a path via expressive vocabulary
($E$), and a direct/residual path.

## Method

- **Phase 0 only** (`phase_mode="itt"`, t1 → t2): the single randomised contrast,
  $n \approx 53$. One row per child.
- **Two count mediators**, $L$ and $E$, each a Beta-Binomial leg conditioned on its own
  baseline; the **outcome** leg adds both standardised post-mediators and the $G\times
  L$, $G\times E$ interactions. Built by `factories.build_two_mediator_model`;
  decomposed by `mediation.decompose_two_mediator` (the g-formula extended to a
  two-mediator block).
- **Adjustment {G, A, R, W_pre, L_t1, E_t1}**: receptive vocabulary $R$ is the
  remaining mediator–outcome confounder (expressive vocab is now a *mediator*), taken
  at baseline (cross-world assumption).
- **Framing.** The **joint** indirect effect through the $\{L, E\}$ block is the robust
  headline; the path-specific $\text{NIE}_L$ / $\text{NIE}_E$ split (sums to the joint
  effect) is **exploratory**, under a stated mediator ordering ($L$ before $E$) and a
  conditional-independence assumption between mediators.

## Results (preliminary, dev config — reporting config is the source of record)

A preliminary fit (intervention-helps; words out of 90) gave, illustratively:

| Quantity | mean (words/90) | P(>0) |
| --- | --- | --- |
| **Total** | ~+3.0 | ~0.98 |
| NDE (direct / residual) | ~+1.1 | ~0.85 |
| **NIE_joint** (via L + E block) | ~+1.9 | ~0.99 |
| NIE_L (exploratory, L-first split) | ~+2.0 | ~1.00 |
| NIE_E (exploratory, given L) | ~0.0 | ~0.45 |

These reconcile with LRP59 (Total +2.92, NDE +1.09, NIE +1.83) and LRP62 (Total +2.72)
in sign and rough magnitude. Re-run with `--config reporting` for the committed
numbers and intervals; `mediation_summary.csv` is the source of record.

## Interpretation

- **The block carries a credible indirect effect**, and the **direct/residual path is
  not credible on its own** — consistent with LRP59/LRP62.
- **Under the (exploratory) split, the indirect effect runs through letter-sounds, not
  through expressive vocabulary *given letter-sounds*.** $\text{NIE}_E \approx 0$ means
  expressive vocabulary adds little *additional* reading-specific indirect path once
  letter-sounds are in the block — **not** that vocabulary is unimportant for reading
  development overall (vocabulary plausibly acts partly *through* letter-sounds, and is
  itself raised by the intervention; LRP54). This is the two-mediator answer to the
  meeting's question: the routes are not cleanly independent — the reading-specific
  indirect path is letter-sound-led.

## Caveats (the binding ones)

- $n \approx 53$: intervals are **wide**, the path split especially so. Lead with the
  joint effect.
- **No unmeasured mediator–outcome confounding** — the binding, unverifiable
  assumption.
- **Contemporaneous measurement** (mediators and outcome both at t2): no temporal
  precedence; baselines mitigate but do not remove this.
- **Sight-reading**: $W$ does not reveal *how* a child read, so a letter-sound indirect
  path is a statistical decomposition, not evidence that decoding produced the words.
- The path split is **ordering-dependent** and assumes conditionally independent
  mediators (no residual $L$–$E$ correlation modelled).

## Reproduce

```
python scripts/fit_statistical_model.py lrp64 --config reporting --render
# -> output/statistical_models/.../lrp64-reporting/mediation_summary.csv
# -> docs/models/lrp64/index.qmd  (rendered report)
```

LRP64 reconciles the earlier single-mediator analyses: its NDE/NIE split is consistent in
sign and magnitude with LRP59 (~62% mediated) and LRP62 (~38%), now expressed as a joint
indirect effect through the letter-sound + expressive-vocabulary block rather than a single ratio.
