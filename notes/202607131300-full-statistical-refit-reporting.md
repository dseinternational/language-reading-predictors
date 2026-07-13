<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Full statistical-model refit (reporting config) — run record and results

Date: 2026-07-13

Related: `METHODS.md` (interpret/reporting standard), `docs/models/README.md` (the model catalogue), #179 (evidence-ladder & ROPE policy), #247 (revised-DAG factor adjustments), #265 (HSGP mechanism reparameterisation), the methodology review at `notes/202607130922-statistical-models-methodology-review.md`.

## What was run

A complete refit of **every registered Bayesian statistical model (115 models across 16 families)** at the `reporting` sampling preset (6 chains × 6000 draws × 6000 tune, `target_accept = 0.95`), each with its Quarto report rendered, followed by the cross-model comparison and publication of all artefacts to the public research site.

```bash
python scripts/fit_statistical_model.py all --config reporting --render
python scripts/compare_statistical_models.py --config reporting
```

- **Data:** `data/rli_data_long.csv`, N = 53 children, 4 timepoints (waitlist-crossover RCT).
- **Fit sweep started 11:34 BST.** The first `all` run was terminated externally at ~12:39 after completing 112 of 115 models (it died mid-sampling on the slow `corr_factor` latent-factor model). No fits were lost — the 112 completed dirs were intact, so the run was **resumed for only the 3 outstanding models** (`lrp-rli-mm-001`, `lrp-rli-mm-101`, `lrp-rlm-hg-001`) rather than re-fitting from scratch. All 3 completed cleanly (exit 0) by 12:58.
- Every model wrote `trace.nc`, `diagnostics_summary.json`, `config.json`, `priors_table.csv`, the family result CSVs, diagnostic plots and a rendered `index.html`.

## How to read these results (a short primer)

This section is for readers who don't work with Bayesian models every day. Skip it if the vocabulary is already familiar.

- **The study design.** RLI (Reading and Language Intervention) is a small **randomised controlled trial** run as a **waitlist crossover**: 53 children were randomly assigned either to start the intervention immediately or to wait and start later, and everyone was assessed at four timepoints. Random assignment is what lets us make a fair, unbiased comparison — the two groups differ only by chance at the start, so a later difference is attributable to the intervention rather than to who happened to be in each group.
- **Causal vs. associational.** Only comparisons that ride on the random assignment support a _causal_ claim ("the intervention raised this skill"). In this collection that means the intervention-group effect **`τ`** (ITT models), the within-person **`δ`** (difference-in-differences), and the on-intervention marginal effect (gain-factor models). Every other quantity — how one skill tracks another, dose–response, mediation paths — is an **adjusted association**: a correlation that could equally be driven by a child's general ability. We never read those as "X drives Y".
- **What we report, and why not p-values.** For each effect we give the posterior **median** (best single estimate), a **95 % credible interval** (the range the true value most plausibly occupies), and the **probability the effect is positive** — e.g. `P(τ > 0) = 0.99` means a 99 % posterior probability the intervention helped. This is a direct probability statement, which is why there are no p-values.
- **The evidence ladder.** We turn that probability into a fixed, plain-language label: **suggestive** (≥ 0.75), **moderate** (≥ 0.91), **strong** (≥ 0.97), **very strong** (≥ 0.99); below 0.75 we call it **inconclusive**. The label describes _how sure we are of the direction_, not how big the effect is — a small effect can be "very strong" evidence and a large effect "inconclusive".
- **"Inconclusive" is not "no effect".** A flat result with a tight interval hugging zero (e.g. standardised vocabulary) is reported as inconclusive-and-probably-negligible, quantified by how much of the posterior sits inside a **region of practical equivalence** (ROPE) around zero — never as "proven null".
- **Two scales.** The models work on the **logit** (log-odds) scale, but we translate effects to the **items / probability** scale because it is far more intuitive: "≈ +3 of 32 letter sounds" is easier to picture than "+0.58 logits".
- **Floor effects.** A few tests (phonetic spelling `P`, nonword reading `N`) had most children scoring zero at the start, so a graded score is uninformative. For these the pre-specified primary question is binary — _did the child come off zero at all?_ — reported as an off-floor risk difference; the graded effect is a flagged secondary.

### Outcome symbols (same letters as the DAG)

| Symbol      | Measure                                      | Notes                                   |
| ----------- | -------------------------------------------- | --------------------------------------- |
| `W`         | Word reading (EWRSWR)                        | primary reading outcome                 |
| `L`         | Letter-sound knowledge (YARC-LSK)            | direct teaching target                  |
| `B`         | Phoneme blending                             | direct teaching target                  |
| `P`         | Phonetic spelling (SPPHON)                   | heavily floored                         |
| `N`         | Nonword reading                              | floored, post-only                      |
| `R`         | Receptive vocabulary (ROWPVT)                | standardised (transfer) measure         |
| `E`         | Expressive vocabulary (EOWPVT)               | standardised (transfer) measure         |
| `F`         | Basic concept knowledge (CELF)               |                                         |
| `T`         | Receptive grammar (TROG-2)                   |                                         |
| `TR` / `TE` | Taught receptive / expressive vocabulary     | the specific words RLI taught (block 1) |
| `UR` / `UE` | Not-taught receptive / expressive vocabulary | generalisation comparators              |

## Convergence gate: 109 / 115 PASS

Before interpreting any model we check it against a convergence gate — diagnostics that tell us the sampler explored the posterior properly. Thresholds: r̂ ≤ 1.01 (chains agree), ESS ≥ 400 (enough effective samples), BFMI ≥ 0.30 (energy well-behaved), and **zero** divergences (no pathological steps). The divergence check fails on any single divergence, so a model can be flagged while r̂ = 1.00 and ESS is in the tens of thousands.

The 6 flagged models are all expected and were triaged, not silently accepted:

| Model              | Failed check(s)     | Divergences  | max r̂  | min ESS | Reading                                                                                                                                                                         |
| ------------------ | ------------------- | ------------ | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-did-007`  | divergences         | 34 (0.094 %) | 1.0015 | 5558    | Divergence-only, well under the METHODS ≤ 1 % guidance — **usable**. Period-resolved letter-sound dose geometry.                                                                |
| `lrp-rli-dose-077` | divergences         | 17 (0.047 %) | 1.001  | 4088    | Divergence-only, **usable**. Cumulative-session dose slope.                                                                                                                     |
| `lrp-rli-dose-177` | divergences         | 13 (0.036 %) | 1.001  | 3792    | Divergence-only, **usable**. Dose slope.                                                                                                                                        |
| `lrp-rli-hs-001`   | divergences         | 2 (0.006 %)  | 1.001  | 9352    | Divergence-only, **usable**. Horseshoe funnel.                                                                                                                                  |
| `lrp-rli-mm-001`   | r̂, ESS, divergences | 1 (0.003 %)  | 1.0195 | 354     | corr_factor latent-factor funnel. Mild r̂/ESS miss — **domain correlations robust; structural leg read cautiously** (as flagged in the methodology review). BFMI healthy (0.91). |
| `lrp-rli-mm-101`   | r̂, ESS, divergences | 57 (0.158 %) | 1.0213 | 260     | corr_factor variant, same funnel — same caution.                                                                                                                                |

No headline causal estimand sits on a flagged model: the four divergence-only fits clear the ≤ 1 % guidance, and the two corr_factor misses affect only the measurement model's structural leg (its **correlations**, which are the deliverable, converged fine). The comparison script correctly excludes all 6 from its interpretable LOO tables.

## Results

Everything below is read per the `METHODS.md` / #179 standard set out in the primer: posterior **median**, 95 % **credible interval**, tail probability, and the fixed evidence ladder. Only `τ`, the DiD `δ` and the gain-factor on-intervention marginal are **causal**; everything else is an adjusted association. Causal `τ`/`δ` figures are on the **probability (risk-difference) scale** at sample-mean baseline unless noted.

### The headline causal question — ITT effect on each outcome (`lrp-rli-itt-001…011`, `025`, `026`)

The intention-to-treat (ITT) effect `τ` is the study's headline: the randomised effect of being assigned to the intervention on each outcome.

Both a 90 % and a 95 % equal-tailed credible interval are shown: the 90 % is the tighter everyday band, the 95 % the more conservative one; read them together, not either alone.

| Outcome                          | `τ` (risk diff.) | 90 % CI          | 95 % CI          | P(τ>0)    | Evidence         |
| -------------------------------- | ---------------- | ---------------- | ---------------- | --------- | ---------------- |
| **Letter sounds (L)**            | **+0.110**       | [+0.051, +0.168] | [+0.040, +0.179] | **0.999** | **very strong**  |
| **Phoneme blending (B)**         | **+0.099**       | [+0.020, +0.177] | [+0.005, +0.192] | **0.980** | **strong**       |
| **Word reading (W)**             | **+0.030**       | [+0.008, +0.052] | [+0.004, +0.057] | **0.986** | **strong**       |
| **Taught expressive vocab (TE)** | **+0.064**       | [+0.016, +0.113] | [+0.006, +0.122] | **0.985** | **strong**       |
| Taught receptive vocab (TR)      | +0.057           | [+0.007, +0.107] | [−0.003, +0.117] | 0.968     | moderate         |
| Untaught receptive vocab (UR)    | +0.050           | [−0.004, +0.105] | [−0.014, +0.116] | 0.937     | moderate         |
| Basic concepts (F)               | +0.048           | [−0.016, +0.112] | [−0.029, +0.124] | 0.891     | suggestive       |
| Nonword reading (N)              | +0.100           | [−0.042, +0.241] | [−0.071, +0.268] | 0.877     | suggestive       |
| Untaught expressive vocab (UE)   | +0.026           | [−0.031, +0.082] | [−0.041, +0.093] | 0.773     | suggestive       |
| Receptive grammar (T)            | +0.020           | [−0.027, +0.068] | [−0.037, +0.077] | 0.760     | suggestive       |
| Phonetic spelling (P)            | +0.041           | [−0.075, +0.159] | [−0.098, +0.183] | 0.724     | inconclusive     |
| **Receptive vocabulary (R)**     | +0.001           | [−0.023, +0.026] | [−0.027, +0.031] | 0.539     | **inconclusive** |
| **Expressive vocabulary (E)**    | +0.001           | [−0.019, +0.021] | [−0.022, +0.025] | 0.534     | **inconclusive** |

Plainly: the strongest, clearest benefit is on **letter-sound knowledge** — about **+3 to +4 of the 32 letter sounds** (the most concrete framing) — followed by **phoneme blending, word reading and taught expressive vocabulary**. These are the skills the intervention teaches directly, so this is the pattern you would hope to see. **Standardised receptive and expressive vocabulary (R/E) is inconclusive and probably negligible** — near-zero medians with tight bands hugging zero (a ROPE reading, not "no effect"); broad standardised vocabulary is a distant transfer target that a phonics-and-language programme is not expected to shift in a short window. Basic concepts (F) and receptive grammar (T) sit in the suggestive/inconclusive middle.

**The floored outcomes (P, N) read better on their primary estimand than the graded `τ` suggests.** The pre-specified primary question for these is "did the child come off the floor?": **29 % of intervention children came off the spelling floor vs 12 % of controls**, and **48 % vs 13 % for nonword reading** — a clear off-floor shift in the intervention's favour, even though the heavily-floored graded score (shown in the table) is noisy.

### Robustness — the story survives adjustment

The ITT effect is identified by an empty adjustment set, but we re-ran the key outcomes adding potential confounders to check the result does not depend on them. It doesn't.

- **General-ability adjusted** (block-design, `itt-017…024`) reproduces the suite almost exactly: L very strong (0.999), W strong (0.980), TE strong (0.982), TR moderate (0.957), UE suggestive (0.777), R/E inconclusive (0.55 / 0.54).
- **SES-adjusted** (mother's education etc., `itt-013` W 0.967, `itt-113` L 0.993) and the **matched SES complete-case** comparators (`itt-014` W 0.977, `itt-114` L 0.998) — holds.
- **Site-adjusted** (North/South, `itt-027` W 0.992, `itt-028` L 0.998) — holds, if anything slightly sharper.

### A second causal view — within-person replication (DiD, `lrp-rli-did-001…012`, `107`)

The difference-in-differences (DiD) design uses the waitlist crossover so that **each child partly acts as their own control** (comparing a child's untreated stretch to their treated stretch), with the immediate-start arm anchoring the natural maturation trend. It's a different, non-randomised-comparison route to the same question, so agreement with the ITT is reassuring. The causal quantity is the DiD contrast `δ`; the separate period term is just the maturation trend and is not the effect.

| Outcome                                   | DiD `δ` (logit) | 90 % CI          | 95 % CI          | P(δ>0)    | Evidence        | Items reading                 |
| ----------------------------------------- | --------------- | ---------------- | ---------------- | --------- | --------------- | ----------------------------- |
| **Letter sounds (L, did-002)**            | **+0.483**      | [+0.164, +0.805] | [+0.100, +0.872] | **0.993** | **very strong** | ≈ +2.8 letter sounds          |
| **Taught expressive vocab (TE, did-004)** | **+0.358**      | [+0.078, +0.635] | [+0.019, +0.691] | **0.981** | **strong**      | ≈ +1.7 words                  |
| Phoneme blending (B, did-003)             | +0.343          | [−0.030, +0.714] | [−0.099, +0.786] | 0.934     | moderate        | ≈ +0.7 items                  |
| Taught receptive vocab (TR, did-008)      | +0.193          | [−0.070, +0.453] | [−0.123, +0.504] | 0.889     | suggestive      | ≈ +1.0 word                   |
| Basic concepts (F, did-010)               | +0.182          | [−0.105, +0.468] | [−0.161, +0.521] | 0.854     | suggestive      | ≈ +0.7 items                  |
| Nonword reading (N, did-012, off-floor)   | +0.302          | [−0.312, +0.927] | [−0.432, +1.046] | 0.791     | suggestive      | off-floor                     |
| Word reading (W, did-001)                 | +0.168          | [−0.116, +0.457] | [−0.176, +0.515] | 0.831     | suggestive      | ≈ +1.3 words                  |
| Expressive vocabulary (E, did-009)        | +0.037          | [−0.116, +0.193] | [−0.147, +0.224] | 0.655     | inconclusive    | —                             |
| Phonetic spelling (P, did-011, off-floor) | +0.024          | [−0.627, +0.679] | [−0.756, +0.803] | 0.524     | inconclusive    | off-floor                     |
| Receptive vocabulary (R, did-005)         | −0.024          | [−0.173, +0.125] | [−0.201, +0.155] | 0.398     | inconclusive    | — (the intended null control) |

The within-person picture matches the ITT's ranking: **letter sounds is the clearest replication (very strong), then taught expressive vocabulary (strong) and phoneme blending (moderate)**, with word reading suggestive and standardised vocabulary (R/E) inconclusive — including `did-005` receptive vocabulary behaving exactly as the pre-specified **null control** should. The word-reading DiD is weaker than its ITT sibling (suggestive vs strong): the within-person contrast has less information than the full randomised comparison, so this is a difference in power, not in direction. The observational **session dose-response** variants are positive (`did-006` word-reading period slope P = 0.992; `did-107` pooled letter-sound dose P = 0.917) but, because attendance is a downstream collider, these are sensitivity views, not causal.

### A third causal view — ANCOVA on each period's gain (gain factors, `lrp-rli-gf-001…011`)

The gain-factor models are a DAG-focused ANCOVA: they model each period's post-score given its own pre-score, and the **on-intervention term averaged over the randomised first transition is the only causal coefficient** (every covariate in these models is an adjusted association). This is the third independent route to the causal effect, and it agrees again:

- **Word reading `W`: P = 0.993, ≈ +2.6 words** — very strong.
- **Letter sounds `L`: P = 0.991, ≈ +3.3 letter sounds** — very strong.
- Phoneme blending `B`: P = 0.903 (≈ +0.8 items) — moderate.
- Basic concepts `F`: 0.865; taught expressive `TE`: 0.818 — suggestive.
- Taught receptive `TR` 0.633, grammar `T` 0.662, nonword `N` 0.621 — inconclusive.
- Standardised vocabulary `R` (0.169) and `E` (0.577) and floored spelling `P` (0.380) — negligible / negative.

Three designs (ITT, DiD, gain-factor ANCOVA), three different sets of assumptions, one story: **the intervention reliably raises the directly-taught code skills (letter sounds, blending) and word reading, and does not shift broad standardised vocabulary in this window.** The treated-only `…b` variants (`gf-101…108`) are companion fits and are not separately interpreted here.

### Generalisation — did benefits reach untaught words? (`lrp-rli-itt-012`, `015`, `016`, `115`)

The joint model (`itt-012`) fits all suite outcomes together and lets us compare effects head-to-head. Its contrast matrix confirms the ranking above: the letter-sound effect is larger than the receptive-vocabulary effect with probability 0.995 and larger than expressive vocabulary with probability 0.997. The dedicated generalisation contrasts ask whether the taught vocabulary gain spilled over to _untaught_ words: taught-vs-untaught expressive vocabulary favours the taught set with probability 0.79 (`itt-015`) and taught-vs-untaught receptive with probability 0.58 (`itt-115`) — i.e. the effect is concentrated on the words actually taught, with only weak evidence of spill-over, as expected for a vocabulary component early in the programme.

### The levels view (`lrp-rli-lf-001…011`)

The level-factor models look at the _score at each timepoint_ rather than the change, with a group×time contrast per timepoint. Only the **first post-baseline timepoint (t2)** contrast is a clean randomised comparison (later timepoints are contaminated by the waitlist arm crossing over). At t2, letter sounds again shows the clearest group difference (`lf-004` P = 0.956, moderate), with word reading (`lf-001` 0.815) and phoneme blending (`lf-006` 0.798) suggestive, and standardised vocabulary flat-to-negative — the same direction as the ITT but weaker, because a single timepoint carries less information than the full model. This view is included for completeness and triangulation, not as a primary estimand.

### Onset-aligned per-protocol view (`lrp-rli-al-001…008`, `101`)

These models align both arms by when they actually started the intervention and fit a single 40-week gain per child. Because the cohort contrast is **not randomised** (it is confounded by age-at-onset and timing), _no_ term here is causal — every coefficient is an association, reported to triangulate. The associations echo the causal families in direction: letter sounds (`al-004` P = 0.961) and word reading (`al-001` 0.906) positive, receptive vocabulary weakly positive (0.832), and expressive vocabulary / grammar / basic concepts flat-to-negative. The dose sensitivity variant (`al-101`) is essentially identical to `al-001`.

### How the skills relate — mechanism models (`lrp-rli-mech-056…058`, `071…073`, `158`, `172`, `173`)

Mechanism models estimate the dose-response of one measured skill on another (e.g. does higher letter-sound knowledge go with higher word reading?). **Every slope here is an adjusted association, latent-ability-confounded — never a causal "this skill drives that one".** Across the phases:

- The **letter-sound → word-reading** coupling is the strongest and clearly positive (`mech-058`: the fitted curve rises monotonically across the observed range; robust in the complete-case comparator `mech-158`).
- The **letter-sound → nonword-decoding** coupling is very strongly positive (`mech-072`), and blending adds to it (moderator P = 0.985); the interaction is mildly _sub-additive_ (P = 0.013), i.e. the letter-sound slope is a little shallower for children who are already strong blenders.
- Vocabulary → word-reading couplings (`mech-056` R, `mech-057` E) are weaker and positive.
- There is **no credible moderation** of the letter-sound → word-reading slope by expressive vocabulary (`mech-071` interaction P = 0.11) or by age (`mech-073` interaction P = 0.07).

### The route of the reading gain — mediation (`lrp-rli-med-059`, `062`, `064`, `066`, `068`, `074…080`)

Mediation asks _how_ the intervention's word-reading gain is produced — how much runs through a given intermediate skill. These are g-formula decompositions into a natural direct effect (NDE, not through the mediator) and a natural indirect effect (NIE, through it). They are not point-identified under the DAG (general ability confounds the same-wave mediator and outcome), so we read them as triangulation, leading with the most robust quantity. The consistent finding: **the word-reading gain runs through letter-sound knowledge, not through vocabulary.**

- `med-059` (single mediator L): **NIE via L P = 0.997, ≈ +1.7 words**; direct effect ≈ 0 (P = 0.56); proportion mediated ≈ 0.82.
- `med-064` (two mediators L + expressive vocabulary E): **NIE via L P = 0.996**; NIE via E ≈ 0 (P = 0.58) — the route is L, not E.
- `med-066` / `med-075` (L + phoneme blending B, and the sequential L→B→reading route): NIE via L P = 0.998; NIE via B ≈ 0 / slightly negative — the mediation loads on letter sounds specifically.
- `med-076` (longitudinal ordering, L measured at t2 carrying the effect on reading at t4): **NIE P = 0.999, ≈ +3.1 words** — the effect survives putting the mediator strictly before the outcome in time.
- `med-068` / `med-080` (taught expressive / receptive vocabulary as mediators): small positive NIE (P = 0.90 / 0.88) but most of the total effect stays in the direct path — vocabulary is not the main conduit.
- `med-062` (code-based route vs lexical share): the code-compatible route carries the larger indirect share (NIE P = 0.96).
- `med-074` (nonword decoding as mediator): floor-limited and direction-uncertain — the total effect stays almost entirely direct (NIE P = 0.70).
- `med-078` is the interventional-effects re-interpretation of `med-059` (same numbers, IDE/IIE labelling).
- `med-079` (**negative-control mediator, grammar `T`** — a path the DAG says should be severed): NIE ≈ 0 (P = 0.71). The control behaves as it should, i.e. there is no spurious route, which calibrates how much residual ability-confounding to worry about (little).

### Associational predictor and dynamics views (`lrp-rli-adj-065`, `lrp-rli-lcsm-067`, `lrp-rli-dose-077`, `177`, `277`)

Three explicitly associational views of word-reading progress, outside the randomised families:

- **Between-child baseline predictors** (`adj-065`): with all baseline skills mutually adjusted, none is a strong independent predictor of subsequent word-reading gain. Baseline letter sounds lean positive (P = 0.89) and hearing status is positive (P = 0.98), while **younger age predicts more gain** (P = 0.003, i.e. a clear negative age association) — younger children had more room to grow. All associations, not levers.
- **Within-child latent change** (`lcsm-067`): a child's prior-wave letter sounds (P = 0.993) and prior-wave expressive vocabulary (P = 0.992) both go with subsequent reading _change_; the negative reading self-feedback term is ordinary regression-to-the-mean.
- **Observational dose-response** (`dose-077`/`177`/`277`): more intervention sessions go with more word-reading gain, clearest in the randomised first period (`dose-077` period-1 slope P = 0.998; pooled `dose-277` P = 0.998). But sessions attended is a **collider** as an exposure, so this is a sensitivity view — never "more sessions cause more gain". (`dose-077`/`177` are the two divergence-flagged, still-usable fits.)

### Measurement structure — correlated domain factors (`lrp-rli-mm-001`, `101`)

This measurement model asks how the three skill domains hang together. The **domain correlations are strong and robust**: vocabulary↔code 0.72, vocabulary↔grammar 0.79, code↔grammar 0.65, each with P(> 0) ≈ 1.0, and stable under the prior-sensitivity variant (`mm-101`). The three domains are highly intercorrelated — consistent with a shared general-ability influence. The model's structural leg (predicting reading gain from the domain factors) had the mild convergence miss noted in the gate and its coefficients (all inconclusive) are held cautiously; the correlations, which are the deliverable, are fine.

### Growth curves — does baseline ability shape trajectories? (`lrp-rli-gc-069`, `070`)

These descriptive, natural-history models fit each child's trajectory across the four waves and ask whether **baseline non-verbal ability** predicts the _shape_ (growth rate, `gamma`) of the verbal/reading trajectories. With only ~54 children the intervals are wide and the answer is mostly **inconclusive**: for word reading, letter sounds, and vocabulary, baseline non-verbal ability does not clearly predict growth rate; the one exception is a modest positive association with **receptive-grammar** growth (`gc-069` P = 0.977). Adding a shared growth-tempo factor (`gc-070`) does not change the picture. All `gamma`/`delta` here are ability-confounded associations, never causal.

### Predictor-ranking cross-check — regularised horseshoe (`lrp-rli-hs-001`, `002`)

The horseshoe is a sparsity-inducing model that ranks predictors, used as a cross-check on the gradient-boosting discovery layer. For the word-reading **level** (`hs-002`) the top selected predictors are **letter sounds `L` (selection p = 0.995) and expressive vocabulary `E` (p = 0.992)**, then receptive grammar `T` (p = 0.886) — corroborating both the gradient-boosting ranking and the mediation finding that letter sounds is central. For the word-reading **gain** (`hs-001`, divergence-flagged but usable) no predictor is strongly selected — gain scores are near-noise after conditioning on baseline, exactly as the discovery layer warned.

### Historical-cohort reproduction (`lrp-rlm-hg-001`)

A separate study, included in the sweep: a reproduction of BAS word-reading growth over three waves from the Byrne et al. historical cohort, comparing children with Down syndrome to typically-developing average readers and to reading-level-matched children. Children with Down syndrome do make reliable word-reading progress (wave 1→3 ≈ +9.7 items, P > 0.999) but far less than average readers (≈ +25.7) and reading-matched peers (≈ +23.7); the Down-syndrome-vs-average gap is ≈ −16 items (P > 0.999). This is descriptive historical context, not part of the RLI trial.

## Cross-model comparison artefacts

`output/statistical_models/comparison/`: `itt_vs_joint_tau.csv`, `tau_forest.png`, `mediation_family.csv` + `mediation_family_forest.png`, and nested PSIS-LOO tables (mechanism / phonics-route / age-moderation / dose / did-dose). **Caveat:** the mechanism τ forest PNG was skipped this run — `lrp-rli-mech-058` had a reconstructed-size mismatch (157 vs 156 obs; confounder-only missingness the keep-mask does not model) and was dropped from that plot and its CSV. All other comparison artefacts wrote normally. Worth a follow-up to teach the mechanism-forest reconstruction about confounder-only missingness so mech-058 rejoins the plot.

## Publication

All 115 model output dirs + the comparison dir published to the **public research site** (anonymously readable), traces excluded — 116 directories, 0 upload failures. Public access verified (`200 text/html`).

- **Publish run id:** `019f5b61-caf8-70f0-9952-11d9121f30b3`
- **Report root:** `https://dseresearch.blob.core.windows.net/public/projects/language-reading-predictors/output/019f5b61-caf8-70f0-9952-11d9121f30b3/`
- **Per-model report:** `<root>/<model-id>-reporting/index.html` — e.g. word reading ITT: `https://dseresearch.blob.core.windows.net/public/projects/language-reading-predictors/output/019f5b61-caf8-70f0-9952-11d9121f30b3/lrp-rli-itt-010-reporting/index.html`
- **Comparison artefacts:** `<root>/comparison/` (CSVs + forest PNGs; no landing HTML).

Uploaded via the `AzureCliCredential` wrapper (`az login` has the write role on `dseresearch`; the VM managed identity does not, so the built-in `--upload` flag would 403 — see `lrp-fit-statistical`). Reports had to be **rendered separately** this run: `fit_statistical_model.py all --render` renders only in a batch after every fit finishes, so the externally-killed first sweep left the 112 completed models un-rendered; they were rendered standalone (`quarto render index.qmd`, 111 OK / 0 fail) before the upload.

This is preliminary work in progress — all data and models are exploratory.
