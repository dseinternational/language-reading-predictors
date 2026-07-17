# Findings — corr_factor & long_corr_factor families (latent measurement models)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Preliminary.

## What these models ask

Individual tests are noisy measures of an underlying ("latent") ability: a child's score on any one task is their true standing plus test-day noise. These models pull the noise out by treating several observed tasks as indicators of one **latent domain**, then ask **how strongly the latent domains correlate** once measurement error is removed. In the RLI models the domains are **vocabulary** (indicators R = receptive and E = expressive vocabulary), **code** (a latent letter-sound / early-decoding factor, indicators L = letter-sound knowledge and B = an early print/blending measure) and **grammar** (indicators F and T). Two of the cross-sectional models also add a **structural leg**: a regression from the latent **code** factor onto **word reading (W)**, adjusting for age (and, in `mm-002`, an errors-in-variables mechanism adjustment set). The separate Byrne historical study (`rlm-mm-001`) uses different domains entirely — reading, language, memory and general ability at wave 3 — and has no structural leg and no word-reading outcome.

`corr_factor` (`mm-*`, `rlm-mm-*`) fits this cross-sectionally (one snapshot); `long_corr_factor` (`lcf-001`) fits it per wave across all four timepoints, which additionally lets it estimate how _stable_ each domain is over time and cross-check the measurement-error correction. Sibling notes: [ITT](202607161800-findings-itt.md) and the gain/level families carry the word-reading _growth_ estimands; these measurement models do not.

A few Bayesian terms used below, in frequentist-friendly language. A **95% credible range** is the interval the parameter lies in with 95% probability _given the data and priors_ — a direct probability statement about the parameter itself, unlike a confidence interval. **P(effect>0)** ("direction probability") is the posterior probability the true value is positive. We grade that direction probability on the project **evidence ladder** (#179): `inconclusive` < 0.75 ≤ `suggestive` < 0.91 ≤ `moderate` < 0.97 ≤ `strong` < 0.99 ≤ `very strong`. These are strengths of _evidence for the sign_, not sizes of effect. Measurement models produce correlations and loadings directly, so there is no ROPE ("region of practical equivalence", a band around zero deemed too small to matter) verdict to report for them.

## Convergence gate — read this before the numbers

Latent-factor models have a known **"funnel" geometry**: when a domain's latent variance can drift near zero, the sampler must explore a narrow neck where tiny changes in the variance imply huge changes in everything scaled by it, and it stalls there. That is exactly what happens here, and it matters _where_ it happens. **All four `corr_factor` / `rlm` models are gate-flagged; only the longitudinal `lcf-001` passed cleanly.**

| Model                        | Divergences | max R̂     | min ESS | What actually failed the gate                                                              |
| ---------------------------- | ----------- | --------- | ------- | ------------------------------------------------------------------------------------------ |
| `mm-002`                     | 0           | **1.048** | 64      | the latent covariances `factor_cov[1–5]` **and the code~grammar correlation itself**       |
| `rlm-mm-001` (Byrne)         | 143 (0.40%) | 1.029     | 64      | latent covariances `factor_cov[2–9]`, three loadings (bpvs/bassim/basnum), one correlation |
| `mm-101` (prior-sensitivity) | 57 (0.16%)  | 1.021     | 260     | latent covariances `factor_cov[1–5]`                                                       |
| `mm-001`                     | 1           | 1.019     | 354     | latent covariances `factor_cov[3–5]`                                                       |
| `lcf-001` (longitudinal)     | 0           | 1.001     | 5157    | **passed** — nothing failed                                                                |

**The crucial correction (this reverses an earlier draft of this note).** In every flagged model, the parameters that fail R̂/ESS are the **latent covariances and correlations themselves** (and a few loadings) — i.e. the very quantities a correlated-factor model exists to report. The **structural `beta_factor` coefficients converged cleanly** — the opposite of what one might assume: `beta_code` (and the other structural betas) have R̂ ≈ 1.001–1.002 and ESS ≈ 7,000–12,000 in `mm-001`/`mm-002`/`mm-101`. So the honest reading is:

- The **cross-sectional latent correlations** (`mm-001`/`mm-002`/`mm-101`/`rlm-mm-001`) are the _flagged_ pieces. Report them as **consistent and indicative but held pending a non-centred reparameterisation or higher `target_accept`**, not as gate-clean.
- The **structural code→word-reading legs** technically _converged_, but we still **hold** them conservatively under the whole-model flag: a model whose covariance block has not mixed cannot be signed off wholesale.
- The **gate-clean anchor for the correlation conclusion is `lcf-001`**, which passed with room to spare (0 divergences, min ESS 5157) and reproduces the same strong-positive pattern per wave. That the flagged cross-sectional models land on the same numbers as the clean longitudinal one is reassuring, but `lcf-001` is what carries the weight.

## Results — all models

Everything below is a **descriptive correlation or an adjusted association**. Nothing is causal (see the final section). "Held" means gate-flagged and not to be signed off from this run.

### Latent domain correlations — cross-sectional RLI models (held, but consistent)

Vocabulary, code and grammar are **strongly and positively correlated** at the latent level in all three RLI cross-sectional fits, and the numbers barely move between them:

| Model    | vocab~code                                  | vocab~grammar                           | code~grammar                                |
| -------- | ------------------------------------------- | --------------------------------------- | ------------------------------------------- |
| `mm-001` | 0.72 (95% 0.44–0.93), P>0 1.00, very strong | 0.79 (0.58–0.94), P>0 1.00, very strong | 0.65 (0.29–0.91), P>0 0.9998, very strong   |
| `mm-002` | 0.74 (0.47–0.93), P>0 1.00, very strong     | 0.78 (0.55–0.93), P>0 1.00, very strong | 0.66 (0.31–0.90), P>0 0.9998, very strong † |
| `mm-101` | 0.71 (0.43–0.92), P>0 0.99997, very strong  | 0.78 (0.57–0.94), P>0 1.00, very strong | 0.64 (0.28–0.90), P>0 0.9996, very strong   |

† In `mm-002` the code~grammar correlation _is itself_ one of the parameters that failed the gate.

`mm-101` is explicitly a **prior-sensitivity replica of `mm-001`** (recalibrated loading and residual priors). It reproduces `mm-001` to within rounding on every correlation and every structural beta, so the substantive conclusion is **prior-robust** — the priors are not doing the work.

### Latent domain correlations — Byrne historical study (`rlm-mm-001`, held)

A separate cohort, separate domains, wave 3 only, **no structural leg and no word-reading outcome**. The correlations are _even higher_ than the RLI models — the latent domains are near-collinear at this age:

| Pair             | Correlation (95% range) | P>0  | Evidence    |
| ---------------- | ----------------------- | ---- | ----------- |
| reading~language | 0.79 (0.67–0.87)        | 1.00 | very strong |
| reading~memory   | 0.79 (0.66–0.89)        | 1.00 | very strong |
| reading~ability  | 0.88 (0.80–0.93)        | 1.00 | very strong |
| language~memory  | 0.85 (0.73–0.94)        | 1.00 | very strong |
| language~ability | **0.92 (0.86–0.98)**    | 1.00 | very strong |
| memory~ability   | 0.90 (0.80–0.97)        | 1.00 | very strong |

Language~~ability (0.92) and memory~~ability (0.90) are so high that latent ability, language and memory are barely separable measurement constructs at wave 3. Loadings are high throughout (reading indicators 0.91–0.94; language bpvs 0.76 / trog 0.93; ability 0.85–0.91), with `basdig` entered as a single-indicator memory measure at a fixed reliability of 0.894.

### Structural code → word-reading legs (converged, but held under the whole-model flag)

These are the pieces most relevant to word reading. **Latent code is a positive adjusted association with word reading (W)**, strongest in the errors-in-variables model:

| Model    | `beta_code` (latent code→W)                   | Age term (`beta_age`)                          | Other notable adjusted associations                                                                                                                                 |
| -------- | --------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mm-002` | **0.34 (95% 0.03–0.65), P>0 0.984, moderate** | −0.35 (−0.56 to −0.13), P<0 0.999, very strong | `beta_hs` 0.23 (0.01–0.44), P>0 0.979, moderate; `beta_G` 0.18, P>0 0.85, suggestive; `beta_deapp_c` 0.06, P>0 0.65, inconclusive; `beta_erbto` −0.02, inconclusive |
| `mm-001` | 0.19 (−0.24 to 0.63), P>0 0.81, suggestive    | −0.32 (−0.53 to −0.10), P<0 0.998, very strong | `beta_grammar` 0.19, P>0 0.83, suggestive; `beta_vocabulary` 0.08, P>0 0.65, inconclusive; `beta_blocks` 0.03, P>0 0.61, inconclusive                               |
| `mm-101` | 0.20 (−0.22 to 0.63), P>0 0.82, suggestive    | −0.32 (−0.53 to −0.10), P<0 0.998, very strong | `beta_grammar` 0.19, P>0 0.83, suggestive; `beta_vocabulary` 0.09, P>0 0.65, inconclusive; `beta_blocks` 0.03, P>0 0.61, inconclusive                               |

The age association is the most certain structural finding in all three: older children in the cross-section sit **lower** on the residualised outcome scale (a robust negative adjusted association, P<0 ≈ 0.998). `mm-002`'s errors-in-variables `beta_code` of 0.34 (moderate) is larger and more certain than the `~0.19` (suggestive) of `mm-001`/`mm-101`, which is expected once the code factor is measured with explicit error rather than as a fixed composite — but again, all three are **held** pending clean convergence of the surrounding covariance block.

### Longitudinal model `lcf-001` — the gate-clean anchor (passed)

**Per-wave latent correlations are strong, positive and remarkably stable across all four waves** (values are the wave-1 figures; waves 2–4 differ only in the second decimal):

| Pair          | Wave-1 correlation (95% range) | Across waves 1–4 | P>0  | Evidence    |
| ------------- | ------------------------------ | ---------------- | ---- | ----------- |
| vocab~grammar | 0.86 (0.76–0.93)               | 0.85–0.86        | 1.00 | very strong |
| vocab~code    | 0.66 (0.49–0.79)               | 0.65–0.66        | 1.00 | very strong |
| code~grammar  | 0.55 (0.36–0.72)               | 0.55–0.56        | 1.00 | very strong |

Vocabulary~grammar (~~0.86) is the tightest coupling and code~~grammar (~0.55) the loosest — a large, meaningful gap.

**Partial (conditional) slopes tell a mediation story.** Holding the third domain constant, the vocabulary links stay strong — grammar→vocabulary 0.71 (P>0 1.00, very strong), vocabulary→grammar 0.88 (1.00, very strong), vocabulary→code 0.71 (P>0 0.997, very strong), code→vocabulary 0.26 (P>0 0.997, very strong) — **but the code↔grammar partial slopes collapse to essentially zero** once vocabulary is conditioned on: code→grammar −0.024 and grammar→code −0.055, both P>0 ≈ 0.42 (inconclusive). Read plainly: the raw code–grammar correlation of ~0.55 appears to run _through_ vocabulary — the grammar–code association is fully mediated by vocabulary. (Descriptive mediation, not an identified natural effect.)

**Stability (trait/state decomposition).** Each domain is dominated by a stable trait rather than occasion-specific state: trait shares are vocabulary 0.95 (95% 0.91–0.99), grammar 0.95 (0.87–0.99), code 0.93 (0.82–0.99). Roughly 93–95% of each domain's latent variance is stable across the four waves — these are highly stable traits, which is _why_ the per-wave correlations barely move. This stability is **not** a word-reading growth trajectory (there is no W indicator here); it is a statement that the domains themselves are steady.

**Disattenuation cross-check — the tangible payoff of the measurement model.** In **all 12** wave × pair cells the latent correlation **exceeds** the raw observed correlation, by gaps of **0.10 to 0.26** (e.g. wave-1 vocab~grammar: latent 0.86 vs observed 0.59, a 0.26 gap). Correcting for measurement error uncovers substantially stronger coupling than the raw scores show — the reason to fit a latent model at all.

**Concurrent-association cross-check.** Where the raw concurrent-association (CA) models produce noisy slopes that often straddle zero, the LCF latent slopes are consistently positive and certain. A concrete items-scale figure: a +1-item shift in latent letter-sound knowledge (L) lines up with **+0.54 items** of latent receptive vocabulary (R) at wave 1 (95% 0.38–0.71; P>0 = **1.00**, very strong), stable at ~0.53 across waves 1–4. Other items-scale slopes: grammar F→vocabulary R ~1.40 and grammar F→code L ~0.81 (both P>0 1.00). (An earlier draft called the L→R direction probability "99.9%" — the latent items-slope P>0 is 1.00 and the corresponding conditional-slope P>0 is 0.997; neither is 99.9%.)

## The one-paragraph story

Once measurement noise is stripped out, the skill domains are **strongly and positively correlated** — vocabulary, code and grammar (and, in the Byrne cohort, reading/language/memory/ability) really do travel together at the latent level, and the coupling is markedly stronger than the raw test scores suggest (latent correlations beat observed in every cell). The trustworthy, gate-clean version of this is `lcf-001`: the domains are highly stable traits, their correlations hold across four waves, and the apparent code–grammar link turns out to run entirely through vocabulary. The cross-sectional `mm` models reproduce the same picture and are prior-robust (`mm-101` replicates `mm-001`), but their correlation/covariance parameters are exactly the ones that failed convergence, so they are held as consistent-but-not-signed-off. The more ambitious "latent code → word reading" structural legs actually _did_ converge (`mm-002` code→W ≈ 0.34, moderate; `mm-001`/`mm-101` ≈ 0.19, suggestive, all alongside a robust negative age association), but we hold them conservatively too, because a model whose covariance block has not mixed cannot be signed off as a whole. Reparameterise, then read the structural legs.

## What is causal

**Nothing.** These are measurement/description tools. Latent correlations are descriptive; the partial slopes and the code→word-reading structural legs are **latent-ability-confounded adjusted associations**, not manipulations — no randomised contrast exists anywhere in a measurement model. The `lcf-001` mediation reading (grammar–code via vocabulary) is a model-based decomposition, not an identified natural effect. Positive numbers mean "these skills co-vary / this predictor is associated with the outcome," never "X drives/causes Y." The causal word-reading estimands live in the ITT / gain-factors / level-factors families, not here.
