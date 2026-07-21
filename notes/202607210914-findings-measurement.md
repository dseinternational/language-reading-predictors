# Correlated-factor measurement findings (2026-07-21)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is one of a set of per-family notes from the full 2026-07-21 re-fit of every Bayesian statistical model in the study (production `reporting` configuration, 6 chains × 6000 draws, 89% credible intervals). Read the findings index and reading guide (`notes/202607210900-findings-00-index-and-reading-guide.md`) first — it explains the study, the outcome measures and their maxima, and the house rules for reading a posterior. This note covers the **correlated-factor measurement** family: **5 models**, of which **1 passes the convergence gate** cleanly (the longitudinal model `lrp-rli-lcf-001`) and **4 are gate-review models** (the cross-sectional `corr_factor` measurement models, which fail on convergence and are reported with an explicit caveat below). **Every quantity in this note is an association or a description of how the skills go together — none is a causal effect.** The study's causal content lives in the randomised ITT, difference-in-differences and gain-factor families, which these models never touch.

## What this family probes

The other families ask "what does the intervention cause?" and "who progresses?". This family asks a prior, structural question: **how do the underlying skills relate to one another once we take measurement error seriously?** Each observed test — receptive vocabulary, letter-sound knowledge, and so on — is treated not as the skill itself but as a **noisy indicator** of an unobserved skill **domain** that sits behind it. A **latent-factor (correlated-factor) model** posits those hidden domains, treats what each test does _not_ share with its domain as measurement noise, and then reports the correlations between the _cleaned-up_ domains rather than between the raw test scores. Because the noise has been stripped out, these latent correlations are usually _higher_ than the plain test-to-test correlations — this upward correction is called **disattenuation**. Some of the models add a **structural leg**: a regression of an outcome (word-reading gain) on the latent factors, holding covariates such as age fixed.

The five models are three views of the same question in two cohorts:

- **`lrp-rli-lcf-001`** — the **longitudinal** correlated-factor model. It fits the three RLI skill domains (vocabulary, code, grammar) across **all four waves at once**, reporting a latent correlation per domain pair per wave, plus directional items-scale conditional slopes and a stable-trait / wave-specific-state split. **This is the model that passes the gate.**
- **`lrp-rli-mm-001`, `lrp-rli-mm-002`, `lrp-rli-mm-101`** — the **cross-sectional RLI** measurement models, one row per child, on the same vocabulary / code / grammar domains. `mm-001` carries a word-reading-gain structural leg; `mm-002` recasts the code factor as an errors-in-variables predictor of word reading (a "mechanism" leg); `mm-101` re-fits `mm-001` under recalibrated loading/residual priors (a prior-sensitivity check).
- **`lrp-rlm-mm-001`** — the **Byrne historical** wave-3 measurement model, a separate cohort of 75 children, with four domains: reading, language, memory and ability.

What each model _adjusts for_: the latent correlation is already an adjusted quantity in the sense that it conditions on the measurement structure (test-specific noise removed). The structural legs additionally hold covariates fixed — chronological age, block-design non-verbal ability, hearing, and nuisance missing-data indicators. Adjustment sets are pre-specified; nothing here is a lever.

## How to read these numbers

The house standard (full version in the reading guide): a Bayesian fit returns a **posterior**, a full probability distribution for each quantity. We summarise it by its **median** (the point estimate) and an **89% equal-tailed credible interval** — "an 89% posterior probability the value lies in this range" — plus, where available, the inner **50%** band. **Direction** is read from the **tail probability** P(>0), never from whether an interval excludes zero and never as a p-value. The fixed **evidence ladder** attaches a claim-strength label to that probability: **inconclusive** (P < 0.75), **suggestive** (≥ 0.75), **moderate** (≥ 0.91), **strong** (≥ 0.97), **very strong** (≥ 0.99) — round odds of 3:1, 10:1, 30:1, 100:1. The label describes strength of _evidence for a directional claim_, oriented to the favoured direction, and **never** describes the size of the effect.

Two family-specific points:

- **The estimand is a correlation (for the headline) and, for the longitudinal model, an items-scale slope.** A latent correlation near +1 with P(>0) ≈ 1.00 says the two hidden domains move together very tightly and the evidence that the association is positive is very strong — but "very strong" is about the _direction_, and the _size_ is read from the correlation value and its interval. The longitudinal model also translates a coupling into **items** (e.g. "+0.54 receptive-vocabulary items per +1 letter-sound item") for tangibility; that is a linearised measurement-model association at the average operating point, not a caused gain.
- **The convergence gate, and the HOLD rule for structural legs.** Before interpretation every fit is checked against a gate: R-hat ≤ 1.01 (chains agree), effective sample size (ESS) ≥ 400 (enough independent draws), and zero divergences (the sampler explored cleanly). Latent-factor models at 50–75 children produce a sampling **"funnel"** — a pinched region of the parameter space the sampler struggles to explore — so the four cross-sectional models fail on R-hat and/or ESS. Following the project's standing policy for measurement models: their **domain correlations remain usable as exploratory descriptions** (they are the robust, reportable quantity and, reassuringly, agree closely across the three RLI re-fits), but their **structural / latent-regression coefficients are HELD as not-yet-reliable, pending a non-centred (funnel-robust) reparameterisation.** The held slopes are shown below only so the fragility is explicit; do not read them as findings.

---

## Per-model findings

### `lrp-rli-lcf-001` — longitudinal correlated-factor model (**gate: PASS**)

The same three RLI domains fitted across all four waves at once: **216 observations from 54 children** (3 on-intervention phases, 4 waves). This is the one model in the family that **passes the convergence gate**, so its estimates are usable — subject to the usual small-sample and prior-dependence caution that attaches to any four-wave latent model at this size.

The headline is a directional, items-scale latent coupling. At **wave 1**, the clearest translated latent coupling linked **+1 letter-sound-knowledge (YARC-LSK) item with +0.54 receptive-vocabulary (ROWPVT) items** (89% credible range **+0.41 to +0.68**). The posterior probability that this association is positive is **99.9%** — **very strong** evidence that the two latent domains tend to move together. This items-scale slope is a linearised measurement-model association evaluated at the average operating point (it conditions on the third domain), **not a caused gain**: it describes how the underlying skills track one another, not what changing one would do to the other.

Read alongside the correlated-factor models below, this passing longitudinal fit is the family's anchor: it says the RLI skill domains are strongly and positively intertwined and that the coupling is stable and detectable wave by wave, using the one specification whose chains mixed cleanly.

### `lrp-rli-mm-001` / `mm-002` / `mm-101` — cross-sectional RLI measurement models (**gate: REVIEW / FAIL**)

Three views of the same RLI vocabulary / code / grammar structure, each **51 children, one row per child**. All three **fail the convergence gate** on the latent-factor funnel — their gate figures are tabulated so the fragility is explicit:

| Model            | Role                                            | N   | Gate     | max R-hat | min ESS | divergences |
| ---------------- | ----------------------------------------------- | --- | -------- | --------- | ------- | ----------- |
| `lrp-rli-mm-001` | reference — word-reading-gain structural leg    | 51  | **FAIL** | 1.019     | 354     | 1 (0.003%)  |
| `lrp-rli-mm-002` | errors-in-variables code→word-reading mechanism | 51  | **FAIL** | 1.048     | 64      | 0           |
| `lrp-rli-mm-101` | prior-sensitivity re-fit of `mm-001`            | 51  | **FAIL** | 1.021     | 260     | 57 (0.158%) |

(Divergence percentages are of the 36,000 total draws; the binding failures here are R-hat above 1.01 and ESS below 400, not the divergences.)

**Domain correlations (the usable headline)** — median [89% CI], with the positive-direction evidence label from the tail probability. The three re-fits agree closely, which is reassuring given the gate failures:

| Domain pair        | `mm-001`          | `mm-002`          | `mm-101`          | P(>0)                  | Label       |
| ------------------ | ----------------- | ----------------- | ----------------- | ---------------------- | ----------- |
| vocabulary–grammar | 0.80 [0.62, 0.92] | 0.79 [0.60, 0.91] | 0.80 [0.62, 0.92] | 1.00                   | very strong |
| vocabulary–code    | 0.74 [0.50, 0.90] | 0.75 [0.53, 0.91] | 0.73 [0.49, 0.89] | 1.00                   | very strong |
| code–grammar       | 0.67 [0.37, 0.87] | 0.67 [0.39, 0.87] | 0.65 [0.36, 0.86] | ≈ 1.00 (0.9996–0.9998) | very strong |

Read plainly: the three underlying RLI skill domains are **strongly and positively intertwined**, with **vocabulary–grammar the tightest coupling** (latent r ≈ 0.79–0.80) and **code (letter-sounds + blending) the most distinct** — yet even code shares roughly two-thirds of a correlation with each language domain. Every pair is very-strong positive evidence, and the stability across the reference fit, the mechanism re-cast and the prior-sensitivity re-fit says the _correlation_ pattern is not an artefact of one prior choice. The intervals are wide (the code–grammar 89% band spans roughly 0.36 to 0.87), so treat the values as provisional and prior-sensitive at n = 51.

**Structural legs — HELD, not findings.** The word-reading-gain slopes (`mm-001`, `mm-101`) and the errors-in-variables code→word-reading leg (`mm-002`) come from the part of each model most distorted by the funnel geometry, so under the project's measurement-model policy they are **held pending a non-centred reparameterisation** and must not be interpreted. For transparency: in `mm-001` the latent code→word-reading-gain slope sits at 0.19 [−0.16, 0.55] (P(>0) = 0.81) and grammar→gain at 0.19 [−0.14, 0.50] (P = 0.83), with vocabulary→gain near zero (0.08, P = 0.65); `mm-101` reproduces these almost exactly. In `mm-002` the errors-in-variables code→word-reading leg is 0.34 [0.09, 0.59] (P(>0) = 0.98) with a hearing term at 0.23 [0.05, 0.40] (P = 0.98). These leg estimates are shown only to document the model; **they are on HOLD and should not be quoted as results** until the reparameterisation clears the gate.

### `lrp-rlm-mm-001` — Byrne historical wave-3 measurement model (**gate: REVIEW / FAIL**)

A separate cohort — the Byrne historical dataset, **75 children at wave 3**, one row per child — with four domains: reading (multiple tests), language, memory and ability. It **fails the gate** on the same funnel (max R-hat = 1.028, min ESS = 64, 143 divergences = 0.397% of draws), so again the **correlations are reported as exploratory description** and any structural leg would be held.

**Domain correlations** — median [89% CI], all with P(>0) = 1.00 (very strong positive):

| Domain pair      | r [89% CI]        |
| ---------------- | ----------------- |
| language–ability | 0.93 [0.87, 0.97] |
| memory–ability   | 0.91 [0.83, 0.96] |
| reading–ability  | 0.88 [0.82, 0.93] |
| language–memory  | 0.86 [0.76, 0.93] |
| reading–memory   | 0.80 [0.69, 0.88] |
| reading–language | 0.79 [0.70, 0.86] |

The correlations are **very high across the board**: ability correlates ≥ 0.88 with every other domain, and the four domains behave almost as a **single general-ability dimension** in this historical cohort. As with the RLI models, only the correlation structure is reported, and only as description — the intervals are tight here (this cohort is larger), but the gate failure still means the fit is provisional.

---

## What to take away

Across both cohorts the measurement models draw a consistent picture of a **tightly inter-correlated skill web**. In the RLI cohort vocabulary and grammar are the closest pair (latent r ≈ 0.79–0.80), the code skills (letter-sounds, blending) are the most distinct but still strongly linked to the language domains (r ≈ 0.65–0.75), and the pattern is stable across the reference fit, the mechanism re-cast and the prior-sensitivity re-fit. In the Byrne historical cohort the four domains collapse almost onto a single general-ability dimension (correlations up to 0.93). The one model that passes the convergence gate — the longitudinal `lrp-rli-lcf-001` — anchors this: it puts the letter-sound → receptive-vocabulary coupling at a very-strong, tangible **+0.54 vocabulary items per letter-sound item** at wave 1.

Two cautions frame everything above. First, **four of the five models fail the gate** on a latent-factor funnel; per project policy their **domain correlations are usable as exploratory descriptions**, but their **structural legs are held** pending a funnel-robust reparameterisation and are not findings. Second — and applying to the passing model too — **none of these numbers is causal**: they describe _who is good at what alongside what_, the descriptive backdrop against which the randomised families are read, and residual confounding by latent general ability is unaddressed throughout. The strong skill inter-correlations are precisely why a single-outcome effect must be read inside a correlated system, and why the concurrent letter-sound → word-reading association is only the observational shadow of the decoding pathway that the mechanism and mediation families probe. Do not read any coefficient here as a lever.
