# Findings — corr_factor & long_corr_factor families (latent measurement models)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary.

## What these models ask

Individual tests are noisy measures of an underlying ("latent") ability. These models estimate the **latent domains** (e.g. reading, language, memory, ability) behind the observed tests and ask **how strongly the domains correlate**, correcting for measurement error. `corr_factor` (`mm-*`) does this cross-sectionally; `long_corr_factor` (`lcf-001`) does it per wave. Some `mm` models also add a **structural leg** (e.g. latent code → word reading).

## Convergence gate — read this before the numbers

Latent-factor models have a known "funnel" geometry that makes sampling hard, and it shows here. **All four `corr_factor` models are gate-flagged:**

| Model                        | Divergences | R̂         | min ESS | Status                                       |
| ---------------------------- | ----------- | --------- | ------- | -------------------------------------------- |
| `mm-002`                     | 0           | **1.048** | 64      | R̂/ESS fail — structural leg **not reliable** |
| `rlm-mm-001` (Byrne)         | 143 (0.40%) | 1.029     | 64      | funnel — structural leg **held**             |
| `mm-101` (prior-sensitivity) | 57 (0.16%)  | 1.021     | 260     | funnel — structural leg **held**             |
| `mm-001`                     | 1           | 1.020     | 354     | marginal — structural leg **cautious**       |

Following METHODS and the model documentation: the **domain correlations from these models are robust and reportable**, but the **structural coefficients (the "code → reading" leg) are held pending a non-centred reparameterisation or higher `target_accept`** — do not interpret them from this run. The `long_corr_factor` model `lcf-001` **passed** cleanly (0 divergences).

## Results we can report

- **`lcf-001`** (per-wave latent correlations, passed): at wave 1, a +1-item shift in latent **letter-sound knowledge** lined up with **+0.54 items** of latent **receptive vocabulary** (95% range +0.38 to +0.71; 99.9% positive) — the latent domains move strongly together, as expected.
- **`mm-001` / `mm-101` / `rlm-mm-001`** domain **correlations**: robust and positive across reading/language/memory/ability (see each model's `factor_correlation_summary.csv`); their **structural legs are on hold** per the gate.

## The one-paragraph story

Once measurement noise is removed, the skill domains are **strongly and positively correlated** — reading, language and memory really do travel together at the latent level. That correlational conclusion is trustworthy. The more ambitious "latent code causes reading" structural piece is **not trustworthy from this run** because those models did not converge cleanly; they need a reparameterisation before their structural coefficients can be read.

## What is causal

**Nothing.** Latent correlations are descriptive. Even the structural leg, once it converges, would be an adjusted association, not a manipulation effect. This is a measurement/description tool, not a causal one.
