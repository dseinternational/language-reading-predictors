<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5), recording the #438 decision by Frank Buckley.
>
> Supersession warning added by a LLM-based AI tool (Codex/GPT-5).

> [!WARNING]
> **Policy clarification added 2026-08-02.** The recorded reparameterisation decision remains active, but the earlier model-spec gate-exception idiom is retired. Any remaining divergence fails closed unless a separate trace- and estimand-bound review satisfies `notes/202608021625-divergence-qualification-policy.md`; nonlinear curve outputs and subsidiary refits remain zero-divergence-only.

# Adopting the thin-support HSGP reparameterisation for six mechanism models (#438)

**Decision (2026-07-25, #438):** `lrp-rli-mech-058`, `071`, `061`, `161`, `063` and `163` adopt the #430/#434 thin-support HSGP reparameterisation — basis count `m = 6` (from 10) and the tighter `ell_prior_mech_tight()` = `InverseGamma(8, 8)` lengthscale prior (from `InverseGamma(5, 5)`). This is a **per-model opt-in, not a change to the shared defaults**. The age-moderation pair `mech-073` / `mech-173` was tested and **rejected**: the same lever regresses `mech-173` from 0 to 10 divergences.

A gate exception for `mech-163` was drafted first and is **withdrawn** — the reparameterisation makes it unnecessary. The reasoning is preserved below because the argument that failed is instructive.

## What the reparameterisation fixes

Not the primary fits, mostly. Five of the six passed the sampling gate already. What fails at the default basis is the **leave-one-out refits**: every HSGP-curve mechanism model carries one or two observations whose Pareto-k exceeds `good_k`, and repairing those by exact refit (`reloo`, #438) requires the n−1 refit to converge. At `m = 10` with `InverseGamma(5, 5)` those refits diverge, so the influential points cannot be repaired and the nested interaction-vs-baseline comparison the pair exists for is unavailable.

| Pair               | primary gate             | max Pareto-k → after repair  | refits | nested `elpd_diff` |
| ------------------ | ------------------------ | ---------------------------- | -----: | ------------------ |
| `mech-058` / `071` | PASS / PASS              | 1.000 → 0.547, 0.998 → 0.555 |  1 + 1 | −3.04 (dse 2.85)   |
| `mech-061` / `161` | PASS / PASS              | 0.999 → 0.668, 0.883 → 0.633 |  1 + 1 | +1.96 (dse 1.38)   |
| `mech-063` / `163` | PASS / PASS              | 0.949 → 0.544, 0.865 → 0.594 |  1 + 1 | −0.02 (dse 0.70)   |
| `mech-073` / `173` | PASS / **FAIL (10 div)** | —                            |      — | **rejected**       |

Every contrast remains inconclusive under the `|elpd_diff| < 4` rule, which is the expected outcome at this sample size and a pass rather than a failure (#438, `METHODS.md`). The point of the change is that the contrasts are now _produced and trustworthy_ rather than absent.

Two incidental gains: `mech-163` passes the gate outright, so its proposed divergences waiver is withdrawn; and `mech-061` / `mech-161` now pass at their declared `target_accept = 0.999`, so the one-off 0.9995 command-line override used on 2026-07-25 is no longer needed.

## It does not flatten the curve

This was the risk #434 named, and the one that would have disqualified the lever for `mech-058`, whose letter-sound knee is a headline deliverable rather than a nuisance. Comparing each model's `mechanism_curve.csv` before and after:

| Model      | amplitude before |  after | change | max pointwise Δ |
| ---------- | ---------------: | -----: | -----: | --------------: |
| `mech-058` |           0.8810 | 0.8941 | +1.5 % |          0.0119 |
| `mech-061` |           0.7840 | 0.8046 | +2.6 % |          0.0188 |
| `mech-073` |           0.9346 | 0.9613 | +2.9 % |          0.0148 |

Amplitude marginally _increases_ in every case, on a curve spanning ~0.9. The moderation coefficients are likewise unmoved: `mech-063`'s `gamma_int` goes −0.050 (89 % −0.130 to +0.030, P(>0) = 0.15) → −0.053 (89 % −0.135 to +0.027, P(>0) = 0.15). The change buys geometry without changing the readout.

## `mech-073` / `mech-173`: comparison abandoned, baseline retired

Neither route works for the age-moderation pair. At the defaults its leave-one-out refits diverge, so the influential points cannot be repaired; under the reparameterisation the baseline's own primary fit regresses from 0 to 10 divergences. The nested PSIS-LOO test is therefore **abandoned** (#438): `age_moderation_loo_compare` is removed from `scripts/compare_statistical_models.py`, and `lrp-rli-mech-173` — which existed _only_ to serve that comparison — is **retired**.

`lrp-rli-mech-073` stays. Its `gamma_int` is a real estimand (−0.058, 89 % −0.119 to +0.006, P(>0) = 0.07 — no credible age moderation of the letter-sound → word-reading effect) and it completes the interaction family's recurring pattern, alongside `mech-071` (vocabulary) and `mech-072` (blending), of apparent moderations collapsing once subject random effects and adjustment are in. It is now read from its own posterior with no predictive-comparison companion, which is the reporting standard the restated criterion points to anyway.

`mech-173`'s canonical number is **reserved, not freed**: `model_ids._RETIRED_VARIANT_NUMBERS` keeps `lrp73base → ("mech", 173)` so a future model cannot silently reuse it. Earlier dated findings notes that quote `mech-173` results (e.g. `notes/202607161800-findings-mechanism.md`) are left as they stand — they are historical records of what was fitted at the time, not live claims.

## Why the defaults stay

`mech-173` is the counter-example that settles it: 0 divergences at the default, 10 under the reparameterisation. A lever that helps three pairs and breaks a fourth is a per-model tool, not a better default, so `_MECH_HSGP_M = 10` with `ell_prior_mech()` remains the shared setting. The `mech_hsgp_m` / `mech_lengthscale_tight` knobs #434 introduced are exactly the right shape for this; the six adopting specs carry the override and a comment recording the measurement.

## The withdrawn `mech-163` waiver, and why its first argument was wrong

Before the reparameterisation was tested on both sides of the pair, `mech-163`'s single divergent transition in 36,000 draws was going to be handled by a signed-off divergences-only gate exception in the `mm-001` / #412 idiom. Two arguments were offered for it, and **both were rejected in review of #439**:

- _Clean R-hat / ESS / BFMI._ These cannot rule out divergence-induced bias at all. If every chain equally fails to enter a funnel neck, the chains agree with each other and R-hat sits at 1.00 — the diagnostic is blind to precisely the failure mode in question.
- _A divergence-influence probe._ Excluding the flagged draw moved every reported median by less than 0.0001 posterior SD, and the draw sat in the bulk of every funnel-prone scale rather than against a boundary. But that measures the influence of the _recorded_ draw, not posterior mass the sampler may never have reached, and lying inside selected one-dimensional marginals does not exclude problematic joint geometry. Persistence at `target_accept` 0.9995 likewise does not support a numerical-artefact reading — if anything it cuts against it.

The evidence that would have licensed the waiver was an independent reparameterised fit returning the same posterior: 0 divergences, max R-hat 1.0000, min ESS 2,800, every reported parameter within 0.03 posterior SD (the sole larger move, `f_mech__ell` at 0.257 SD, being the intended effect of tightening its own prior). Once that fit existed the better move was obvious — adopt the reparameterisation on **both** sides of the pair rather than waive a check on one.

Reparameterising only the baseline would have broken the exact nesting the pair depends on. The two models would then differ by the interaction _and_ by basis count _and_ by lengthscale prior, and since more regularisation plausibly improves out-of-sample fit at n ≈ 54, the baseline could have won for reasons having nothing to do with `gamma_int`.

The general lesson worth carrying: when a fit is flagged, prefer fixing the geometry over waiving the check, and reserve the waiver for cases where the geometry genuinely cannot be fixed (`mm-001`'s near-singular wave-3 domain-correlation matrix remains the standing example).

## Reproducing

```bash
python notes/assets/202607251500-mech-hsgp-reparameterisation-probe.py --sensitivity
```

The default (no flag) runs the divergence-influence probe described above. It is retained only because it is cheap and bounds one failure mode; per the review it does **not** establish that a divergence is benign, and nothing should rest on it.
