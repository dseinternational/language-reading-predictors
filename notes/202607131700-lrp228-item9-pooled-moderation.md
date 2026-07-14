# Pooled treatment × baseline moderation (#228 item 9)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

## Question

Does the intervention **flatten the baseline gradient** — help children who start lower more than children who start higher — on average across skills? Each gain-factor model estimates a treatment × own-baseline interaction (`gamma_int_trt_own`); it is negative in 7 of the 8 outcomes, but every single interval spans or nearly spans zero, so no one model settles the equity question. Pooling gives it far more power.

## Method (second stage)

A Bayesian random-effects meta-analysis over the eight fitted interaction posteriors:

```
theta_k ~ Normal(mu, tau)          # per-outcome true interaction
d_k     ~ Normal(theta_k, se_k)    # each gain-factor model's estimate (mean, sd)
```

`mu` is the pooled interaction (negative = flattening); `tau` is how much it varies across skills. Priors `mu ~ Normal(0, 0.5)`, `tau ~ HalfNormal(0.3)`, non-centred. Code: `statistical_models/pooled_moderation.py`; reproduce with `python scripts/pooled_moderation.py --suffix reporting`.

## Result (from the 2026-07-08 reporting fits)

**Pooled interaction `mu` = −0.11 (95% CrI −0.22 to −0.02), P(mu < 0) = 0.99.** Moderate but consistent evidence that the intervention narrows the baseline gap. **Heterogeneity `tau` = 0.05 (95% CrI 0.00 to 0.18)** — low, so the interaction is fairly consistent across skills.

| Outcome                     | raw `gamma_int_trt_own` (mean, sd) | shrunken |
| --------------------------- | ---------------------------------- | -------- |
| W word reading              | −0.095 (0.127)                     | −0.109   |
| R receptive vocabulary      | −0.084 (0.072)                     | −0.102   |
| E expressive vocabulary     | −0.053 (0.069)                     | −0.093   |
| L letter-sounds             | −0.121 (0.170)                     | −0.110   |
| P phonetic spelling         | +0.046 (0.277)                     | −0.105   |
| B blending (phoneme aware.) | −0.293 (0.176)                     | −0.124   |
| F basic concepts            | −0.221 (0.164)                     | −0.120   |
| T receptive grammar         | −0.187 (0.104)                     | −0.124   |

With low heterogeneity the model pools hard: the noisiest estimates (B, F) and the lone positive (P) all shrink toward the −0.11 pooled mean. Read the shrunken column, not the raw one.

## Caveats (load-bearing)

- **Shared subjects.** The eight outcomes are measured on the same ~54 children, so the eight estimates are correlated, not independent studies. This meta-analysis treats them as independent, which makes `mu`'s interval **somewhat too narrow**. Read the direction and rough size, not a precise bound.
- **Associational.** The across-period interaction is an adjusted association, not the randomised effect; only the gain-factor on-intervention marginal is quasi-causal. Exploratory at n≈54, per `METHODS.md`.

## Recommended follow-on: the joint model (removes the shared-subject caveat)

Fit a single joint gain model over all eight outcomes with the interaction pooled directly in the likelihood, rather than as a second stage:

- Stack every outcome's gain-factor observation (post-count conditional on its own baseline, per period, masked Beta-Binomial), in the style of `build_joint_model`.
- Per-outcome interaction `delta_k = mu_delta + tau_delta · z_k` (non-centred), with a shared child random effect across outcomes to carry the cross-outcome correlation the second stage ignores.
- Everything non-centred, `target_accept` raised — this is funnel-prone at n≈54 (the mm-001 precedent), so it needs the same reparameterisation care and should be pre-registered before the reporting fit.

The second stage above is the tractable interim; the joint model is the honest final version. Deciding whether to build it is the #228-item-9 follow-up.
