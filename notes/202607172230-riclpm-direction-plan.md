<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Plan (proposal): a random-intercept cross-lagged model for the LS ↔ WR direction

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). **Status: plan for review — nothing built yet.** This is the "carefully plan before we build" document; the design decisions in §7 need sign-off, and the build is explicitly **gated on a simulation feasibility check** (§5) that may return "do not fit on real data at this n".

## 1. Purpose

Give the letter-sound ↔ word-reading **direction** question a within-person estimate that nets out the stable between-child ability confound, as the observational partner to the randomisation-anchored mediation contrast already built (`med-076` forward, `med-176`/`med-276` reverse; see `notes/202607172100-reverse-mediation-wr-ls-direction-spec.md`). The vehicle is a **random-intercept cross-lagged panel model (RI-CLPM)** (Hamaker, Kuiper & Grasman 2015, [doi:10.1037/a0038889](https://doi.org/10.1037/a0038889)), adapted to this study's bounded counts, four waves, waitlist-crossover design and n ≈ 54.

## 2. Why this is not already covered by `lcsm` or `med-176`

- **`med-076`/`med-176`** are two-wave g-formula decompositions. They are ID-2 / GA-confounded: a positive reverse cross-lag is exactly what a shared stable ability manufactures, which is why the reverse NIE (+0.45) is "not separable from ability-leakage".
- **`lcsm`** (the coupled latent change-score family) carries between-person differences through the **initial latent condition** `x_m[i, t_1] ~ Normal(mu1_m, sigma1_m)` propagated by the AR recursion, and its couplings `g`, `h` act on the latent **levels** `x[t-1]`. It therefore has **no time-invariant random intercept**, and its own honesty box (ID-3) states the couplings "partly carry stable ability". This is the CLPM/ALT confound, not the RI-CLPM decomposition.
- **RI-CLPM's defining feature** — a per-domain **time-invariant** random intercept `RI^m_i` (constant across all waves), with the cross-lagged paths estimated only on the **within-person deviations** from each child's own stable level — is genuinely absent. That is the confound-removing step. (For the formal relationship among CLPM / LCS / ALT / RI-CLPM see Usami, Murayama & Hamaker 2019, [doi:10.1037/met0000210](https://doi.org/10.1037/met0000210).)

**Decision D1 (recommend): build a new `riclpm` family, do not flag `lcsm`.** The `lcsm` code is already intricate (change recursion, process noise, arm×window intercepts, lagged-change couplings) and its between-person structure is the initial condition, not an RI. Bolting an alternative decomposition onto it would obscure both. A small dedicated family is cleaner and keeps the two decompositions legibly distinct.

## 3. Model specification (generalised RI-CLPM on the logit–Beta-Binomial scale)

Bivariate, letter sounds `L` and word reading `W`, four waves `t = 1..4`. On the **latent logit scale** for domain `m ∈ {L, W}`, child `i`, wave `t`:

```
theta^m[i,t] = mu^m_t  +  arm-shift (see §4)  +  RI^m[i]  +  w^m[i,t]
```

- `mu^m_t` — a per-wave, per-domain fixed mean (the grand trajectory).
- `RI^m[i]` — the **time-invariant** random intercept. `(RI^L[i], RI^W[i]) ~ MvNormal(0, Σ_RI)`, `Σ_RI` from an LKJ prior. **`corr(RI^L, RI^W)` is the between-person structure** ("ability travels together") — reported, and explicitly _not_ the dynamics.
- `w^m[i,t]` — the **within-person deviation**, carrying the dynamics:
  ```
  w^L[i,t] = α_L · w^L[i,t-1] + β · w^W[i,t-1] + ζ^L[i,t]
  w^W[i,t] = α_W · w^W[i,t-1] + δ · w^L[i,t-1] + ζ^W[i,t]
  ```
  with `w^m[i,1]` a free initial deviation (correlated across domains) and `ζ ~ Normal(0, σ_ζ^m)` process noise (residual innovation correlation `ρ_ζ` optional — see §7 D4).
- **Observed counts:** `y^m[i,t] ~ BetaBinomial(n_trials^m, logit^{-1}(theta^m[i,t]), kappa^m)`, reusing the suite's `beta_binomial_from_logit`. Missing cells masked (the panel already carries `obs_mask`).

**The estimand: `δ` (L→W within-person) vs `β` (W→L within-person).** Report the posterior of **`δ − β`** and **`P(δ > β)`**, plus each path's own credible interval and direction probability on the #179 ladder. Paths are **time-invariant** (pooled over transitions) — see §4 — so `δ`, `β`, `α_L`, `α_W` are four scalars, not per-wave vectors.

## 4. The crossover — asset and complication

The intervention shifts trajectories on a **randomised, staggered** schedule (immediate arm t1→t2, waitlist t2→t3). Two things follow:

- **Mean structure must absorb treatment timing** so the RI captures stable _ability_, not "which arm got treated when". Add an **arm × wave** shift to `mu^m_t` (mirroring `lcsm`'s `arm_window_intercepts`), i.e. `mu^m_t + τ^m_{arm,t}`. Without it, the RI would soak up the between-arm mean difference and bias the variance decomposition.
- **The randomised shock is the identification asset.** Because the arm assignment is random, the _timing_ of the letter-sound jump is exogenous. The cleanest causal-flavoured readout is therefore **not** the pooled `δ` but the **period-1 within-person propagation**: does the immediate arm's randomised t1→t2 letter-sound deviation predict its t2→t3 word-reading deviation (with the waitlist as the untreated-timing comparator)? Report this as the anchored companion to the pooled `δ` — the RI-CLPM analogue of the mediation family's period-1 readout. (Note there is **no** randomised early _word-reading_ shock, so the reverse `β` has no equivalent anchor — the same power asymmetry flagged for `med-176`.)

## 5. Feasibility gate — SIMULATION FIRST (build step 0; go/no-go)

RI-CLPM is data-hungry and this panel is small (n ≈ 54, T = 4), and the counts are floored/ceilinged (less Fisher information than Gaussians). The RI variance and the within-person variance are known to **trade off** badly with few waves, and the wave-1 deviation is weakly identified (Mulder & Hamaker 2021, [doi:10.1080/10705511.2020.1784738](https://doi.org/10.1080/10705511.2020.1784738)). **Before any real-data fit** we therefore run a recovery study:

1. Simulate ~200 datasets at the true design (n = 54, T = 4, the real `n_trials`, the real missingness pattern, arm sizes) from the RI-CLPM at **plausible parameter values** taken from the fitted `lcsm`/`med-176` (a modest `δ`, a small `β`, a strong `corr(RI^L, RI^W)`, realistic `σ_ζ`, `κ`).
2. Fit the model to each; assess: (a) **bias and CI coverage** of `δ`, `β`, and `δ − β`; (b) **simulation-based calibration** (SBC rank uniformity) for the direction contrast; (c) whether `Σ_RI` and `σ_ζ` are separable (the trade-off) or collapse.
3. **Go/no-go:** proceed to real data only if `δ − β` is recoverable with roughly nominal coverage and no pathological RI/within trade-off. **If it is not, the deliverable is a documented "RI-CLPM is not identifiable for this direction contrast at n ≈ 54 × 4 waves" — a legitimate, honest result**, and we stop rather than over-interpret a fragile fit. This gate is the single most important part of the plan.

## 6. Parameter economy (what makes it estimable at all)

Every choice below spends degrees of freedom deliberately:

- **One bivariate pair only** (L, W). A trivariate (add PA or E) is almost certainly unaffordable at this n — defer unless the feasibility study says otherwise.
- **Time-invariant** `α_L, α_W, δ, β` (pooled over the three transitions) — the largest df saving; per-wave paths are not affordable.
- **Regularising priors** on the cross-lags (§7).
- **Consider fixing** the process-noise innovation correlation `ρ_ζ = 0` and/or equating `σ_ζ` across waves if the feasibility study shows they are not separable from `Σ_RI`.
- **No covariate block** beyond the arm×wave means and age in `mu` (HS/SP/RW enter only if the feasibility study shows headroom — unlikely).

## 7. Design decisions needing sign-off

- **D1 — new `riclpm` family vs `lcsm` flag.** Recommend **new family** (§2).
- **D2 — pair.** Recommend **L ↔ W only** (§6). Trivariate deferred.
- **D3 — first-wave treatment.** Free `w^m[i,1]` with a correlated bivariate prior (the standard RI-CLPM choice) vs constraining it. Recommend free-but-correlated, revisited by the feasibility study.
- **D4 — process-noise correlation `ρ_ζ`.** Estimate vs fix at 0. Recommend **start fixed at 0** (parameter economy) and only free it if the feasibility study shows it is both identifiable and material.
- **D5 — reliability / measurement.** The Beta-Binomial measurement variance is the built-in guard against the reliability-asymmetry artefact (LS 32 items vs WR 79 items); a separate latent-indicator layer is **out of scope** for this n. Note the residual asymmetry as a limit.
- **D6 — go/no-go authority.** Confirm that a "not identifiable at this n" feasibility result is an acceptable (indeed valuable) deliverable, so we are not pushed to force a fragile real-data fit.

## 8. Reporting & contrast

On a passing feasibility gate and a gate-passing real fit, report: `δ`, `β`, **`δ − β` with `P(δ > β)`**; the **period-1 anchored** `δ` readout (§4); `corr(RI^L, RI^W)` (the between-person "ability travels together" quantity); the within-person residual and RI variances; and a **time-varying-ability sensitivity** (see §9). Then a single table contrasting the three direction reads — `med-076/176` (2-wave g-formula), and RI-CLPM pooled + period-1 — written into the mediation findings note and the word-reading growth synthesis note.

## 9. Honest limits (to state in the report regardless of outcome)

- The RI removes only the **time-invariant** slab of general ability. If ability itself _grows_ (time-varying GA), the within-person cross-lags remain confounded — so `δ`, `β` are **adjusted associations, not causal**. Report a sensitivity (e.g. an ability-trend term, or an E-value-style argument).
- **Power:** n ≈ 54 × 4 waves is marginal; expect wide intervals. The feasibility gate (§5) is precisely to keep us honest about this.
- **Asymmetric anchoring:** the design randomises an early L shock but no early W shock, so `δ` has a randomised-timing anchor and `β` does not — a weak `β` is not strong evidence against `W → L`.
- This is a **triangulation partner** to the randomisation-anchored evidence, never a decisive causal direction test.

## 10. Build sequence (only after §7 sign-off)

1. [ ] Implement the model as a standalone builder + a `riclpm` pipeline entry (reuse `LongitudinalPanel`/`load_wave_panel`, `beta_binomial_from_logit`, the shared priors, the gate, the CSV/plot writers).
2. [ ] **Feasibility study (§5)** — simulate, fit, assess recovery + SBC. Write the results to a dated note. **Stop here if no-go.**
3. [ ] On go: register `lrp-rli-riclpm-001` (L↔W), fit `dev` → `reporting`; gate.
4. [ ] Report per §8; contrast with `med-076/176`; update the two synthesis notes.
5. [ ] `ruff` / `format:check` / `spellcheck` / CI green; report template under `docs/models/`.

## 11. Rough effort / risk

Larger than the `med-176` build: a new likelihood-bearing family plus a simulation study. Main risk is that the feasibility gate returns no-go — in which case the _value_ is the documented negative result and the clean contrast against the mediation reads, not a new headline estimate. That is an acceptable and honest outcome and should be treated as success of the _plan_, not failure.
