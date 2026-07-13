# LCSM change-on-change extension (#229): spec and assumption set

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8). This is a specification and assumption set for discussion, per #229's request to write these out **before** committing to a fit. No model is fitted here.

## Question

Does a child's prior-wave **change** in letter-sounds (ΔL) or expressive vocabulary (ΔE) predict their later **change** in word reading (ΔW)? This is the literal reading of #229 ("do W gains depend on L gains vs E gains"), and the one specification the suite does not yet fit.

## What already exists

- **`lrp-rli-lcsm-067`** fits prior **level** to change (specification 1): `mean_ΔW[t] = a_W + b_W·x_W[t-1] + g_L·x_L[t-1] + g_E·x_E[t-1] + d_age·age[t-1]`, with `g_L = +0.135` and `g_E = +0.284` (both P ≈ 0.99). This says where a child _stands_ on L/E predicts reading growth, not how much L _grew_.
- **Gain-factor upstream couplings** are a level-to-level ANCOVA stacked across periods.
- **Neither fits change to change.** That is the gap #229 points at.

## Target specification (spec 2: lagged change to change)

Add the previous transition's latent change to the reading-change equation, keeping the level terms:

```
mean_ΔW[t] = a_W + b_W·x_W[t-1]
           + Σ_c g_c·x_c[t-1]          (prior level, as now)
           + Σ_c h_c·Δx_c[t-1]         (prior change, new)
           + d_age·age[t-1]
```

for c in {L, E}, where `Δx_c[t-1] = x_c[t-1] - x_c[t-2]` is the previous transition's latent change in c. `h_L` and `h_E` are the new headline coefficients: does prior letter-sound / vocabulary **growth** predict later reading **growth**, over and above prior level. The McArdle latent true-score layer is kept, so measurement floors (P, N) and noise stay out of the change estimates.

## Factory sketch (`build_lcsm_model`)

The recursion already computes each transition's latent change (the `delta` term in the loop, `x[s][t] - x[s][t-1]`). Minimal changes:

1. Store the per-transition latent change `Delta_c[i, k]` for the cross measures.
2. Add coefficients `h_cross = {c: Normal(0, coupling_prior_sigma) for c in cross}`.
3. In the reading-change equation, for transitions `k >= 1` only, add `Σ_c h_cross[c]·Delta_c[:, k-1]`. The first transition (`k = 0`) has no prior change, so the `h` terms are omitted there.
4. Expose it behind a flag (`lagged_change_coupling=True`) and register a new module (e.g. `lrp-rli-lcsm-068`) so `lcsm-067` (spec 1) stays untouched for side-by-side comparison.

Everything else (non-centred parameterisation, masked Beta-Binomial, `kappa`, process noise, the fallback ladder) is unchanged.

## Assumptions and limits (to carry into the report)

- **Associational, across all periods.** Over the four waves the waitlist arm crosses over, so between-child treatment is no longer randomised. Any skill-to-skill coupling here describes how L, E and W co-develop, not that L drives W. Only the Phase-0 randomised mediation (`med-059` / `med-064`) carries a causal label.
- **Two transitions only.** Four waves give three change transitions; lagged change-to-change needs a prior transition, so only two are usable. At n≈54 power is thin. The deliverable is **direction agreement** with the Phase-0 mediation, not a precise gain split. This is almost certainly why `lcsm-067` couples to prior levels (three transitions) rather than prior changes.
- **Between- and within-child variance are not separable.** RI-CLPM is not estimable at this sample size (recorded in the `lcsm-067` docstring), so `g_c` and `h_c` partly carry the fact that abler children grow on everything. Do not read a large `h_E` as "vocabulary growth drives reading growth".
- **Temporal order is preserved** (prior change then later change), which keeps it honest. Do **not** fit specification 3 (contemporaneous `ΔW ~ ΔL + ΔE`): reading practice builds letter-sounds within the same window, so direction is ambiguous; the shared measurement occasion inflates the covariance; and it is regression-to-the-mean prone. That is covariation, not mechanism.

## What it delivers

A triangulation point. Does the **sign** of `h_L` (prior letter-sound growth to later reading growth) agree with the Phase-0 randomised mediation, which moved reading through letter-sounds? Agreement strengthens the "letter-sounds are the active ingredient" reading. A large `h_E` with a small `h_L` would instead echo `lcsm-067`'s level result and its ability-confounding caveat, and should be reported that way, not as "vocabulary growth drives reading".

## Open decision for the team (the #229 next step)

1. **(a) Build the lagged-change LCSM (this spec).** Recommended: it reuses the `lcsm-067` scaffold, answers the literal question, and is honest about the thin power.
2. **(b) Period-stacked g-formula on the gain-factor scaffold**, with the exposure being the per-period on-intervention indicator instead of the Phase-0 randomised group. This reuses every on/off period but leans on the gain-factor ignorability assumption rather than randomisation. A companion, not a substitute.
3. **(c) Both.**

No fit is committed until this is agreed.
