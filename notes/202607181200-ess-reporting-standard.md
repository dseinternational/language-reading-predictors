<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Standard: how we handle and report effective sample size (ESS)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Records the ESS handling/reporting convention reviewed and tightened 2026-07-18. Companion to [`202607172359-credible-interval-standard.md`](202607172359-credible-interval-standard.md) (the median + 50 % + 89 % ETI decision): that note fixes the interval _width_; this one fixes what ESS is required to estimate that interval and how we report it.

## Summary

Every threshold we quote for ESS encodes an unstated target precision (Monte-Carlo standard error, MCSE, as a fraction of the posterior SD). This note states the target, maps each reported quantity to the diagnostic that governs it, and records what the suite actually attains. The headline: **400 is a reliability floor, not a precision target.** Our reporting fits clear the precision target for the quantities we report by a wide margin, and the recent move to an 89 % _equal-tailed_ interval makes the standard Tail-ESS the near-exact diagnostic for our reported limits.

## The floor: ESS ≥ 400

Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021, [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)) recommend at least **100 effective draws per chain** for both Bulk-ESS and Tail-ESS. With the default four chains that is the familiar **≥ 400** figure. The per-chain framing is the actual rationale: below ~100 effective draws in a chain the ESS and R̂ estimators _themselves_ become too noisy to trust, so 400 marks the point below which the diagnostics stop being informative — not the point at which an estimate is precise enough to report. Our convergence gate (`diagnostics_summary.json`, thresholds owned by `dse_research_utils.statistics.diagnostics`) fails a fit when the minimum of Bulk-ESS and Tail-ESS over the model's free RVs plus the curated headline terms drops below 400. Taking the minimum of the two is deliberate: it binds on whichever is worse, which for interval reporting is usually Tail-ESS.

## Mapping each reported quantity to its diagnostic

- **Mean and median → Bulk-ESS.** The median is the 50 % quantile, where the posterior density is highest, so it is as easy to pin down as the mean; both are central summaries and Bulk-ESS (computed on rank-normalised draws) is exactly their diagnostic. Bulk-ESS ≥ 400 already gives MCSE ≈ SD/√400 = 5 % of the posterior SD, fine for a couple of significant figures; ~1,000–2,000 makes the point estimate visually stable across reruns. Central summaries are never the binding constraint.
- **89 % interval limits → Tail-ESS.** The limits are tail quantiles, Monte-Carlo-noisier per effective draw, so they set the binding target. A convenient alignment makes this exact for us: the `posterior`/Stan Tail-ESS is defined as the minimum of the ESS at the **5 % and 95 %** quantiles — i.e. it is calibrated to a **90 %** interval. For our **89 %** interval (5.5 %/94.5 % limits) the standard Tail-ESS is therefore essentially the exactly-right diagnostic. (For a 95 % interval it slightly _understates_ the difficulty, because 2.5 %/97.5 % are more extreme than the 5 %/95 % points Tail-ESS measures.)

## The precision target for the 89 % limits

Kruschke's Bayesian Analysis Reporting Guidelines (2021, [doi:10.1038/s41562-021-01177-7](https://doi.org/10.1038/s41562-021-01177-7)) state: _"For reasonably stable estimates of limits of highest-density intervals (HDIs), I recommend that ESS ≥ 10,000. For stable estimates of limits of equal-tailed intervals, ESS can be lower."_ His 10,000 is calibrated to a **95 % HDI**, so it is conservative for us on two counts (89 %, and equal-tailed). Normal-theory scaling shows how much: quantile MCSE per effective draw goes as √[p(1−p)]/φ(z_p) — ≈ 2.67 (in SD/√ESS units) at the 2.5 % point, ≈ 2.06 at the 5.5 % point (our 89 % endpoint). The ratio is ~1.30, and required ESS scales with its square, ~1.7, so matching Kruschke's stability standard at 89 % needs roughly 10,000 / 1.7 ≈ **6,000** rather than 10,000.

Practically: **target Tail-ESS ≈ 10,000** for a clean, safe, defensible number (it costs little at our sampler speed and matches the published guideline); **accept down toward ~6,000** if compute-bound; **never below the 400 floor**.

**HDI vs ETI.** For a symmetric, unimodal posterior the HDI and ETI limits nearly coincide and need the same Tail-ESS. They diverge when the posterior is skewed: the ETI endpoints are exactly the 5.5 % and 94.5 % quantiles, so Tail-ESS applies directly; the HDI limits sit wherever the density is equal, which pushes one limit further into a tail and makes it a bit noisier — and the HDI is not transformation-invariant. We report the **ETI** as the headline and keep the HPDI only as a per-scale sensitivity companion, so the favourable regime is the one that governs our headline numbers; the conservative 10,000 is the figure to hold the HPDI companion to, especially on visibly skewed bounded-scale posteriors.

## What the suite actually attains (reporting config, 6 chains × 6,000 draws)

Measured across the 176 reporting fits on 2026-07-18:

- **Headline causal terms** (τ / β_trt / δ / the t2 group contrast): Tail-ESS median ≈ **22,000**, 5th percentile ≈ 12,300, minimum ≈ 9,200 — above Kruschke's conservative 10,000 on essentially every fit, so the 89 % limits on the quantities we actually report are stable well beyond the precision we quote them at.
- **Gate min-ESS** (the worst single parameter, including random-effect scales and HSGP basis weights): median ≈ 6,468. Four fits fall below 400 — all the latent-factor _measurement_ models (`mm-001/002/101`, `rlm-mm-001`), which the gate is designed to flag (a latent-factor funnel; hold their structural coefficients, their correlations are fine).

So the ESS argument for 89 % is **not** that we are ESS-starved — we are not. It is that (a) 95 % is an arbitrary NHST import, (b) per effective draw the 89 % limits are cheaper and more stable to estimate, and (c) the diagnostic we already gate and report (Tail-ESS) is the near-exact one for the 89 % limits.

## Per reported quantity, including derived quantities

Everything above is **per reported quantity** — the parameters and derived quantities we actually put a median/interval on, not every node in the model. A derived quantity can have materially worse Tail-ESS than its parents. Our headlines are frequently _derived_: the probability-scale average marginal effect (AME) / off-floor risk difference, the g-formula NDE/NIE, and the readiness knee are post-processed from draws rather than being PyMC deterministics, so `az.summary` and the convergence gate never see them.

As of 2026-07-18 the reporting layer computes each primary derived estimand's own **Bulk-ESS, Tail-ESS and MCSE** and writes them alongside the estimate (`reporting.derived_mc_diagnostics`, wired into `tau_summary_itt`, the mediation g-formula `_effect_row`, and the readiness-knee post-processor). The draws are reshaped back to `(chain, draw)` so `az.ess`/`az.mcse` see the between-chain information both need. Measured on the existing traces:

- **AME** (a monotone transform of τ): Tail-ESS ≈ its parent τ (≈ 22,000), MCSE ≈ 0.0002 on the risk-difference scale (~0.02 percentage points) — far finer than we report.
- **Readiness knee** (a non-smooth argmax over binned draws — the case most at risk of mixing worse): still ≈ 33,000 Tail-ESS on `lrp-rli-mech-058`; its `mcse_median` can read 0 when the discrete knee median sits on a mass point (an edge-pinned knee, already flagged for some models).
- **g-formula NDE/NIE**: carry mediator re-simulation noise on top of posterior autocorrelation, so this is where a derived-vs-parent ESS gap would first appear; now reported per decomposition row.

These columns are written at fit time, so — like the 89 % interval change — they populate on **re-fit**; they render automatically in the reports' whole-table displays and are called out in the ITT results partial.

## Open item: scale the floor with chain count (upstream)

The gate floor is a fixed 400 regardless of chain count, but the Vehtari et al. rationale is ≥ 100 effective draws _per chain_. Our reporting preset uses **6** chains, for which the principled floor is 600 (400 total ≈ 67 per chain). This never bites in practice — the headline terms attain thousands — and the threshold lives in the shared `dse_research_utils.statistics.diagnostics` package (cross-project), so it is **flagged here, not changed locally**: the clean fix is a `100 × n_chains` floor upstream so the gate scales with the sampling preset. Recorded for a future `dse-research-utils` change rather than a per-repo override.

## What to report (checklist)

For every reported estimate, quote the achieved **Bulk-ESS, Tail-ESS and MCSE** rather than treating any single ESS number as a pass/fail line. `diagnostics.csv` already carries `ess_bulk`, `ess_tail`, `mcse_mean` and `mcse_sd` per model parameter and is shown in every report; the derived-estimand columns above extend that to the post-processed headlines. The convergence gate's 400 remains the reliability floor below which none of these numbers is trustworthy.

## References

- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC. _Bayesian Analysis_, 16(2), 667–718. [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
- Kruschke, J. K. (2021). Bayesian analysis reporting guidelines. _Nature Human Behaviour_, 5(10), 1282–1291. [doi:10.1038/s41562-021-01177-7](https://doi.org/10.1038/s41562-021-01177-7) (PMID 34400814)
