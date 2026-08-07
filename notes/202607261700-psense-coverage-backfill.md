<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).
>
> Supersession warning added by a LLM-based AI tool (Codex/GPT-5).

> [!WARNING]
> **Superseded in part on 2026-08-02.** The power-scaling coverage record remains valid, but sensitivity diagnostics cannot release scientific quantities from a failed sampling gate. The former `rlm-mm-001` exemption is retired, and posterior conclusions below from gate-failed measurement or horseshoe fits are withheld under `notes/202608021625-divergence-qualification-policy.md`.

> [!WARNING]
> **Superseded in part on 2026-08-07.** The statement below that "`rlm-mm-001` remains the one true exemption … its posterior has not converged" no longer holds: the model converges since the corr_factor `LKJCorr` repair and its reporting fit carries a measured `psense_summary.csv`. See the closing section "The `rlm-mm-001` exemption is closed".

# Power-scaling sensitivity is now measured for 192 of 194 reporting fits (#381)

**What changed (2026-07-26, #381):** every stored `--config reporting` fit whose trace carries the `log_prior` and `log_likelihood` groups now has a `psense_summary.csv` and `psense.png`, backfilled by `scripts/regenerate_psense.py` without refitting anything — coverage 86 → 170. A second pass the same day refitted the 21 fits whose traces predated that wiring, taking coverage to **192 of 194**; see the closing section for what that changed (nothing) and for the two that remain.

## Why this was a gap rather than a clean result

Power-scaling sensitivity was wired into the family pipelines in #408 and #416, so the emission code has been correct for some time — `run_psense` is called from 22 fit functions covering every family. What was missing was **artefacts for fits stored before those merged**. A report with no psense table showed no flags, and a reader had no way to tell "measured, no concern" from "never measured". #381 named that distinction as its central meta-finding, and it is the whole value of this pass: it converts a large block of unverified verdicts into verified ones, in both directions.

## Why no refit was needed

Power-scaling is importance reweighting over the draws already in hand, not resampling. Given a trace with `log_prior` and `log_likelihood`, `arviz_stats.psense_summary` reconstructs the sensitivity after the fact, and the numbers belong to exactly the posterior the published report was written from — which is a stronger guarantee than a fresh fit would give, since a refit would produce a different (if equivalent) chain.

The reported-parameter set for each backfill comes from that fit's own `diagnostics.csv` rather than from re-deriving it off the spec. That file _is_ the fit's record of which parameters it reported, so the backfill covers exactly what the report shows, and it stays correct for a model whose spec has been edited since.

## What the measurement found

Across 1,647 parameters in 170 fits, 46 % carry some power-scaling flag. That headline number should **not** be read as "46 % of the suite is unsound". At n ≈ 54 a weakly-informative prior against a small sample will often show sensitivity, and most of the flagged parameters are nuisance or scale terms whose prior dependence changes no reported claim. The decision-relevant question is narrower: what happens to the terms the suite actually reports as estimands?

Restricting to each family's headline causal term — 155 such terms across 77 models:

| Family | term         |   ✓ | prior–data conflict | strong prior / weak likelihood |
| ------ | ------------ | --: | ------------------: | -----------------------------: |
| `itt`  | `tau`        |  28 |                  15 |                          **0** |
| `lf`   | `b_grp_time` |  20 |                  11 |                             13 |
| `hs`   | `beta`       |  19 |                  17 |                              2 |
| `did`  | `tau_t2`     |   4 |                   4 |                              3 |
| `mech` | `beta_mech`  |  10 |                   5 |                              0 |
| `bx`   | `delta`      |   2 |                   2 |                              0 |

**The ITT suite — the primary deliverable — has no headline `tau` in the serious class.** Fifteen show potential prior–data conflict, which at this sample size is unsurprising and informational; none is prior-driven. That is a genuinely reassuring result that was previously unmeasured for a third of the suite.

**Five models have their one randomised causal quantity flagged `potential strong prior / weak likelihood`** — the class where the prior is doing more work than the data:

- `lrp-rli-lf-005` and `lrp-rli-lf-006` at `b_grp_time[1]`, the clean randomised t2 contrast (phase coordinates are 0-based, so index 1 is t2). Both are floor-limited outcomes — phonetic spelling off-floor, and phoneme blending — so a thin likelihood is explicable rather than alarming, but it means the reported interval reflects the prior substantially and should be read that way.
- `lrp-rli-did-001` (word reading), `lrp-rli-did-003` (phoneme blending) and `lrp-rli-did-013` at `tau_t2`, the clean randomised t2 contrast.

These are precisely the refits #382 already proposes, and this pass supplies the measurement that motivates them rather than assuming it.

Two other results worth recording. `lrp-rli-hs-001`'s slab scale `hs_c2` comes back at prior 1.12 against likelihood 0.028 — `potential strong prior / weak likelihood`, exactly the dependence #381 predicted for horseshoe rankings ("the reported P(|β| > 0.1) is a direct function of `tau0`/`slab_scale`"). And `rli-lcf` and `rli-surv` flag on 100 % of their measured parameters, on small parameter sets.

## The remaining 23, resolved (2026-07-26, later the same day)

Their traces predate the `log_prior` / `log_likelihood` wiring, so power-scaling could not be reconstructed from them. **Twenty-one were refitted at `--config reporting` and now carry psense; coverage is 192 of 194.** The two that remain are not refit gaps, and this section's original claim that they were is corrected below.

`scripts/regenerate_psense.py` exits non-zero while any target is in this state, so a sweep cannot report success while leaving estimands unmeasured.

### The refits changed no published result

Nineteen mediation and measurement fits were overwritten, so the first thing checked was whether anything moved. Nothing did. The mediation headlines reproduce the values already quoted in #404 and the findings notes — `med-066` NIE_B −0.030 words, `med-075` −0.033, `med-074` proportion mediated 0.7 % — and the three `rli-mm` models reproduce their **gate failures to the digit** against `notes/202607210914-findings-measurement.md`: `mm-001` R-hat 1.0195 / ESS 354 / 1 divergence, `mm-002` 1.0478 / 64 / 0, `mm-101` 1.0213 / 260 / 57. Those failures are pre-existing and documented, their structural legs already on HOLD; the exact reproduction across a two-week gap and a `dse-research-utils` v0.8.0 bump is itself a reproducibility check the suite had not previously run.

One consequence for reading the new numbers: the three `rli-mm` fits flag ~96 % of their parameters (168/176, 170/179, 169/176), which corroborates #383's diagnosis that `HalfNormal(1)` on loadings-and-residuals puts ~32 % of prior mass on loadings > 1. But those posteriors **fail the gate**, so the same reasoning that exempts `rlm-mm-001` applies: treat this as corroboration of #383, not as an independent measurement.

### `rlm-jc-001` was a code gap, not a refit gap — and is now a diagnosed one

This note originally listed `rlm-jc-001` among the fits needing a refit. That was wrong. `fit_rlm_joint_growth` calls `_run_sampling_and_loo(ctx, compute_loo=False)` and never called `compute_log_likelihood_and_prior` **or** `run_psense` at all — #416 added that wiring to `fit_mediation` and `fit_correlated_factor` but not to this family — so a refit alone would have burned sampling time and produced nothing.

The wiring is now added, and the refit shows the real obstacle, which is neither what this note assumed nor what was predicted before running it. Both groups are skipped with:

```
exact match required for all data variable names, but
['eta_cell', 'sigma_subject', 'measure_corr_chol_cholesky', 'z_subject', 'kappa'] !=
['eta_cell', 'sigma_subject', 'kappa', 'measure_corr_chol', 'z_subject']
```

The model draws `pm.LKJCorr("measure_corr_chol", ...)` — chosen deliberately over `LKJCholeskyCov` to avoid its unidentified nuisance sd scales — and PyMC stores the **transformed value variable** in the posterior as `measure_corr_chol_cholesky` while the model's free RV keeps the base name. `compute_log_prior` and `compute_log_likelihood` both require an exact name match, so both refuse. It is a naming seam, not an intractable likelihood, and it blocks the prior group as well as the likelihood group.

That makes it plausibly fixable rather than an intrinsic exemption, and the value of adding the wiring is precisely that the absence is now **diagnosed with a specific error** instead of silent. Fixing it means reconciling the LKJCorr value-variable name across the `compute_log_*` seam, which reaches beyond psense coverage and is left as a follow-up.

`rlm-mm-001` remains the one true exemption #381 already records: its posterior has not converged, so a sensitivity diagnostic on it would not mean anything.

## The `rlm-jc-001` seam is closed (2026-08-05, #453)

The follow-up left open above is done, and the diagnosis it rested on was right: a naming seam, not an intractable density.

**The cause is upstream and is not `LKJCorr`-specific.** `pymc.util.get_untransformed_name` recovers a variable's base name by dropping a **fixed three** trailing underscore-separated components — one for the transform name and two for the `__` marker — which is only correct when `transform.name` itself contains no underscore. `LKJCorr`'s default `cholesky_corr` transform contains one, so `measure_corr_chol_cholesky_corr__` comes back as `measure_corr_chol_cholesky`. `LogExpM1` (`log_exp_m1`) breaks identically; those two are the only shipped transforms with an underscore in the name, checked exhaustively. The behaviour is byte-identical in PyMC 6.1.0 and 6.2.0, so waiting for an upgrade was never an option. A bug report is drafted at `notes/assets/draft-pymc-issue-get-untransformed-name.md` for a human to review and post.

**The local repair is a rename, not a rescale**, which was the one substantive risk in closing this at our end — a wrong-scale value would have produced plausible numbers rather than an error. `diagnostics.log_density_model` builds the un-transformed model itself, restores each value variable's name from its RV and hands that model to PyMC. The posterior already stores the constrained draw, which is exactly what the un-transformed log-density expects; only the label was wrong. Two guards pin this: a model whose names already round-trip is returned **unchanged** (identity, so the ordinary path is bit-for-bit what it was), and the repaired `LKJCorr` log-prior is asserted equal — `atol=0`, `rtol=0` — to `pm.logp` evaluated directly on the bare distribution at the same draws. A third test pins the upstream round-trip itself, so a PyMC release that fixes `get_untransformed_name` shows up as a failing test rather than as silence.

**Verified end to end.** A `--config dev` fit of `lrp-rlm-jc-001` now emits `psense_summary.csv` and `psense.png` with no skipped-group warnings, covering all three headline `measure_corr_pairs` correlations alongside `eta_cell`, `sigma_subject` and `kappa`. The dev-tier _numbers_ are not reportable — the reporting artefact lands with the next reporting-tier fit — but the wiring is proven, so this is a coverage gap closed rather than deferred.

One incidental corroboration worth carrying to #383, which proposes widening `sigma_subject` from `HalfNormal(0.5)` to `HalfNormal(1.0)` for exactly these high-variance measures: every one of the nine `sigma_subject[measure, group]` entries in the new dev-tier summary carries a flag, at prior sensitivities 0.21–0.52 — the largest values anywhere in that table. That is a dev fit and so is indicative only, but it is the first direct measurement on the parameter #383 names, and it points the same way.

## The `rlm-mm-001` exemption is closed (2026-08-07, #383 follow-up)

The one remaining "true exemption" above rested on a single ground — a non-converged posterior — and that ground is gone. The corr_factor `LKJCorr` repair showed the divergences were the discarded, unidentified `sd_dist` scales of `LKJCholeskyCov`, not near-singular geometry (see the supersession notices on `notes/202607241200-mm001-gate-exception.md`), and the current `rlm-mm-001` reporting fit passes the full gate: 0 divergences, max R-hat 1.0004, min ESS 5,183, BFMI ≥ 0.84 on every chain. The psense wiring for `fit_rlm_corr_factor` landed with #480, so the fit now carries a measured `psense_summary.csv`: 22 parameters, 18 clear, with 4 of the wave-3 `factor_corr_pairs` correlations at "potential prior-data conflict" — informational for a latent correlation under an LKJ prior at this sample size, and consistent with the family's standing fragility caveat. Both exemptions this note recorded (`rlm-jc-001`, `rlm-mm-001`) are therefore closed with measurements, not waivers.

## Not addressed here

#381 also asks for `prior_pushforward.csv` across every estimand-reporting family, and for an indicator-scale prior-predictive check on the measurement/CFA families. That is **not** the same shape of job. `prior_pushforward` pushes a treatment term through the items-scale average marginal effect and is ITT-shaped by construction — it takes the arm indicator and a trial count. Families whose estimand is a curve (`mech`), a ranking (`hs`), a latent change (`lcsm`) or a correlation (`lcf`) have no single "estimand scale" to push a prior onto, so extending it needs a per-family decision about what the estimand-scale check even is, not a call-site added. 51 of 193 fits currently carry one. The outcome-scale prior-predictive check, by contrast, is near-universal already (169 of 193).

## Reproducing

```bash
python scripts/regenerate_psense.py all --dry-run
```

Drop `--dry-run` to write. `--force` recomputes where a summary already exists.
