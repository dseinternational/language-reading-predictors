<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

# Power-scaling sensitivity is now measured for 170 of 193 reporting fits (#381)

**What changed (2026-07-26, #381):** every stored `--config reporting` fit whose trace carries the `log_prior` and `log_likelihood` groups now has a `psense_summary.csv` and `psense.png`, backfilled by `scripts/regenerate_psense.py` without refitting anything. Coverage goes from 86 to 170 of 193 fits. The remaining 23 need a refit and are listed below.

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

## Still uncovered: 23 fits needing a refit

Their traces predate the `log_prior` / `log_likelihood` wiring, so power-scaling cannot be reconstructed from them: the `med` family (18 of 19), all three `rli-mm`, `rlm-jc-001` and `rlm-mm-001`. `rlm-mm-001` is the one legitimate exemption #381 already records — its posterior has not converged, so a sensitivity diagnostic on it would not mean anything. The rest need a `--config reporting` refit, at which point the fit-time wiring emits psense on its own with no further change.

`scripts/regenerate_psense.py` exits non-zero while any target is in this state, so a sweep cannot report success while leaving estimands unmeasured.

## Not addressed here

#381 also asks for `prior_pushforward.csv` across every estimand-reporting family, and for an indicator-scale prior-predictive check on the measurement/CFA families. That is **not** the same shape of job. `prior_pushforward` pushes a treatment term through the items-scale average marginal effect and is ITT-shaped by construction — it takes the arm indicator and a trial count. Families whose estimand is a curve (`mech`), a ranking (`hs`), a latent change (`lcsm`) or a correlation (`lcf`) have no single "estimand scale" to push a prior onto, so extending it needs a per-family decision about what the estimand-scale check even is, not a call-site added. 51 of 193 fits currently carry one. The outcome-scale prior-predictive check, by contrast, is near-universal already (169 of 193).

## Reproducing

```bash
python scripts/regenerate_psense.py all --dry-run
```

Drop `--dry-run` to write. `--force` recomputes where a summary already exists.
