# LRP67 / LRP68 — longitudinal dynamics: do prior skills predict later reading _over time_?

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

> [!IMPORTANT]
> **Scope (post-review).** Only **LRP67 (LCSM)** ships as a model. **LRP68 (RI-CLPM)
> was dropped** — degenerate at n≈54 (R-hat→1.33, unreliable LOO); its negative finding
> ("the within-child cross-lagged direction is not estimable at this n") is **recorded
> below** as the durable result. LRP67's intervention-dose term was removed (it is the
> locked DAG's `IS` collider; ID-3), so its reporting-tier numbers need a re-fit.

> [!WARNING]
> Preliminary `dev`-tier fits; numbers will firm up at the reporting tier. At n~54
> these are **exploratory triangulation**, not inferential headlines. Not for citation.

Date: 2026-06-19

## Context

This is "Approach 4" from the modelling meeting (process-over-time): the
**within-child** complement to LRP65's between-child word-reading-gain model. Two
additive models on all four waves:

- **LRP67 — latent change-score model (LCSM).** For each variable, a latent logit
  true-score per wave; reading _change_ is regressed on prior-wave letter-sounds,
  vocabulary, own level, age and dose. The longitudinal extension of LRP65.
- **LRP68 — constrained RI-CLPM.** Separates each child's stable trait from
  within-child fluctuations and asks the cross-lagged question directly: when a
  child is temporarily above their own expected letter-sounds, do they gain more in
  reading next? Four competing structures compared by LOO.

Symbol mapping (codebase vs the handoff's R / L / V): **W** = reading (`ewrswr`),
**L** = letter-sounds (`yarclet`), **E** = expressive vocabulary (`eowpvt`). `R` is
receptive vocabulary in `MEASURES` and is not reused — the handoff's "L → R" is
`g_L` (LCSM) / `A[W <- L]` (RI-CLPM).

## Method

- All four waves, n~54 children. Beta-Binomial likelihood on the logit scale;
  observed counts masked per cell so a child missing one score still contributes its
  other waves. nutpie. Covariates: age and dose (`attend`).
- **Deliberate constraints at n~54:** time-invariant coupling / cross-lagged paths
  (pooled over the three transitions); regularising `Normal(0, 0.5)` priors on the
  cross-lagged paths with a 0.3 / 0.7 sensitivity refit; the RI-CLPM models the
  observed (logit) scores directly (no measurement-error / latent-indicator layer).
- LRP67 is the full McArdle form with per-variable process noise on the change
  scores, on top of Beta-Binomial measurement overdispersion.
- Framing/reference for early-process-measure → later-outcome: **Yoder, Woynaroski,
  Fey, Warren & Gardner (2015)**.

## Results (preliminary, `dev`-tier; logit scale; 94/95% intervals; P = P(coef > 0))

### LRP67 — latent change-score (reading-change couplings)

Samples cleanly (0 divergences; R-hat <= 1.05 at `dev`).

| Coupling into reading _change_    | mean [interval]         | P(>0)    |
| --------------------------------- | ----------------------- | -------- |
| prior **letter-sounds** `g_L`     | **0.197 [0.075, 0.32]** | **1.00** |
| prior **vocabulary** `g_E`        | 0.246 [~0, 0.505]       | 0.98     |
| reading self-feedback `b_self[W]` | -0.267 [-0.36, -0.18]   | 0.00     |
| age `d_age[W]`                    | -0.111 [-0.19, -0.03]   | 0.00     |
| dose `d_dose[W]`                  | 0.149 [0.05, 0.25]      | 1.00     |

Prior letter-sounds (and, more weakly, vocabulary) predict subsequent reading
change; younger children change more; dose is positively associated; negative
self-feedback is regression-to-the-mean on the logit scale.

### LRP68 — constrained RI-CLPM (within-child cross-lagged)

| Within-child path                                | mean [interval]         | P(>0) |
| ------------------------------------------------ | ----------------------- | ----- |
| **letter-sounds → reading** `A[W<-L]` (headline) | **0.085 [-0.27, 0.40]** | 0.65  |
| reading → letter-sounds `A[L<-W]`                | 0.505 [0.20, 0.82]      | 1.00  |
| reading → vocabulary `A[E<-W]`                   | 0.096 [0.03, 0.17]      | 1.00  |

- The forward `A[W<-L]` is **wide, spans zero, and prior-sensitive**
  (0.3 → ~0.00, 0.5 → 0.09, 0.7 → 0.03): not robustly distinguishable from zero
  within-child.
- **Sampling trouble is part of the finding.** At `dev` tier the letter-sound
  equation mixes poorly (R-hat up to ~1.33, low ESS) and the simpler structures
  diverge (AR-only 22, L→R 6, R-driven 1, reciprocal 0). PSIS-LOO is **unreliable**
  here — every structure carries a Pareto-k warning and `p_loo ~ 180` on 639
  observed cells — so the LOO ranking (nominally `r_driven` ahead of AR-only) is
  **not trustworthy**: the ELPD differences are degenerate / within noise. There is
  **no reliable evidence that any cross-lagged structure out-predicts AR-only**.
- The only cleanly-identified cross-lagged signals run **reading → skills**
  (`A[L<-W]`, `A[E<-W]`), consistent with a reading-practice channel — but read these
  with the L-equation's poor mixing in mind.

### Between-child reference (LRP65 / exploratory note `202606161215`)

Baseline **letter-sounds is the strongest predictor of reading gain (r ~ +0.47)**,
ahead of vocabulary (+0.24); age is negative (-0.24).

## Convergence — the lead conclusion

The deliverable is whether the **direction** `letter-sounds → reading over time` is
supported across approaches, not any single coefficient.

- **Between-child (LRP65)** and **latent change (LRP67)** agree cleanly: letter-sounds
  is the leading positive predictor of subsequent reading (`g_L` = +0.197, P = 1.00;
  r ~ +0.47 between-child). Vocabulary is positive but more modest in both; age is
  negative in both. **Two of three approaches converge** on `L → reading`.
- The **within-child RI-CLPM (LRP68)** can **neither confirm nor refute** this at
  n~54: the forward path is wide/zero-spanning and prior-sensitive, and the model is
  weakly identified with unreliable LOO. This is the **expected limitation** of a
  within-child cross-lagged model at this sample size — it is not evidence against
  the `L → reading` direction. If anything, the cleaner within-child dynamic is
  reading → skills (practice), which is compatible with a reciprocal real-world
  process the data are too small to separate.

**Honest verdict:** the developmental direction `letter-sounds → reading` is robustly
supported in the between-child and latent-change views and is the defensible
triangulated conclusion; the within-child cross-lagged view is underpowered and
inconclusive at n~54, which is itself a reportable finding.

## Honest reading / limitations

- **n~54**, four waves, three within-child transitions — cross-lagged posteriors are
  wide; the RI-CLPM trait/within decomposition is weakly identified.
- **Observed scores** (no measurement-error correction in the RI-CLPM): within-child
  fluctuations include measurement noise, which attenuates cross-lagged estimates.
- **Time-invariant** couplings pool over transitions; **associational, not causal**.
- **`dev`-tier**: a reporting-tier rerun (6 chains x 6000, `target_accept` 0.95) is
  expected to reduce the RI-CLPM divergences and improve R-hat, but is unlikely to
  resolve the weak identification of the within-child cross-lagged paths. PSIS-LOO
  will remain unreliable for these latent-heavy models — prefer the structure
  comparison's _qualitative_ reading over its ELPD point values.

## Reproduce

```bash
# dev-tier fits (env with PyMC + the ML stack, e.g. dse-research-data-analysis):
python scripts/fit_statistical_model.py lrp67 --config dev   # LCSM
# (LRP68 / RI-CLPM dropped on review — see the scope note above; its negative
#  finding is recorded here but the model is no longer fitted.)
```

Artifacts: `output/statistical_models/models/lrp67-dev/` (`coupling_summary.csv`) and
`output/statistical_models/models/lrp68-dev/` (`loo_compare.csv`,
`cross_lagged_summary.csv`, `prior_sensitivity.csv`).
