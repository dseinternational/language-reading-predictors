<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Full rebuild of both model layers, 2026-09-08

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

Like the 2026-09-01 batch, this is **not a refresh**: the repository's `output/` tree was absent at the start, so every number below comes from a fit performed in this batch. Unlike that batch, `.venv` and `node_modules` were present and current, so no environment rebuild was needed. There is no "before" tree on disk to diff against, so this note carries no movement record; what it carries instead is one independent reproduction check and four cross-batch reproductions of previously recorded results (see **Validation**).

Every fit in both layers was produced at a single commit, `173dc0aa` (#665), from a clean working tree.

## Execution record

- **Environment**: `uv sync` reported no changes; the full `pytest` suite was green before any compute.
- **Gradient boosting**: 50/50 at `reporting`, one driver stream, **1 h 25 m 54 s**, 0 failures — within seconds of the previous batch's 1 h 26 m.
- **Statistical**: 276/276 at `reporting`. The fit phase spanned **8.62 h** wall and consumed **13.6 h** of stream time.
- **Registry growth**: 276 statistical models against the previous batch's 269, and 50 GB models unchanged.
- The tree was held clean throughout, so **no fit records `dirty: true`** and all 276 record commit `173dc0aa`. This is a single-commit batch, improving on the previous batch's accepted two-commit split.

## An operational finding: balance sweeps by cost, not by family count

The statistical sweep was first split into two streams of 138 by alternating the sorted registry, with the split checked for balance across the slow families (`med`, `mm`, `gc`/`hg`, `lcsm`, `hs`). **That check was the wrong one and the split was badly unbalanced.** Balancing the _count_ of mediation models per stream does not balance their cost: within `med` alone, this batch measured `med-092` at 10 min, `med-066` at 24 min, and `med-064` and `med-075` at roughly 95 and 130 min. Stream A finished its 138 in 6 h 47 m while stream B still held every remaining g-formula fit.

The cause is worth recording because it is not obvious from the model definitions: **the expensive part of a mediation fit is single-threaded.** The g-formula counterfactual integration and the per-leg tipping analysis run after sampling, so a stream sitting in that phase uses one core. With one stream in the tail, measured load average fell to 2.0 on a 16-core machine — 14 cores idle behind a queue.

The fix used, twice, was to **stop the driver but not its running child**: `pkill` matches the driver's command line and not the `fit_statistical_model.py` subprocess, so the in-flight fit runs to completion, writes its own artefacts and renders its own report, while fresh drivers take the remaining models. `med-075` and `lrp-rlm-jc-002` completed this way and both passed their gates (`med-075`: 0 divergences, r̂ 1.00038, ESS 14,595). The measured payoff: 13 models that had been queued behind the mediation tail completed in 39 m 33 s once split off, and a later batch of 3 in 2 m 57 s.

The cost of this manoeuvre is a **journal gap**, and it is worth stating precisely: the journal holds 275 records over 274 unique models — `mech-104` appears twice (original plus remediation refit), and `med-075` and `lrp-rlm-jc-002` appear not at all, because their drivers were stopped before they could write a record. Both fits are complete on disk with full artefacts, passing gates and correct provenance. **For statistical fits the journal is not the completeness authority** — the fit directories are, and the audit walks those. That is not true of the GB layer; see the next finding.

## Two defects in what the sweep can see

**1. A convergence-gate failure is invisible to the sweep journal.** `lrp-rli-mech-104` exited 0 and was journalled `ok`, because a failed sampling-quality gate does not make `fit_statistical_model.py` exit non-zero. It was caught only by re-evaluating the stored artefacts. A batch verified by its own journal — "276 fitted, 0 failed" — would have published an unconverged causal-adjacent fit. **Read the gate, not the exit code.**

**2. The GB layer records no fit-time provenance at all.** `output/models/<id>/config.json` has no commit, no dirty flag, no data digest and no environment lock, where the statistical layer records all four. An early audit pass of mine printed "0 dirty" for GB, which was vacuous — the field is absent, not false. GB provenance for this batch is established instead from the sweep journal, whose 50 records carry one identity: commit `173dc0aa`, `dirty: false`, environment digest `a2e5c1a6…`. So the batch is properly attributed, but **for GB the journal is the only provenance record**, and it lives in a gitignored tree. This is a new residual, not previously recorded.

## Remediations, each following recorded precedent

Neither was a blanket escalation; each is strictly above that fit's own recorded value, and each is recorded in its `config.json`.

| Model      | Problem                                                                   | Resolution                                                          |
| ---------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `mech-104` | 1 divergence in 36,000 draws; r̂ 1.0026, ESS 2,745, BFMI 0.88 all clean    | refit at `target_accept` 0.98 → 0 divergences, r̂ 1.00088, ESS 4,841 |
| `did-007`  | 1 of 21 prior-sweep cells (`mu_dose`, `tau_sigma` 1.5) with 6 divergences | `--cell-target-accept 0.99` → all 3 cells 0 divergences             |

Both were verified to fix computation without moving the science: `did-007`'s swept `tau_logit_mean` went +0.134 → +0.133 with an essentially identical interval, and `mech-104`'s `mechanism_curve` moved by at most 0.019 across all numeric columns. The `mech-104` pre-remediation directory is retained at `output/statistical_models/_backups/`. It was initially copied beside the fit, where `regenerate_key_findings.py all` promptly picked it up as a real fit directory — **backups must live outside `output/statistical_models/models/`**, since the tooling globs that path.

## Evidence loop

| Sweep                  | Result                                                                          |
| ---------------------- | ------------------------------------------------------------------------------- |
| gain-factor prior      | 6 rows, all cells converged, attached to `gf-005` / `gf-011`                    |
| level-factor prior     | 15 rows, all cells converged (W, L, P, B, N)                                    |
| floored P/N `tau` grid | validated manifest published; report-local copies to `itt-009` / `itt-011`      |
| blending link          | 2 B link fits validated; trace-bound key findings regenerated for `itt-008/108` |
| dispersion             | 12 cells, all converged                                                         |
| DiD prior              | 21/21 cells converged (after the `did-007` remediation above)                   |
| horseshoe prior        | **15/20 cells** — 5 unconverged on `hs-002`, `hs-004`, `rlm-hs-001`             |
| predictor ranking      | 12 model directories rebuilt under `output/ranking/`                            |

`compare_statistical_models.py` wrote 24 artefacts; `compare_gb_vs_statistical.py` and the four `compare_horseshoe_vs_gb.py` pairings followed.

## Final state

- **276/276 statistical fits and 50/50 GB models**, each with config, trace, key findings and a rendered report.
- **276/276 convergence-gate passes and 0 divergences anywhere**, after the two remediations. **0 fits from a dirty tree. All 276 at one commit.**
- **270 publishable / 6 withheld.** The 6 are exactly the #338 Byrne/RLM ports blocked at the `inputs` stage on unconfirmed `basspel` / `woco` / `basnum` denominators — `rlm-adj-001`, `rlm-hg-002`, `rlm-hg-003`, `rlm-hg-008`, `rlm-hs-001`, `rlm-mm-001`. The same 6 as the previous two batches; the release contract working as designed.
- 50 publishable fits carry a robustness qualification note (predominantly DiD treatment-prior leverage warnings). These are published _with_ the qualification, not withheld.
- **All 8 phoneme-blending response-link pairs are both-publishable**, so no `B` result is blocked by its companion.
- Rendered output corresponds exactly to the decisions: 276/276 reports rendered with 0 failures, **0 containing a traceback or import error** in either layer, and **precisely 6 showing "Findings withheld"** — the same 6 the audit withheld.

Verification was by **re-evaluating `release.evaluate_publication()` over all 276 stored directories**, not by reading the `release_decision.json` written at fit time, which goes stale the moment the evidence sweeps attach.

## Validation

`lrp-rli-med-059`'s total effect comes out at **2.3199 words (mean) / 2.3153 (median)**, against the **2.3195** the 2026-09-01 rebuild reported and the 2.319 published by the 2026-08-27 closing pass. The MCSE on that quantity is 0.000112 on the probability scale, about 0.0088 words, so the prior figure sits within 0.001 words of this batch's mean and within roughly half an MCSE of its median. #635 measured `med-059` as invariant under the ERB quarantine, so it is exactly the model that should reproduce — and it does, from a destroyed and rebuilt artefact tree, across the three intervening dependency and feature PRs (#661, #663, #664, #665).

Three further results reproduce their previously recorded values exactly, which matters because each is a structural property rather than a headline anyone was tracking:

1. **The horseshoe/GB top-3 construct overlaps are 2/3, 2/3, 1/3 and 2/3** across the four pairings each `hs-*` module names for itself — identical to the previous batch. The Bayesian sparse-regression ranking broadly corroborates the boosting ranking without matching it term for term.
2. **The L × N nested LOO** (`joint_readiness_lxn_w_loo_compare.csv`) is again `comparison_valid=True` via `psis+reloo`, verdict inconclusive (|elpd_diff| < 4) — the same answer as the 2026-08-27 and 2026-09-01 runs.
3. **The same 5 of 20 horseshoe sensitivity cells fail to converge, on the same three models.** Reproducing the identical count and set across a full rebuild indicates a stable property of those geometries, not sampling noise.

## Results

Nothing here is new science; it is the same suite re-estimated. It is recorded so the batch has a readable headline and so the next rebuild has something to reproduce against.

**Available-case modified ITT suite** (`itt-001`–`011`), outcome-specific average marginal on the probability scale, median with 89 % equal-tailed interval, probability oriented to the favoured direction:

| Model     | Outcome                        | Marginal | 89 % ETI         | P(>0) | Evidence        |
| --------- | ------------------------------ | -------- | ---------------- | ----- | --------------- |
| `itt-007` | Letter sounds (L)              | +0.110   | [+0.053, +0.166] | 0.999 | **very strong** |
| `itt-010` | Word reading (W)               | +0.030   | [+0.009, +0.052] | 0.986 | **strong**      |
| `itt-002` | Taught expressive vocab (TE)   | +0.064   | [+0.018, +0.111] | 0.985 | **strong**      |
| `itt-008` | Phoneme blending (B)           | +0.099   | [+0.022, +0.174] | 0.980 | **strong**      |
| `itt-001` | Taught receptive vocab (TR)    | +0.057   | [+0.008, +0.106] | 0.968 | moderate        |
| `itt-003` | Untaught receptive vocab (UR)  | +0.050   | [−0.002, +0.103] | 0.937 | moderate        |
| `itt-011` | Nonword reading (N)            | +0.100   | [−0.038, +0.237] | 0.877 | suggestive      |
| `itt-004` | Untaught expressive vocab (UE) | +0.026   | [−0.029, +0.080] | 0.773 | suggestive      |
| `itt-009` | Phonics/floored (P)            | +0.041   | [−0.071, +0.155] | 0.724 | inconclusive    |
| `itt-005` | Broad receptive vocab (R)      | +0.001   | [−0.022, +0.025] | 0.539 | inconclusive    |
| `itt-006` | Broad expressive vocab (E)     | +0.001   | [−0.014, +0.016] | 0.529 | inconclusive    |

Only `τ` is causal, and only under the suite's stated design and analysis-set assumptions. The pattern is the one the project has been carrying: directional evidence concentrated on the directly taught reading skills, with broad standardised receptive and expressive vocabulary imprecise — R and E are **inconclusive**, which is a statement about evidence, not a demonstration of absence; their intervals still permit material effects.

**Waitlist-crossover DiD**, `tau_t2` (the randomisation-anchored contrast), on the items scale:

| Model     | Outcome           | `tau_t2` items | 89 % ETI       | P(>0) | Evidence        |
| --------- | ----------------- | -------------- | -------------- | ----- | --------------- |
| `did-002` | Letter sounds (L) | +3.53          | [+1.18, +5.81] | 0.991 | **very strong** |
| `did-001` | Word reading (W)  | +2.22          | [−0.31, +4.69] | 0.920 | moderate        |
| `did-003` | Blending (B)      | +0.88          | [+0.06, +1.69] | 0.956 | moderate        |

These reuse the same randomised t2 information under a different specification and are **not** independent replications of the ITT estimates.

**Gradient boosting**, pooled cross-validated R²:

| Family        | n   | median R² | range         |
| ------------- | --- | --------- | ------------- |
| Level (`gbl`) | 28  | 0.603     | 0.185 – 0.998 |
| Gain (`gbg`)  | 22  | 0.128     | 0.021 – 0.267 |

This is the expected split and a useful check on a from-scratch rebuild: level models are autoregressive and predict well, gain models sit near noise. GB output is a ranking, not evidence about causal claims.

## This batch discharges the HSGP refits pending under #660

Issue [#660](https://github.com/dseinternational/language-reading-predictors/issues/660) had two parts. The `v0.13.0` code migration landed in #661; the **20 HSGP mechanism reporting refits and their held-out comparisons stayed open**, because the boundary correction changed the basis — the library moved from a `max(abs(X))` domain to half the input range about the midpoint — and identical basis weights against a different basis describe a different curve. Old posteriors could not be rescored, only resampled. The later `v0.14.0` upgrade (#662 / PR #663) explicitly did **not** absorb that work: it expressed the same arithmetic through `HSGPDesign.from_domain`, verified to reproduce the retired boundary exactly on 2,000 random domains, and recorded that the 47 disagreeing stored mechanism identities "belong to the refits already pending under #660".

This batch refit all 20 as part of the full sweep. Measured against `notes/assets/20260906-hsgp-refit-inventory-660.csv`, now updated in place with the post-refit state beside the preserved `old_*` audit columns:

- **20/20 refit** at commit `173dc0aa`, all carrying `hsgp_basis_version = midpoint-half-range-v1`.
- **20/20 stored `hsgp_m` / `hsgp_L` / `hsgp_center` and exposure scalers reproduce the migrated inventory exactly**, to a tolerance of 1e-12 — so the fitted basis is the one the migration predicted, not a reconstruction.
- **20/20 pass the convergence gate with 0 divergences and are publishable**, so no publication or comparison reader is withholding them as pending. Eighteen declare `target_accept` 0.999 in-module; `mech-073` and `mech-204` sit at the 0.95 preset, and `mech-104` is the batch's remediation at 0.98.
- **8 of the 20 carry a nested held-out comparison**, and every one is `comparison_valid=True` under `psis+reloo` with one exact refit: `mech-058`/`071`, `061`/`161`, `063`/`163` and `104`/`204`. The remaining 12 have no registered nested comparator, so per-fit PSIS-LOO is the whole of their held-out evidence.

Worth recording because it supersedes an earlier working assumption: nested LOO over HSGP pairs used to degrade to per-model ELPD, every pair hitting Pareto-k around 0.9–1.0. The #438 exact-refit repair now fixes those influential points, so these comparisons are valid rather than abandoned. All four verdicts are inconclusive (|elpd_diff| < 4), which is a statement about discrimination, not about the refits.

The implementation checklist in #660 is a separate matter and this note does not speak to it; what is discharged here is the scientific obligation — the fresh fits, their recorded identities and their comparison outcomes.

## Open residuals

1. **14 fits record no `data_sha256`** — `surv-009/011`, the six `rlm-adj-*`, `rlm-ca-001/002`, `rlm-hs-001/002/003`, `rlm-mm-001`. **Byte-identical to the set the 2026-09-01 batch recorded.** Nothing consumes the field for these families, so this is a provenance gap, not a correctness one.
2. **5 of 20 horseshoe prior-sensitivity cells are unconverged**, on `hs-002`, `hs-004` and `rlm-hs-001` — again the same count and the same three models. The four RLI horseshoe primaries remain publishable; this limits the ranking cross-check, not the primaries.
3. **The gain reports still cite a ranking view that is never produced for them.** `gbg-017/018/020/021` tell readers that `ranking_excluding_same_skill.csv` separates same-instrument from cross-domain signal, but `SAME_SKILL_SIBLINGS` is keyed on the _target_ and gain targets are deliberately absent. Confirmed again by direct inspection: the file exists for `gbl-017/018/020/021` and for none of the four gain models. Left for a decision rather than silently patched.
4. **The ERB source-archive question remains open** (#631 follow-up). The t4 record for `ID_FDCBDCF29AC0BF03` is still unverified against the source archive, so this batch fits against the **quarantined (missing)** values, which is what current code does.
5. **NEW — the GB layer writes no fit-time provenance.** See the second defect above. Unlike residual 1 this is not a missing field in an otherwise-complete record; there is no provenance block at all, and the only attribution is the gitignored sweep journal.

Residuals 1–4 all reproduce the 2026-09-01 batch exactly. Residual 5 is new to this note only in the sense that it was not previously looked for.
