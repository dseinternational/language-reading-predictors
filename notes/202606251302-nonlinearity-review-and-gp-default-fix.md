<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Non-linearity handling review, and the ITT GP-default fix

> [!WARNING]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8) via a multi-agent review
> (inventory -> four assessment lenses -> adversarial verification). Figures and
> file:line references were checked against the code on this branch, but re-verify
> before relying on them.

Date: 2026-06-25 — relates to issue #119 (the LRPITT suite). Branch: `fix/stats-nonlinearity-review` off post-#117 `main`.

## Why this note

A review of how the project models **non-linear relations** surfaced one real
trap and a few doc-hygiene errors. This note records the review's conclusions
and the small fix made here; the larger recommendations are logged as follow-ups
(some folded into #119).

## How non-linearity is handled (the two scales)

- **Link-scale (always on).** Every Bayesian model pushes a linear predictor
  through one sigmoid into a Beta-Binomial on the bounded count
  (`likelihood.py`). Even a purely linear predictor is non-linear in
  probability — coefficients shrink toward the 0/1 asymptotes, largest at
  mu = 0.5. Counts enter via a Haldane logit then z-score (`preprocessing.py`);
  the outcome stays a raw count, so the floor is undistorted in the likelihood.
- **Latent-predictor (optional, HSGP).** ExpQuad Hilbert-space GPs (`hsgp.py`):
  ITT age and own-baseline GPs, an age tau-modifier, a joint per-outcome age GP,
  and the mechanism dose-response `f_mech`. Interactions/moderation are **linear
  only** (`gamma_int`/`gamma_mod`); a GP-varying slope is deferred.
- **Per-family choice.** `f_mech` is the one GP on by default (≈159 stacked obs
  make it identifiable); ITT/joint GPs and the LKJ residual default off;
  `linear_mechanism` swaps a single slope in for floored counts.
- **Gradient-boosting explore step.** LightGBM is non-linear by construction;
  SHAP reads direction, permutation importance reads magnitude. Its job is to
  _find_ shapes, not estimate them.

## What's sound (keep)

- **GPs reserved for where they're identifiable.** Disabling ITT/joint GPs was
  _proven_, not assumed: GP-on gave ~1-8 % divergences from the eta -> basis-weight
  funnel while LOO marginally preferred linear (notes 202604181445, 202604181700,
  202604181730). Keeping `f_mech` only where phases stack is the right boundary.
- **AME reporting is the right antidote to the link.** `tau_summary_itt`
  (`reporting.py`) is a true per-observation g-computation —
  `expit(eta0_i + delta) - expit(eta0_i)` averaged per draw, each child at its own
  sigmoid position — reported on both logit and probability scales.
- **Link-saturation is reasoned about** (LRP72 refuses to over-read a negative
  `gamma_int` on a 57 %-floored count, note 202606161416).

## The fix made here (behaviour-preserving)

1. **Flip the ITT GP defaults to `False`** — `build_itt_model`
   `use_age_gp`/`use_own_baseline_gp` (`factories.py`) and the matching
   `pipeline.fit_itt` `.get(..., True)` defaults (`pipeline.py`). They defaulted
   **on**, opposite the `build_joint_model` / `build_mechanism_model` convention
   and the documented policy. Safe only because every current ITT spec
   (lrp52/53/54/60/60a/74/75) sets both flags explicitly — so a _new_ spec that
   forgot would silently fit two unidentifiable GPs into the known funnel. With
   the flip, the suite's no-GP stance can't drift, and the LRPITT models inherit
   it. **No behaviour change** for existing models or tests (the GP-on path keeps
   its explicit `use_age_gp=True` test).
2. **Doc-hygiene:** corrected the stale `hsgp.py` docstrings (`HalfNormal(1.0)` ->
   `HalfNormal(0.3)`, the real `eta_main_prior`) and the inaccurate
   `pipeline.py` "Mechanism curve ... on both scales" comment (only the
   logit-contribution curve is written).

Verified: `ruff` clean; `tests/statistical_models/test_factories.py` all pass
against the edited code (PYTHONPATH-pinned to this worktree).

## Recommended follow-ups (not done here)

- **LRPITT (#119): pair the binary off-floor estimand with a conditional
  above-floor mean (hurdle-style).** `Pr(score>0)` alone relocates the saturation
  problem and discards above-floor dose information. (Folded into #119.)
- **Make the linearity stance self-documenting in the LRPITT spec** — linearity is
  a deliberate identifiability choice at n≈53, not an oversight. (Folded into #119.)
- **Extend the AME g-computation to the mechanism family** — `f_mech` and
  `gamma_int`/`gamma_mod` are reported on the **logit scale only** (`reporting.py`,
  `_write_mechanism_curve`), exactly where saturation bites. A probability-scale
  `f_mech` curve + a marginal-interaction contrast would remove the verbal-only
  caveat.
- **HSGP adequacy check.** `m`/`c` are hardcoded and the data-driven
  `approx_hsgp_hyperparams` path is dead code (no caller passes `ls_range`,
  `hsgp.py`). Wire one caller, or add a post-fit basis-sufficiency assertion.
- **One line in METHODS.md** acknowledging GB gain models are near-noise (note
  202606201500), so the DAG-motivated (not GB-discovered) smooths are a documented
  choice, not a gap in the explore->confirm loop.

## A verification note on the sign convention

The review's first draft flagged `METHODS.md`'s tau-sign wording as "stale". The
adversarial verifier **removed** that: on `main` (post-#117, `G = 2 - group`,
positive tau = benefit) the prose and code are consistent. The earlier
_pre-#117_ worktrees still carry `G = group - 1` (negative tau = benefit) and are
also internally consistent. There is nothing to fix in either place — recorded
here because it was a tempting-but-wrong "gap".

## Related

- `notes/202604181445-lrp52-gp-sensitivity.md` — the LRP52 GP-off decision (LOO vs funnel).
- `notes/202604181700-lrp55-age-gp-drop.md`, `notes/202604181730-mechanism-models-age-gp-drop-and-docs.md`.
- `notes/202606161416-lrp72-phonics-route-decoding.md` — link-saturation on floored counts.
- `notes/202606201500-gb-replication-findings.md` — GB gain models near-noise.
- `notes/202606251124-lrpitt-floored-outcomes-nonword-spelling.md` — the floored-outcome estimand.
