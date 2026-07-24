<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Repair: `mech-190` blending → word-reading knee-test (issue #430)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8). Records the reparameterisation that clears `mech-190`'s 31 reporting-tier divergences (a geometry funnel, not a mixing failure) so the blending → word-reading curve is usable, and the honest reading of the resulting curve: no knee is resolved for blending at this sample size. Also fixes the misleading docstring alias headers in `mech-190`/`mech-191`.

## The problem

`lrp-rli-mech-190` — the only model in the suite whose focal exposure is phoneme blending (`B`) with word reading (`W`) as the outcome, and so the only one that fits that relationship as a **flexible curve** rather than a single adjusted coefficient — failed the sampling-quality gate on divergences alone. From the stored reporting fit: **31 divergences**, but max R-hat 1.0015, min ESS 2621 and BFMI 0.92–0.94 all healthy. Mixing was fine and the chains were long enough, so this was a **boundary-geometry problem, not a length problem** — the same diagnosis and the same class of fix as #265, which reparameterised the HSGP mechanism curves for `mech-058`/`071`/`158`. `target_accept` was already at 0.999 (per `mech-058`), so raising it was not available.

## The fix — a thin-support HSGP reparameterisation

`mech-058`/`071`/`158` were cleared by #265's shared defaults — a standardised input, `_MECH_HSGP_M = 10` basis functions, and an `InverseGamma(5, 5)` mechanism lengthscale prior. Blending has far thinner support than letter sounds: a 10-item three-alternative picture-pointing task with a chance floor near 3.3 and roughly 19% of children at ceiling by t4, against letter sounds' 32 items. So it needs **more** taming than the shared defaults, and that taming must be **model-specific** — moving the global default would regress the models #265 already fixed.

Two opt-in knobs, both defaulting to the shared behaviour so every other mechanism model is byte-identical:

- **Fewer basis functions** — `mech_hsgp_m = 6` (from 10). A lower-rank HSGP shrinks the parameter space feeding the funnel; at n ≈ 53 with a smooth curve, 6 is ample resolution for the blending support.
- **A tighter lengthscale prior** — `mech_lengthscale_tight` selects `ell_prior_mech_tight()` = `InverseGamma(8, 8)` (vs the default `InverseGamma(5, 5)`). This barely moves the central lengthscale (mode 0.89 vs 0.83) but roughly halves the spread (sd ≈ 0.47 vs ≈ 0.72), thinning the short-lengthscale tail that drives the wiggle-and-diverge geometry. It is a genuine tightening, not a flat prior: the mode is essentially unchanged, so real curvature is still permitted where the data support it.

The HSGP is already non-centred (a basis-coefficient parameterisation), so that #265 lever was already spent; the remaining two levers are the ones applied here.

## Result

Refitted at `--config reporting`:

| Check | Baseline (m=10, IG(5,5)) | Repaired (m=6, IG(8,8)) |
| --- | --- | --- |
| divergences | **31** (fail) | **0** (pass) |
| max R-hat | 1.0015 | 1.0029 |
| min ESS | 2621 | 2699 |
| gate | fail | **PASS** |

The reparameterisation clears the geometry without forcing the curve flat. The fitted curve is then **flat and wide on its own terms**: the posterior-mean amplitude spans ≈ 0.14 on the logit scale against an 89% band ≈ 0.45 wide throughout, with near-zero slope in both halves (≈ 0.03 lower, ≈ 0.01 upper). So the honest knee-test answer is that **no knee is resolved** for blending at this sample size and measurement support — unlike the letter-sound knee `mech-058` resolves at ≈ 29.5 of 32 (itself wide and partly manufactured by the bounded scale). The curve should be read as "shape unresolved", not "shape flat"; it carries the same bounded-scale / logit-link caveat as `mech-058`.

This satisfies the issue's decision tree: the linear-fallback route (`linear_mechanism=True`) is reserved for when the geometry **cannot** be fixed, and here it can. Keeping the curve lets the report show *why* the knee is unanswerable (the flat, wide band is the evidence) rather than sidestepping to a linearity assumption that cannot test for a knee at all.

## Second item — docstring alias headers

`lrp_rli_mech_190.py` opened `"""LRP91 ...` and `lrp_rli_mech_191.py` opened `"""LRP92 ...`, but `lrp91` is the registry alias of `lcsm-091` (the change-on-change LCSM) and `lrp92` is a registered mediation model — a live trap for anyone grepping by alias. The correct aliases are `lrp190`/`lrp191`, which `definitions.MODEL_REGISTRY` already carries. Both headers are corrected, and a sweep of all 50 module docstrings that carry an `LRP<n>` header (comparing each against `model_ids.to_legacy` of its own canonical id) confirms no other module claims an alias that resolves to a different model.

## Files

- `src/.../statistical_models/priors.py` — new `ell_prior_mech_tight()` (`InverseGamma(8, 8)`).
- `src/.../statistical_models/factories.py` — `build_mechanism_model` gains `mech_hsgp_m` / `mech_lengthscale_prior` (both default to the shared behaviour; applied to the standard and phase-specific `f_mech` builds).
- `src/.../statistical_models/pipeline.py` — `fit_mechanism` reads `mech_hsgp_m` / `mech_lengthscale_tight` from `spec.extra` and forwards them.
- `src/.../statistical_models/lrp_rli_mech_190.py` — SPEC sets `mech_hsgp_m = 6` + `mech_lengthscale_tight = True`; docstring header + body updated. `lrp_rli_mech_191.py` — docstring header corrected.
- `docs/models/lrp-rli-mech-190/index.qmd` — overview prose updated to the "shape unresolved" reading.
- `tests/statistical_models/test_factories.py` — the knobs default to the shared behaviour, verified byte-identical by the existing mechanism factory tests.

## Related

- #265 — HSGP reparameterisation precedent (`mech-058`/`071`/`158`).
- `notes/202607241600-findings-word-reading-bands.md` — pass 4 (raised the repair; updated with this result).
- `notes/202607171215-findings-skill-thresholds.md` — the letter-sound knee the blending curve was to be compared against; six exposures swept, only `LS` has a resolved knee.
