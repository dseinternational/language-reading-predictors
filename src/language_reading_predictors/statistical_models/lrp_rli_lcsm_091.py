# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LCSM-091 - lagged change-on-change: do L / E *gains* predict later W *gains*?

The literal reading of #229 ("to what extent do word-reading gains depend on
gains in letter-sounds vs gains in expressive vocabulary"), built to the
signed-off specification ``notes/202607131530-lrp229-lcsm-change-change-spec.md``
(#296). LRP67 answers the prior-*level* form of that question (where a child
*stands* on L/E predicts how much reading grows next); this model adds the
prior-*change* form: does how much a child's letter-sounds / vocabulary just
*grew* predict how much their reading grows next, over and above their levels?

Model (the LRP67 McArdle latent change-score scaffold; see
:func:`factories.build_lcsm_model`). Reading-change equation, transitions
w = 1..3::

    Delta_W[i,w] = a_W[arm_i, w] + b_W * x_W[w-1]
                 + g_L * x_L[w-1] + g_E * x_E[w-1]        (prior level, as LRP67)
                 + h_L * Delta_L[w-1] + h_E * Delta_E[w-1]  (prior change, new; w >= 2)
                 + d_age * age[w-1]                         (+ process noise)

``h_L`` / ``h_E`` are the headline coefficients. ``Delta_c[w-1]`` is the
previous transition's **latent** change (the McArdle true-score layer is kept,
so measurement floors and noise stay out of the change estimates).

Two deviations from the merged spec sketch, both required by machinery that
landed after it was written (#250, ``notes/202607141030-time-lagged-model-designs.md``):

1. **Arm x window change intercepts** replace the pooled per-measure intercept.
   The h terms exist only on transitions 2-3 (the first transition has no prior
   change) — both post-crossover, exactly where the verified d-separation result
   says the randomised arm confounds every pooled coupling. The factory enforces
   this pairing. The window-1 cell contrast is reported as the usual ITT-suite
   consistency check (``itt_window1_contrast.csv``).
2. **No adjuster covariate block.** Unlike LCSM-081's reverse couplings, no
   fittable verified measured backdoor set exists for couplings *into* reading
   (the LCSM-082 derivation put it at ~13 nodes), so hs/rw/sp slopes would spend
   scarce parameters without closing any backdoor. The model stays in LRP67's
   exploratory-association class, and keeps its W/L/E shape for the side-by-side
   g-vs-h comparison.

**Reading rules** (spec assumption set; carry into the report):

- **Associational, post-crossover only.** ``h_L`` / ``h_E`` are identified
  entirely from the t2->t3 and t3->t4 transitions — two usable transitions at
  n ~ 54, so intervals will be wide. The deliverable is **direction agreement**
  with the Phase-0 randomised mediation (``med-059``/``064``), not a precise
  gain split.
- **Do not read a large ``h_E`` as "vocabulary growth drives reading growth"**:
  between- and within-child variance are not separable at this n (the RI-CLPM is
  not estimable; LRP67 docstring), so the couplings partly carry stable ability.
- Specification 3 (contemporaneous ``Delta_W ~ Delta_L + Delta_E``) is
  deliberately **not** fitted: same-window change-on-change is direction-
  ambiguous, occasion-correlated and regression-to-the-mean-prone.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_lcsm

SPEC = ModelSpec(
    model_id="lrp-rli-lcsm-091",
    kind="lcsm",
    title=(
        "Lagged change-on-change: prior letter-sound (L) and vocabulary (E) "
        "change as within-child predictors of subsequent reading (W) change"
    ),
    outcome_symbol="W",
    extra={
        # LRP67's measure set, kept for the side-by-side g-vs-h comparison.
        "outcomes": ["W", "L", "E"],
        # Prior-level couplings retained (the spec keeps the level terms) ...
        "couplings": {"W": ["L", "E"]},
        # ... plus the new prior-change couplings, the headline h_L / h_E.
        "lagged_change_couplings": {"W": ["L", "E"]},
        # Mandatory with pooled post-crossover couplings (see module docstring).
        "arm_window_intercepts": True,
        # Shared association scale (prior-critical-review 2026-07-07, rec. 3).
        "coupling_prior_sigma": 0.3,
        "use_process_noise": True,
        "shared_process_noise": False,
    },
)


def fit(config: str = "dev"):
    return fit_lcsm(SPEC, config=config)
