# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP68 - constrained random-intercept cross-lagged panel model (RI-CLPM).

The **within-child** triangulation of LRP67 (latent change) and LRP65
(between-child). It separates each child's stable trait from their within-child
fluctuations and asks the cross-lagged question directly: when a child is
temporarily **above their own expected letter-sounds**, do they make greater
**subsequent reading** gains?

Form (see :func:`factories.build_riclpm_model`), logit scale, observed scores
modelled directly (no measurement-error / latent-indicator layer at n~54)::

    m_m[i, t] = u_m[i] + w_m[i, t] + d_age_m * age[i, t] + d_dose_m * dose[i, t]
    w[i, t]   = A @ w[i, t-1] + innovation[i, t]            (time-invariant VAR(1))

``u_m[i]`` is the stable between-child trait (non-centred random intercept per
measure); ``w`` is the within-child deviation. ``A[target, source]`` holds the
autoregressive (diagonal) and cross-lagged (off-diagonal) coefficients, pooled
across the three transitions. The headline is ``A[W, L]`` (letter-sounds ->
reading within child).

Competing-structure comparison (the deliverable, by PSIS-LOO):

1. ``ar``         - AR-only: is apparent prediction just stability?
2. ``l_to_r``     - + letter-sounds -> reading (``A[W, L]``).
3. ``r_driven``   - + reading -> letter-sounds / vocabulary (reverse / practice).
4. ``reciprocal`` - all cross-lagged paths (the headline model for diagnostics).

Constraints for tractability (all deliberate at n~54): time-invariant paths;
regularising ``Normal(0, 0.5)`` cross-lagged priors with a 0.3 / 0.7 sensitivity
refit; no measurement-error layer; covariates age + dose. **Expect wide
posteriors on the cross-lagged paths - that is the honest result.** If the
RI-CLPM will not sample cleanly even constrained, that is reported as the finding
rather than forced (watch R-hat / ESS / divergences).

Symbol mapping (codebase vs handoff R / L / V): W = reading (ewrswr), L =
letter-sounds (yarclet), E = expressive vocabulary (eowpvt). ``R`` is receptive
vocabulary in ``MEASURES`` and is not reused; the handoff's "L -> R" is ``A[W, L]``.

Coherence: the cross-lagged paths are the shared DAG's developmental edges
(language -> letter-sounds -> reading) unrolled over time; variable roles are kept
consistent with that DAG and with LRP65 / LRP67.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_riclpm

SPEC = ModelSpec(
    model_id="lrp68",
    kind="riclpm",
    title=(
        "Constrained RI-CLPM: within-child cross-lagged letter-sounds (L), "
        "vocabulary (E) and reading (W), with an AR-only vs cross-lagged LOO "
        "comparison"
    ),
    outcome_symbol="W",
    extra={
        "outcomes": ["W", "L", "E"],
        "letter_symbol": "L",
        # Competing within-RI-CLPM structures compared by LOO; reciprocal is the
        # headline model for the detailed posterior / diagnostics.
        "structures": ["ar", "l_to_r", "r_driven", "reciprocal"],
        "headline_structure": "reciprocal",
        # Regularising cross-lagged prior SD + the sensitivity refit values.
        "cross_prior_sigma": 0.5,
        "prior_sensitivity_sigmas": [0.3, 0.7],
    },
)


def fit(config: str = "dev"):
    return fit_riclpm(SPEC, config=config)
