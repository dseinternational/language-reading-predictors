# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared design validation for mechanism settings and factory calls.

This module avoids factory dependencies so both entry points can use the same
cross-field rules without an import cycle.
"""

from __future__ import annotations

from language_reading_predictors.statistical_models.itt import KAPPA_PRIOR_FAMILIES

__all__ = ["validate_mechanism_design"]


def validate_mechanism_design(
    *,
    linear_mechanism: bool,
    phase_specific_mechanism: bool,
    phase_varying_slope: bool,
    decompose_between_within: bool,
    mechanism_is_covariate: bool,
    mechanism_at_pre: bool,
    moderator_symbol: str | None,
    moderator_is_covariate: bool,
    mech_hsgp_m: int | None,
    hsgp_lengthscale_declared: bool,
    kappa_prior_family: str,
    default_hsgp_m: int | None = None,
) -> None:
    """Reject every mechanism design the family cannot build **as declared**.

    ``hsgp_lengthscale_declared`` is the one argument the two entry points spell
    differently: the settings declare ``mech_lengthscale_tight`` (a flag selecting
    a thinner short-lengthscale tail), the factory takes the resolved
    ``mech_lengthscale_prior`` object. Both mean "an HSGP lengthscale setting was
    asked for", which is what the rules below are about.

    ``default_hsgp_m`` only enriches the basis-count message with the shared
    default the factory would otherwise have used.

    Raises ``TypeError`` for a value of the wrong type and ``ValueError`` for a
    combination that is well-typed but not constructible.
    """

    if mech_hsgp_m is not None:
        if isinstance(mech_hsgp_m, bool) or not isinstance(mech_hsgp_m, int):
            raise TypeError("mech_hsgp_m must be a positive integer or None")
        if mech_hsgp_m < 1:
            # Resolved with ``is None``, not ``or``, so a mistyped falsy value is
            # never read as "use the shared default": ``mech_hsgp_m=0`` is a
            # misconfiguration, not a request for the default basis count.
            default = (
                f" (or None for the shared default {default_hsgp_m})" if default_hsgp_m is not None else " (or None)"
            )
            raise ValueError(f"mech_hsgp_m must be a positive HSGP basis count{default}; got {mech_hsgp_m!r}.")

    if kappa_prior_family not in KAPPA_PRIOR_FAMILIES:
        raise ValueError(
            f"kappa_prior_family must be one of {sorted(KAPPA_PRIOR_FAMILIES)}, got {kappa_prior_family!r}"
        )

    if moderator_is_covariate and moderator_symbol is None:
        raise ValueError("moderator_is_covariate requires moderator_symbol")

    if mechanism_at_pre and mechanism_is_covariate:
        raise ValueError(
            "mechanism_at_pre is incompatible with mechanism_is_covariate: a "
            "standardised covariate exposure has no separate period-start score."
        )

    if linear_mechanism and (mech_hsgp_m is not None or hsgp_lengthscale_declared):
        raise ValueError("linear_mechanism cannot declare HSGP basis or lengthscale settings")

    if linear_mechanism and phase_specific_mechanism:
        raise ValueError(
            "linear_mechanism cannot be combined with phase_specific_mechanism; "
            "the factory's linear branch would silently ignore the phase-specific "
            "declaration"
        )

    # These sensitivities split or vary a scalar linear slope, not an HSGP curve.
    if decompose_between_within and not linear_mechanism:
        raise ValueError(
            "decompose_between_within requires linear_mechanism=True: a "
            "between/within split of a nonparametric curve is a separate design "
            "question, not a reparameterisation of this one"
        )
    if phase_varying_slope and not linear_mechanism:
        raise ValueError(
            "phase_varying_slope requires linear_mechanism=True; a per-period "
            "HSGP curve is phase_specific_mechanism, which this family cannot "
            "report"
        )
    if phase_varying_slope and phase_specific_mechanism:
        raise ValueError(
            "phase_varying_slope and phase_specific_mechanism are mutually "
            "exclusive: the first varies one linear slope by period, the second "
            "builds a separate curve per period"
        )

    # Moderation still uses the pooled exposure, so these sensitivities cannot
    # also promise decomposed or period-specific interactions.
    if moderator_symbol is not None and (decompose_between_within or phase_varying_slope):
        which = "decompose_between_within" if decompose_between_within else "phase_varying_slope"
        raise ValueError(
            f"{which} cannot be combined with moderator_symbol: the interaction "
            "term would still be built on the pooled exposure"
        )
