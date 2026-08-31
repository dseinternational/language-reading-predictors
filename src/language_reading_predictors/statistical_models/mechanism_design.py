# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One cross-field design validator for the ``mechanism`` family (#637 stage 1).

``MechanismModelSettings`` and :func:`factories.build_mechanism_model` each
carried their own list of forbidden design combinations, and the lists had drifted
apart in both directions:

* the settings rejected ``linear_mechanism=True`` with
  ``phase_specific_mechanism=True``; the direct factory did not, and its ``if
  linear_mechanism:`` branch runs first — so a caller asking for per-period curves
  got one pooled ``beta_mech`` slope and no warning;
* the factory rejected ``mechanism_at_pre`` beside ``mechanism_is_covariate``; the
  settings did not, so that combination survived resolution and failed only after
  the output directory had been reset and the data loaded — the ordering #455
  exists to prevent.

Both entry points now call :func:`validate_mechanism_design`, so a design is
rejected identically whichever way it is declared, and a new rule cannot be added
to one path alone.

The module deliberately depends on nothing that depends on ``factories``: the
settings layer imports the factory, so a shared validator living in either would
close a cycle.
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
                f" (or None for the shared default {default_hsgp_m})"
                if default_hsgp_m is not None
                else " (or None)"
            )
            raise ValueError(
                "mech_hsgp_m must be a positive HSGP basis count"
                f"{default}; got {mech_hsgp_m!r}."
            )

    if kappa_prior_family not in KAPPA_PRIOR_FAMILIES:
        raise ValueError(
            "kappa_prior_family must be one of "
            f"{sorted(KAPPA_PRIOR_FAMILIES)}, got {kappa_prior_family!r}"
        )

    if moderator_is_covariate and moderator_symbol is None:
        raise ValueError("moderator_is_covariate requires moderator_symbol")

    if mechanism_at_pre and mechanism_is_covariate:
        raise ValueError(
            "mechanism_at_pre is incompatible with mechanism_is_covariate: a "
            "standardised covariate exposure has no separate period-start score."
        )

    if linear_mechanism and (mech_hsgp_m is not None or hsgp_lengthscale_declared):
        raise ValueError(
            "linear_mechanism cannot declare HSGP basis or lengthscale settings"
        )

    if linear_mechanism and phase_specific_mechanism:
        raise ValueError(
            "linear_mechanism cannot be combined with phase_specific_mechanism; "
            "the factory's linear branch would silently ignore the phase-specific "
            "declaration"
        )

    # #603 / #604: both sensitivities restructure the single linear slope. On an
    # HSGP design there is no scalar to split or vary, so the declaration could
    # only be honoured by building a different model than the one declared.
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

    # Neither sensitivity carries the moderation terms. ``gamma_int`` multiplies
    # the *undecomposed* standardised exposure, so a moderated split fit would
    # report a between/within decomposition beside an interaction built on the
    # blend it exists to reject; a moderated period-varying fit would likewise vary
    # the main slope by period while its interaction stayed pooled. Both are
    # coherent designs, but neither is this one, and the report would describe the
    # wrong model.
    if moderator_symbol is not None and (
        decompose_between_within or phase_varying_slope
    ):
        which = (
            "decompose_between_within"
            if decompose_between_within
            else "phase_varying_slope"
        )
        raise ValueError(
            f"{which} cannot be combined with moderator_symbol: the interaction "
            "term would still be built on the pooled exposure"
        )
