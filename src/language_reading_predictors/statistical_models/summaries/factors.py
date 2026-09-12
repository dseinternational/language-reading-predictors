# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Factors calculations and summaries."""

from __future__ import annotations

from language_reading_predictors.statistical_models.posteriors import REPORTING_CI_PROB

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr
from dse_research_utils.statistics.evidence import (
    evidence_label,
    favoured_direction,
)
from scipy.special import expit
from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
    apply_score_mean_link,
)
from language_reading_predictors.statistical_models.posteriors import (
    band50,
)


def factor_summary(
    trace: xr.DataTree,
    coef_names: list[str],
    *,
    ci_prob: float,
    causal_terms: tuple[str, ...] = (),
    role_overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Per-coefficient posterior summary for a factor model (LRPGF / LRPLF, #127).

    One row per coefficient in ``coef_names`` present in the trace: posterior
    ``median`` (the house headline statistic), posterior ``mean`` (secondary),
    equal-tailed central interval at coverage ``ci_prob`` (``lo``/``hi``, same
    convention as :func:`tau_summary_itt`), and ``prob_positive`` =
    ``P(coef > 0)``. The ``role`` column labels each term **causal** (the
    randomised treatment terms named in ``causal_terms``) or **association** —
    under the locked DAG every non-randomised coefficient is an adjusted
    association confounded by latent general ability and must never be read as
    "drives". ``role_overrides`` (term or base name -> role) names the further
    roles the level family carries under its t1-referenced arm-gap
    parameterisation (#552): ``balance`` for the pre-randomisation t1 arm gap,
    ``levels_view`` for the derived per-wave arm gaps, and ``regime`` for the
    t3/t4 arm-gap changes — randomised early-start-versus-delayed-start
    treatment-schedule contrasts, not treated-versus-untreated effects and not
    ordinary adjusted associations (#631 finding 13; the DiD arm_gap_t3 idiom).

    A vector coefficient is expanded to one row per element, labelled by the
    element's coordinate value (``b_grp_time[1]`` for the integer ``phase``
    coordinate, ``d_grp_time[t2]`` for the labelled ``post_phase`` one) — the
    same label ArviZ's summaries and ``psense_summary.csv`` use, so a focal term
    resolves to the same string everywhere.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    overrides = dict(role_overrides or {})

    def _row(term: str, base: str, d: np.ndarray) -> dict[str, object]:
        causal = term in causal_terms or base in causal_terms
        if causal:
            role = "causal"
        else:
            role = overrides.get(term, overrides.get(base, "association"))
        prob_pos = float(np.mean(d > 0))
        lo50, hi50 = band50(d)
        return {
            "term": term,
            "role": role,
            "median": float(np.median(d)),
            "mean": float(np.mean(d)),
            "lo": float(np.quantile(d, lo_q)),
            "hi": float(np.quantile(d, hi_q)),
            "lo50": lo50,
            "hi50": hi50,
            "prob_positive": prob_pos,
            "direction_label": evidence_label(prob_pos),
            **favoured_direction(prob_pos),
        }

    rows: list[dict[str, object]] = []
    for name in coef_names:
        if name not in posterior:
            continue
        da = posterior[name]
        extra_dims = [dd for dd in da.dims if dd not in ("chain", "draw")]
        if not extra_dims:
            d = da.stack(sample=("chain", "draw")).values.ravel()
            rows.append(_row(name, name, d))
        else:
            # Vector coefficient (e.g. the level model's per-timepoint b_grp_time):
            # one row per element, so an element can be labelled causal on its own
            # (e.g. only the t2 group contrast is the clean randomised effect).
            dim = extra_dims[0]
            labels = (
                [str(v) for v in da.coords[dim].values]
                if dim in da.coords
                else [str(i) for i in range(int(da.sizes[dim]))]
            )
            for i, label in enumerate(labels):
                d = da.isel({dim: i}).stack(sample=("chain", "draw")).values.ravel()
                rows.append(_row(f"{name}[{label}]", name, d))
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class AssociationTerm:
    """One adjusted-association covariate for the gain-factor items-scale marginals (#310).

    Describes how a single covariate enters the gain-factor linear predictor, so
    :func:`association_marginals` can push a ``+1 SD`` (and, for bounded-count
    covariates, a ``+k items``) perturbation of it through the fitted posterior onto
    the probability / items scales — the covariate analogue of the treatment
    marginal. The pipeline (which holds the prepared design) builds these; the
    reporting helper stays agnostic about the gain-factor internals.

    Attributes
    ----------
    label
        Human covariate name for the report row (e.g. ``"own"``, ``"L"``, ``"age"``).
    coef
        Posterior variable name of the covariate's main-effect coefficient
        (e.g. ``"gamma_own"``, ``"gamma_A"``, ``"gamma_ability"``, ``"gamma_L"``).
    main_scale
        Data-scale shift of the covariate's *main-effect* input per ``+1`` standardised
        unit (``b_t``). For age / cognitive ability the main effect already enters on
        the standardised scale, so ``main_scale = 1``. For the own baseline and skill
        baselines the main effect enters on the **raw logit** scale while the fitted
        interactions use the standardised vector, so ``main_scale`` is the SD of that
        raw logit — ``+1 SD`` shifts the raw-logit input by ``main_scale``.
    interactions
        ``(gamma_int_name, z_partner)`` pairs for every fitted interaction this term
        participates in. Because the interaction inputs are plain elementwise products
        of standardised vectors (``z_a · z_b``), a ``+1`` shift in this term's
        standardised value changes the product by exactly the partner's standardised
        vector — so the per-row interaction contribution to ``Δη`` is
        ``gamma_int · z_partner``. Treatment interactions are included: the covariate
        marginal holds the treatment indicator fixed and perturbs the covariate, so a
        ``trt × covariate`` term does move with the covariate.
    n_items
        Denominator of the covariate when it is a bounded-count measure (own / skill
        baselines); enables the ``+k items`` variant. ``None`` for age / ability /
        continuous adjusters.
    mean_prop
        Mean baseline proportion of a bounded-count covariate on the fitted rows — the
        operating point at which the ``+k items`` perturbation is evaluated (the logit
        shift for ``+k items`` is level-dependent, so it is anchored at the mean).
    sd_items
        Informational: how many items ``+1 SD`` of a bounded-count covariate is,
        evaluated at ``mean_prop`` — so a reader can translate the opaque ``+1 SD``.
    perturbation_label
        Optional override for the ``scale`` column of the unit perturbation row
        (default ``"+1 SD"``). Used when the term is not a standardised continuous
        covariate — e.g. the gain family's off-floor binary off-floor-at-pre
        indicator, whose ``+1`` perturbation is the at-floor -> off-floor switch
        (#391 finding 2 decision) and would be mislabelled as a ``+1 SD`` shift.
    toggle_vector
        The observed 0/1 indicator vector (aligned with ``eta``'s ``obs_id`` axis)
        for a **binary** covariate entered raw — the off-floor path's
        off-floor-at-pre indicator. When set, the marginal uses the
        net-out-and-toggle idiom of :func:`_itt_ame_draws`: per row the observed
        contribution ``x_i·Δη_i`` is removed (``η0 = η − x_i·Δη_i``, exact because
        the main effect and any interaction product are linear in the indicator)
        and the full 0 -> 1 switch is contrasted at that baseline for every row.
        The default forward shift ``expit(η + Δη) − expit(η)`` would instead
        evaluate an out-of-support 1 -> 2 move on rows whose indicator is already
        1, understating the switch the label promises (gain-factors code review
        2026-08-20, finding 2). ``None`` (default) keeps the forward-shift
        convention, which IS the documented estimand for standardised continuous
        covariates. Incoherent with ``n_items`` (a ``+k items`` increment of a
        0/1 indicator has no meaning), and rejected loudly if both are set.
    """

    label: str
    coef: str
    main_scale: float
    interactions: tuple[tuple[str, np.ndarray], ...] = ()
    n_items: int | None = None
    mean_prop: float | None = None
    sd_items: float | None = None
    perturbation_label: str | None = None
    toggle_vector: np.ndarray | None = None
    #: Per-measure items increment for the bounded-count companion row (#575
    #: finding 3): ``None`` falls back to :func:`association_marginals`'
    #: ``k_items`` argument. A fixed suite-wide ``+5`` was a third of a 6-item
    #: scale and half a 10-item one; per-measure ``max(1, round(n/10))`` matches
    #: the concurrent family's convention.
    k_items: int | None = None


def association_marginals(
    trace: xr.DataTree,
    *,
    terms: Sequence[AssociationTerm],
    n_trials: int,
    off_floor: bool = False,
    k_items: int = 5,
    eta_name: str = "eta",
    ci_prob: float = REPORTING_CI_PROB,
    row_mask: np.ndarray | None = None,
    group: str = "posterior",
    score_mean_link: ScoreMeanLink = "logit",
) -> pd.DataFrame:
    """Per-covariate items-scale association marginals for the gain family (#310).

    The adjusted-association analogue of :func:`treatment_marginal_effect`: for each
    covariate in ``terms`` it forms the per-draw change in the linear predictor from a
    ``+1 SD`` perturbation of that covariate, holding everything else at its observed
    value, and averages the response-scale change ``m(η + Δη) − m(η)`` over
    observations, where ``m`` is the fitted score mean ``score_mean_link ∘ expit``.
    Reported on the probability and items scales (``n_trials`` ×
    probability), with an equal-tailed ``ci_prob`` interval and an inner 50 % band.

    ``score_mean_link`` must be the link the model was **built** with: under the
    phoneme-blending guessing floor the same ``Δη`` maps to a smaller response-scale
    change than under the ordinary logit, so summarising a floor-link posterior at
    the default would overstate every association in items (#596).

    Per draw ``s`` and observation ``i`` the perturbation's linear-predictor shift is

        Δη_{i,s} = γ_{c,s} · (main_scale) + Σ_k γ^{int}_{k,s} · z^{partner}_{k,i},

    i.e. the covariate's main-effect coefficient scaled to the ``+1 SD`` data shift,
    plus each fitted interaction's contribution (the interaction inputs are elementwise
    products of standardised vectors, so a ``+1`` standardised shift moves the product
    by the partner's standardised vector). For a continuous covariate the contrast is
    the **forward shift** from each row's observed ``η`` — the documented estimand.
    A **binary 0/1 indicator** term (``toggle_vector`` set, e.g. the off-floor path's
    off-floor-at-pre indicator) instead uses the treatment marginal's full
    "net out and toggle" idiom (:func:`_itt_ame_draws`): the observed contribution is
    removed per row and the 0 -> 1 switch contrasted at that baseline, since the
    forward shift would evaluate an out-of-support 1 -> 2 move on rows already at 1
    (gain-factors code review 2026-08-20, finding 2).

    For **bounded-count** covariates (``n_items`` set) a second ``+{k_items} items`` row
    is emitted, evaluated at the covariate's mean baseline proportion (``mean_prop``):
    the raw-logit shift ``Δraw = logit(p̄ + k/N) − logit(p̄)`` replaces the ``+1 SD``
    shift, and the interaction contribution scales by ``Δz = Δraw / main_scale`` (the
    same shift in standardised units). ``+1 SD`` is opaque to readers; ``+k items`` is
    the interpretable companion.

    For ``off_floor`` outcomes (``n_trials`` should be passed as ``1``) the items scale
    collapses to the off-floor probability delta, mirroring the treatment marginal's
    floor-rule handling.

    ``row_mask`` (default ``None`` = **all** stacked rows): the covariate associations
    are descriptive, so the natural averaging population is every fitted observation —
    unlike the treatment marginal, which restricts to the randomised period-1 rows. The
    choice is pre-specified in the design note and recorded in ``config.json``.

    Every row carries ``role = "association"`` — none of these terms is causal, per the
    gain family's documented estimand structure.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
    )

    posterior = getattr(trace, group)
    eta = (
        posterior[eta_name]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    n_obs = eta.shape[0]

    mask: np.ndarray | None = None
    if row_mask is not None:
        m = np.asarray(row_mask)
        if m.ndim != 1:
            raise ValueError(f"row_mask must be 1-D, got a {m.ndim}-D array.")
        if m.dtype == bool:
            if m.shape[0] != n_obs:
                raise ValueError(
                    f"boolean row_mask has {m.shape[0]} entries but eta has "
                    f"{n_obs} observations."
                )
        elif np.issubdtype(m.dtype, np.integer):
            if m.size and (int(m.min()) < 0 or int(m.max()) >= n_obs):
                raise ValueError(f"integer row_mask has indices outside [0, {n_obs}).")
        else:
            raise ValueError(
                "row_mask must be a boolean mask or integer index array, got dtype "
                f"{m.dtype}."
            )
        mask = m

    eta_sel = eta if mask is None else eta[mask]
    if eta_sel.shape[0] == 0:
        raise ValueError("row_mask selects no observations for the marginal effect.")

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    rows: list[dict[str, float | str]] = []

    for term in terms:
        coef = posterior[term.coef].stack(sample=("chain", "draw")).values.ravel()  # (S,)

        if term.toggle_vector is not None and term.n_items:
            raise ValueError(
                f"{term.label!r}: toggle_vector marks a binary 0/1 indicator; a "
                "+k items perturbation is incoherent with it — set one or the other."
            )

        # (scale label, standardised shift Δz). +1 SD is Δz = 1; +k items maps the
        # bounded-count increment to standardised units at the mean operating point.
        # A term may override the unit label (e.g. a 0/1 indicator switch).
        perturbations: list[tuple[str, float]] = [
            (term.perturbation_label or "+1 SD", 1.0)
        ]
        if term.n_items and term.mean_prop is not None and term.main_scale > 0:
            # The fitted baselines are Haldane logits, log((y+0.5)/(n-y+0.5)),
            # whose proportion p* = (y+0.5)/(n+1) is affine in the count — so the
            # mean fitted proportion inverts EXACTLY to the mean baseline count.
            # The former arithmetic added k/N to p* and clipped at 1, which is
            # wrong for this transform and, at the ceiling, silently manufactured
            # a huge logit shift (a "+5 items" row of ~15 logits on the 6-item
            # nonword scale). The correct increment is the Haldane-logit
            # difference of feasible counts, with the increment capped at the
            # items the scale has left — the concurrent family's idiom
            # (#575 finding 3).
            n = float(term.n_items)
            y_mean = float(np.clip(term.mean_prop * (n + 1.0) - 0.5, 0.0, n))
            k_req = int(term.k_items if term.k_items is not None else k_items)
            k_eff = min(k_req, int(np.floor(n - y_mean)))
            if k_eff >= 1:
                dz = float(
                    logit_safe(np.asarray([y_mean + k_eff]), int(n))[0]
                    - logit_safe(np.asarray([y_mean]), int(n))[0]
                ) / term.main_scale
                perturbations.append((f"+{k_eff} items", dz))

        for scale_label, dz in perturbations:
            # Main-effect shift: γ_c scaled to the requested data increment. Broadcast
            # over observations (shape (1, S)); promoted to (n_obs, S) by interactions.
            delta_eta = (coef * (dz * term.main_scale))[None, :]
            for gi_name, z_partner in term.interactions:
                gi = posterior[gi_name].stack(sample=("chain", "draw")).values.ravel()  # (S,)
                zp = np.asarray(z_partner, dtype=float)
                if zp.shape[0] != n_obs:
                    raise ValueError(
                        f"interaction partner for {term.label!r}/{gi_name!r} has "
                        f"{zp.shape[0]} rows but eta has {n_obs} observations."
                    )
                delta_eta = delta_eta + np.outer(zp, gi) * dz  # (n_obs, S)

            de_sel = (
                delta_eta
                if delta_eta.shape[0] == 1
                else (delta_eta if mask is None else delta_eta[mask])
            )
            if term.toggle_vector is not None:
                # Binary-indicator toggle (gain-factors code review 2026-08-20,
                # finding 2): net the observed contribution out per row — exact,
                # because the main effect and any interaction product are linear in
                # the indicator — then contrast the full 0 -> 1 switch at that
                # baseline, mirroring _itt_ame_draws. The forward shift below would
                # evaluate an out-of-support 1 -> 2 move on rows already at 1,
                # understating the switch on the flattened part of the expit curve.
                x = np.asarray(term.toggle_vector, dtype=float)
                if x.shape[0] != n_obs:
                    raise ValueError(
                        f"toggle_vector for {term.label!r} has {x.shape[0]} rows "
                        f"but eta has {n_obs} observations."
                    )
                x_sel = x if mask is None else x[mask]
                eta_base = eta_sel - de_sel * x_sel[:, None]
            else:
                eta_base = eta_sel
            # Map both arms through the fitted score mean before differencing: under
            # a non-identity link the response-scale change is not the logit-scale
            # one rescaled, so differencing raw inverse-logits would report a
            # quantity the likelihood never modelled (#596).
            ame_prob = (
                apply_score_mean_link(expit(eta_base + de_sel), score_mean_link)
                - apply_score_mean_link(expit(eta_base), score_mean_link)
            ).mean(axis=0)  # (S,)
            ame_items = float(n_trials) * ame_prob
            prob_lo50, prob_hi50 = band50(ame_prob)
            items_lo50, items_hi50 = band50(ame_items)
            rows.append(
                {
                    "term": term.label,
                    "role": "association",
                    "scale": scale_label,
                    "prob_median": float(np.median(ame_prob)),
                    "prob_lo": float(np.quantile(ame_prob, lo_q)),
                    "prob_hi": float(np.quantile(ame_prob, hi_q)),
                    "prob_lo50": prob_lo50,
                    "prob_hi50": prob_hi50,
                    "items_median": float(np.median(ame_items)),
                    "items_lo": float(np.quantile(ame_items, lo_q)),
                    "items_hi": float(np.quantile(ame_items, hi_q)),
                    "items_lo50": items_lo50,
                    "items_hi50": items_hi50,
                    "prob_pos": float(np.mean(ame_items > 0)),
                    "off_floor": bool(off_floor),
                    "sd_items": (
                        float(term.sd_items)
                        if term.sd_items is not None
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)
