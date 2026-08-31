# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""g-formula mediation model construction (single, two-mediator, period-stacked).

Carved out of the 8,506-line ``factories.py`` by #637 stage 3, which is why
every name here is still re-exported from ``factories``. Every family module
depends only on :mod:`factories.base`; nothing crosses between families.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pymc as pm

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.fitted_payloads import (
    EmptyPayload,
    MediationPayload,
)
from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
    beta_binomial_from_logit,
    beta_binomial_from_score_mean_link,
)
from language_reading_predictors.statistical_models.mediation_parameter_names import (
    outcome_confounder_coefficient,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
)

@dataclass
class MediationData:
    """Row-aligned phase-0 arrays + mediator metadata for the g-formula.

    Carried alongside the BuiltModel so :func:`mediation.decompose` can
    re-simulate counterfactuals from the posterior using the exact inputs the
    model saw. ``mediator_kind`` selects which mediator sub-model to simulate:

    - ``"beta_binomial"`` (LRP59): a single count mediator (``L_t2`` out of
      ``n_trials_L``) conditioned on ``logit(L_t1)``; ``med_mean`` / ``med_sd``
      standardise its logit for the outcome model.
    - ``"gaussian_composite"`` (LRP62): a continuous standardised code-based-route
      composite; the baseline composite is ``M_pre_std`` and the mediator is
      drawn from a Normal, so the count-specific fields are unused.

    Confounders are carried generically in ``conf_logit`` (baseline t1 logits
    keyed by symbol), with ``confounder_symbols`` recording the fitted set, so
    :func:`mediation.decompose` adjusts for exactly the confounders the model
    was fitted with — no symbol can drift between the fit and the g-formula.
    """

    # Shared across mediator kinds.
    G: np.ndarray
    W1_logit: np.ndarray
    A_std: np.ndarray
    conf_logit: dict[str, np.ndarray]
    n_trials_W: int
    mediator_kind: str = "beta_binomial"
    confounder_symbols: tuple[str, ...] = ()
    #: The mediator's data symbol (L for LRP59, TE for LRP68, N for LRP74); the
    #: own-baseline coefficient node is ``a_{mediator_symbol}``.
    mediator_symbol: str = "L"
    # Beta-Binomial single mediator (LRP59).
    L1_logit: np.ndarray | None = None
    L2_count: np.ndarray | None = None
    W2_count: np.ndarray | None = None
    n_trials_L: int = 0
    med_mean: float = 0.0
    med_sd: float = 1.0
    # Gaussian composite mediator (LRP62).
    M_pre_std: np.ndarray | None = None
    route_symbols: tuple[str, ...] = ()
    #: Off-floor (Bernoulli) OUTCOME (#228 item 12, e.g. nonword N): the outcome leg
    #: is a Bernoulli on the off-floor indicator (post > 0) with no own-baseline term,
    #: so ``decompose`` reads no ``b_W`` and reports NIE/NDE on the off-floor
    #: risk-difference (probability) scale (``n_trials_W`` is set to 1, collapsing the
    #: ``words_*`` columns onto the risk difference). Default False = graded outcome.
    off_floor: bool = False
    #: Cross-leg baseline regressors restored by #585, keyed by coefficient name.
    #: ``decompose`` reads these so the counterfactual simulation conditions on
    #: exactly the common vector each fitted leg saw.
    mediator_cross_values: dict[str, np.ndarray] = field(default_factory=dict)
    outcome_cross_values: dict[str, np.ndarray] = field(default_factory=dict)
    #: Binary off-floor-at-baseline indicator for an off-floor outcome leg.
    own_offfloor: np.ndarray | None = None


def _baseline_confounder_value(prepared: PreparedData, symbol: str) -> np.ndarray:
    """Baseline value column for a mediation-family confounder (#246).

    Bounded-count measures enter on their t1 logit scale (``prepared.pre_logit``);
    the revised-DAG raw confounders that are not measures — hearing (``hs`` /
    ``hs_missing``), speech (``deapp_c``), phonological memory (``erbto``) — enter
    from ``prepared.covariates`` (already standardised, taken from the t1 pre-row in
    the ITT phase, so treatment-unaffected as the cross-world assumption requires).
    """
    if symbol in prepared.pre_logit:
        return prepared.pre_logit[symbol]
    return prepared.covariates[symbol]


def _cross_baseline_arrays(prepared: PreparedData, terms) -> dict[str, np.ndarray]:
    """Row-aligned t1 values for a mediation leg's cross-baseline terms (#585).

    ``terms`` are :class:`mediation_settings.BaselineTerm` records. A measure the
    fit declares off-floor enters as the **binary off-floor-at-baseline
    indicator** (``pre > 0``) rather than its near-degenerate floored logit,
    mirroring the project's ``gamma_own_offfloor`` convention. Call after any
    row subsetting so the arrays stay aligned with the fitted rows.
    """
    values: dict[str, np.ndarray] = {}
    for term in terms:
        if term.form == "offfloor":
            if term.symbol not in prepared.pre_counts:
                raise KeyError(
                    f"Off-floor baseline {term.symbol!r} has no pre-count column"
                )
            values[term.coefficient] = (
                np.asarray(prepared.pre_counts[term.symbol]) > 0
            ).astype(float)
        else:
            values[term.coefficient] = _baseline_confounder_value(prepared, term.symbol)
    return values


def _add_cross_baselines(eta, terms, values: dict[str, np.ndarray]):
    """Add a leg's cross-baseline regressors to ``eta`` inside a model context.

    Restores the common pre-exposure vector the g-formula integrates over: before
    #585 the mediator leg never saw the outcome baseline and the outcome leg never
    saw the mediator baseline, so the two design matrices implemented neither the
    declared adjustment set nor a documented reduction of it.
    """
    for term in terms:
        data = pm.Data(f"{term.coefficient}_x", values[term.coefficient], dims="obs_id")
        prior = (
            _priors.gamma_own_offfloor_prior()
            if term.form == "offfloor"
            else _priors.gamma_cross_prior()
        )
        eta = eta + prior.to_pymc(term.coefficient) * data
    return eta


def _build_outcome_leg(
    *,
    mediator_node,
    G_d,
    W1_d,
    A_d,
    conf_d: dict,
    confounder_symbols: Iterable[str],
    N_out,
    W2_count,
    outcome_kind: str = "beta_binomial",
    cross_baselines=(),
    cross_values: dict | None = None,
    own_offfloor=None,
    # #619: the phoneme-blending response link, applied to the OUTCOME's score mean.
    # A mediator is a separate leg with its own measure and is unaffected.
    score_mean_link: ScoreMeanLink = "logit",
):
    """Shared outcome leg for the single-mediator-design factories.

    Both LRP59 (:func:`build_mediation_model`) and LRP62
    (:func:`_build_route_composite_model`) regress ``logit(W_t2)`` on treatment,
    the standardised post mediator and its ``G`` interaction, baseline word
    reading, age, and the baseline confounders — identical save for
    ``mediator_node`` (``z_med`` for LRP59, the route composite for LRP62). Must
    be called inside an open ``pm.Model`` context so the nodes register.

    ``outcome_kind="beta_binomial"`` (default) fits the graded post count and is
    **byte-identical** to the original build. ``"bernoulli_offfloor"`` (#228 item 12,
    a heavily-floored outcome such as nonword N) instead fits a Bernoulli on the
    off-floor indicator ``post > 0`` (node ``y_offfloor``, no ``kappa_Y``) and
    **drops the own-baseline term** ``b_W * W1`` — mirroring the off-floor ITT / DiD /
    gain-factor convention (the ``Normal(1, ·)`` autoregressive prior does not
    transfer to a binary indicator, and a floored baseline logit is degenerate). In
    that case ``W1_d`` is unused (may be ``None``).
    """
    off_floor = outcome_kind == "bernoulli_offfloor"
    cross_values = dict(cross_values or {})
    b0 = _priors.alpha_prior().to_pymc("b0")
    b_G = _priors.tau_prior().to_pymc(
        "b_G",
        role="association",
        rationale=(
            "Randomised-arm coefficient in one g-formula leg, carried on the "
            "treatment prior tau ~ Normal(0, 0.5). Reported as an association: "
            "this family's causal deliverables are the NDE/NIE decomposition "
            "the legs compose, not a single leg's coefficient."
        ),
    )
    b_M = _priors.b_path_prior().to_pymc("b_M")
    b_GM = _priors.gamma_cross_prior().to_pymc("b_GM")
    if not off_floor:
        # Own-baseline coefficient — created before b_A so the graded path's free-RV
        # order (and therefore its sampling) is byte-identical to the original.
        b_W = _priors.gamma_own_prior().to_pymc("b_W")
    b_A = _priors.gamma_age_prior().to_pymc("b_A")
    if off_floor:
        # #585 finding 4: the off-floor outcome leg no longer drops its own
        # baseline outright. The graded Normal(1, 0.25) autoregressive prior still
        # does not transfer to a binary indicator, so the baseline enters as the
        # binary off-floor-at-baseline contrast (gamma_own_offfloor ~ Normal(0, 1))
        # — the same convention the off-floor ITT / DiD / gain-factor models use.
        # The sample rule and the likelihood now require the same measurement.
        b_own_ff = _priors.gamma_own_offfloor_prior().to_pymc("b_own_offfloor")
        own_off_d = pm.Data("own_pre_offfloor", own_offfloor, dims="obs_id")
        eta_Y = (
            b0
            + b_G * G_d
            + b_M * mediator_node
            + b_GM * (G_d * mediator_node)
            + b_own_ff * own_off_d
            + b_A * A_d
        )
    else:
        eta_Y = (
            b0
            + b_G * G_d
            + b_M * mediator_node
            + b_GM * (G_d * mediator_node)
            + b_W * W1_d
            + b_A * A_d
        )
    for s in confounder_symbols:
        b_c = _priors.gamma_cross_prior().to_pymc(
            outcome_confounder_coefficient(s)
        )
        eta_Y = eta_Y + b_c * conf_d[s]
    eta_Y = _add_cross_baselines(eta_Y, cross_baselines, cross_values)
    eta_Y = pm.Deterministic("eta", eta_Y, dims="obs_id")
    if off_floor:
        off = (np.asarray(W2_count) > 0).astype(np.int64)
        return pm.Bernoulli("y_offfloor", logit_p=eta_Y, observed=off, dims="obs_id")
    kappa_Y = _priors.kappa_prior().to_pymc("kappa_Y")
    return beta_binomial_from_score_mean_link(
        "y_post", eta_Y, n_trials=N_out, kappa=kappa_Y,
        score_mean_link=score_mean_link,
        observed=W2_count, dims="obs_id",
    )


def build_mediation_model(
    prepared: PreparedData,
    *,
    mediator_symbol: str = "L",
    outcome_symbol: str = "W",
    confounder_symbols: Iterable[str] = ("E", "R"),
    mediator_kind: str = "beta_binomial",
    route_symbols: Iterable[str] = (),
    outcome_kind: str = "beta_binomial",
    mediator_cross_baselines=(),
    outcome_cross_baselines=(),
    # #619: the phoneme-blending response link for the outcome leg. B outcomes only,
    # graded only, and released only beside the paired ordinary-link fit.
    score_mean_link: ScoreMeanLink = "logit",
) -> tuple[BuiltModel[MediationPayload], MediationData]:
    """Joint mediator + outcome model for the ITT-phase (phase 0) decomposition.

    ``mediator_kind`` selects the mediator sub-model:

    - ``"beta_binomial"`` (LRP59, default): a single count mediator
      (``mediator_symbol``, e.g. letter-sound L) — documented below.
    - ``"gaussian_composite"`` (LRP62): the mediator is an equal-weight
      standardised-logit composite of ``route_symbols`` (the code-based route,
      e.g. ``("L", "B")``), modelled as ``Normal(mu_M, sigma_M)``. The outcome
      model is identical to the LRP59 case; only the mediator leg changes. See
      :func:`_build_route_composite_model`.

    Two Beta-Binomial likelihoods on the logit scale share the randomised
    treatment ``G`` and a baseline-covariate adjustment set:

    - Mediator: ``logit(L_t2) ~ a0 + a_G·G + a_L·logit(L_t1) + sum a_c·C_t1``
    - Outcome:  ``logit(W_t2) ~ b0 + b_G·G + b_M·z(logit L_t2)
                  + b_GM·G·z(logit L_t2) + b_W·logit(W_t1) + sum b_c·C_t1``

    The ``G×M`` interaction (``b_GM``) is included so the natural direct/indirect
    decomposition admits exposure-mediator interaction (the general g-formula
    case). NDE/NIE are NOT read off coefficients — they are computed by
    counterfactual simulation from the posterior (see ``mediation.decompose``);
    this factory only builds the joint likelihood and returns the row-aligned
    inputs needed for that simulation.

    Confounders ``C`` are taken at **baseline (t1)** on their logit scale, not at
    post (t2): a mediator-outcome confounder must not itself be affected by
    treatment (the cross-world assumption), so post-treatment values are
    inadmissible here. Documented in the report.

    The resulting NDE/NIE are **not identified natural effects**: beyond the
    latent-general-ability confounding of the mediator->outcome path, dose ``IS``
    is a treatment-induced (exposure-induced) mediator-outcome confounder, so the
    decomposition is model-based under stated (cross-world) assumptions. An
    interventional estimand (``decompose(..., interventional=True)``;
    MED-078/186/187) drops
    the cross-world requirement and so escapes the ``IS`` obstacle, but it still
    assumes no unmeasured mediator-outcome confounding, which latent general
    ability violates here — a weaker-assumption target, not an identified one. See
    the :mod:`mediation` module docstring and the report assumptions sections.

    Requires ``prepared.phase_mode == 'itt'`` (the single randomised t1->t2
    transition; one row per child, so no subject random intercept).
    """
    if prepared.phase_mode != "itt":
        raise ValueError("Mediation factory requires phase_mode='itt'")
    confounder_symbols = tuple(confounder_symbols)
    if mediator_kind == "gaussian_composite":
        return _build_route_composite_model(
            prepared,
            outcome_symbol=outcome_symbol,
            confounder_symbols=confounder_symbols,
            route_symbols=tuple(route_symbols),
            mediator_cross_baselines=tuple(mediator_cross_baselines),
            outcome_cross_baselines=tuple(outcome_cross_baselines),
            score_mean_link=score_mean_link,
        )
    if mediator_kind != "beta_binomial":
        raise ValueError(f"Unknown mediator_kind {mediator_kind!r}")
    if outcome_kind not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(f"Unknown outcome_kind {outcome_kind!r}")
    off_floor = outcome_kind == "bernoulli_offfloor"
    # Both legs need both baselines (#585 finding 1): the mediator law conditions on
    # the outcome baseline and the outcome law on the mediator baseline, so the
    # g-formula integrates over one common pre-exposure vector. An off-floor outcome
    # still needs its pre-score — as the binary off-floor-at-baseline indicator
    # (#585 finding 4), not as a degenerate floored logit.
    required_pre = (mediator_symbol, outcome_symbol)
    for s in required_pre:
        if s not in prepared.pre_logit:
            raise KeyError(f"Symbol {s!r} missing from prepared data")
    if outcome_symbol not in prepared.post_counts:
        raise KeyError(f"Outcome {outcome_symbol!r} post-count missing from prepared data")
    for s in confounder_symbols:
        if s not in prepared.pre_logit and s not in prepared.covariates:
            raise KeyError(f"Confounder {s!r} not in prepared pre_logit or covariates")

    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    med_post = prepared.post_counts[mediator_symbol]
    out_post = prepared.post_counts[outcome_symbol]
    keep = ~(np.isnan(med_post) | np.isnan(out_post))
    if not keep.all():
        prepared = _subset(prepared, keep)

    N_med = prepared.n_trials[mediator_symbol]
    N_out = prepared.n_trials[outcome_symbol]
    L2_count = prepared.post_counts[mediator_symbol].astype(np.int64)
    W2_count = prepared.post_counts[outcome_symbol].astype(np.int64)

    # Standardised mediator (z of logit L_t2) — the regressor in the outcome
    # model; the standardiser is reused for counterfactual mediator draws.
    med_logit = logit_safe(L2_count, N_med)
    z_med, med_scaler = standardise(med_logit)

    L1 = prepared.pre_logit[mediator_symbol]
    # Off-floor outcome: the graded logit is still unusable, so ``W1`` stays a
    # never-referenced placeholder there and the baseline enters through
    # ``own_offfloor`` instead (#585 finding 4).
    W1 = np.zeros(prepared.n_obs) if off_floor else prepared.pre_logit[outcome_symbol]
    own_offfloor = (
        (np.asarray(prepared.pre_counts[outcome_symbol]) > 0).astype(float)
        if off_floor
        else None
    )
    conf_logit = {
        s: _baseline_confounder_value(prepared, s) for s in confounder_symbols
    }
    mediator_cross_baselines = tuple(mediator_cross_baselines)
    outcome_cross_baselines = tuple(outcome_cross_baselines)
    med_cross_values = _cross_baseline_arrays(prepared, mediator_cross_baselines)
    out_cross_values = _cross_baseline_arrays(prepared, outcome_cross_baselines)

    coords = {"obs_id": np.arange(prepared.n_obs)}
    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        G_d = pm.Data("G", G_f, dims="obs_id")
        # Mediator baseline / own-baseline coef / likelihood are parameterised by
        # ``mediator_symbol`` so a non-L mediator (LRP68 TE, LRP74 N) gets correctly
        # labelled nodes; when mediator_symbol == 'L' every name is byte-identical to
        # the original LRP59 build.
        L1_d = pm.Data(f"{mediator_symbol}_pre_logit", L1, dims="obs_id")
        # Outcome own-baseline data node only for the graded path; the off-floor
        # outcome leg drops the b_W term, so no baseline node is created. The node
        # is named by ``outcome_symbol`` (was hardcoded "W_pre_logit"): byte-identical
        # for every W-outcome model, and distinct from the mediator node when the
        # mediator is itself word reading (LRP176 reverse WR->LS direction test),
        # which the hardcoded name collided with.
        W1_d = None if off_floor else pm.Data(f"{outcome_symbol}_pre_logit", W1, dims="obs_id")
        A_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        conf_d = {
            s: pm.Data(f"{s}_pre_logit", conf_logit[s], dims="obs_id")
            for s in confounder_symbols
        }
        z_med_d = pm.Data("z_med", z_med, dims="obs_id")

        # --- Mediator model: logit(mediator_t2) ---
        a0 = _priors.alpha_prior().to_pymc("a0")
        a_G = _priors.tau_prior().to_pymc(
            "a_G",
            role="association",
            rationale=(
                "Randomised-arm coefficient in one g-formula leg, carried on the "
                "treatment prior tau ~ Normal(0, 0.5). Reported as an association: "
                "this family's causal deliverables are the NDE/NIE decomposition "
                "the legs compose, not a single leg's coefficient."
            ),
        )
        a_L = _priors.gamma_own_prior().to_pymc(f"a_{mediator_symbol}")
        a_A = _priors.gamma_age_prior().to_pymc("a_A")
        mu_M = a0 + a_G * G_d + a_L * L1_d + a_A * A_d
        for s in confounder_symbols:
            a_c = _priors.gamma_cross_prior().to_pymc(f"a_{s}")
            mu_M = mu_M + a_c * conf_d[s]
        mu_M = _add_cross_baselines(mu_M, mediator_cross_baselines, med_cross_values)
        mu_M = pm.Deterministic("mu_M", mu_M, dims="obs_id")
        kappa_M = _priors.kappa_prior().to_pymc("kappa_M")
        beta_binomial_from_logit(
            f"{mediator_symbol}_post", mu_M, n_trials=N_med, kappa=kappa_M,
            observed=L2_count, dims="obs_id",
        )

        # --- Outcome model: logit(W_t2) (graded) or logit P(off-floor) (Bernoulli) ---
        _build_outcome_leg(
            mediator_node=z_med_d,
            G_d=G_d,
            W1_d=W1_d,
            A_d=A_d,
            conf_d=conf_d,
            confounder_symbols=confounder_symbols,
            N_out=N_out,
            W2_count=W2_count,
            outcome_kind=outcome_kind,
            cross_baselines=outcome_cross_baselines,
            cross_values=out_cross_values,
            own_offfloor=own_offfloor,
            score_mean_link=score_mean_link,
        )

    med_data = MediationData(
        G=prepared.G.astype(float),
        L1_logit=L1,
        W1_logit=W1,
        A_std=prepared.A_std,
        conf_logit={s: conf_logit[s] for s in confounder_symbols},
        confounder_symbols=confounder_symbols,
        L2_count=L2_count,
        W2_count=W2_count,
        n_trials_L=int(N_med),
        # Off-floor outcome: n_trials_W = 1 so decompose's ``words_* = prob_* · N_W``
        # collapses onto the off-floor risk difference (the outcome is the binary
        # off-floor indicator, reported on the probability scale).
        n_trials_W=(1 if off_floor else int(N_out)),
        med_mean=float(med_scaler.mean),
        med_sd=float(med_scaler.sd),
        mediator_symbol=mediator_symbol,
        off_floor=off_floor,
        mediator_cross_values=med_cross_values,
        outcome_cross_values=out_cross_values,
        own_offfloor=own_offfloor,
    )
    built = BuiltModel(
        model=model,
        prepared=prepared,
        payload=MediationPayload(score_mean_link=score_mean_link),
    )
    return built, med_data


def _build_route_composite(
    prepared: PreparedData, route_symbols: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight standardised-logit code-based-route composite.

    For each route symbol, the Haldane-logit of the count is standardised on its
    *post* (t2) distribution and that same scaler is applied to the pre (t1)
    value, so the baseline and post composites share one scale. Each child's
    composite is the equal-weight mean across symbols; the post composite is then
    standardised to mean 0 / SD 1 (the scaler reused for the baseline composite,
    so ``a_comp`` is a like-for-like autoregressive coupling). Returns
    ``(C_pre_std, C_post_std)``.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    pre_cols, post_cols = [], []
    for s in route_symbols:
        post_logit = logit_safe(prepared.post_counts[s], prepared.n_trials[s])
        z_post, scaler = standardise(post_logit)
        pre_cols.append(scaler(prepared.pre_logit[s]))  # same scaler maps baseline
        post_cols.append(z_post)
    c_pre_raw = np.mean(np.stack(pre_cols, axis=1), axis=1)
    c_post_raw = np.mean(np.stack(post_cols, axis=1), axis=1)
    c_post_std, comp_scaler = standardise(c_post_raw)
    return comp_scaler(c_pre_raw), c_post_std


def _build_route_composite_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    confounder_symbols: tuple[str, ...],
    route_symbols: tuple[str, ...],
    mediator_cross_baselines=(),
    outcome_cross_baselines=(),
    score_mean_link: ScoreMeanLink = "logit",
) -> tuple[BuiltModel[MediationPayload], MediationData]:
    """LRP62 reading-route mediation: a continuous code-based-route composite mediator.

    Same ITT-phase joint design and the *same* Beta-Binomial outcome model as
    :func:`build_mediation_model`, but the single count mediator is replaced by
    an equal-weight standardised composite of ``route_symbols`` modelled as
    ``Normal(mu_M, sigma_M)``. NDE/NIE are still computed by counterfactual
    simulation in :func:`mediation.decompose` (the ``gaussian_composite`` branch),
    not from coefficients. Confounders are taken at baseline (cross-world
    assumption), exactly as in LRP59.
    """
    if not route_symbols:
        raise ValueError("gaussian_composite requires non-empty route_symbols")
    if score_mean_link != "logit":
        # The composite route's outcome is word reading, never phoneme blending,
        # and its outcome mean is built with the plain inverse logit below — a
        # non-logit link would be silently ignored, so it is rejected instead.
        raise ValueError(
            "gaussian_composite supports only the ordinary logit score mean, "
            f"got {score_mean_link!r}"
        )
    for s in (outcome_symbol, *route_symbols):
        if s not in prepared.pre_logit:
            raise KeyError(f"Symbol {s!r} missing from prepared data")
    for s in confounder_symbols:
        if s not in prepared.pre_logit and s not in prepared.covariates:
            raise KeyError(f"Confounder {s!r} not in prepared pre_logit or covariates")

    # Keep rows with the outcome post and every route post observed.
    keep = ~np.isnan(prepared.post_counts[outcome_symbol])
    for s in route_symbols:
        keep = keep & ~np.isnan(prepared.post_counts[s])
    if not keep.all():
        prepared = _subset(prepared, keep)

    N_out = prepared.n_trials[outcome_symbol]
    W2_count = prepared.post_counts[outcome_symbol].astype(np.int64)
    c_pre_std, c_post_std = _build_route_composite(prepared, route_symbols)

    W1 = prepared.pre_logit[outcome_symbol]
    conf_logit = {
        s: _baseline_confounder_value(prepared, s) for s in confounder_symbols
    }
    mediator_cross_baselines = tuple(mediator_cross_baselines)
    outcome_cross_baselines = tuple(outcome_cross_baselines)
    med_cross_values = _cross_baseline_arrays(prepared, mediator_cross_baselines)
    # The outcome leg conditions on the COMPOSITE baseline (#585 finding 1): one
    # term matching the mediator leg's ``a_comp``, rather than the route symbols
    # entered separately. The resolver emits a single synthetic ``b_base_M`` term.
    out_cross_values = {
        term.coefficient: (
            c_pre_std
            if term.symbol == "M"
            else _cross_baseline_arrays(prepared, (term,))[term.coefficient]
        )
        for term in outcome_cross_baselines
    }

    coords = {"obs_id": np.arange(prepared.n_obs)}
    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        G_d = pm.Data("G", G_f, dims="obs_id")
        Mpre_d = pm.Data("M_pre_std", c_pre_std, dims="obs_id")
        Mpost_d = pm.Data("M_post_std", c_post_std, dims="obs_id")
        W1_d = pm.Data(f"{outcome_symbol}_pre_logit", W1, dims="obs_id")
        A_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        conf_d = {
            s: pm.Data(f"{s}_pre_logit", conf_logit[s], dims="obs_id")
            for s in confounder_symbols
        }

        # --- Mediator model: standardised route composite ~ Normal ---
        a0 = _priors.alpha_prior().to_pymc("a0")
        a_G = _priors.tau_prior().to_pymc(
            "a_G",
            role="association",
            rationale=(
                "Randomised-arm coefficient in one g-formula leg, carried on the "
                "treatment prior tau ~ Normal(0, 0.5). Reported as an association: "
                "this family's causal deliverables are the NDE/NIE decomposition "
                "the legs compose, not a single leg's coefficient."
            ),
        )
        a_comp = _priors.gamma_own_prior().to_pymc("a_comp")
        a_A = _priors.gamma_age_prior().to_pymc("a_A")
        mu_M = a0 + a_G * G_d + a_comp * Mpre_d + a_A * A_d
        for s in confounder_symbols:
            a_c = _priors.gamma_cross_prior().to_pymc(f"a_{s}")
            mu_M = mu_M + a_c * conf_d[s]
        mu_M = _add_cross_baselines(mu_M, mediator_cross_baselines, med_cross_values)
        mu_M = pm.Deterministic("mu_M", mu_M, dims="obs_id")
        sigma_M = _priors.sigma_mediator_prior().to_pymc("sigma_M")
        pm.Normal("M_post", mu=mu_M, sigma=sigma_M, observed=Mpost_d, dims="obs_id")

        # --- Outcome model: logit(W_t2) (shared with the LRP59 outcome leg) ---
        _build_outcome_leg(
            mediator_node=Mpost_d,
            G_d=G_d,
            W1_d=W1_d,
            A_d=A_d,
            conf_d=conf_d,
            confounder_symbols=confounder_symbols,
            N_out=N_out,
            W2_count=W2_count,
            cross_baselines=outcome_cross_baselines,
            cross_values=out_cross_values,
        )

    med_data = MediationData(
        G=prepared.G.astype(float),
        W1_logit=W1,
        A_std=prepared.A_std,
        conf_logit={s: conf_logit[s] for s in confounder_symbols},
        confounder_symbols=tuple(confounder_symbols),
        n_trials_W=int(N_out),
        mediator_kind="gaussian_composite",
        W2_count=W2_count,
        M_pre_std=c_pre_std,
        route_symbols=route_symbols,
        mediator_cross_values=med_cross_values,
        outcome_cross_values=out_cross_values,
    )
    # The pipeline reads the fitted score-mean link off the payload for the
    # g-formula (#619); the composite branch returned EmptyPayload when that
    # contract landed and so failed at fit time on the first refit (#575 batch,
    # 2026-08-26). Always "logit" here, by the validation above.
    built = BuiltModel(
        model=model,
        prepared=prepared,
        payload=MediationPayload(score_mean_link=score_mean_link),
    )
    return built, med_data


@dataclass
class TwoMediatorData:
    """Row-aligned phase-0 arrays + scalers for the two-mediator g-formula (LRP64).

    Two count mediators — letter-sound knowledge ``L`` and expressive vocabulary
    ``E`` — are each modelled with a Beta-Binomial leg conditioned on their own
    baseline; the outcome leg adds both standardised post-mediators and their
    treatment interactions. :func:`mediation.decompose_two_mediator` re-simulates
    each mediator under each treatment arm to compute the joint indirect effect,
    the direct effect, and the (ordering-dependent) path-specific indirect effects.
    """

    G: np.ndarray
    A_std: np.ndarray
    W1_logit: np.ndarray
    conf1_logit: dict[str, np.ndarray]
    n_trials_W: int
    # Mediator L (letter-sound).
    L1_logit: np.ndarray
    n_trials_L: int
    zL_mean: float
    zL_sd: float
    # Mediator E (expressive vocabulary).
    E1_logit: np.ndarray
    n_trials_E: int
    zE_mean: float
    zE_sd: float
    mediator_symbols: tuple[str, str] = ("L", "E")
    confounder_symbols: tuple[str, ...] = ("R",)
    #: Sequential code route (LRP75): the second mediator regresses on post-L, so
    #: the g-formula must draw it conditional on the simulated first mediator.
    chain: bool = False
    #: Off-floor (Bernoulli) second mediator (e.g. floored nonword decoding N, med-081):
    #: its leg models P(mediator > 0) with no dispersion / denominator / own-baseline,
    #: and the g-formula draws it as a Bernoulli indicator, not a Beta-Binomial count.
    second_mediator_offfloor: bool = False
    #: Per-leg cross baseline regressors restored by #585, keyed by mediator
    #: symbol then coefficient name (the outcome leg's are flat).
    mediator_cross_values: dict[str, dict[str, np.ndarray]] = field(
        default_factory=dict
    )
    outcome_cross_values: dict[str, np.ndarray] = field(default_factory=dict)
    #: Binary off-floor-at-baseline indicator for an off-floor second mediator.
    second_mediator_offfloor_pre: np.ndarray | None = None


def build_two_mediator_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    mediator_symbols: tuple[str, str] = ("L", "E"),
    confounder_symbols: Iterable[str] = ("R",),
    chain: bool = False,
    second_mediator_offfloor: bool = False,
    mediator_cross_baselines: dict | None = None,
    outcome_cross_baselines=(),
) -> tuple[BuiltModel[EmptyPayload], TwoMediatorData]:
    """Joint two-mediator + outcome model for the ITT-phase decomposition (LRP64).

    Generalises :func:`build_mediation_model` to **two named count mediators** so
    the word-reading effect can be split into a path via letter-sound knowledge, a
    path via a second mediator (expressive vocabulary ``E`` in LRP64, phoneme
    blending ``B`` in LRP66), and a direct/residual path. The first leg is fixed to
    ``L``; the second is parameterised by ``mediator_symbols[1]``. Three
    Beta-Binomial legs share the randomised treatment ``G`` and a baseline-covariate
    adjustment::

        L_t2 ~ aL0 + aL_G·G + aL_L·logit(L_t1) + aL_A·A + sum aL_c·C_t1
        E_t2 ~ aE0 + aE_G·G + aE_E·logit(E_t1) + aE_A·A + sum aE_c·C_t1
        W_t2 ~ b0 + b_G·G + b_L·zL + b_E·zE + b_GL·G·zL + b_GE·G·zE
               + b_W·logit(W_t1) + b_A·A + sum b_c·C_t1

    where ``zL`` / ``zE`` are the standardised post-mediator logits. The two
    treatment×mediator interactions admit exposure-mediator interaction; the
    natural (in)direct effects are computed by counterfactual simulation in
    :func:`mediation.decompose_two_mediator`, **not** from coefficients.

    Confounders ``C`` (e.g. receptive vocab ``R``) are taken at **baseline (t1)**
    on the logit scale (cross-world assumption). Expressive vocab is a *mediator*
    here, not a confounder, so only its baseline enters (autoregressively in the
    ``E`` leg). Requires ``prepared.phase_mode == 'itt'`` (one row per child).
    """
    if prepared.phase_mode != "itt":
        raise ValueError("Two-mediator factory requires phase_mode='itt'")
    confounder_symbols = tuple(confounder_symbols)
    mL, mE = mediator_symbols
    # The FIRST mediator leg's node/coefficient names are hard-coded to L
    # (L_pre_logit, z_L, aL_*, b_L, b_GL, L_post); the SECOND leg is
    # parameterised by its symbol ``mE`` ({mE}_pre_logit, z_{mE}, a{mE}_*,
    # b_{mE}, b_G{mE}, {mE}_post, kappa_{mE}). When mE == 'E' every generated
    # name is byte-identical to the original LRP64 build, so ('L', 'E') is
    # unchanged; ('L', 'B') etc. get correctly-labelled second-leg variables.
    if mL != "L":
        raise NotImplementedError(
            "build_two_mediator_model hard-codes the first leg to L; "
            f"mediator_symbols[0] must be 'L', got {mediator_symbols!r}"
        )
    # Every leg conditions on the common baseline vector (#585 finding 1), and an
    # off-floor second mediator now carries its baseline as the binary
    # off-floor-at-baseline indicator rather than dropping it (#585 finding 4), so
    # all three pre-scores are required.
    _need_pre = (outcome_symbol, mL, mE)
    for s in _need_pre:
        if s not in prepared.pre_logit:
            raise KeyError(f"Symbol {s!r} missing from prepared data")
    if mE not in prepared.post_counts:
        raise KeyError(f"Second mediator {mE!r} missing a post score in prepared data")
    for s in confounder_symbols:
        if s not in prepared.pre_logit and s not in prepared.covariates:
            raise KeyError(f"Confounder {s!r} not in prepared pre_logit or covariates")

    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    keep = ~np.isnan(prepared.post_counts[outcome_symbol])
    for s in (mL, mE):
        keep = keep & ~np.isnan(prepared.post_counts[s])
    if not keep.all():
        prepared = _subset(prepared, keep)

    N_W = prepared.n_trials[outcome_symbol]
    N_L = prepared.n_trials[mL]
    N_E = prepared.n_trials[mE]
    L2 = prepared.post_counts[mL].astype(np.int64)
    E2 = prepared.post_counts[mE].astype(np.int64)
    W2 = prepared.post_counts[outcome_symbol].astype(np.int64)

    zL, zL_scaler = standardise(logit_safe(L2, N_L))
    if second_mediator_offfloor:
        # The regressor entering the outcome leg for an off-floor mediator is the
        # off-floor INDICATOR (mediator > 0), not a standardised count logit — it has
        # no dispersion, denominator or own-baseline. zE_mean/zE_sd are placeholders
        # (the g-formula draws a Bernoulli indicator, so no destandardisation applies).
        zE = (E2 > 0).astype(float)
        zE_mean, zE_sd = 0.0, 1.0
    else:
        zE, zE_scaler = standardise(logit_safe(E2, N_E))
        zE_mean, zE_sd = float(zE_scaler.mean), float(zE_scaler.sd)

    L1 = prepared.pre_logit[mL]
    E1 = (
        np.zeros(prepared.n_obs, dtype=float)
        if second_mediator_offfloor
        else prepared.pre_logit[mE]
    )
    W1 = prepared.pre_logit[outcome_symbol]
    conf_logit = {
        s: _baseline_confounder_value(prepared, s) for s in confounder_symbols
    }
    mediator_cross_baselines = dict(mediator_cross_baselines or {})
    outcome_cross_baselines = tuple(outcome_cross_baselines)
    med_cross_values = {
        symbol: _cross_baseline_arrays(prepared, terms)
        for symbol, terms in mediator_cross_baselines.items()
    }
    out_cross_values = _cross_baseline_arrays(prepared, outcome_cross_baselines)
    # Binary off-floor-at-baseline indicator restoring the second mediator's own
    # baseline term (#585 finding 4).
    E1_off = (
        (np.asarray(prepared.pre_counts[mE]) > 0).astype(float)
        if second_mediator_offfloor
        else None
    )

    coords = {"obs_id": np.arange(prepared.n_obs)}
    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        G_d = pm.Data("G", G_f, dims="obs_id")
        A_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        L1_d = pm.Data("L_pre_logit", L1, dims="obs_id")
        if not second_mediator_offfloor:
            E1_d = pm.Data(f"{mE}_pre_logit", E1, dims="obs_id")
        W1_d = pm.Data(f"{outcome_symbol}_pre_logit", W1, dims="obs_id")
        conf_d = {
            s: pm.Data(f"{s}_pre_logit", conf_logit[s], dims="obs_id")
            for s in confounder_symbols
        }
        zL_d = pm.Data("z_L", zL, dims="obs_id")
        zE_d = pm.Data(f"z_{mE}", zE, dims="obs_id")

        # --- Mediator L (letter-sound) ---
        aL0 = _priors.alpha_prior().to_pymc("aL0")
        aL_G = _priors.tau_prior().to_pymc(
            "aL_G",
            role="association",
            rationale=(
                "Randomised-arm coefficient in one g-formula leg, carried on the "
                "treatment prior tau ~ Normal(0, 0.5). Reported as an association: "
                "this family's causal deliverables are the NDE/NIE decomposition "
                "the legs compose, not a single leg's coefficient."
            ),
        )
        aL_L = _priors.gamma_own_prior().to_pymc("aL_L")
        aL_A = _priors.gamma_age_prior().to_pymc("aL_A")
        mu_L = aL0 + aL_G * G_d + aL_L * L1_d + aL_A * A_d
        for s in confounder_symbols:
            aL_c = _priors.gamma_cross_prior().to_pymc(f"aL_{s}")
            mu_L = mu_L + aL_c * conf_d[s]
        mu_L = _add_cross_baselines(
            mu_L,
            mediator_cross_baselines.get(mL, ()),
            med_cross_values.get(mL, {}),
        )
        mu_L = pm.Deterministic("mu_L", mu_L, dims="obs_id")
        kappa_L = _priors.kappa_prior().to_pymc("kappa_L")
        beta_binomial_from_logit(
            "L_post", mu_L, n_trials=N_L, kappa=kappa_L, observed=L2, dims="obs_id"
        )

        # --- Mediator 2 (``mE``; expressive vocabulary in LRP64, blending in LRP66;
        #     off-floor nonword decoding N in med-081) ---
        aE0 = _priors.alpha_prior().to_pymc(f"a{mE}0")
        aE_G = _priors.tau_prior().to_pymc(
            f"a{mE}_G",
            role="association",
            rationale=(
                "Randomised-arm coefficient in one g-formula leg, carried on the "
                "treatment prior tau ~ Normal(0, 0.5). Reported as an association: "
                "this family's causal deliverables are the NDE/NIE decomposition "
                "the legs compose, not a single leg's coefficient."
            ),
        )
        if second_mediator_offfloor:
            # Off-floor mediator: the graded Normal(1, 0.25) autoregressive prior does
            # not transfer to a binary leg, but the baseline is not dropped either
            # (#585 finding 4) — it enters as the binary off-floor-at-baseline
            # contrast, so the complete-case rule and the likelihood agree.
            aE_off = _priors.gamma_own_offfloor_prior().to_pymc(f"a{mE}_own_offfloor")
            aE_off_d = pm.Data(f"{mE}_pre_offfloor", E1_off, dims="obs_id")
            aE_A = _priors.gamma_age_prior().to_pymc(f"a{mE}_A")
            mu_E = aE0 + aE_G * G_d + aE_off * aE_off_d + aE_A * A_d
        else:
            aE_E = _priors.gamma_own_prior().to_pymc(f"a{mE}_{mE}")
            aE_A = _priors.gamma_age_prior().to_pymc(f"a{mE}_A")
            mu_E = aE0 + aE_G * G_d + aE_E * E1_d + aE_A * A_d
        for s in confounder_symbols:
            aE_c = _priors.gamma_cross_prior().to_pymc(f"a{mE}_{s}")
            mu_E = mu_E + aE_c * conf_d[s]
        if chain:
            # Sequential code route (LRP75 / med-081): the second mediator is downstream
            # of the first (L -> B, or L -> N), so post-L (``z_L``) enters the mE leg. The
            # coefficient a{mE}_L is the L->mE coupling; the g-formula then draws the
            # second mediator conditional on the *simulated* L.
            aE_L = _priors.gamma_cross_prior().to_pymc(f"a{mE}_{mL}")
            mu_E = mu_E + aE_L * zL_d
        mu_E = _add_cross_baselines(
            mu_E,
            mediator_cross_baselines.get(mE, ()),
            med_cross_values.get(mE, {}),
        )
        mu_E = pm.Deterministic(f"mu_{mE}", mu_E, dims="obs_id")
        if second_mediator_offfloor:
            # Bernoulli off-floor leg: models P(mediator > 0); no dispersion, no
            # denominator, no own-baseline (mirrors the off-floor outcome leg).
            pm.Bernoulli(
                f"{mE}_offfloor",
                logit_p=mu_E,
                observed=(E2 > 0).astype(np.int64),
                dims="obs_id",
            )
        else:
            kappa_E = _priors.kappa_prior().to_pymc(f"kappa_{mE}")
            beta_binomial_from_logit(
                f"{mE}_post", mu_E, n_trials=N_E, kappa=kappa_E, observed=E2, dims="obs_id"
            )

        # --- Outcome W ---
        b0 = _priors.alpha_prior().to_pymc("b0")
        b_G = _priors.tau_prior().to_pymc(
        "b_G",
        role="association",
        rationale=(
            "Randomised-arm coefficient in one g-formula leg, carried on the "
            "treatment prior tau ~ Normal(0, 0.5). Reported as an association: "
            "this family's causal deliverables are the NDE/NIE decomposition "
            "the legs compose, not a single leg's coefficient."
        ),
    )
        b_L = _priors.b_path_prior().to_pymc("b_L")
        b_E = _priors.b_path_prior().to_pymc(f"b_{mE}")
        b_GL = _priors.gamma_cross_prior().to_pymc("b_GL")
        b_GE = _priors.gamma_cross_prior().to_pymc(f"b_G{mE}")
        b_W = _priors.gamma_own_prior().to_pymc("b_W")
        b_A = _priors.gamma_age_prior().to_pymc("b_A")
        eta_Y = (
            b0
            + b_G * G_d
            + b_L * zL_d
            + b_E * zE_d
            + b_GL * (G_d * zL_d)
            + b_GE * (G_d * zE_d)
            + b_W * W1_d
            + b_A * A_d
        )
        for s in confounder_symbols:
            b_c = _priors.gamma_cross_prior().to_pymc(f"b_{s}")
            eta_Y = eta_Y + b_c * conf_d[s]
        eta_Y = _add_cross_baselines(eta_Y, outcome_cross_baselines, out_cross_values)
        eta_Y = pm.Deterministic("eta", eta_Y, dims="obs_id")
        kappa_Y = _priors.kappa_prior().to_pymc("kappa_Y")
        beta_binomial_from_logit(
            "y_post", eta_Y, n_trials=N_W, kappa=kappa_Y, observed=W2, dims="obs_id"
        )

    med_data = TwoMediatorData(
        G=prepared.G.astype(float),
        A_std=prepared.A_std,
        W1_logit=W1,
        conf1_logit={s: conf_logit[s] for s in confounder_symbols},
        n_trials_W=int(N_W),
        L1_logit=L1,
        n_trials_L=int(N_L),
        zL_mean=float(zL_scaler.mean),
        zL_sd=float(zL_scaler.sd),
        E1_logit=E1,
        n_trials_E=int(N_E),
        zE_mean=zE_mean,
        zE_sd=zE_sd,
        mediator_symbols=(mL, mE),
        confounder_symbols=confounder_symbols,
        chain=chain,
        second_mediator_offfloor=second_mediator_offfloor,
        mediator_cross_values=med_cross_values,
        outcome_cross_values=out_cross_values,
        second_mediator_offfloor_pre=E1_off,
    )
    built = BuiltModel(model=model, prepared=prepared, payload=EmptyPayload())
    return built, med_data


@dataclass
class PeriodStackedMediationData:
    """Row-aligned stacked-period arrays for the period-stacked g-formula (MED-092).

    The stacked analogue of :class:`MediationData`: one row per child x period
    transition (``phase_mode="all"``), so the g-formula needs the per-row phase
    index and child index to reconstruct the phase intercepts and per-leg child
    random intercepts alongside the coefficient draws. Confounder values are
    carried generically in ``conf_values`` (period-start logits for bounded-count
    measures; loader-timed standardised values for raw covariates), keyed by
    symbol, so :func:`mediation.decompose_period_stacked` adjusts for exactly the
    set the model was fitted with.
    """

    trt: np.ndarray
    """Per-row on-intervention indicator (the exposure being decomposed)."""
    phase_idx: np.ndarray
    child_idx: np.ndarray
    n_phases: int
    n_children: int
    L1_logit: np.ndarray
    W1_logit: np.ndarray
    A_std: np.ndarray
    conf_values: dict[str, np.ndarray]
    confounder_symbols: tuple[str, ...]
    L2_count: np.ndarray
    W2_count: np.ndarray
    n_trials_L: int
    n_trials_W: int
    med_mean: float
    med_sd: float
    mediator_symbol: str = "L"
    #: Cross-leg baseline regressors restored by #585, keyed by coefficient name.
    mediator_cross_values: dict[str, np.ndarray] = field(default_factory=dict)
    outcome_cross_values: dict[str, np.ndarray] = field(default_factory=dict)


def build_period_stacked_mediation_model(
    prepared: PreparedData,
    *,
    mediator_symbol: str = "L",
    outcome_symbol: str = "W",
    confounder_symbols: Iterable[str] = (),
    sigma_child_prior_sigma: float = 0.5,
    mediator_cross_baselines=(),
    outcome_cross_baselines=(),
) -> tuple[BuiltModel[EmptyPayload], PeriodStackedMediationData]:
    """Joint mediator + outcome model over all stacked periods (MED-092, #229).

    The LRP59 mediation design transplanted onto the **gain-factor scaffold**
    (``phase_mode="all"``): every on-intervention and untreated period transition
    is stacked, the exposure is the per-period **on-intervention** indicator
    ``T`` (the term the gain-factor models already treat as ignorable) rather
    than the Phase-0 randomised group, and both legs carry the gain-factor
    machinery — per-phase intercepts and a per-leg child random intercept::

        Mediator: logit(M_post) ~ a0 + a_phase[p] + a_trt*T + a_M*logit(M_pre)
                                  + a_A*age + sum a_c*C + u_child_M
        Outcome:  logit(W_post) ~ b0 + b_phase[p] + b_trt*T + b_M*z(logit M_post)
                                  + b_trtM*T*z + b_W*logit(W_pre) + b_A*age
                                  + sum b_c*C + u_child_Y

    ``T`` varies between arms **only in period 1** (after the waitlist crossover
    both arms are on the programme), so the exposure contrast is still anchored
    on the randomised window; what the stacking buys is that the
    mediator -> outcome leg (``b_M``, the coefficient that carries the indirect
    effect) and every covariate coupling are informed by **all** periods instead
    of the single t1->t2 window. The price is stated plainly: the estimand is a
    **per-period** decomposition under the gain-factor family's ignorability
    assumption (on-intervention exchangeable given period-start state, phase,
    age and the child intercept), not the Phase-0 randomised contrast — read it
    as a triangulation companion to ``med-059``/``064``, never a replacement.

    Confounders mirror LRP59's set: bounded-count measures (E, R) enter at the
    **period start** (``prepared.pre_logit``, the sequential g-formula's
    history-adjustment rule); raw covariates keep the loader's gain-factor
    timing (hearing contemporaneous, speech/phonological memory at the t1
    baseline — the A1 timing decision, 2026-07-13). Per-period baselines at
    post-crossover transitions are descendants of *earlier* periods' exposure —
    admissible for the per-period estimand, but exactly why no cumulative
    (multi-period) effect is decomposed here.

    NDE/NIE are computed by counterfactual simulation
    (:func:`mediation.decompose_period_stacked`), not from coefficients, and are
    **not identified natural effects** — every LRP59 obstacle (latent-GA
    mediator-outcome confounding; treatment-induced dose ``IS``) carries over,
    plus the ignorability trade above. Requires ``phase_mode == 'all'``; the
    graded Beta-Binomial legs only (no off-floor branch — W and L are graded).
    """
    if prepared.phase_mode != "all":
        raise ValueError(
            "build_period_stacked_mediation_model requires phase_mode='all'"
        )
    confounder_symbols = tuple(confounder_symbols)
    for s in (mediator_symbol, outcome_symbol):
        if s not in prepared.pre_logit or s not in prepared.post_counts:
            raise KeyError(f"Symbol {s!r} needs pre+post scores in prepared data")
    for s in confounder_symbols:
        if s not in prepared.pre_logit and s not in prepared.covariates:
            raise KeyError(f"Confounder {s!r} not in prepared pre_logit or covariates")

    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    # Complete cases on the mediator/outcome pre+post and every confounder value.
    keep = (
        ~np.isnan(prepared.post_counts[mediator_symbol])
        & ~np.isnan(prepared.pre_logit[mediator_symbol])
        & ~np.isnan(prepared.post_counts[outcome_symbol])
        & ~np.isnan(prepared.pre_logit[outcome_symbol])
    )
    for s in confounder_symbols:
        keep = keep & ~np.isnan(_baseline_confounder_value(prepared, s))
    prepared = _subset(prepared, keep)

    N_med = prepared.n_trials[mediator_symbol]
    N_out = prepared.n_trials[outcome_symbol]
    # Graded Beta-Binomial legs only (no off-floor branch): this factory has no
    # off-floor likelihood, and the g-formula labels every effect off_floor=False
    # and reports words = prob * N_out. A unit denominator would silently make
    # that "items" scale a risk difference while the flag still said graded — the
    # inconsistency the reviewer flagged — so fail fast instead of building it.
    for sym, n in ((mediator_symbol, N_med), (outcome_symbol, N_out)):
        if n <= 1:
            raise ValueError(
                f"period-stacked mediation is graded-only, but {sym!r} has "
                f"n_trials={n}; a floored/off-floor outcome is out of scope here"
            )
    L2_count = prepared.post_counts[mediator_symbol].astype(np.int64)
    W2_count = prepared.post_counts[outcome_symbol].astype(np.int64)
    trt = ((prepared.G == 1) | (prepared.phase >= 1)).astype(float)

    # Standardised post-mediator logit — the outcome-leg regressor; the scaler is
    # reused for the counterfactual mediator draws (as in build_mediation_model).
    med_logit = logit_safe(L2_count, N_med)
    z_med, med_scaler = standardise(med_logit)

    L1 = prepared.pre_logit[mediator_symbol]
    W1 = prepared.pre_logit[outcome_symbol]
    conf_values = {
        s: _baseline_confounder_value(prepared, s) for s in confounder_symbols
    }

    mediator_cross_baselines = tuple(mediator_cross_baselines)
    outcome_cross_baselines = tuple(outcome_cross_baselines)
    med_cross_values = _cross_baseline_arrays(prepared, mediator_cross_baselines)
    out_cross_values = _cross_baseline_arrays(prepared, outcome_cross_baselines)

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": np.arange(prepared.n_phases),
        "child": np.arange(prepared.n_children),
    }
    with pm.Model(coords=coords) as model:
        phase_d = pm.Data("phase_idx", prepared.phase.astype(np.int64), dims="obs_id")
        child_d = pm.Data("child_idx", prepared.child_idx.astype(np.int64), dims="obs_id")
        trt_d = pm.Data("on_intervention", trt, dims="obs_id")
        L1_d = pm.Data(f"{mediator_symbol}_pre_logit", L1, dims="obs_id")
        W1_d = pm.Data(f"{outcome_symbol}_pre_logit", W1, dims="obs_id")
        A_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        conf_d = {
            s: pm.Data(f"{s}_conf", conf_values[s], dims="obs_id")
            for s in confounder_symbols
        }
        z_med_d = pm.Data("z_med", z_med, dims="obs_id")

        # --- Mediator leg: logit(M_post) over all stacked periods ---
        a0 = _priors.alpha_prior().to_pymc("a0")
        a_phase = _priors.declare(
            pm.Normal("a_phase", mu=0.0, sigma=0.5, dims="phase"),
            role="association",
            rationale=(
                "Per-phase intercept/period offset on the mediator leg "
                "(Normal(0, 0.5)); an age/maturation/period association, not a "
                "cross-baseline skill coupling and not a mediator a-path."
            ),
        )
        a_trt = _priors.tau_prior().to_pymc("a_trt")
        a_M = _priors.gamma_own_prior().to_pymc(f"a_{mediator_symbol}")
        a_A = _priors.gamma_age_prior().to_pymc("a_A")
        mu_M = a0 + a_phase[phase_d] + a_trt * trt_d + a_M * L1_d + a_A * A_d
        for s in confounder_symbols:
            a_c = _priors.gamma_cross_prior().to_pymc(f"a_{s}")
            mu_M = mu_M + a_c * conf_d[s]
        mu_M = _add_cross_baselines(mu_M, mediator_cross_baselines, med_cross_values)
        sigma_child_M = _priors.declare(
            pm.HalfNormal("sigma_child_M", sigma=sigma_child_prior_sigma),
            role="nuisance",
            rationale=(
                "Between-child SD of the mediator leg's random intercept "
                "(HalfNormal); partially pools stable child heterogeneity across "
                "the stacked periods."
            ),
        )
        u_child_M_raw = _priors.declare(
            pm.Normal("u_child_M_raw", mu=0.0, sigma=1.0, dims="child"),
            role="nuisance",
            rationale=(
                "Non-centred standard-normal per-child offsets (Normal(0, 1)); "
                "scaled by sigma_child_M to form the mediator leg's child random "
                "intercept u_child_M."
            ),
        )
        u_child_M = pm.Deterministic(
            "u_child_M", sigma_child_M * u_child_M_raw, dims="child"
        )
        mu_M = pm.Deterministic("mu_M", mu_M + u_child_M[child_d], dims="obs_id")
        kappa_M = _priors.kappa_prior().to_pymc("kappa_M")
        beta_binomial_from_logit(
            f"{mediator_symbol}_post", mu_M, n_trials=N_med, kappa=kappa_M,
            observed=L2_count, dims="obs_id",
        )

        # --- Outcome leg: logit(W_post) over all stacked periods ---
        b0 = _priors.alpha_prior().to_pymc("b0")
        b_phase = _priors.declare(
            pm.Normal("b_phase", mu=0.0, sigma=0.5, dims="phase"),
            role="association",
            rationale=(
                "Per-phase intercept/period offset on the outcome leg "
                "(Normal(0, 0.5)); an age/maturation/period association, not a "
                "cross-baseline skill coupling."
            ),
        )
        b_trt = _priors.tau_prior().to_pymc("b_trt")
        b_M = _priors.b_path_prior().to_pymc("b_M")
        b_trtM = _priors.gamma_cross_prior().to_pymc("b_trtM")
        b_W = _priors.gamma_own_prior().to_pymc("b_W")
        b_A = _priors.gamma_age_prior().to_pymc("b_A")
        eta_Y = (
            b0
            + b_phase[phase_d]
            + b_trt * trt_d
            + b_M * z_med_d
            + b_trtM * (trt_d * z_med_d)
            + b_W * W1_d
            + b_A * A_d
        )
        for s in confounder_symbols:
            b_c = _priors.gamma_cross_prior().to_pymc(f"b_{s}")
            eta_Y = eta_Y + b_c * conf_d[s]
        eta_Y = _add_cross_baselines(eta_Y, outcome_cross_baselines, out_cross_values)
        sigma_child_Y = _priors.declare(
            pm.HalfNormal("sigma_child_Y", sigma=sigma_child_prior_sigma),
            role="nuisance",
            rationale=(
                "Between-child SD of the outcome leg's random intercept "
                "(HalfNormal); partially pools stable child heterogeneity across "
                "the stacked periods."
            ),
        )
        u_child_Y_raw = _priors.declare(
            pm.Normal("u_child_Y_raw", mu=0.0, sigma=1.0, dims="child"),
            role="nuisance",
            rationale=(
                "Non-centred standard-normal per-child offsets (Normal(0, 1)); "
                "scaled by sigma_child_Y to form the outcome leg's child random "
                "intercept u_child_Y."
            ),
        )
        u_child_Y = pm.Deterministic(
            "u_child_Y", sigma_child_Y * u_child_Y_raw, dims="child"
        )
        eta_Y = pm.Deterministic("eta", eta_Y + u_child_Y[child_d], dims="obs_id")
        kappa_Y = _priors.kappa_prior().to_pymc("kappa_Y")
        beta_binomial_from_logit(
            "y_post", eta_Y, n_trials=N_out, kappa=kappa_Y,
            observed=W2_count, dims="obs_id",
        )

    med_data = PeriodStackedMediationData(
        trt=trt,
        phase_idx=prepared.phase.astype(np.int64),
        child_idx=prepared.child_idx.astype(np.int64),
        n_phases=int(prepared.n_phases),
        n_children=int(prepared.n_children),
        L1_logit=L1,
        W1_logit=W1,
        A_std=prepared.A_std,
        conf_values=conf_values,
        confounder_symbols=confounder_symbols,
        L2_count=L2_count,
        W2_count=W2_count,
        mediator_cross_values=med_cross_values,
        outcome_cross_values=out_cross_values,
        n_trials_L=int(N_med),
        n_trials_W=int(N_out),
        med_mean=float(med_scaler.mean),
        med_sd=float(med_scaler.sd),
        mediator_symbol=mediator_symbol,
    )
    built = BuiltModel(model=model, prepared=prepared, payload=EmptyPayload())
    return built, med_data
