# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-fit reporting helpers shared across the statistical models."""

from __future__ import annotations

import json
import os

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)


def tau_summary_itt(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    G: np.ndarray,
) -> dict[str, float]:
    """Summarise the treatment effect ``tau`` on both scales for an ITT model.

    Logit scale: the posterior summary of ``tau`` directly.

    Probability scale: the **average marginal effect** of randomised
    assignment over the fitted sample. For every posterior draw and every
    observation ``i`` we form the counterfactual baseline linear predictor
    ``η0_i = η_i − δ_i · G_i`` from the model's stored per-observation ``eta``
    (the treatment contribution removed; ``δ_i`` is ``tau`` for a constant
    effect, or ``tau_i`` when the effect varies with age), then average
    ``expit(η0_i + δ_i) − expit(η0_i)`` over observations. Each observation's
    effect is therefore evaluated at its *actual* covariate profile —
    including the cross-baseline, adjuster and GP terms carried in ``eta`` —
    rather than at a single constructed baseline point, and the average is
    taken per draw so the posterior uncertainty of the marginal effect is
    preserved.

    ``G`` is the per-observation treatment indicator from the *fitted* prepared
    data (``built.prepared.G``), aligned with ``eta``'s ``obs_id`` axis.

    ``ci_prob`` names the *coverage* probability — the returned ``_lo`` /
    ``_hi`` values are equal-tailed central quantiles, not highest-density
    intervals. For ArviZ-style HDI use :func:`arviz.hdi` directly.
    """
    posterior = trace.posterior
    tau_draws = posterior["tau"].stack(sample=("chain", "draw")).values  # (S,)
    eta = (
        posterior["eta"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)

    G = np.asarray(G, dtype=float)
    if G.shape[0] != eta.shape[0]:
        raise ValueError(
            f"G has {G.shape[0]} rows but eta has {eta.shape[0]} observations; "
            "pass built.prepared.G (aligned with the fitted subset)."
        )

    # Per-observation treatment contribution δ_i: age-varying tau_i if the
    # model has it, otherwise the constant tau broadcast over observations.
    if "tau_i" in posterior:
        delta = (
            posterior["tau_i"]
            .stack(sample=("chain", "draw"))
            .transpose("obs_id", "sample")
            .values
        )  # (n_obs, S)
    else:
        delta = tau_draws[None, :]  # (1, S)

    eta0 = eta - delta * G[:, None]  # untreated baseline (G=0 = control) per obs, per draw
    # Average marginal effect over observations, per draw.
    marginal = (expit(eta0 + delta) - expit(eta0)).mean(axis=0)  # (S,)

    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    tau_mean = float(np.mean(tau_draws))
    lower, upper = np.quantile(tau_draws, [lo_q, hi_q])
    marg_mean = float(np.mean(marginal))
    marg_lo, marg_hi = np.quantile(marginal, [lo_q, hi_q])
    prob_pos = float(np.mean(tau_draws > 0))

    return {
        "tau_logit_mean": tau_mean,
        "tau_logit_lo": float(lower),
        "tau_logit_hi": float(upper),
        "tau_prob_mean": marg_mean,
        "tau_prob_lo": float(marg_lo),
        "tau_prob_hi": float(marg_hi),
        "prob_tau_pos": prob_pos,
    }


def tau_summary_offfloor(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    G: np.ndarray,
) -> dict[str, float]:
    """Summarise the binary off-floor treatment effect (floor-rule PRIMARY, #119).

    For the ``bernoulli_offfloor`` model, ``expit(eta)`` is ``Pr(post > 0 at t2)``
    (the probability of coming off the floor), so the marginal-effect machinery of
    :func:`tau_summary_itt` returns exactly the off-floor quantities: the logit
    scale is the log-odds of coming off the floor, and the probability scale is
    the average **risk difference** in off-floor probability between the
    intervention and control arms. The keys match :func:`tau_summary_itt` (so the
    report and CSV share a schema); the off-floor interpretation is documented in
    the floored-outcome report.
    """
    return tau_summary_itt(trace, ci_prob=ci_prob, G=G)


def offfloor_mover_table(prepared, symbol: str) -> pd.DataFrame:
    """Per-arm off-floor "mover" counts for a floored outcome (floor-rule, #119).

    Returns, for each randomised arm, the number of children with a non-missing
    post-score, how many came **off the floor** (``post > 0`` at t2), how many
    stayed at the floor, and the off-floor proportion. ``prepared.G`` uses the
    positive-benefit coding (1 = intervention, 0 = wait-list control).
    """
    post = np.asarray(prepared.post_counts[symbol], dtype=float)
    G = np.asarray(prepared.G, dtype=int)
    rows = []
    for g, label in ((1, "intervention"), (0, "control")):
        mask = (G == g) & np.isfinite(post)
        n = int(mask.sum())
        off = int(np.sum(post[mask] > 0))
        rows.append(
            {
                "arm": label,
                "n": n,
                "off_floor": off,
                "at_floor": n - off,
                "prop_off_floor": (off / n) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def tau_moderation_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
) -> dict[str, float]:
    """Summarise the ITT tau-moderator coefficients ``gamma_tau_int`` / ``gamma_tau_mod``.

    Part B (HTE) analogue of :func:`gamma_interaction_summary`, but for the
    treatment-moderator path of :func:`factories.build_itt_model`: ``gamma_tau_int``
    is the effect modification (how the treatment effect ``tau`` changes per 1 SD
    of the pre-randomisation moderator), and ``gamma_tau_mod`` is the moderator's
    main effect. Equal-tailed central interval at coverage ``ci_prob`` and
    ``P(coef > 0)`` for each coefficient present in the trace.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    out: dict[str, float] = {}
    for name in ("gamma_tau_int", "gamma_tau_mod"):
        if name not in posterior:
            continue
        d = posterior[name].stack(sample=("chain", "draw")).values
        out[f"{name}_mean"] = float(np.mean(d))
        out[f"{name}_lo"] = float(np.quantile(d, lo_q))
        out[f"{name}_hi"] = float(np.quantile(d, hi_q))
        out[f"prob_{name}_pos"] = float(np.mean(d > 0))
    return out


def proportion_at_zero_ppc(
    prepared,
    symbol: str,
    trace: xr.DataTree,
    *,
    node: str = "y_post",
) -> dict[str, float]:
    """Posterior-predictive check on the proportion-at-zero (floor-rule diagnostic).

    Compares the observed fraction of zero post-scores to the posterior-predictive
    distribution of that fraction under the graded Beta-Binomial model — the check
    that reveals whether the graded model reproduces the floor (it typically does
    not for ``P``/``N``, which is the motivation for the binary primary estimand).
    Returns the observed proportion, the predictive mean, and the
    posterior-predictive p-value ``P(rep >= observed)``; the per-draw replicated
    proportions are returned under ``"rep"`` for plotting.
    """
    post = np.asarray(prepared.post_counts[symbol], dtype=float)
    finite = post[np.isfinite(post)]
    obs_p0 = float(np.mean(finite == 0.0)) if finite.size else float("nan")
    pp = trace.posterior_predictive[node]
    yrep = (
        pp.stack(sample=("chain", "draw")).transpose("sample", "obs_id").values
    )  # (S, n_obs)
    rep_p0 = np.mean(yrep == 0.0, axis=1)  # (S,)
    return {
        "obs_prop_at_zero": obs_p0,
        "ppc_mean_prop_at_zero": float(np.mean(rep_p0)),
        "ppc_p_value": float(np.mean(rep_p0 >= obs_p0)),
        "rep": rep_p0,
    }


def did_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    n_trials: int,
    dose: bool = False,
) -> dict[str, float]:
    """Summarise the waitlist-crossover / difference-in-differences effect (kind="did").

    For the binary model ``delta`` is the treatment effect on the logit scale, and
    ``delta_items_*`` is the average marginal effect of toggling ``Treated`` 0 -> 1
    across the fitted rows (per draw), times ``n_trials`` — directly comparable to
    the ITT ``tau_summary_itt`` items figures. ``beta_period`` is the period
    (time / maturation) anchor estimated from the immediate arm. Equal-tailed
    central intervals at coverage ``ci_prob``. With ``dose=True`` the key
    coefficient is ``beta_dose`` (effect per 1 SD of intervention sessions) and no
    items translation is produced.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q

    def _summ(name: str) -> dict[str, float]:
        d = posterior[name].stack(sample=("chain", "draw")).values
        return {
            f"{name}_mean": float(np.mean(d)),
            f"{name}_lo": float(np.quantile(d, lo_q)),
            f"{name}_hi": float(np.quantile(d, hi_q)),
            f"prob_{name}_pos": float(np.mean(d > 0)),
        }

    out: dict[str, float] = {}
    out.update(_summ("beta_period"))
    if dose:
        out.update(_summ("beta_dose"))
        return out

    out.update(_summ("delta"))
    # Items-scale average marginal effect: toggle Treated 0 -> 1 per fitted row.
    delta = posterior["delta"].stack(sample=("chain", "draw")).values  # (S,)
    eta_base = (
        posterior["eta_base"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    from scipy.special import expit  # local import

    eff = (expit(eta_base + delta[None, :]) - expit(eta_base)).mean(axis=0) * n_trials
    out["delta_items_mean"] = float(np.mean(eff))
    out["delta_items_lo"] = float(np.quantile(eff, lo_q))
    out["delta_items_hi"] = float(np.quantile(eff, hi_q))
    return out


def tau_summary_joint(
    trace: xr.DataTree,
    outcomes: list[str],
    ci_prob: float,
) -> pd.DataFrame:
    """Return a DataFrame summarising tau_k for each outcome (logit scale).

    ``tau_lo`` / ``tau_hi`` are equal-tailed central quantiles at coverage
    ``ci_prob``. See :func:`tau_summary_itt` for the convention.
    """
    draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values  # (K, n_sample)
    out = []
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    for k, s in enumerate(outcomes):
        d = draws[k]
        out.append(
            {
                "outcome": s,
                "tau_mean": float(np.mean(d)),
                "tau_lo": float(np.quantile(d, lo_q)),
                "tau_hi": float(np.quantile(d, hi_q)),
                "prob_pos": float(np.mean(d > 0)),
            }
        )
    return pd.DataFrame(out)


def gamma_interaction_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
) -> dict[str, float]:
    """Summarise the linear-moderation coefficients ``gamma_int`` / ``gamma_mod``.

    Reports the posterior mean, equal-tailed central interval at coverage
    ``ci_prob`` (same convention as :func:`tau_summary_itt`), and ``P(coef > 0)``
    for each coefficient present in the trace. ``gamma_int`` is the moderation
    (>0: the standardised mechanism effect strengthens with the moderator);
    ``gamma_mod`` is the moderator main effect at the mean of the mechanism.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    out: dict[str, float] = {}
    for name in ("gamma_int", "gamma_mod"):
        if name not in posterior:
            continue
        d = posterior[name].stack(sample=("chain", "draw")).values
        out[f"{name}_mean"] = float(np.mean(d))
        out[f"{name}_lo"] = float(np.quantile(d, lo_q))
        out[f"{name}_hi"] = float(np.quantile(d, hi_q))
        out[f"prob_{name}_pos"] = float(np.mean(d > 0))
    return out


def tau_contrast_matrix(
    trace: xr.DataTree,
    outcomes: list[str],
) -> pd.DataFrame:
    """Compute P(tau_k > tau_j) for every outcome pair."""
    draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values  # (K, n_sample)
    K = draws.shape[0]
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                M[i, j] = np.nan
            else:
                M[i, j] = float(np.mean(draws[i] > draws[j]))
    return pd.DataFrame(M, index=outcomes, columns=outcomes)


def tau_difference_summary(
    trace: xr.DataTree,
    outcomes: list[str],
    pair: tuple[str, str],
    *,
    ci_prob: float,
) -> dict[str, float | str]:
    """Summarise the difference ``tau[a] - tau[b]`` between two joint outcomes.

    The difference is computed per posterior draw and then summarised, so the
    reported interval propagates the full joint posterior (including any residual
    correlation between the two outcomes) rather than combining two marginal
    summaries. ``pair = (a, b)`` names the contrast ``tau[a] - tau[b]``.

    Sign convention: ``tau`` is the coefficient on ``G = 2 - group``, and group 1
    receives the intervention from t1, so a *positive* ``tau`` means the
    intervention raised that outcome (see the "Sign convention" section of
    METHODS.md). For the LRPITT15/15b generalisation contrast the pair is therefore
    ``("TE", "UE")``: ``tau_TE - tau_UE`` equals the intervention benefit on
    taught words minus the benefit on not-taught words, so a *positive* difference
    means the directly-taught words moved *more* than the not-taught comparison
    words - i.e. limited generalisation.

    ``_lo`` / ``_hi`` are equal-tailed central quantiles at coverage ``ci_prob``
    (same convention as :func:`tau_summary_itt`).
    """
    a, b = pair
    draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values  # (K, n_sample)
    ia, ib = outcomes.index(a), outcomes.index(b)
    diff = draws[ia] - draws[ib]
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    return {
        "contrast": f"{a}_minus_{b}",
        "diff_logit_mean": float(np.mean(diff)),
        "diff_logit_lo": float(np.quantile(diff, lo_q)),
        "diff_logit_hi": float(np.quantile(diff, hi_q)),
        "prob_diff_pos": float(np.mean(diff > 0)),
    }


def write_run_metadata(context: StatisticalFitContext, extra: dict | None = None) -> None:
    """Persist a ``config.json`` and basic metrics for the report."""
    out = context.output_dir
    os.makedirs(out, exist_ok=True)
    spec = context.spec
    cfg = {
        "model_id": spec.model_id,
        "kind": spec.kind,
        "title": spec.title,
        "outcome_symbol": spec.outcome_symbol,
        "mechanism_symbol": spec.mechanism_symbol,
        "adjustment": spec.adjustment,
        "n_obs": context.prepared.n_obs if context.prepared else None,
        "n_children": context.prepared.n_children if context.prepared else None,
        "n_phases": context.prepared.n_phases if context.prepared else None,
        "dropped_rows": context.prepared.dropped_rows if context.prepared else None,
        "ci_prob": context.reporting.hdi,
        "sampling": {
            "draws": context.sampling.draws,
            "tune": context.sampling.tune,
            "chains": context.sampling.chains,
            "target_accept": context.sampling.target_accept,
            "random_seed": context.sampling.random_seed,
        },
        "extra": extra or {},
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def write_loo_summary(context: StatisticalFitContext) -> None:
    if context.loo is None:
        return
    out = context.output_dir
    path = os.path.join(out, "loo.txt")
    with open(path, "w") as f:
        f.write(str(context.loo))


def loo_delta(loo_a: az.ELPDData, loo_b: az.ELPDData) -> dict[str, float]:
    """Delta-ELPD between two models using ArviZ compare.

    arviz 1.x ``az.compare`` reports the ELPD in an ``elpd`` column (the 0.x
    ``elpd_loo`` was renamed); ``dse`` is unchanged.
    """
    df = az.compare({"a": loo_a, "b": loo_b})
    return {
        "d_elpd": float(df.loc["a", "elpd"] - df.loc["b", "elpd"]),
        "d_se": float(df.loc["a", "dse"]) if "dse" in df.columns else float("nan"),
    }


def factor_summary(
    trace: xr.DataTree,
    coef_names: list[str],
    *,
    ci_prob: float,
    causal_terms: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Per-coefficient posterior summary for a factor model (LRPGF / LRPLF, #127).

    One row per coefficient in ``coef_names`` present in the trace: posterior
    ``mean``, equal-tailed central interval at coverage ``ci_prob`` (``lo``/``hi``,
    same convention as :func:`tau_summary_itt`), and ``prob_positive`` =
    ``P(coef > 0)``. The ``role`` column labels each term **causal** (the
    randomised treatment terms named in ``causal_terms``) or **association** —
    under the locked DAG every non-randomised coefficient is an adjusted
    association confounded by latent general ability and must never be read as
    "drives".
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q

    def _row(term: str, base: str, d: np.ndarray) -> dict[str, object]:
        causal = term in causal_terms or base in causal_terms
        return {
            "term": term,
            "role": "causal" if causal else "association",
            "mean": float(np.mean(d)),
            "lo": float(np.quantile(d, lo_q)),
            "hi": float(np.quantile(d, hi_q)),
            "prob_positive": float(np.mean(d > 0)),
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
            for i in range(int(da.sizes[dim])):
                d = da.isel({dim: i}).stack(sample=("chain", "draw")).values.ravel()
                rows.append(_row(f"{name}[{i}]", name, d))
    return pd.DataFrame(rows)


def treatment_marginal_effect(
    trace: xr.DataTree,
    *,
    trt: np.ndarray,
    n_trials: int,
    term: str = "beta_trt",
    eta_name: str = "eta",
    ci_prob: float = 0.95,
) -> dict[str, float]:
    """Items-scale average marginal effect of the treatment term (LRPGF, #127).

    The gain model's logit predictor contains ``term * trt`` (``trt`` = the
    on-intervention indicator). Per posterior draw, the counterfactual
    operating point is ``eta0 = eta - term * trt`` (all off) and
    ``eta1 = eta0 + term`` (all on); the average marginal effect is the mean over
    observations of ``expit(eta1) - expit(eta0)``, reported on the probability
    scale and the items scale (``n_trials`` x probability) with an equal-tailed
    interval. ``prob_trt_pos`` is ``P(term > 0)`` on the logit scale.
    """
    post = trace.posterior
    eta = (
        post[eta_name]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "obs_id")
        .values
    )  # (S, n_obs)
    b = post[term].stack(sample=("chain", "draw")).values.ravel()  # (S,)
    trt = np.asarray(trt, dtype=float)  # (n_obs,)
    eta0 = eta - b[:, None] * trt[None, :]
    eta1 = eta0 + b[:, None]
    ame_prob = (expit(eta1) - expit(eta0)).mean(axis=1)  # (S,)
    ame_items = n_trials * ame_prob
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    return {
        "trt_prob_mean": float(np.mean(ame_prob)),
        "trt_prob_lo": float(np.quantile(ame_prob, lo_q)),
        "trt_prob_hi": float(np.quantile(ame_prob, hi_q)),
        "trt_items_mean": float(np.mean(ame_items)),
        "trt_items_lo": float(np.quantile(ame_items, lo_q)),
        "trt_items_hi": float(np.quantile(ame_items, hi_q)),
        "prob_trt_pos": float(np.mean(b > 0)),
    }
