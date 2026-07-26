# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prototype likelihood sensitivities for the bounded-count working model.

Two one-off sensitivity fits accompanying
notes/202607261405-binomial-exchangeability-item-difficulty-review.md. Both mirror
the uniform DAG-faithful single-outcome ITT structure (``factories.build_itt_model``:
``eta = alpha + gamma_own * pre_logit + gamma_A * A_std + tau * G``), use the shared
priors from ``priors.py``, load rows with
``preprocessing.load_and_prepare(phase_mode="itt", outcomes=(symbol,))``, and differ
ONLY in the observation model — so any difference in the treatment summaries is
attributable to the likelihood alone.

``blending-chance-floor``
    ITT B (``lrp-rli-itt-008`` structure). Standard Beta-Binomial versus a
    guessing-floor link ``mu = 1/3 + (2/3) * sigmoid(eta)``: blending is
    three-alternative forced choice, so the expected score cannot fall below chance
    (about 3.3 of 10), yet the standard logit link admits sub-chance means. The
    shared priors are kept identical in both arms so the comparison isolates the
    link; note ``gamma_own``'s autoregressive centring at 1 was calibrated for the
    identity relation between pre- and post-logit and is retained unchanged.

``word-reading-cmb``
    ITT W (``lrp-rli-itt-010`` structure). Beta-Binomial versus Conway-Maxwell-
    binomial (Kadane 2016, DOI 10.1214/15-BA955), which can express UNDER- as well
    as over-dispersion relative to the Binomial (``nu`` > 1 underdispersed, = 1
    Binomial, < 1 overdispersed). Heterogeneous item difficulty makes the score
    conditionally underdispersed (Poisson-binomial), which the Beta-Binomial cannot
    represent — its variance floor is the Binomial.

These are NOT registered models: no diagnostics gate, no report artefacts, and
lighter sampling than the reporting preset. Promotion path if adopted: a
``likelihood=``/link option in ``factories.py`` with guard tests, a registered
variant spec, and the usual pipeline artefacts.

Run from the repo root (worktree checkouts need ``PYTHONPATH=src``):
    python scripts/likelihood_sensitivity_prototypes.py blending-chance-floor
    python scripts/likelihood_sensitivity_prototypes.py word-reading-cmb
"""

from __future__ import annotations

import argparse

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import gammaln
from scipy.stats import betabinom as sp_betabinom

from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.preprocessing import load_and_prepare

EPSILON = 1e-6
SEED = 20260726
Q50 = (0.25, 0.75)
Q89 = (0.055, 0.945)


def _summary(x: np.ndarray, label: str) -> str:
    med = np.median(x)
    lo50, hi50 = np.quantile(x, Q50)
    lo89, hi89 = np.quantile(x, Q89)
    return (f"{label}: median {med:+.3f}  50% [{lo50:+.3f}, {hi50:+.3f}]  "
            f"89% [{lo89:+.3f}, {hi89:+.3f}]  P(>0) = {(x > 0).mean():.3f}")


def _diagnostics(idata) -> str:
    # Mirrors statistical_models.diagnostics: ``round_to="none"`` (the string) genuinely
    # disables rounding, so R-hat/ESS are read unrounded — ``round_to=None`` falls through
    # to rcParams' 2 significant figures and would report a rounded R-hat.
    summ = az.summary(idata, round_to="none", kind="diagnostics")
    rhat = float(summ["r_hat"].max())
    ess = float(min(summ["ess_bulk"].min(), summ["ess_tail"].min()))
    div = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    return f"max R-hat {rhat:.4f}, min ESS {ess:.0f}, divergences {div}"


def _fit(symbol: str, observation: str, draws: int, tune: int, chains: int,
         target_accept: float, sampler: str):
    """Fit one ITT-structure variant; ``observation`` in {bb, bb_chance_floor, cmb}."""
    prepared = load_and_prepare(phase_mode="itt", outcomes=(symbol,))
    post = prepared.post_counts[symbol]
    keep = ~np.isnan(post)
    pre = prepared.pre_logit[symbol][keep]
    a_std = prepared.A_std[keep]
    g = prepared.G.astype(float)[keep]
    y = post[keep].astype(np.int64)
    n_trials = prepared.n_trials[symbol]
    n_obs = int(keep.sum())

    with pm.Model(coords={"obs_id": np.arange(n_obs)}) as model:
        alpha = _priors.alpha_prior().to_pymc("alpha")
        tau = _priors.tau_prior().to_pymc("tau")
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
        gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
        eta = alpha + gamma_own * pre + gamma_A * a_std + tau * g

        if observation in ("bb", "bb_chance_floor"):
            kappa = _priors.kappa_prior().to_pymc("kappa")
            sig = pm.math.sigmoid(eta)
            mu_raw = 1.0 / 3.0 + (2.0 / 3.0) * sig if observation == "bb_chance_floor" else sig
            mu = pm.math.clip(mu_raw, EPSILON, 1 - EPSILON)
            pm.BetaBinomial("y_post", n=n_trials, alpha=mu * kappa, beta=(1 - mu) * kappa,
                            observed=y, dims="obs_id")
        else:  # cmb
            nu = pm.LogNormal("nu", mu=0.0, sigma=0.5)
            j = np.arange(n_trials + 1)
            log_c = gammaln(n_trials + 1) - gammaln(j + 1) - gammaln(n_trials + 1 - j)
            terms = nu * pt.constant(log_c)[None, :] + eta[:, None] * pt.constant(j.astype(float))[None, :]
            m = pt.max(terms, axis=1, keepdims=True)
            log_z = (m + pt.log(pt.sum(pt.exp(terms - m), axis=1, keepdims=True)))[:, 0]
            loglik = nu * pt.constant(log_c[y]) + y * eta - log_z
            pm.Potential("y_cmb", loglik.sum())

        kwargs = dict(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                      random_seed=SEED, progressbar=False)
        if sampler == "nutpie":
            try:
                idata = pm.sample(nuts_sampler="nutpie", **kwargs)
            except Exception as exc:  # pragma: no cover - environment-dependent
                print(f"  [nutpie unavailable for {observation} ({exc!r}); falling back to pymc]")
                idata = pm.sample(**kwargs)
        else:
            idata = pm.sample(**kwargs)
        if observation != "cmb":
            pm.compute_log_likelihood(idata)

    data = {"pre": pre, "a_std": a_std, "g": g, "y": y, "n_trials": n_trials, "model": model}
    return data, idata


def _draws(idata, names, thin_to=1500):
    post = idata.posterior
    flat = {k: post[k].values.reshape(-1) for k in names}
    n = len(next(iter(flat.values())))
    idx = np.linspace(0, n - 1, min(thin_to, n)).astype(int)
    return {k: v[idx] for k, v in flat.items()}


def _eta_by_arm(dr, data):
    base = (dr["alpha"][:, None] + dr["gamma_own"][:, None] * data["pre"][None, :]
            + dr["gamma_A"][:, None] * data["a_std"][None, :])
    return base + dr["tau"][:, None], base  # (eta with G=1, eta with G=0)


def _cmb_pmf(eta, nu, n_trials):
    """Predictive pmf per (draw, obs) over 0..n; eta (D,N), nu (D,)."""
    j = np.arange(n_trials + 1)
    log_c = gammaln(n_trials + 1) - gammaln(j + 1) - gammaln(n_trials + 1 - j)
    logw = nu[:, None, None] * log_c[None, None, :] + eta[:, :, None] * j[None, None, :]
    logw -= logw.max(axis=2, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=2, keepdims=True)


def _coverage_from_pmf(pmf_rows, y, level):
    """Equal-tailed predictive-interval coverage from a per-row pmf over 0..n."""
    cdf = np.cumsum(pmf_rows, axis=1)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    lo = (cdf < lo_q).sum(axis=1)
    hi = (cdf < hi_q).sum(axis=1)
    return float(((y >= lo) & (y <= hi)).mean())


def _ppc_coverage(observation, dr, data, thin_to=400):
    idx = np.linspace(0, len(dr["tau"]) - 1, min(thin_to, len(dr["tau"]))).astype(int)
    sub = {k: v[idx] for k, v in dr.items()}
    eta1, eta0 = _eta_by_arm(sub, data)
    eta = np.where(data["g"][None, :] == 1.0, eta1, eta0)
    n_trials = data["n_trials"]
    if observation == "cmb":
        pmf = _cmb_pmf(eta, sub["nu"], n_trials).mean(axis=0)
    else:
        sig = 1 / (1 + np.exp(-eta))
        mu = 1 / 3 + (2 / 3) * sig if observation == "bb_chance_floor" else sig
        kk = np.arange(n_trials + 1)[None, None, :]
        a = (mu * sub["kappa"][:, None])[:, :, None]
        b = ((1 - mu) * sub["kappa"][:, None])[:, :, None]
        pmf = sp_betabinom.pmf(kk, n_trials, a, b).mean(axis=0)
    return {lvl: _coverage_from_pmf(pmf, data["y"], lvl) for lvl in (0.5, 0.9)}


def _ame_items(observation, dr, data):
    """Average marginal effect on the items scale, toggling G over all analysis rows."""
    eta1, eta0 = _eta_by_arm(dr, data)
    n_trials = data["n_trials"]
    if observation == "cmb":
        m1 = (_cmb_pmf(eta1, dr["nu"], n_trials) * np.arange(n_trials + 1)).sum(axis=2)
        m0 = (_cmb_pmf(eta0, dr["nu"], n_trials) * np.arange(n_trials + 1)).sum(axis=2)
        return (m1 - m0).mean(axis=1)
    sig1, sig0 = 1 / (1 + np.exp(-eta1)), 1 / (1 + np.exp(-eta0))
    if observation == "bb_chance_floor":
        sig1, sig0 = 1 / 3 + (2 / 3) * sig1, 1 / 3 + (2 / 3) * sig0
    return n_trials * (sig1 - sig0).mean(axis=1)


def run_pair(symbol, variants, labels, args):
    results = {}
    for obs, label in zip(variants, labels, strict=True):
        print(f"\n--- fitting {label} ({obs}) for outcome {symbol} ---")
        data, idata = _fit(symbol, obs, args.draws, args.tune, args.chains,
                           args.target_accept, args.sampler)
        print(f"  n_obs = {len(data['y'])}, n_trials = {data['n_trials']};  {_diagnostics(idata)}")
        names = ["alpha", "tau", "gamma_own", "gamma_A"] + (["nu"] if obs == "cmb" else ["kappa"])
        dr = _draws(idata, names)
        print("  " + _summary(dr["tau"], "tau (logit)"))
        print("  " + _summary(_ame_items(obs, dr, data), "AME (items)"))
        print("  " + _summary(dr["gamma_A"], "gamma_A"))
        print("  " + _summary(dr["gamma_own"], "gamma_own"))
        disp = "nu" if obs == "cmb" else "kappa"
        print("  " + _summary(dr[disp], disp))
        cov = _ppc_coverage(obs, dr, data)
        print(f"  predictive coverage: 50% band {cov[0.5]:.3f}, 90% band {cov[0.9]:.3f}")
        results[label] = (data, idata)
    both_ll = [id_ for (_, id_) in results.values() if hasattr(id_, "log_likelihood")]
    if len(both_ll) == len(results) == 2:
        try:
            comp = az.compare({lbl: id_ for lbl, (_, id_) in results.items()})
            cols = [c for c in ("rank", "elpd_loo", "elpd_diff", "dse", "p_loo") if c in comp.columns]
            print("\nPSIS-LOO comparison:")
            print(comp[cols].to_string())
        except Exception as exc:  # pragma: no cover - ArviZ version dependent
            print(f"\n[PSIS-LOO comparison unavailable: {exc!r}]")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", choices=["blending-chance-floor", "word-reading-cmb"])
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.92)
    ap.add_argument("--sampler", choices=["nutpie", "pymc"], default="nutpie")
    args = ap.parse_args()

    if args.task == "blending-chance-floor":
        run_pair("B", ["bb", "bb_chance_floor"],
                 ["standard Beta-Binomial", "chance-floor Beta-Binomial"], args)
    else:
        run_pair("W", ["bb", "cmb"],
                 ["standard Beta-Binomial", "Conway-Maxwell-binomial"], args)


if __name__ == "__main__":
    main()
