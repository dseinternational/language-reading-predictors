# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Probes behind the mechanism-family HSGP reparameterisation (#438).

Reproducible record for ``notes/202607251500-mech-hsgp-reparameterisation.md``.

**Read the note before reading the numbers.** The decision rests on the *sensitivity
fit* (``--sensitivity`` below), not on the descriptive probe this file runs by
default. That probe answers two questions about the recorded divergent transition —
where it sits, and whether excluding it moves anything — and both answers are weak
evidence about the thing that actually matters. Excluding the flagged draw measures
that draw's influence, not posterior mass the sampler may never have reached; and
sitting inside one-dimensional marginals does not exclude problematic joint geometry.
They are retained because they are cheap and they bound one failure mode, not because
they establish the waiver (this distinction was the substance of the #439 review).

The evidence behind the decision is that an **independently reparameterised fit**
— HSGP basis count 6 and the tighter lengthscale prior, per #430/#434, which changes
the geometry and so the sampler's dynamics — samples with zero divergences and
reproduces every reported parameter. ``--sensitivity`` runs it and tabulates the
comparison; it costs a full reporting-tier fit (six chains, a few minutes).

Run from the repository root with the environment's Python:

    python notes/assets/202607251500-mech-hsgp-reparameterisation-probe.py
    python notes/assets/202607251500-mech-hsgp-reparameterisation-probe.py --sensitivity
"""

from __future__ import annotations

import warnings
from pathlib import Path

import arviz as az
import numpy as np

warnings.filterwarnings("ignore")

TRACE = Path("output/statistical_models/models/lrp-rli-mech-163-reporting/trace.nc")
FUNNEL_PRONE = ("sigma_child", "f_mech__ell", "f_mech__eta", "kappa")


def main() -> None:
    if not TRACE.exists():
        raise SystemExit(f"no trace at {TRACE}; fit the model at --config reporting first")

    idata = az.from_netcdf(TRACE)
    div = idata.sample_stats["diverging"].values
    n_div = int(div.sum())
    print(f"divergent draws: {n_div} of {div.size}")
    if n_div == 0:
        raise SystemExit("no divergences in this trace — nothing to probe")

    for chain, draw in np.argwhere(div):
        print(f"  located: chain {chain}, draw {draw} of {div.shape[1]}")

    post = idata.posterior
    scalars = sorted(
        v
        for v in post.data_vars
        if post[v].ndim == 2 and not v.endswith("_raw") and not v.startswith("f_mech__g")
    )

    print(f"\n{'parameter':<24}{'median (all)':>14}{'median (excl.)':>16}{'shift / SD':>12}")
    worst = 0.0
    for name in scalars:
        values = post[name].values
        flat = values.reshape(-1)
        kept = values[~div]
        sd = flat.std()
        shift = abs(np.median(flat) - np.median(kept)) / sd if sd else 0.0
        worst = max(worst, shift)
        print(f"{name:<24}{np.median(flat):>14.4f}{np.median(kept):>16.4f}{shift:>12.5f}")

    print(f"\nmax |median shift|: {worst:.5f} posterior SD")

    print("\nlocation of the divergent draw(s) on the funnel-prone scales:")
    for name in FUNNEL_PRONE:
        if name not in post:
            continue
        values = post[name].values
        lo, mid, hi = (np.quantile(values, q) for q in (0.055, 0.5, 0.945))
        at = ", ".join(f"{values[c, d]:.4f}" for c, d in np.argwhere(div))
        print(f"  {name:<14} divergent at {at:<10} posterior 5.5/50/94.5% = {lo:.4f} / {mid:.4f} / {hi:.4f}")


def sensitivity() -> None:
    """Refit under the #430/#434 reparameterisation and compare posterior summaries.

    Different geometry means different sampler dynamics, so a clean run here that
    agrees with the flagged fit is evidence the flagged fit explored the space — which
    the descriptive probe above cannot show.
    """
    import importlib

    import pymc as pm

    from language_reading_predictors.statistical_models import mechanism as M
    from language_reading_predictors.statistical_models import priors as _priors

    spec = importlib.import_module(
        "language_reading_predictors.statistical_models.lrp_rli_mech_163"
    ).SPEC
    plan = M.resolve_mechanism_plan(spec)
    kwargs = dict(plan.factory_kwargs)
    kwargs["mech_hsgp_m"] = 6
    kwargs["mech_lengthscale_prior"] = _priors.ell_prior_mech_tight()
    alt = M.MechanismPlan(
        spec=spec,
        prepared=plan.prepared,
        factory_kwargs=kwargs,
        confounders=plan.confounders,
        adjust_for=plan.adjust_for,
    )
    built = M.build_mechanism_for_plan(alt)
    with built.model:
        refit = pm.sample(
            draws=6000,
            tune=6000,
            chains=6,
            target_accept=0.999,
            random_seed=20260726,
            nuts_sampler="nutpie",
            progressbar=False,
            compute_convergence_checks=False,
        )

    divergences = int(np.asarray(refit.sample_stats["diverging"].values).sum())
    summary = az.summary(refit, round_to=None)
    print(
        f"reparameterised fit: divergences={divergences} "
        f"max R-hat={np.nanmax(summary['r_hat']):.4f} "
        f"min ESS={np.nanmin(summary['ess_bulk']):.0f}"
    )

    original = az.from_netcdf(TRACE)
    print(f"\n{'parameter':<24}{'original':>13}{'reparam':>13}{'shift / SD':>12}")
    for name in sorted(
        v
        for v in original.posterior.data_vars
        if original.posterior[v].ndim == 2
        and not v.endswith("_raw")
        and not v.startswith("f_mech__g")
    ):
        if name not in refit.posterior:
            continue
        a = original.posterior[name].values.reshape(-1)
        b = refit.posterior[name].values.reshape(-1)
        shift = abs(np.median(a) - np.median(b)) / a.std() if a.std() else 0.0
        print(f"{name:<24}{np.median(a):>13.4f}{np.median(b):>13.4f}{shift:>12.3f}")


if __name__ == "__main__":
    import sys

    if "--sensitivity" in sys.argv:
        sensitivity()
    else:
        main()
