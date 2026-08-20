# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Is the uniform negative ``gamma_int`` moderation, or letter-sound ceiling curvature?

Every moderated letter-sound → word-reading fit returns a negative ``gamma_int``
of about the same size whatever the moderator. ``notes/202608191600-moderation-
items-scale-not-2d-surface.md`` records a competing explanation: the
Haldane-corrected logit stretches the top of the letter-sound scale, so the
standardised exposure ``z_L`` is dominated by the rows at the ceiling, and
``z_L·z_M`` correlates about 0.7 with ``z_L²``. A negative interaction could
therefore be curvature in the exposure that a six-function HSGP with a tight
lengthscale cannot express, rather than moderation.

That note names the cheap check: **one moderated refit with the interaction built
on the count-standardised or top-clipped letter-sound score**. This script is
that check (issue #554, item 6).

It is exact rather than approximate because ``z_mech_logit`` enters *only* the
interaction in a curve-mechanism moderated fit — the mechanism main effect is an
HSGP on the raw logit — so replacing that one data vector changes the interaction
basis and nothing else. The model, its rows, its priors and its sampling settings
are the registered fit's own.

Usage::

    python scripts/mech_ceiling_check.py                    # mech-061, reporting
    python scripts/mech_ceiling_check.py --models lrp-rli-mech-061 lrp-rli-mech-063

Writes ``ceiling_check.csv`` to ``output/statistical_models/mech_ceiling_check/``.
Diagnostic only: it writes nothing into a fit directory and changes no release
decision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import dse_research_utils.environment.setup as setup  # noqa: E402
import dse_research_utils.statistics.models.sampling as _sampling  # noqa: E402

from language_reading_predictors import paths  # noqa: E402
from language_reading_predictors.statistical_models import factories  # noqa: E402
from language_reading_predictors.statistical_models.preprocessing import (  # noqa: E402
    load_and_prepare,
    logit_safe,
)
from language_reading_predictors.statistical_models.context import (  # noqa: E402
    spec_target_accept,
)
from language_reading_predictors.statistical_models.mechanism import (  # noqa: E402
    resolve_mechanism_run_plan,
)
from language_reading_predictors.statistical_models.registry import (  # noqa: E402
    discover_models,
)

DEFAULT_MODELS = ("lrp-rli-mech-061",)


def _standardise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=0)
    if sd == 0:
        raise ValueError("cannot standardise a constant exposure")
    return (values - values.mean()) / sd


def interaction_bases(counts: np.ndarray, n_trials: int) -> dict[str, np.ndarray]:
    """The registered basis and the two alternatives the note names.

    ``logit`` reproduces the fit as registered. ``count`` standardises the raw
    count, which gives the top items no more room than the middle ones.
    ``clipped`` keeps the logit but caps the count at the 90th percentile first,
    so the ceiling rows stop dominating the squared term.
    """
    cap = float(np.quantile(counts, 0.9))
    clipped = np.minimum(counts, cap)
    return {
        "logit": _standardise(logit_safe(counts, n_trials)),
        "count": _standardise(counts),
        "clipped": _standardise(logit_safe(clipped, n_trials)),
    }


def check_model(model_id: str, config: str, spec) -> list[dict[str, Any]]:
    plan = resolve_mechanism_run_plan(spec)
    print(f"{model_id} ({plan.outcome_symbol} moderated by {plan.moderator_symbol})")
    loaded = load_and_prepare(**plan.prepare_kwargs())
    built = factories.build_mechanism_model(loaded, **plan.factory_kwargs())
    model = built.model
    # The factory drops rows with a missing confounder post-score, so the basis
    # must be built from the rows it actually kept, not from the loaded frame.
    prepared = built.prepared

    counts = np.asarray(prepared.post_counts[plan.mechanism_symbol], dtype=float)
    n_trials = int(prepared.n_trials[plan.mechanism_symbol])
    bases = interaction_bases(counts, n_trials)

    sampling = _sampling.get_sampling_configuration(config)
    # Honour the module's own target_accept. mech-061 declares 0.999 because its
    # HSGP geometry needs it; taking the preset's value would LOWER acceptance and
    # make the comparison a comparison of samplers rather than of bases.
    declared = spec_target_accept(spec)
    target_accept = float(declared if declared is not None else sampling.target_accept)
    z_moderator = np.asarray(model["z_moderator"].get_value(), dtype=float)
    rows: list[dict[str, Any]] = []
    for name, z in bases.items():
        with model:
            pm.set_data({"z_mech_logit": z})
            trace = pm.sample(
                draws=sampling.draws,
                tune=sampling.tune,
                chains=sampling.chains,
                target_accept=target_accept,
                random_seed=sampling.random_seed,
                nuts_sampler="nutpie",
                progressbar=False,
            )
        draws = np.asarray(trace.posterior["gamma_int"].values).reshape(-1)
        divergences = int(np.asarray(trace.sample_stats["diverging"].values).sum())
        rows.append(
            {
                "model_id": model_id,
                "config": config,
                "interaction_basis": name,
                "n_rows": int(prepared.n_obs),
                "gamma_int_median": float(np.median(draws)),
                "gamma_int_lo": float(np.quantile(draws, 0.055)),
                "gamma_int_hi": float(np.quantile(draws, 0.945)),
                "prob_negative": float((draws < 0).mean()),
                # The note's proposed mechanism needs the interaction column to
                # be largely the squared exposure: it reports corr(z_L·z_M, z_L²)
                # of about 0.7 on the registered basis.
                "corr_interaction_with_zsq": float(
                    np.corrcoef(z * z_moderator, z**2)[0, 1]
                ),
                "share_of_interaction_ss_top_decile": float(
                    np.sort((z * z_moderator) ** 2)[-max(1, len(z) // 10):].sum()
                    / ((z * z_moderator) ** 2).sum()
                ),
                "target_accept": target_accept,
                "n_divergences": divergences,
            }
        )
        print(f"  {name:8s} gamma_int {rows[-1]['gamma_int_median']:+.3f} "
              f"[{rows[-1]['gamma_int_lo']:+.3f}, {rows[-1]['gamma_int_hi']:+.3f}] "
              f"P(neg)={rows[-1]['prob_negative']:.3f} div={divergences}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--config", default="reporting")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    setup.init_script()
    paths.set_output_root(args.output_dir)
    registry = discover_models()

    rows: list[dict[str, Any]] = []
    for model_id in args.models:
        entry = registry.get(model_id)
        if entry is None:
            print(f"unknown model {model_id}", file=sys.stderr)
            return 2
        spec = entry.load().SPEC if hasattr(entry, "load") else entry.SPEC
        rows.extend(check_model(model_id, args.config, spec))

    out_dir = paths.stat_dir() / "mech_ceiling_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    path = out_dir / "ceiling_check.csv"
    frame.to_csv(path, index=False)
    print(f"\nWrote {path}")
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
