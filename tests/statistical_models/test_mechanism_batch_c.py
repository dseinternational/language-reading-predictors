# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""#586 Batch C regressions for the ``mechanism`` family (#602, #603, #604, #605).

Four separate defects, one test module because they share a factory and a set of
writers:

* **#602** — the family published *two* different natural-scale answers to the same
  question in the same report, over different exposure intervals and under different
  standardisations, with nothing declaring which was the headline. There is now one
  declared estimand, and the tests below pin that ``mechanism_summary.csv``, the
  items worked example and ``key_findings.json`` all carry the *same* number.
* **#603** — one exposure coefficient over a child random intercept is a
  precision-weighted blend of the between-child and within-child associations. The
  Mundlak split is pinned to be built on the *fitted* rows and to sum back to the
  exposure exactly.
* **#604** — the pooled exposure slope assumes stability across three substantively
  different treatment histories. The per-period slopes are pinned to be
  partially-pooled and linear-only.
* **#605** — ``kappa ~ HalfNormal(50)`` enforces a floor on overdispersion at high
  denominators. The alternative family is pinned to reach the near-Binomial limit
  that the registered one cannot.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models import mechanism_items as mi
from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
    resolve_mechanism_run_plan,
)
from language_reading_predictors.statistical_models.pipelines import (
    mechanism as mech_pipeline,
)
from language_reading_predictors.statistical_models.preprocessing import (
    logit_safe,
    standardise,
)
from language_reading_predictors.statistical_models.reporting import (
    generate_key_findings,
)


# ---------------------------------------------------------------------------
# Fixtures: a synthetic linear mechanism fit, as the writers see one
# ---------------------------------------------------------------------------


def _dataset(spec: dict) -> xr.Dataset:
    coords: dict[str, np.ndarray] = {}
    for dims, values in spec.values():
        arr = np.asarray(values)
        for k, dim in enumerate(dims):
            coords.setdefault(dim, np.arange(arr.shape[k]))
    return xr.Dataset({n: (d, v) for n, (d, v) in spec.items()}, coords=coords)


def _linear_fit(tmp_path: Path, *, n_trials_exposure: int = 32):
    """A synthetic pooled-linear mechanism fit and a context the writers accept."""
    rng = np.random.default_rng(11)
    # Chosen so the 25th and 75th percentiles land exactly on observed values
    # (v[2] == v[3] and v[8] == v[9] for a 12-row sample), which makes the
    # worked-example points rows of the plotted curve rather than interpolations.
    counts = np.array(
        [4.0, 8.0, 16.0, 16.0, 18.0, 20.0, 20.0, 22.0, 24.0, 24.0, 28.0, 32.0]
    )
    z_mech, scaler = standardise(logit_safe(counts, n_trials_exposure))
    n_obs = counts.size
    beta = rng.normal(0.4, 0.05, size=(2, 25))
    child_idx = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    u_child = rng.normal(0.0, 0.4, size=(2, 25, 4))
    baseline = rng.normal(0.0, 0.3, size=(2, 25, n_obs))
    eta = (
        baseline
        + beta[:, :, None] * z_mech[None, None, :]
        + u_child[:, :, child_idx]
    )
    posterior = _dataset(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "beta_mech": (("chain", "draw"), beta),
            "u_child": (("chain", "draw", "child"), u_child),
            "kappa": (("chain", "draw"), np.full((2, 25), 60.0)),
        }
    )
    constant = _dataset(
        {
            "z_mech_logit": (("obs_id",), z_mech),
            "child_idx": (("obs_id",), child_idx),
            "phase_idx": (("obs_id",), np.array([0, 1, 2] * 4)),
        }
    )
    spec = ModelSpec(
        model_id="lrp-rli-mech-test",
        kind="mechanism",
        title="test",
        outcome_symbol="W",
        mechanism_symbol="L",
        adjustment=["G", "A", "W_pre"],
        model_settings=MechanismModelSettings(
            outcomes=("W", "L"), linear_mechanism=True
        ),
    )
    ctx = SimpleNamespace(
        trace=SimpleNamespace(posterior=posterior, constant_data=constant),
        spec=spec,
        resolved_plan=resolve_mechanism_run_plan(spec),
        prepared=SimpleNamespace(
            post_counts={"L": counts},
            n_trials={"L": n_trials_exposure, "W": 79},
            covariates={},
            covariate_scalers={},
            n_obs=n_obs,
            phase=np.array([0, 1, 2] * 4),
            G=np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),
            child_idx=child_idx,
        ),
        reporting=SimpleNamespace(ci_prob=0.89),
        output_dir=str(tmp_path),
        tables={},
    )
    return ctx, counts, z_mech, scaler, beta


# ---------------------------------------------------------------------------
# #602 — one declared estimand, used consistently
# ---------------------------------------------------------------------------


def test_headline_matches_across_summary_worked_example_and_key_findings(
    tmp_path, monkeypatch
):
    """The one number a reader quotes must be one number, wherever they read it.

    Before #602 the key-findings headline came from a full-observed-range,
    probability-averaged contrast and the items curve's worked example from an
    interquartile, typical-child one — +6.64 against +2.32 items on the same
    ``lrp-rli-mech-058`` report, with nothing saying which to quote.
    """
    monkeypatch.setattr(mi, "save_styled_figure", lambda *_a, **_k: None)
    ctx, *_ = _linear_fit(tmp_path)

    worked = mech_pipeline._write_mechanism_items(ctx)

    summary = pd.read_csv(tmp_path / "mechanism_summary.csv")
    headline = summary.iloc[0]
    assert headline["contrast"] == "headline_interquartile"
    assert headline["estimand"] == mi.HEADLINE_ESTIMAND
    # The worked example annotated on the figure is the same contrast, not another.
    assert headline["items_median"] == pytest.approx(
        worked["outcome_difference_median"]
    )
    assert headline["exposure_low"] == pytest.approx(worked["exposure_ref_low"])
    assert headline["exposure_high"] == pytest.approx(worked["exposure_ref_high"])

    # ... and so is the key-findings headline, which reads the first summary row.
    _write_publishable_fit_dir(tmp_path, ctx.spec.model_id)
    payload = generate_key_findings(tmp_path)
    assert payload["status"] == "ok"
    assert payload["headline_estimand"]["estimand"] == mi.HEADLINE_ESTIMAND
    assert payload["headline_estimand"]["items_median"] == pytest.approx(
        worked["outcome_difference_median"]
    )
    text = payload["sentences"][0]["text"]
    assert f"{worked['outcome_difference_median']:+.1f} items" in text
    assert "75th and 25th percentile" in text


def test_secondary_contrast_is_present_and_labelled(tmp_path, monkeypatch):
    """The observed-range contrast is retained, explicitly as a secondary."""
    monkeypatch.setattr(mi, "save_styled_figure", lambda *_a, **_k: None)
    ctx, counts, *_ = _linear_fit(tmp_path)

    mech_pipeline._write_mechanism_items(ctx)

    summary = pd.read_csv(tmp_path / "mechanism_summary.csv")
    assert list(summary["contrast"]) == [
        "headline_interquartile",
        "secondary_observed_range",
    ]
    secondary = summary.iloc[1]
    assert secondary["estimand"] == mi.SECONDARY_ESTIMAND
    assert secondary["exposure_low"] == pytest.approx(counts.min())
    assert secondary["exposure_high"] == pytest.approx(counts.max())
    # Same reference population on both rows: only the interval differs.
    assert set(summary["reference_population"]) == {"fitted_rows"}
    assert set(summary["child_intercept"]) == {"retained_at_fitted_value"}


def test_the_curve_and_the_worked_example_agree_at_the_reference_points(
    tmp_path, monkeypatch
):
    """The annotated points lie **on** the plotted curve.

    They did not before #602: the curve and its worked example were the same
    quantity, but the headline the report published beside them was a different
    standardisation, so a reader comparing the figure with the box saw two numbers
    that could not both be read off the same line. Checked against a hand-computed
    standardisation, so it pins the definition and not merely internal agreement.
    """
    monkeypatch.setattr(mi, "save_styled_figure", lambda *_a, **_k: None)
    ctx, counts, z_mech, _scaler, beta = _linear_fit(tmp_path)

    worked = mech_pipeline._write_mechanism_items(ctx)
    curve = pd.read_csv(tmp_path / "mechanism_curve_items.csv")

    eta = (
        ctx.trace.posterior["eta"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )
    beta_flat = beta.reshape(-1)
    eta_base = eta - z_mech[:, None] * beta_flat[None, :]

    def y_at(count: float) -> np.ndarray:
        j = int(np.flatnonzero(counts == count)[0])
        return 79.0 * expit(eta_base + z_mech[j] * beta_flat[None, :]).mean(axis=0)

    # The reference points fall on observed exposure values, so the curve's own rows
    # carry them and the contrast is the gap between two points on the plotted line.
    for ref, column in (
        (worked["exposure_ref_low"], "predicted_low_median"),
        (worked["exposure_ref_high"], "predicted_high_median"),
    ):
        row = curve.loc[curve["exposure"] == ref]
        assert len(row) == 1
        draws = y_at(ref)
        assert float(row["outcome_mean"].iloc[0]) == pytest.approx(draws.mean())
        assert worked[column] == pytest.approx(float(np.median(draws)))
    assert worked["outcome_difference_median"] == pytest.approx(
        float(np.median(y_at(worked["exposure_ref_high"]) - y_at(worked["exposure_ref_low"])))
    )


def test_standardisation_is_not_the_link_of_an_average(tmp_path, monkeypatch):
    """``mean_i expit(eta_i)`` is not ``expit(mean_i eta_i)`` on a nonlinear link.

    The pre-#602 items curve applied the link to a row-averaged linear predictor,
    which is why the two published numbers would have disagreed even on an identical
    exposure interval. Pinned with a deliberately dispersed baseline, where Jensen's
    inequality makes the two visibly different.
    """
    monkeypatch.setattr(mi, "save_styled_figure", lambda *_a, **_k: None)
    ctx, counts, z_mech, _scaler, beta = _linear_fit(tmp_path)
    eta = (
        ctx.trace.posterior["eta"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )
    beta_flat = beta.reshape(-1)
    eta_base = eta - z_mech[:, None] * beta_flat[None, :]

    curve, _worked = mi.mechanism_items_curve(
        ctx.trace,
        x_exposure=counts,
        n_trials_outcome=79,
        exposure_n_trials=32,
        ci_prob=0.89,
    )
    row = curve.loc[curve["exposure"] == counts[0]].iloc[0]
    row_standardised = 79.0 * expit(eta_base + z_mech[0] * beta_flat[None, :]).mean(
        axis=0
    )
    typical_child = 79.0 * expit(eta_base.mean(axis=0) + z_mech[0] * beta_flat)
    assert row["outcome_mean"] == pytest.approx(row_standardised.mean())
    assert row["outcome_mean"] != pytest.approx(typical_child.mean(), rel=1e-6)


def test_items_scale_steepest_interval_is_published_beside_the_logit_one(
    tmp_path, monkeypatch
):
    """"Where do outcome *items* rise fastest?" gets its own answer (#602).

    ``_readiness_knee`` locates the steepest interval on the outcome-*logit* scale;
    the expected-items derivative carries an extra ``p (1 - p)`` factor from the
    inverse link and can peak elsewhere. Publishing the items-scale interval under
    the declared reference population is the alternative to relabelling the logit
    statistic — so the two must be present together and be separately labelled.
    """
    monkeypatch.setattr(mi, "save_styled_figure", lambda *_a, **_k: None)
    rng = np.random.default_rng(3)
    counts = np.arange(0, 33, dtype=float)
    n_obs = counts.size
    ell = logit_safe(counts, 32)
    # A curve that is flat, then rises: a real steepest interval to find.
    f = np.broadcast_to(
        1.4 / (1.0 + np.exp(-(counts - 20.0) / 2.0)), (2, 40, n_obs)
    ).copy()
    baseline = rng.normal(-1.0, 0.3, size=(2, 40, n_obs))
    posterior = _dataset(
        {
            "eta": (("chain", "draw", "obs_id"), baseline + f),
            "f_mech": (("chain", "draw", "obs_id"), f),
        }
    )
    constant = _dataset({"mech_post_logit": (("obs_id",), ell)})
    spec = ModelSpec(
        model_id="lrp-rli-mech-test-gp",
        kind="mechanism",
        title="test",
        outcome_symbol="W",
        mechanism_symbol="L",
        adjustment=["G", "A", "W_pre"],
        model_settings=MechanismModelSettings(outcomes=("W", "L")),
    )
    ctx = SimpleNamespace(
        trace=SimpleNamespace(posterior=posterior, constant_data=constant),
        spec=spec,
        resolved_plan=resolve_mechanism_run_plan(spec),
        prepared=SimpleNamespace(
            post_counts={"L": counts},
            n_trials={"L": 32, "W": 79},
            covariates={},
            covariate_scalers={},
            n_obs=n_obs,
        ),
        reporting=SimpleNamespace(ci_prob=0.89),
        output_dir=str(tmp_path),
        tables={},
    )
    monkeypatch.setattr(
        mech_pipeline, "save_styled_figure", lambda *_a, **_k: None
    )

    mech_pipeline._write_readiness_threshold(ctx)

    row = pd.read_csv(tmp_path / "readiness_threshold.csv").iloc[0]
    assert row["scale"] == "latent_logit"
    assert row["items_scale"] == "expected_items"
    assert np.isfinite(row["items_knee_count_median"])
    # Both are located over the same observed exposure range.
    assert 0.0 <= row["items_knee_count_median"] <= 32.0


def _write_publishable_fit_dir(d: Path, model_id: str) -> None:
    """The minimum stored-fit inventory ``generate_key_findings`` requires."""
    from language_reading_predictors.statistical_models import release as _release

    with open(d / "config.json", "w") as fh:
        json.dump(
            {
                "model_id": model_id,
                "kind": "mechanism",
                "outcome_symbol": "W",
                "mechanism_symbol": "L",
                "title": "test",
                "extra": {},
            },
            fh,
        )
    with open(d / "diagnostics_summary.json", "w") as fh:
        json.dump(
            {
                "passed": True,
                "checks": {
                    "rhat": True,
                    "ess": True,
                    "divergences": True,
                    "bfmi": True,
                    "diagnostics_assessable": True,
                },
                "divergences": 0,
                "max_rhat": 1.001,
                "min_ess": 1000.0,
                "bfmi_per_chain": [0.8, 0.9],
            },
            fh,
        )
    for name in _release._CORE_ARTIFACTS_BASE:
        path = d / name
        if not path.exists():
            path.write_bytes(b"fixture")
    with open(d / "artifact_manifest.json", "w") as fh:
        json.dump(
            {
                "artifacts": [
                    {"filename": name, "status": "written", "required": True}
                    for name in _release._CORE_ARTIFACTS_BASE
                ]
            },
            fh,
        )


# ---------------------------------------------------------------------------
# #603 — the between/within (Mundlak) split
# ---------------------------------------------------------------------------


def _synthetic_frame(tmp_path: Path, n_children: int = 20):
    # Relative, not ``tests.statistical_models.test_factories``: pytest imports these
    # modules as the ``statistical_models`` package (``tests/`` has no ``__init__.py``,
    # so it is the basedir pytest prepends to sys.path, not a package). The absolute
    # form only resolves when the repo root happens to be on sys.path — which
    # ``python -m pytest`` arranges and the ``pytest`` entry point CI runs does not.
    from .test_factories import _write_synthetic

    return _write_synthetic(tmp_path, n_children=n_children)


def test_mundlak_vectors_are_built_on_fitted_rows_and_sum_back(tmp_path):
    """The split must be of the *fitted* exposure, and lossless.

    Two properties, one test: the child mean is taken over the rows the model keeps
    (not the pre-keep-mask frame the loader returns), and ``child_mean +
    within_dev == z`` exactly, so the reparameterisation adds a question rather than
    changing the model's fit.
    """
    from language_reading_predictors.statistical_models.factories import (
        build_mechanism_model,
    )
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    path = _synthetic_frame(tmp_path)
    prepared = load_and_prepare(path=path, phase_mode="all")
    built = build_mechanism_model(
        prepared,
        mechanism_symbol="L",
        outcome_symbol="W",
        confounder_symbols=(),
        linear_mechanism=True,
        decompose_between_within=True,
    )
    mean = built.model["mech_child_mean"].get_value()
    dev = built.model["mech_within_dev"].get_value()
    z = built.model["z_mech_logit"].get_value()
    child = np.asarray(built.prepared.child_idx, dtype=int)

    assert mean.shape == z.shape == dev.shape
    assert mean.shape[0] == len(built.model.coords["obs_id"])
    np.testing.assert_allclose(mean + dev, z, atol=1e-12)
    for c in np.unique(child):
        rows = child == c
        # The child mean is the mean of that child's *fitted* rows, so the
        # deviations within each child sum to zero.
        np.testing.assert_allclose(dev[rows].sum(), 0.0, atol=1e-12)
        np.testing.assert_allclose(mean[rows], z[rows].mean(), atol=1e-12)

    free = {rv.name for rv in built.model.free_RVs}
    assert {"beta_between", "beta_within"} <= free
    assert "beta_mech" not in free


def test_the_split_contrast_is_the_within_child_one(tmp_path):
    """A decomposed fit's natural-scale contrast moves the *deviation*, not the mean.

    Setting a row's exposure to ``x`` holds that child's study average fixed, so the
    between term cancels and the contrast is driven by ``beta_within`` alone. That is
    what makes the headline a within-child quantity on this design.
    """
    rng = np.random.default_rng(7)
    counts = np.array([4.0, 12.0, 20.0, 8.0, 16.0, 28.0])
    z, _ = standardise(logit_safe(counts, 32))
    child = np.array([0, 0, 0, 1, 1, 1])
    mbar = np.array([z[child == c].mean() for c in child])
    dev = z - mbar
    between = rng.normal(0.9, 0.05, size=(2, 30))
    within = rng.normal(0.1, 0.02, size=(2, 30))
    baseline = rng.normal(0.0, 0.2, size=(2, 30, counts.size))
    eta = (
        baseline
        + between[:, :, None] * mbar[None, None, :]
        + within[:, :, None] * dev[None, None, :]
    )
    trace = SimpleNamespace(
        posterior=_dataset(
            {
                "eta": (("chain", "draw", "obs_id"), eta),
                "beta_between": (("chain", "draw"), between),
                "beta_within": (("chain", "draw"), within),
            }
        ),
        constant_data=_dataset(
            {
                "z_mech_logit": (("obs_id",), z),
                "mech_child_mean": (("obs_id",), mbar),
                "mech_within_dev": (("obs_id",), dev),
            }
        ),
    )
    terms = mi.resolve_mechanism_terms(
        trace, x_exposure=counts, exposure_n_trials=32
    )
    assert terms.kind == "linear_between_within"
    np.testing.assert_allclose(
        terms.fitted,
        mbar[:, None] * between.reshape(-1)[None, :]
        + dev[:, None] * within.reshape(-1)[None, :],
        atol=1e-12,
    )
    # Every row's contrast between two exposures is beta_within * dz — the between
    # term is identical at both and cancels.
    lo, hi = 8.0, 24.0
    to_z = mi._exposure_to_z(counts, z, 32)
    shift = terms.contribution_at(hi) - terms.contribution_at(lo)
    expected = within.reshape(-1)[None, :] * (to_z(hi) - to_z(lo))
    np.testing.assert_allclose(shift, np.broadcast_to(expected, shift.shape), atol=1e-12)


def test_between_within_is_rejected_for_a_curve_before_any_io():
    """A split of a nonparametric curve is a different design, not this one."""
    with pytest.raises(ValueError, match="decompose_between_within requires"):
        MechanismModelSettings(decompose_between_within=True)


def test_between_within_is_rejected_alongside_a_moderator():
    """The interaction would still be built on the pooled exposure the split rejects."""
    with pytest.raises(ValueError, match="cannot be combined with moderator_symbol"):
        MechanismModelSettings(
            linear_mechanism=True,
            decompose_between_within=True,
            moderator_symbol="E",
        )


# ---------------------------------------------------------------------------
# #604 — the phase-stability sensitivity
# ---------------------------------------------------------------------------


def test_phase_varying_slope_is_partially_pooled_and_linear_only(tmp_path):
    """Three free slopes at ~52 rows each are noisy; the deviations are shrunk.

    Pinned structurally: the per-period vector is a Deterministic of a shared mean
    and a scaled standard-normal offset, so it cannot silently become three
    independent slopes.
    """
    from language_reading_predictors.statistical_models.factories import (
        build_mechanism_model,
    )
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    path = _synthetic_frame(tmp_path)
    prepared = load_and_prepare(path=path, phase_mode="all")
    built = build_mechanism_model(
        prepared,
        mechanism_symbol="L",
        outcome_symbol="W",
        confounder_symbols=(),
        linear_mechanism=True,
        phase_varying_slope=True,
    )
    free = {rv.name for rv in built.model.free_RVs}
    assert {"mu_mech", "sigma_mech_phase", "beta_mech_phase_raw"} <= free
    assert "beta_mech" not in free
    assert "beta_mech_phase" in built.model.named_vars
    assert "beta_mech_phase" not in free  # a Deterministic, i.e. partially pooled

    with pytest.raises(ValueError, match="phase_varying_slope requires"):
        build_mechanism_model(
            prepared,
            mechanism_symbol="L",
            outcome_symbol="W",
            confounder_symbols=(),
            phase_varying_slope=True,
        )


def test_phase_varying_slope_is_rejected_with_phase_specific_curves(tmp_path):
    """A per-period *slope* must not be confusable with a per-period *curve*.

    Both entry points now share one design validator (#637 stage 1), so they refuse
    the combination for the same stated reason. Without ``linear_mechanism`` there
    is no scalar slope to vary at all; with it, a phase-specific curve is not a
    linear design. The factory used to reach neither rule and quietly built the
    pooled linear slope instead.
    """
    from language_reading_predictors.statistical_models.factories import (
        build_mechanism_model,
    )
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    with pytest.raises(ValueError, match="phase_varying_slope requires"):
        MechanismModelSettings(
            linear_mechanism=False,
            phase_varying_slope=True,
            phase_specific_mechanism=True,
        )
    with pytest.raises(ValueError, match="linear_mechanism cannot be combined"):
        MechanismModelSettings(
            linear_mechanism=True,
            phase_varying_slope=True,
            phase_specific_mechanism=True,
        )

    prepared = load_and_prepare(path=_synthetic_frame(tmp_path), phase_mode="all")
    for kwargs, match in (
        (
            {"phase_varying_slope": True, "phase_specific_mechanism": True},
            "phase_varying_slope requires",
        ),
        (
            {
                "linear_mechanism": True,
                "phase_varying_slope": True,
                "phase_specific_mechanism": True,
            },
            "linear_mechanism cannot be combined",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            build_mechanism_model(
                prepared,
                mechanism_symbol="L",
                outcome_symbol="W",
                confounder_symbols=(),
                **kwargs,
            )


def test_phase_varying_contribution_uses_each_row_s_own_period():
    """The standardised contrast averages the per-period slopes over the fitted rows."""
    rng = np.random.default_rng(5)
    counts = np.array([4.0, 12.0, 20.0, 8.0, 16.0, 28.0])
    z, _ = standardise(logit_safe(counts, 32))
    phase = np.array([0, 1, 2, 0, 1, 2])
    per_phase = rng.normal(0.3, 0.05, size=(2, 20, 3))
    baseline = rng.normal(0.0, 0.2, size=(2, 20, counts.size))
    eta = baseline + per_phase[:, :, phase] * z[None, None, :]
    trace = SimpleNamespace(
        posterior=_dataset(
            {
                "eta": (("chain", "draw", "obs_id"), eta),
                "beta_mech_phase": (("chain", "draw", "phase"), per_phase),
            }
        ),
        constant_data=_dataset(
            {
                "z_mech_logit": (("obs_id",), z),
                "phase_idx": (("obs_id",), phase),
            }
        ),
    )
    terms = mi.resolve_mechanism_terms(
        trace, x_exposure=counts, exposure_n_trials=32
    )
    assert terms.kind == "linear_phase_varying"
    flat = per_phase.reshape(-1, 3)
    np.testing.assert_allclose(
        terms.fitted, flat[:, phase].T * z[:, None], atol=1e-12
    )
    # Row-specific: at one exposure value the contribution still differs by period.
    at_20 = terms.contribution_at(20.0)
    assert at_20.shape[0] == counts.size
    assert not np.allclose(at_20[0], at_20[1])


# ---------------------------------------------------------------------------
# #605 — a dispersion prior that can reach the near-Binomial limit
# ---------------------------------------------------------------------------


def _kappa_prior_draws(family: str, sigma: float, *, draws: int = 200_000):
    rng = np.random.default_rng(20260824)
    if family == "halfnormal_concentration":
        return np.abs(rng.normal(0.0, sigma, size=draws))
    u = np.abs(rng.normal(0.0, sigma, size=draws))
    return 1.0 / (u**2 + 1e-6)


@pytest.mark.parametrize("n_trials", [79, 170])
def test_registered_kappa_prior_excludes_the_near_binomial_limit(n_trials):
    """``HalfNormal(50)`` gives "no extra-Binomial variation" no prior mass.

    The variance inflation is ``(kappa + n) / (kappa + 1)``, so being within 10% of
    Binomial needs ``kappa >= 10 (n - 1) - 1`` — 779 at n = 79 and 1689 at n = 170.
    This is the substantive assumption #605 is about: the registered prior enforces a
    floor on overdispersion, and the dispersion-scale alternative does not.
    """
    threshold = 10.0 * (n_trials - 1) - 1.0
    registered = _kappa_prior_draws("halfnormal_concentration", 50.0)
    alternative = _kappa_prior_draws("halfnormal_inverse_sqrt", 0.25)

    assert float(np.mean(registered >= threshold)) < 1e-4
    assert float(np.mean(alternative >= threshold)) > 0.02
    # At the registered prior's own median the enforced inflation is substantial.
    inflation = (np.median(registered) + n_trials) / (np.median(registered) + 1.0)
    assert inflation > (3.0 if n_trials == 79 else 5.0)


def test_mechanism_factory_threads_the_dispersion_prior_family(tmp_path):
    from language_reading_predictors.statistical_models.factories import (
        build_mechanism_model,
    )
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    path = _synthetic_frame(tmp_path)
    prepared = load_and_prepare(path=path, phase_mode="all")
    default = build_mechanism_model(
        prepared,
        mechanism_symbol="L",
        outcome_symbol="W",
        confounder_symbols=(),
        linear_mechanism=True,
    )
    assert "kappa" in {rv.name for rv in default.model.free_RVs}

    alternative = build_mechanism_model(
        prepared,
        mechanism_symbol="L",
        outcome_symbol="W",
        confounder_symbols=(),
        linear_mechanism=True,
        kappa_prior_family="halfnormal_inverse_sqrt",
    )
    free = {rv.name for rv in alternative.model.free_RVs}
    assert "inv_sqrt_kappa" in free
    assert "kappa" not in free  # a Deterministic of the sampled dispersion
    assert "kappa" in alternative.model.named_vars

    with pytest.raises(ValueError, match="kappa_prior_family must be"):
        build_mechanism_model(
            prepared,
            mechanism_symbol="L",
            outcome_symbol="W",
            confounder_symbols=(),
            linear_mechanism=True,
            kappa_prior_family="lognormal",
        )


def test_dispersion_settings_are_validated_before_data_loading():
    """A misspelled family or a non-positive scale must fail in pure resolution."""
    with pytest.raises(ValueError, match="kappa_prior_family must be one of"):
        MechanismModelSettings(kappa_prior_family="halfcauchy")
    with pytest.raises(ValueError, match="kappa_sigma must be"):
        MechanismModelSettings(kappa_sigma=0.0)
    settings = MechanismModelSettings(
        kappa_prior_family="halfnormal_inverse_sqrt", kappa_sigma=0.5
    )
    assert settings.kappa_sigma == 0.5
    assert _priors.inv_sqrt_kappa_prior(sigma=0.5) is not None


def test_dispersion_summary_records_the_prior_and_the_implied_inflation(
    tmp_path, monkeypatch
):
    """The report must be able to show whether the data moved ``kappa`` at all."""
    monkeypatch.setattr(mi, "save_styled_figure", lambda *_a, **_k: None)
    ctx, *_ = _linear_fit(tmp_path)

    frame = mech_pipeline._write_dispersion_summary(ctx)

    assert frame is not None
    row = frame.iloc[0]
    assert row["kappa_prior_label"] == "kappa ~ HalfNormal(50)"
    assert bool(row["reaches_near_binomial"]) is False
    assert row["kappa_for_10pct_of_binomial"] == pytest.approx(10.0 * 78 - 1)
    # kappa is pinned at 60 in the fixture, n_trials = 79.
    assert row["variance_inflation_median"] == pytest.approx((60 + 79) / 61)
    assert row["prob_within_10pct_of_binomial"] == pytest.approx(0.0)
