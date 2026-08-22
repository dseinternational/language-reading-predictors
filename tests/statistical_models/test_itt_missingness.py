# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contracts and algebra for the trace-backed word-reading missing-data bundle."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models import itt_missingness as missing


def _archive_frame() -> pd.DataFrame:
    group = np.asarray([1] * 29 + [2] * 28)
    included = np.asarray([1] * 28 + [0] + [1] * 26 + [0] * 2)
    post = np.full(57, np.nan)
    post[:28] = np.arange(28) % 20
    post[29 : 29 + 25] = np.arange(25) % 17
    index = np.arange(57)
    return pd.DataFrame(
        {
            "group": group,
            "area": 1 + (index % 2),
            "gender": 1 + ((index // 2) % 2),
            "included": included,
            "age_ts": 60 + index,
            "expr_vocab_raw_ts": 6 + (index % 60),
            "recep_vocab_raw_ts": 5 + (index % 58),
            "word_reading_raw_ts": index % 31,
            "letter_sound_raw_ts": index % 33,
            "word_reading_t2": post,
        }
    )


def _write_archive(path: Path) -> str:
    _archive_frame().to_csv(path, index=False)
    return missing.sha256_file(path)


def _loaded(tmp_path: Path):
    path = tmp_path / "archive.csv"
    digest = _write_archive(path)
    return missing.load_randomised_w_archive(
        path,
        expected_sha256=digest,
        local_wide_path=None,
    )


def _trace(data, *, p0: float = 0.10, p1: float = 0.20):
    shape_target = (2, 3, missing.RANDOMISED_N)
    shape_observed = (2, 3, data.n_obs)
    posterior = xr.Dataset(
        {
            "p0_target": (("chain", "draw", "target_id"), np.full(shape_target, p0)),
            "p1_target": (("chain", "draw", "target_id"), np.full(shape_target, p1)),
            "p0_observed_profiles": (
                ("chain", "draw", "obs_id"),
                np.full(shape_observed, p0),
            ),
            "p1_observed_profiles": (
                ("chain", "draw", "obs_id"),
                np.full(shape_observed, p1),
            ),
        }
    )
    return SimpleNamespace(posterior=posterior)


def test_loader_keeps_53_likelihood_rows_and_all_57_target_profiles(tmp_path):
    data = _loaded(tmp_path)

    assert data.n_obs == data.n_children == 53
    assert data.target_X.shape == (57, 2)
    assert data.covariate_names == ("screening_age", "screening_word_reading")
    assert int(data.G.sum()) == 28
    assert int((data.G == 0).sum()) == 25
    assert int((~data.target_outcome_observed).sum()) == 4
    assert int((~data.target_in_original_analysis).sum()) == 3
    assert data.local_wide_sha256 is None
    assert not any("subject" in value.casefold() for value in data.subject_ids)


def test_loader_rejects_a_wrong_source_hash(tmp_path):
    path = tmp_path / "archive.csv"
    _write_archive(path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        missing.load_randomised_w_archive(
            path,
            expected_sha256="0" * 64,
            local_wide_path=None,
        )


def test_screening_model_has_53_observations_and_57_prediction_profiles(tmp_path):
    built = missing.build_screening_w_model(_loaded(tmp_path))

    assert built.prepared.n_obs == 53
    assert len(built.model.coords["obs_id"]) == 53
    assert len(built.model.coords["target_id"]) == 57
    assert {rv.name for rv in built.model.free_RVs} == {
        "alpha",
        "tau",
        "beta_screening_age",
        "beta_screening_word",
        "kappa",
    }
    assert {rv.name for rv in built.model.observed_RVs} == {"y_post"}


def test_mar_j2r_and_delta_grid_use_the_explicit_missing_profiles(tmp_path):
    data = _loaded(tmp_path)
    result = missing.summarise_missingness_sensitivity(_trace(data), data)
    indexed = result.set_index("scenario")

    assert len(result) == 28
    assert indexed.loc["mar_all_57", "effect_items_median"] == pytest.approx(7.9)
    assert indexed.loc["mar_all_57", "estimand_class"] == (
        "common_profile_standardisation"
    )
    # Only the unique intervention nonstarter loses the 7.9-item model contrast,
    # but the factual treatment mean has the randomised-arm denominator 29.
    expected_j2r = 7.9 - (7.9 / 29)
    assert indexed.loc[
        "jump_to_reference_intervention_nonstarter", "effect_items_median"
    ] == pytest.approx(expected_j2r)
    assert indexed.loc[
        "jump_to_reference_intervention_nonstarter", "estimand_class"
    ] == "randomised_arm_factual_completion"
    assert indexed.loc["delta_i_+0_c_+0", "effect_items_median"] == pytest.approx(
        7.9
    )
    # +4 applies to one of 29 intervention profiles; +4 control applies to three
    # of 28 control profiles.
    assert indexed.loc["delta_i_+4_c_+0", "effect_items_median"] == pytest.approx(
        7.9 + (4 / 29)
    )
    assert indexed.loc["delta_i_+0_c_+4", "effect_items_median"] == pytest.approx(
        7.9 - (12 / 28)
    )


def test_factual_completion_is_distinct_from_common_profile_mar(tmp_path):
    data = _loaded(tmp_path)
    draws = 6
    p0_target = np.linspace(0.02, 0.58, missing.RANDOMISED_N)
    p1_target = p0_target + 0.10
    observed = data.target_outcome_observed
    trace = SimpleNamespace(
        posterior=xr.Dataset(
            {
                "p0_target": (
                    ("chain", "draw", "target_id"),
                    np.broadcast_to(p0_target, (1, draws, len(p0_target))),
                ),
                "p1_target": (
                    ("chain", "draw", "target_id"),
                    np.broadcast_to(p1_target, (1, draws, len(p1_target))),
                ),
                "p0_observed_profiles": (
                    ("chain", "draw", "obs_id"),
                    np.broadcast_to(p0_target[observed], (1, draws, data.n_obs)),
                ),
                "p1_observed_profiles": (
                    ("chain", "draw", "obs_id"),
                    np.broadcast_to(p1_target[observed], (1, draws, data.n_obs)),
                ),
            }
        )
    )

    indexed = missing.summarise_missingness_sensitivity(trace, data).set_index(
        "scenario"
    )
    intervention = data.target_G == 1
    control = ~intervention
    expected_factual_mar = (
        p1_target[intervention].mean() - p0_target[control].mean()
    ) * missing.WORD_READING_N

    assert indexed.loc["mar_all_57", "effect_items_median"] == pytest.approx(7.9)
    assert indexed.loc[
        "delta_i_+0_c_+0", "effect_items_median"
    ] == pytest.approx(expected_factual_mar)
    assert indexed.loc["delta_i_+0_c_+0", "effect_items_median"] != pytest.approx(
        indexed.loc["mar_all_57", "effect_items_median"]
    )
    assert indexed.loc["delta_i_+0_c_+0", "target_population"].startswith(
        "randomised-arm factual completion"
    )


def test_delta_effect_is_monotone_by_arm_and_reports_bound_clipping(tmp_path):
    data = _loaded(tmp_path)
    result = missing.summarise_missingness_sensitivity(_trace(data), data)
    grid = result.loc[result["scenario_class"] == "arm_specific_delta_grid"]

    at_control_zero = grid.loc[grid["delta_control_items"].eq(0)].sort_values(
        "delta_intervention_items"
    )
    assert np.all(np.diff(at_control_zero["effect_items_median"]) >= 0)
    at_intervention_zero = grid.loc[
        grid["delta_intervention_items"].eq(0)
    ].sort_values("delta_control_items")
    assert np.all(np.diff(at_intervention_zero["effect_items_median"]) <= 0)
    assert grid["clipped_control_fraction"].max() > 0


def test_screening_prior_is_baseline_anchored_and_traceable(tmp_path, monkeypatch):
    data = _loaded(tmp_path)
    built = missing.build_screening_w_model(data)
    prior = missing.sample_missingness_prior_predictive(
        built,
        draws=missing.MISSINGNESS_PRIOR_DRAWS,
        random_seed=47,
    )
    frame = missing.missingness_prior_check(prior, data)
    monkeypatch.setattr(missing, "RLI_ARCHIVE_CSV_SHA256", data.data_sha256)

    assert {"/prior", "/prior_predictive"}.issubset(set(prior.groups))
    assert set(frame["estimand"]) == {
        "common_profile_all_57",
        "randomised_arm_factual_mar",
    }
    assert frame["alpha_anchor_logit"].eq(
        data.covariate_scalers["screening_word_reading"]["mean"]
    ).all()
    # The anchor is exactly the pre-randomisation screening proportion mapped to
    # the 79-item outcome scale; no t2 outcome statistic enters it.
    expected_anchor_items = missing.WORD_READING_N / (
        1
        + np.exp(
            -data.covariate_scalers["screening_word_reading"]["mean"]
        )
    )
    assert frame["alpha_anchor_items"].eq(expected_anchor_items).all()
    assert missing.validate_missingness_prior_check(frame) == ()

    trace = xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {"tau": (("chain", "draw"), np.zeros((1, 2)))}
            )
        }
    )
    missing.attach_missingness_prior_groups(trace, prior)
    trace_path = tmp_path / missing.MISSINGNESS_TRACE_FILENAME
    trace.to_netcdf(trace_path)
    reopened = xr.open_datatree(trace_path)
    assert {"/prior", "/prior_predictive"}.issubset(set(reopened.groups))


def test_bundle_validator_rejects_an_incomplete_delta_grid(tmp_path, monkeypatch):
    data = _loaded(tmp_path)
    frame = missing.summarise_missingness_sensitivity(_trace(data), data)
    trace_path = tmp_path / missing.MISSINGNESS_TRACE_FILENAME
    trace_path.write_bytes(b"trace")
    trace_hash = missing.sha256_file(trace_path)
    monkeypatch.setattr(missing, "RLI_ARCHIVE_CSV_SHA256", data.data_sha256)
    frame["converged"] = True
    frame["trace_file"] = missing.MISSINGNESS_TRACE_FILENAME
    frame["trace_sha256"] = trace_hash

    assert missing.validate_missingness_summary(frame, trace_path=trace_path) == ()
    broken = frame.drop(frame.index[-1])
    errors = missing.validate_missingness_summary(broken, trace_path=trace_path)
    assert any("scenario count" in error for error in errors)
    assert any("delta grid" in error for error in errors)


def test_bundle_validator_can_record_a_failed_subfit_before_release_withholds(
    tmp_path, monkeypatch
):
    data = _loaded(tmp_path)
    frame = missing.summarise_missingness_sensitivity(_trace(data), data)
    trace_path = tmp_path / missing.MISSINGNESS_TRACE_FILENAME
    trace_path.write_bytes(b"trace")
    monkeypatch.setattr(missing, "RLI_ARCHIVE_CSV_SHA256", data.data_sha256)
    frame["converged"] = False
    frame["trace_file"] = missing.MISSINGNESS_TRACE_FILENAME
    frame["trace_sha256"] = missing.sha256_file(trace_path)

    strict_errors = missing.validate_missingness_summary(
        frame, trace_path=trace_path
    )
    write_time_errors = missing.validate_missingness_summary(
        frame,
        trace_path=trace_path,
        require_converged=False,
    )

    assert any("failed or was not checked" in error for error in strict_errors)
    assert write_time_errors == ()


# ---------------------------------------------------------------------------
# 2026-08-22 ITT audit regressions (issue #577, finding 9)
# ---------------------------------------------------------------------------


def _write_mutated_archive(path: Path, mutate) -> str:
    frame = _archive_frame()
    mutate(frame)
    frame.to_csv(path, index=False)
    return missing.sha256_file(path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("group", 1.7, "whole numbers"),
        ("included", 0.4, "whole numbers"),
        ("word_reading_t2", 12.5, "whole item counts"),
    ],
)
def test_the_archive_validator_rejects_fractional_values(
    tmp_path, column, value, message
):
    """Integrality is checked before the cast, not silently truncated by it.

    ``_validate_binary_codes`` compared ``set(values.astype(int))`` against the
    allowed codes, and ``astype`` truncates toward zero - so 1.7 became a valid 1
    and 0.4 a valid 0. ``word_reading_t2`` was bounds-checked only, then cast with
    ``astype(np.int64)``, so 12.5 was modelled as 12. The pinned archive checksum
    protects production, but this validator is what a future archive revision is
    checked against.
    """
    path = tmp_path / "archive.csv"

    def mutate(frame):
        # Widen first: the fixture's integer columns reject a float in place.
        frame[column] = frame[column].astype(float)
        frame.loc[0, column] = value

    digest = _write_mutated_archive(path, mutate)
    with pytest.raises(ValueError, match=message):
        missing.load_randomised_w_archive(
            path, expected_sha256=digest, local_wide_path=None
        )


def test_the_provenance_records_every_sampled_coefficient_prior():
    """``tau`` and ``kappa`` were missing from ``coefficient_priors``.

    The record described a model with no treatment effect and no dispersion
    parameter, while the release check's own convergence scan covered both. The
    three shared-constructor entries are rendered from the constructors so the
    record cannot drift from the model the factory builds.
    """
    import inspect

    from language_reading_predictors.statistical_models import priors as p

    source = inspect.getsource(missing.run_missingness_subfit)
    assert '"tau": str(_provenance_priors.tau_prior())' in source
    assert '"kappa": str(_provenance_priors.kappa_prior())' in source
    assert (
        '"beta_screening_age": str(_provenance_priors.gamma_age_prior())' in source
    )
    # And the constructors render something a reader can act on.
    assert "0.5" in str(p.tau_prior())
    assert "50" in str(p.kappa_prior())


# ---------------------------------------------------------------------------
# 2026-08-22 ITT audit regressions (issue #577, finding 8)
# ---------------------------------------------------------------------------


def test_the_design_record_pins_the_all_57_target_design(tmp_path):
    """Counts and a digest, so a stored fit can be checked without the archive.

    Fresh generation validated the target count, the likelihood rows and the
    arm / missingness masks, but none of it reached the persisted artefacts — so
    stored release evaluation could only check that the trace carried variables
    of the right *names*.
    """
    record = missing.missingness_design_record(_loaded(tmp_path))
    assert record["target_profile_n"] == missing.RANDOMISED_N == 57
    assert record["observed_outcome_n"] == 53
    assert record["target_by_arm"] == {"intervention": 29, "control": 28}
    assert record["target_observed_by_arm"] == {"intervention": 28, "control": 25}
    assert record["target_in_original_analysis_n"] == 54
    assert len(record["target_design_sha256"]) == 64


def test_the_design_digest_separates_two_different_target_sets(tmp_path):
    """Counts alone cannot establish that two runs completed the same profiles."""
    base = missing.missingness_design_record(_loaded(tmp_path))

    other = tmp_path / "other"
    other.mkdir()
    path = other / "archive.csv"

    def mutate(frame):
        # Every arm size, mask and count held fixed; one child's screening
        # covariate moved. The completion problem is a different one, and only
        # the digest can see it.
        frame["age_ts"] = frame["age_ts"].astype(float)
        frame.loc[0, "age_ts"] = float(frame.loc[0, "age_ts"]) + 11.0

    digest = _write_mutated_archive(path, mutate)
    moved = missing.missingness_design_record(
        missing.load_randomised_w_archive(
            path, expected_sha256=digest, local_wide_path=None
        )
    )
    assert moved["target_by_arm"] == base["target_by_arm"]
    assert moved["target_observed_by_arm"] == base["target_observed_by_arm"]
    assert moved["observed_outcome_n"] == base["observed_outcome_n"]
    assert moved["target_in_original_analysis_n"] == base["target_in_original_analysis_n"]
    assert moved["target_design_sha256"] != base["target_design_sha256"]


@pytest.mark.parametrize(
    ("targets", "observations", "fragment"),
    [
        (57, 53, None),
        (57, 1, "y_post covers 1 observations"),
        (1, 53, "p0_target covers 1 target profiles"),
    ],
)
def test_stored_evaluation_checks_the_trace_design_dimensions(
    targets, observations, fragment
):
    """Names are not a design: a one-target trace used to qualify."""
    import numpy as np
    import xarray as xr

    from language_reading_predictors.statistical_models import release as R

    coords = {"chain": [0], "draw": np.arange(4)}
    tree = xr.DataTree.from_dict(
        {
            "prior": xr.Dataset(
                {
                    name: (("chain", "draw", "target_id"), np.zeros((1, 4, targets)))
                    for name in ("p0_target", "p1_target")
                },
                coords={**coords, "target_id": np.arange(targets)},
            ),
            "prior_predictive": xr.Dataset(
                {"y_post": (("chain", "draw", "obs_id"), np.zeros((1, 4, observations)))},
                coords={**coords, "obs_id": np.arange(observations)},
            ),
        }
    )
    error = R._missingness_design_dimension_error(
        tree, expected_targets=57, expected_observations=53
    )
    if fragment is None:
        assert error is None
    else:
        assert error is not None and fragment in error
