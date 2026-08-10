# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Full-cohort sensitivity for the word-reading available-case modified ITT estimate.

The primary word-reading model uses the 53 children with both a t1 baseline and
a t2 outcome.  The public trial archive additionally contains complete screening
profiles for all 57 randomised children, but no t1 or later measurements for the
three children lost to follow-up.  One further waiting-control child in the
54-child analysed archive has no word-reading score at t1 or t2.

This module fits a deliberately secondary outcome model to the 53 observed t2
scores using only complete pre-randomisation screening predictors.  Its posterior
is then standardised over either the 53 observed profiles or all 57 randomised
profiles.  MAR, jump-to-reference and fixed arm-specific delta scenarios are
posterior transforms: the four unavailable outcomes are never sampled as if the
data identified them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import pandas as pd

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.preprocessing import (
    logit_safe,
    standardise,
)

if TYPE_CHECKING:
    from language_reading_predictors.statistical_models.factories import BuiltModel

RLI_ARCHIVE_DOI = "10.5255/UKDA-SN-852291"
RLI_ARCHIVE_URL = "https://reshare.ukdataservice.ac.uk/852291/"
RLI_ARCHIVE_ZIP_URL = (
    "https://reshare.ukdataservice.ac.uk/852291/1/DSE_Data.zip"
)
RLI_ARCHIVE_ZIP_SHA256 = (
    "a015edd19d0d35e325f3a14a06cc5894e1beb3cc95ce6db6513b0c763b4a7d3b"
)
RLI_ARCHIVE_CSV_SHA256 = (
    "7c6cda3634c302d6b2b253ba01a9043bd0762d3f35a66027f9a2f1f2dbdc5ae7"
)
RLI_ARCHIVE_CSV_NAME = "dse-rli-trial-data-archive.csv"
RLI_LOCAL_WIDE_SHA256 = (
    "2c47eb49a96013a0283a225dcd8460ceb62720fdca60bcaeb3811345e5b7c99c"
)
RLI_RECONCILIATION_DIGEST = (
    "22b745ee81a32a5c654cfd0d480c0ba87dd9aea3a3f51a74ce3447e56ac37c0e"
)

WORD_READING_N = 79
RANDOMISED_N = 57
RANDOMISED_INTERVENTION_N = 29
RANDOMISED_CONTROL_N = 28
OBSERVED_INTERVENTION_N = 28
OBSERVED_CONTROL_N = 25
LOST_TO_FOLLOW_UP_N = 3
WITHIN_ARCHIVE_W_MISSING_N = 1

# A diagnostic grid, not a posterior over the unidentified departure.  Each value
# says how many items higher/lower the missing-pattern mean is than the observed-
# pattern outcome surface in the same arm.  Clipping at the test bounds is reported.
DEFAULT_DELTA_ITEMS: tuple[float, ...] = (-8.0, -4.0, 0.0, 4.0, 8.0)
SCREENING_ALPHA_SIGMA = 1.0
MISSINGNESS_PRIOR_DRAWS = 1000
MISSINGNESS_TRACE_FILENAME = "trace_screening_missingness.nc"
MISSINGNESS_SUMMARY_FILENAME = "itt_missingness_sensitivity.csv"
MISSINGNESS_PROVENANCE_FILENAME = "itt_missingness_provenance.json"
MISSINGNESS_PPC_FILENAME = "itt_missingness_ppc.csv"
MISSINGNESS_PRIOR_FILENAME = "itt_missingness_prior_check.csv"
MISSINGNESS_RENDERED_SCIENTIFIC_ARTIFACTS: tuple[str, ...] = (
    MISSINGNESS_SUMMARY_FILENAME,
    MISSINGNESS_PPC_FILENAME,
    MISSINGNESS_PRIOR_FILENAME,
)
MISSINGNESS_SUBFIT_LABEL = "lrp-rli-itt-010 screening-baseline missingness sensitivity"
MISSINGNESS_SCENARIOS: tuple[str, ...] = (
    "screening_model_observed_profiles",
    "mar_all_57",
    "jump_to_reference_intervention_nonstarter",
    "arm_specific_delta_grid",
)

_REQUIRED_COLUMNS = (
    "group",
    "area",
    "gender",
    "included",
    "age_ts",
    "expr_vocab_raw_ts",
    "recep_vocab_raw_ts",
    "word_reading_raw_ts",
    "letter_sound_raw_ts",
    "word_reading_t2",
)


def _waves(source: str, local: str, times: Sequence[int] = range(1, 5)) -> dict[str, str]:
    return {f"{source}{time}": f"{local}{time}" for time in times}


ARCHIVE_TO_LOCAL_WIDE: dict[str, str] = {
    "group": "group",
    "area": "area",
    "gender": "gender",
    **_waves("age_t", "age"),
    "block_design_raw_t1": "blocks1",
    **_waves("word_reading_t", "ewrswr"),
    **_waves("expr_grammar_t", "aptgram"),
    **_waves("expr_info_t", "aptinfo"),
    **_waves("taught_expr_vocab_1_t", "b1extau"),
    **_waves("taught_recep_vocab_1_t", "b1retau"),
    **_waves("taught_expr_vocab_2_t", "b2extau", (2, 3, 4)),
    **_waves("taught_recep_vocab_2_t", "b2retau", (2, 3, 4)),
    **_waves("basic_concepts_t", "celf"),
    **_waves("expr_vocab_t", "eowpvt"),
    **_waves("nonword", "nonword"),
    **_waves("blending", "blending"),
    **_waves("recep_vocab_t", "rowpvt"),
    **_waves("phon_spell_t", "spphon"),
    **_waves("recep_grammar_t", "trog"),
    **_waves("letter_sound_t", "yarclet"),
    "attend_t1_t2": "attend1",
    "attend_t2_t3": "attend2",
    "attend_t3_t4": "attend3",
    "attend_t1_t3": "attendto",
    "attend_t1_t4": "attendall",
}
if len(ARCHIVE_TO_LOCAL_WIDE) != 71:  # pragma: no cover - import-time guard
    raise RuntimeError("RLI archive reconciliation mapping must contain 71 fields")

SCREENING_COVARIATES: tuple[str, ...] = (
    "screening_age",
    "screening_word_reading",
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one source or trace file."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ScreeningWordReadingData:
    """Observed likelihood rows plus all-randomised screening target profiles."""

    subject_ids: np.ndarray
    child_idx: np.ndarray
    phase: np.ndarray
    G: np.ndarray
    X: np.ndarray
    post_counts: dict[str, np.ndarray]
    n_trials: dict[str, int]
    n_obs: int
    n_children: int
    target_subject_ids: np.ndarray
    target_G: np.ndarray
    target_X: np.ndarray
    target_outcome_observed: np.ndarray
    target_in_original_analysis: np.ndarray
    covariate_names: tuple[str, ...]
    covariate_scalers: dict[str, dict[str, float]]
    data_sha256: str
    local_wide_sha256: str | None
    reconciled_included_n: int | None
    reconciliation_digest: str | None
    source_path: str


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values


def _validate_binary_codes(values: np.ndarray, *, name: str, allowed: set[int]) -> None:
    if not np.isfinite(values).all() or set(values.astype(int)) != allowed:
        raise ValueError(f"{name} must contain exactly the codes {sorted(allowed)}")


def _z(values: np.ndarray, *, name: str) -> tuple[np.ndarray, dict[str, float]]:
    standardised, scaler = standardise(values)
    if not np.isfinite(standardised).all():
        raise ValueError(f"screening covariate {name!r} is incomplete")
    return standardised, {"mean": scaler.mean, "sd": scaler.sd}


def _row_fingerprints(frame: pd.DataFrame) -> pd.Series:
    numeric = frame.apply(pd.to_numeric, errors="raise")
    return numeric.apply(
        lambda row: hashlib.sha256(
            "\x1f".join(
                "<NA>" if pd.isna(value) else float(value).hex() for value in row
            ).encode("ascii")
        ).hexdigest(),
        axis=1,
    )


def _reconcile_included_rows(
    archive: pd.DataFrame,
    local_wide_path: str | Path,
    *,
    expected_local_sha256: str,
) -> tuple[str, int, str]:
    local_path = Path(local_wide_path)
    local_sha256 = sha256_file(local_path)
    if local_sha256 != expected_local_sha256:
        raise ValueError(
            "local RLI wide-data checksum mismatch: "
            f"expected {expected_local_sha256}, observed {local_sha256}"
        )
    wide = pd.read_csv(local_path, skipinitialspace=True, na_values=["", " "])
    included = archive.loc[pd.to_numeric(archive["included"]).eq(1)].copy()
    if len(included) != 54 or len(wide) != 54:
        raise ValueError("reconciliation requires 54 included archive and local rows")
    source_columns = list(ARCHIVE_TO_LOCAL_WIDE)
    local_columns = list(ARCHIVE_TO_LOCAL_WIDE.values())
    missing_source = sorted(set(source_columns) - set(included.columns))
    missing_local = sorted(set(local_columns) - set(wide.columns))
    if missing_source or missing_local:
        raise ValueError(
            "archive/local reconciliation columns are missing: "
            f"archive={missing_source}, local={missing_local}"
        )
    source_values = included[source_columns].rename(columns=ARCHIVE_TO_LOCAL_WIDE)
    local_values = wide[local_columns]
    source_fingerprints = _row_fingerprints(source_values)
    local_fingerprints = _row_fingerprints(local_values)
    if not source_fingerprints.is_unique or not local_fingerprints.is_unique:
        raise ValueError("71-field reconciliation fingerprints are not one-to-one")
    if set(source_fingerprints) != set(local_fingerprints):
        raise ValueError("the 54 archive rows do not reconcile with the repository")
    digest = hashlib.sha256(
        "\n".join(sorted(source_fingerprints)).encode("ascii")
    ).hexdigest()
    return local_sha256, len(source_fingerprints), digest


def load_randomised_w_archive(
    path: str | Path,
    *,
    expected_sha256: str = RLI_ARCHIVE_CSV_SHA256,
    local_wide_path: str | Path | None = _paths.DATA_DIR / "rli_data_wide.csv",
    expected_local_sha256: str = RLI_LOCAL_WIDE_SHA256,
    expected_reconciliation_digest: str = RLI_RECONCILIATION_DIGEST,
) -> ScreeningWordReadingData:
    """Load and strictly validate the checksum-pinned 57-row public archive.

    Source subject identifiers are intentionally discarded.  Stable internal row
    labels are generated only after every allocation, screening and outcome-count
    assertion passes; no source-to-repository ID crosswalk is written.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"RLI randomised archive not found: {source}")
    observed_sha256 = sha256_file(source)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "RLI randomised archive checksum mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )

    frame = pd.read_csv(
        source,
        encoding="utf-8-sig",
        skipinitialspace=True,
        na_values=["", " "],
    )
    missing_columns = [column for column in _REQUIRED_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"RLI randomised archive is missing columns: {missing_columns}")
    if len(frame) != RANDOMISED_N:
        raise ValueError(
            f"RLI randomised archive must contain {RANDOMISED_N} rows, got {len(frame)}"
        )

    group_raw = _numeric(frame, "group")
    area_raw = _numeric(frame, "area")
    gender_raw = _numeric(frame, "gender")
    included = _numeric(frame, "included")
    _validate_binary_codes(group_raw, name="group", allowed={1, 2})
    _validate_binary_codes(area_raw, name="area", allowed={1, 2})
    _validate_binary_codes(gender_raw, name="gender", allowed={1, 2})
    _validate_binary_codes(included, name="included", allowed={0, 1})

    G_target = (group_raw == 1).astype(np.int64)
    included_bool = included.astype(bool)
    allocation = {
        1: int(np.sum(G_target == 1)),
        0: int(np.sum(G_target == 0)),
    }
    if allocation != {1: RANDOMISED_INTERVENTION_N, 0: RANDOMISED_CONTROL_N}:
        raise ValueError(f"published randomised allocation mismatch: {allocation}")
    analysed = {
        1: int(np.sum((G_target == 1) & included_bool)),
        0: int(np.sum((G_target == 0) & included_bool)),
    }
    if analysed != {1: 28, 0: 26}:
        raise ValueError(f"published analysed allocation mismatch: {analysed}")

    age = _numeric(frame, "age_ts")
    word = _numeric(frame, "word_reading_raw_ts")
    letter = _numeric(frame, "letter_sound_raw_ts")
    expr = _numeric(frame, "expr_vocab_raw_ts")
    recep = _numeric(frame, "recep_vocab_raw_ts")
    screening = {
        "screening_age": age,
        "screening_word_reading": logit_safe(word, 30),
    }
    bounds = {
        "word_reading_raw_ts": (word, 30),
        "letter_sound_raw_ts": (letter, 32),
        "expr_vocab_raw_ts": (expr, 170),
        "recep_vocab_raw_ts": (recep, 170),
    }
    if not np.isfinite(age).all():
        raise ValueError("screening age must be observed for all 57 children")
    for name, (values, maximum) in bounds.items():
        if not np.isfinite(values).all() or np.any((values < 0) | (values > maximum)):
            raise ValueError(f"{name} must be complete and lie in [0, {maximum}]")

    z_columns: list[np.ndarray] = []
    scalers: dict[str, dict[str, float]] = {}
    for name in SCREENING_COVARIATES:
        values, scaler = _z(screening[name], name=name)
        z_columns.append(values)
        scalers[name] = scaler
    X_target = np.column_stack(z_columns).astype(float)

    post = _numeric(frame, "word_reading_t2")
    observed_mask = np.isfinite(post)
    if np.any((post[observed_mask] < 0) | (post[observed_mask] > WORD_READING_N)):
        raise ValueError(f"word_reading_t2 must lie in [0, {WORD_READING_N}]")
    observed_counts = {
        1: int(np.sum((G_target == 1) & observed_mask)),
        0: int(np.sum((G_target == 0) & observed_mask)),
    }
    expected_observed = {
        1: OBSERVED_INTERVENTION_N,
        0: OBSERVED_CONTROL_N,
    }
    if observed_counts != expected_observed:
        raise ValueError(
            "word-reading t2 observation counts mismatch: "
            f"expected {expected_observed}, observed {observed_counts}"
        )
    lost = int(np.sum(~included_bool))
    internal_missing = int(np.sum(included_bool & ~observed_mask))
    if lost != LOST_TO_FOLLOW_UP_N or internal_missing != WITHIN_ARCHIVE_W_MISSING_N:
        raise ValueError(
            "word-reading missingness pattern mismatch: "
            f"lost={lost}, within_archive_missing={internal_missing}"
        )

    if local_wide_path is None:
        local_wide_sha256 = None
        reconciled_included_n = None
        reconciliation_digest = None
    else:
        (
            local_wide_sha256,
            reconciled_included_n,
            reconciliation_digest,
        ) = _reconcile_included_rows(
            frame,
            local_wide_path,
            expected_local_sha256=expected_local_sha256,
        )
        if reconciliation_digest != expected_reconciliation_digest:
            raise ValueError(
                "archive/local reconciliation digest mismatch: "
                f"expected {expected_reconciliation_digest}, "
                f"observed {reconciliation_digest}"
            )

    target_ids = np.asarray(
        [f"randomised-{index:03d}" for index in range(1, RANDOMISED_N + 1)],
        dtype=object,
    )
    observed_ids = target_ids[observed_mask]
    observed_n = int(observed_mask.sum())
    return ScreeningWordReadingData(
        subject_ids=observed_ids,
        child_idx=np.arange(observed_n, dtype=np.int64),
        phase=np.zeros(observed_n, dtype=np.int64),
        G=G_target[observed_mask],
        X=X_target[observed_mask],
        post_counts={"W": post[observed_mask].astype(np.int64)},
        n_trials={"W": WORD_READING_N},
        n_obs=observed_n,
        n_children=observed_n,
        target_subject_ids=target_ids,
        target_G=G_target,
        target_X=X_target,
        target_outcome_observed=observed_mask,
        target_in_original_analysis=included_bool,
        covariate_names=SCREENING_COVARIATES,
        covariate_scalers=scalers,
        data_sha256=observed_sha256,
        local_wide_sha256=local_wide_sha256,
        reconciled_included_n=reconciled_included_n,
        reconciliation_digest=reconciliation_digest,
        source_path=str(source),
    )


def build_screening_w_model(data: ScreeningWordReadingData) -> BuiltModel:
    """Build the regularised screening-baseline Beta-Binomial companion."""

    import pymc as pm

    from language_reading_predictors.statistical_models import priors as _priors
    from language_reading_predictors.statistical_models.factories import BuiltModel
    from language_reading_predictors.statistical_models.likelihood import (
        beta_binomial_from_logit,
    )

    coords = {
        "obs_id": np.arange(data.n_obs),
        "target_id": np.arange(RANDOMISED_N),
        "screening_covariate": list(data.covariate_names),
    }
    with pm.Model(coords=coords) as model:
        G = pm.Data("G", data.G.astype(float), dims="obs_id")
        X = pm.Data("X_screening", data.X, dims=("obs_id", "screening_covariate"))
        X_target = pm.Data(
            "X_screening_target",
            data.target_X,
            dims=("target_id", "screening_covariate"),
        )
        # Unlike the model of record, this companion has no unstandardised t1-W
        # term whose slope prior carries the outcome level.  Its intercept is the
        # t2-W logit at the mean screening profile, so the shared zero-centred
        # ANCOVA intercept prior would put its median at 39.5/79 items.  Anchor it
        # instead at the mean *pre-randomisation* screening-W logit: a conservative
        # no-mean-change-on-proportion-correct reference that uses no t2 outcome.
        # The unit SD allows large learning, test-difficulty and timing departures
        # from that reference (about 1.4--23.8 items over the central 89% interval
        # before the other coefficients are integrated).
        alpha_anchor = data.covariate_scalers["screening_word_reading"]["mean"]
        alpha = pm.Normal(
            "alpha",
            mu=alpha_anchor,
            sigma=SCREENING_ALPHA_SIGMA,
        )
        tau = _priors.tau_prior().to_pymc("tau")
        beta_age = _priors.gamma_age_prior().to_pymc("beta_screening_age")
        # Screening W is standardised after its bounded-count logit transform.
        # It is the key prognostic bridge but is not the same 79-item t1 measure,
        # so it gets a zero-centred, wider slope rather than gamma_own's 1:1 anchor.
        beta_word = pm.Normal("beta_screening_word", mu=0.0, sigma=1.0)
        kappa = _priors.kappa_prior().to_pymc("kappa")

        baseline_eta = alpha + beta_age * X[:, 0] + beta_word * X[:, 1]
        eta = pm.Deterministic("eta", baseline_eta + tau * G, dims="obs_id")
        beta_binomial_from_logit(
            "y_post",
            eta,
            n_trials=WORD_READING_N,
            kappa=kappa,
            observed=data.post_counts["W"],
            dims="obs_id",
        )

        target_eta0 = alpha + beta_age * X_target[:, 0] + beta_word * X_target[:, 1]
        pm.Deterministic("p0_target", pm.math.sigmoid(target_eta0), dims="target_id")
        pm.Deterministic(
            "p1_target", pm.math.sigmoid(target_eta0 + tau), dims="target_id"
        )
        pm.Deterministic(
            "p0_observed_profiles",
            pm.math.sigmoid(baseline_eta),
            dims="obs_id",
        )
        pm.Deterministic(
            "p1_observed_profiles",
            pm.math.sigmoid(baseline_eta + tau),
            dims="obs_id",
        )

    return BuiltModel(
        model=model,
        prepared=data,  # type: ignore[arg-type]
        extras={
            "source_sha256": data.data_sha256,
            "target_n": RANDOMISED_N,
            "observed_n": data.n_obs,
        },
    )


def _draw_matrix(trace: Any, variable: str, *, group: str = "posterior") -> np.ndarray:
    values = np.asarray(getattr(trace, group)[variable].values, dtype=float)
    if values.ndim != 3:
        raise ValueError(
            f"{group} variable {variable!r} must be chain x draw x row"
        )
    return values.reshape((-1, values.shape[-1]))


def _summarise_draws(draws: np.ndarray, *, prefix: str) -> dict[str, float]:
    q50 = np.quantile(draws, [0.25, 0.75])
    q89 = np.quantile(draws, [0.055, 0.945])
    return {
        f"{prefix}_median": float(np.median(draws)),
        f"{prefix}_lo50": float(q50[0]),
        f"{prefix}_hi50": float(q50[1]),
        f"{prefix}_lo89": float(q89[0]),
        f"{prefix}_hi89": float(q89[1]),
    }


def _scenario_row(
    *,
    scenario: str,
    scenario_class: str,
    estimand_class: str,
    target_population: str,
    intervention_mean: np.ndarray,
    control_mean: np.ndarray,
    source_sha256: str,
    delta_intervention_items: float | None = None,
    delta_control_items: float | None = None,
    clipped_intervention: float = 0.0,
    clipped_control: float = 0.0,
) -> dict[str, Any]:
    effect = (intervention_mean - control_mean) * WORD_READING_N
    intervention_items = intervention_mean * WORD_READING_N
    control_items = control_mean * WORD_READING_N
    return {
        "scenario": scenario,
        "scenario_class": scenario_class,
        "estimand_class": estimand_class,
        "target_population": target_population,
        "delta_intervention_items": delta_intervention_items,
        "delta_control_items": delta_control_items,
        "clipped_intervention_fraction": clipped_intervention,
        "clipped_control_fraction": clipped_control,
        **_summarise_draws(effect, prefix="effect_items"),
        **_summarise_draws(intervention_items, prefix="intervention_mean_items"),
        **_summarise_draws(control_items, prefix="control_mean_items"),
        "prob_effect_positive": float(np.mean(effect > 0.0)),
        "randomised_n": RANDOMISED_N,
        "randomised_intervention_n": RANDOMISED_INTERVENTION_N,
        "randomised_control_n": RANDOMISED_CONTROL_N,
        "observed_intervention_n": OBSERVED_INTERVENTION_N,
        "observed_control_n": OBSERVED_CONTROL_N,
        "missing_intervention_n": RANDOMISED_INTERVENTION_N - OBSERVED_INTERVENTION_N,
        "missing_control_n": RANDOMISED_CONTROL_N - OBSERVED_CONTROL_N,
        "lost_to_follow_up_n": LOST_TO_FOLLOW_UP_N,
        "within_archive_w_missing_n": WITHIN_ARCHIVE_W_MISSING_N,
        "n_trials": WORD_READING_N,
        "source_sha256": source_sha256,
    }


def summarise_missingness_sensitivity(
    trace: Any,
    data: ScreeningWordReadingData,
    *,
    delta_items: Sequence[float] = DEFAULT_DELTA_ITEMS,
) -> pd.DataFrame:
    """Summarise the bridge, MAR, J2R and arm-specific pattern-mixture grid.

    The delta grid shifts the missing-pattern expected score directly on the
    79-item scale and clips only to the physical test bounds.  The fraction of
    profile-by-draw predictions clipped is carried in every grid row so a wide
    stress test cannot masquerade as an unconstrained location shift.

    The bridge and MAR rows are common-profile standardisations: both potential
    outcome surfaces are averaged over the same 53 or 57 profiles.  J2R and the
    delta grid answer a different, explicitly factual completion question.  They
    average the intervention surface over the 29 children randomised to
    intervention and the control surface over the 28 children randomised to
    control, modifying only the unavailable outcome(s) in that factual arm.  The
    zero-delta grid row is therefore the randomised-arm MAR completion; it need
    not equal the common-profile all-57 MAR row when baseline profiles are
    imbalanced by chance.
    """

    grid = tuple(float(value) for value in delta_items)
    if not grid or 0.0 not in grid or len(set(grid)) != len(grid):
        raise ValueError("delta_items must be unique, non-empty and include 0")
    if tuple(sorted(grid)) != grid:
        raise ValueError("delta_items must be sorted")

    p0_target = _draw_matrix(trace, "p0_target")
    p1_target = _draw_matrix(trace, "p1_target")
    p0_observed = _draw_matrix(trace, "p0_observed_profiles")
    p1_observed = _draw_matrix(trace, "p1_observed_profiles")
    if p0_target.shape[1] != RANDOMISED_N or p0_observed.shape[1] != data.n_obs:
        raise ValueError("posterior target/profile dimensions do not match the archive")

    missing_intervention = (~data.target_outcome_observed) & (data.target_G == 1)
    missing_control = (~data.target_outcome_observed) & (data.target_G == 0)
    intervention_nonstarter = (
        (~data.target_in_original_analysis) & (data.target_G == 1)
    )
    intervention_arm = data.target_G == 1
    control_arm = data.target_G == 0
    if (
        int(missing_intervention.sum()) != 1
        or int(missing_control.sum()) != 3
        or int(intervention_nonstarter.sum()) != 1
        or not np.array_equal(missing_intervention, intervention_nonstarter)
    ):
        raise ValueError("archive missing-pattern masks do not match the trial contract")
    rows: list[dict[str, Any]] = []

    rows.append(
        _scenario_row(
            scenario="screening_model_observed_profiles",
            scenario_class="bridge",
            estimand_class="common_profile_standardisation",
            target_population="53 children with observed t2 word reading",
            intervention_mean=p1_observed.mean(axis=1),
            control_mean=p0_observed.mean(axis=1),
            source_sha256=data.data_sha256,
        )
    )
    mar_intervention = p1_target.mean(axis=1)
    mar_control = p0_target.mean(axis=1)
    rows.append(
        _scenario_row(
            scenario="mar_all_57",
            scenario_class="missing_at_random",
            estimand_class="common_profile_standardisation",
            target_population="all 57 randomised screening profiles",
            intervention_mean=mar_intervention,
            control_mean=mar_control,
            source_sha256=data.data_sha256,
        )
    )

    # The factual-arm completion uses the randomised-arm denominators (29 and 28),
    # not the 57-profile common standardisation.  The one intervention child who
    # never started and was lost is assigned the control surface; control-arm
    # missing outcomes remain MAR.  The four followed discontinuers are observed
    # and stay in the likelihood under assignment.
    j2r_p1 = p1_target[:, intervention_arm].copy()
    j2r_p0 = p0_target[:, intervention_arm]
    nonstarter_in_intervention = intervention_nonstarter[intervention_arm]
    j2r_p1[:, nonstarter_in_intervention] = j2r_p0[
        :, nonstarter_in_intervention
    ]
    j2r_intervention = j2r_p1.mean(axis=1)
    factual_mar_control = p0_target[:, control_arm].mean(axis=1)
    rows.append(
        _scenario_row(
            scenario="jump_to_reference_intervention_nonstarter",
            scenario_class="reference_based",
            estimand_class="randomised_arm_factual_completion",
            target_population=(
                "randomised-arm factual completion: 29 intervention versus "
                "28 control profiles"
            ),
            intervention_mean=j2r_intervention,
            control_mean=factual_mar_control,
            source_sha256=data.data_sha256,
        )
    )

    missing_i_in_intervention = missing_intervention[intervention_arm]
    missing_c_in_control = missing_control[control_arm]
    for delta_i in grid:
        completed_i = p1_target[:, intervention_arm].copy()
        raw_i = (
            completed_i[:, missing_i_in_intervention]
            + delta_i / WORD_READING_N
        )
        missing_i = np.clip(raw_i, 0.0, 1.0)
        completed_i[:, missing_i_in_intervention] = missing_i
        clipped_i = float(np.mean(raw_i != missing_i))
        intervention = completed_i.mean(axis=1)
        for delta_c in grid:
            completed_c = p0_target[:, control_arm].copy()
            raw_c = (
                completed_c[:, missing_c_in_control]
                + delta_c / WORD_READING_N
            )
            missing_c = np.clip(raw_c, 0.0, 1.0)
            completed_c[:, missing_c_in_control] = missing_c
            clipped_c = float(np.mean(raw_c != missing_c))
            control = completed_c.mean(axis=1)
            rows.append(
                _scenario_row(
                    scenario=f"delta_i_{delta_i:+g}_c_{delta_c:+g}",
                    scenario_class="arm_specific_delta_grid",
                    estimand_class="randomised_arm_factual_completion",
                    target_population=(
                        "randomised-arm factual completion: 29 intervention "
                        "versus 28 control profiles"
                    ),
                    intervention_mean=intervention,
                    control_mean=control,
                    source_sha256=data.data_sha256,
                    delta_intervention_items=delta_i,
                    delta_control_items=delta_c,
                    clipped_intervention=clipped_i,
                    clipped_control=clipped_c,
                )
            )
    return pd.DataFrame(rows)


def sample_missingness_prior_predictive(
    built: BuiltModel,
    *,
    draws: int = MISSINGNESS_PRIOR_DRAWS,
    random_seed: int | None = None,
) -> Any:
    """Draw all prior parameters, estimands and observed-scale replications."""

    import pymc as pm

    names: list[str] = []
    names += [rv.name for rv in built.model.free_RVs]
    names += [rv.name for rv in built.model.deterministics]
    names += [rv.name for rv in built.model.observed_RVs]
    with built.model:
        return pm.sample_prior_predictive(
            draws=draws,
            var_names=list(dict.fromkeys(names)),
            random_seed=random_seed,
        )


def attach_missingness_prior_groups(trace: Any, prior_samples: Any) -> None:
    """Attach required prior groups to the persisted sensitivity trace."""

    for group in ("prior", "prior_predictive"):
        source = getattr(prior_samples, group, None)
        if source is None or not len(source.data_vars):
            raise ValueError(f"missingness prior samples have no populated {group} group")
        trace[group] = prior_samples[group]


def missingness_prior_check(
    prior_samples: Any,
    data: ScreeningWordReadingData,
) -> pd.DataFrame:
    """Summarise prior implications on the two reported estimand scales."""

    p0 = _draw_matrix(prior_samples, "p0_target", group="prior")
    p1 = _draw_matrix(prior_samples, "p1_target", group="prior")
    replicated = _draw_matrix(prior_samples, "y_post", group="prior_predictive")
    if p0.shape[1] != RANDOMISED_N or replicated.shape[1] != data.n_obs:
        raise ValueError("prior draws do not align with the screening data")

    intervention_arm = data.target_G == 1
    control_arm = data.target_G == 0
    replicated_mean = replicated.mean(axis=1)
    replicated_floor = np.mean(replicated == 0, axis=1)
    replicated_ceiling = np.mean(replicated == WORD_READING_N, axis=1)
    alpha_anchor = data.covariate_scalers["screening_word_reading"]["mean"]
    common = {
        **_summarise_draws(
            replicated_mean,
            prefix="prior_predictive_mean_items",
        ),
        **_summarise_draws(
            replicated_floor,
            prefix="prior_predictive_floor_fraction",
        ),
        **_summarise_draws(
            replicated_ceiling,
            prefix="prior_predictive_ceiling_fraction",
        ),
        "alpha_anchor_logit": alpha_anchor,
        "alpha_anchor_items": float(
            WORD_READING_N / (1.0 + np.exp(-alpha_anchor))
        ),
        "alpha_sigma": SCREENING_ALPHA_SIGMA,
        "prior_draws": int(p0.shape[0]),
        "source_sha256": data.data_sha256,
    }
    rows: list[dict[str, Any]] = []
    for estimand, target_population, intervention, control in (
        (
            "common_profile_all_57",
            "all 57 randomised screening profiles under both arms",
            p1.mean(axis=1),
            p0.mean(axis=1),
        ),
        (
            "randomised_arm_factual_mar",
            "29 intervention-arm versus 28 control-arm screening profiles",
            p1[:, intervention_arm].mean(axis=1),
            p0[:, control_arm].mean(axis=1),
        ),
    ):
        rows.append(
            {
                "estimand": estimand,
                "target_population": target_population,
                **_summarise_draws(
                    (intervention - control) * WORD_READING_N,
                    prefix="effect_items",
                ),
                **_summarise_draws(
                    intervention * WORD_READING_N,
                    prefix="intervention_mean_items",
                ),
                **_summarise_draws(
                    control * WORD_READING_N,
                    prefix="control_mean_items",
                ),
                "prob_effect_positive": float(np.mean(intervention > control)),
                **common,
            }
        )
    return pd.DataFrame(rows)


def validate_missingness_prior_check(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return schema and traceability errors for the estimand-scale prior check."""

    expected_estimands = {
        "common_profile_all_57",
        "randomised_arm_factual_mar",
    }
    required = {
        "estimand",
        "target_population",
        "effect_items_median",
        "effect_items_lo50",
        "effect_items_hi50",
        "effect_items_lo89",
        "effect_items_hi89",
        "intervention_mean_items_median",
        "intervention_mean_items_lo89",
        "intervention_mean_items_hi89",
        "control_mean_items_median",
        "control_mean_items_lo89",
        "control_mean_items_hi89",
        "prob_effect_positive",
        "prior_predictive_mean_items_median",
        "prior_predictive_mean_items_lo89",
        "prior_predictive_mean_items_hi89",
        "prior_predictive_floor_fraction_median",
        "prior_predictive_ceiling_fraction_median",
        "alpha_anchor_logit",
        "alpha_anchor_items",
        "alpha_sigma",
        "prior_draws",
        "source_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return (f"missing columns: {', '.join(missing)}",)

    errors: list[str] = []
    estimands = frame["estimand"].astype(str)
    if len(frame) != 2 or set(estimands) != expected_estimands or not estimands.is_unique:
        errors.append("prior estimand rows are incomplete or duplicated")
    numeric_columns = sorted(required - {"estimand", "target_population", "source_sha256"})
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        errors.append("prior check contains non-finite values")
    probability = pd.to_numeric(frame["prob_effect_positive"], errors="coerce")
    if not probability.between(0.0, 1.0, inclusive="both").all():
        errors.append("prior direction probability lies outside [0, 1]")
    for column in (
        "prior_predictive_floor_fraction_median",
        "prior_predictive_ceiling_fraction_median",
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.between(0.0, 1.0, inclusive="both").all():
            errors.append(f"{column} lies outside [0, 1]")
    if not pd.to_numeric(frame["alpha_sigma"], errors="coerce").eq(
        SCREENING_ALPHA_SIGMA
    ).all():
        errors.append("screening intercept scale does not match the registered prior")
    if not pd.to_numeric(frame["prior_draws"], errors="coerce").eq(
        MISSINGNESS_PRIOR_DRAWS
    ).all():
        errors.append("prior check does not contain the registered draw count")
    if not frame["source_sha256"].astype(str).eq(RLI_ARCHIVE_CSV_SHA256).all():
        errors.append("prior check is not bound to the registered UKDS file")
    return tuple(errors)


def expected_delta_pairs(
    delta_items: Sequence[float] = DEFAULT_DELTA_ITEMS,
) -> set[tuple[float, float]]:
    """The complete Cartesian delta grid required by the release contract."""

    grid = tuple(float(value) for value in delta_items)
    return {(left, right) for left in grid for right in grid}


def missingness_ppc_summary(trace: Any, data: ScreeningWordReadingData) -> pd.DataFrame:
    """Observed-scale posterior-predictive calibration for the 53-row sub-fit."""

    values = np.asarray(trace.posterior_predictive["y_post"].values, dtype=float)
    if values.ndim != 3 or values.shape[-1] != data.n_obs:
        raise ValueError("missingness posterior predictive does not align with 53 rows")
    draws = values.reshape((-1, values.shape[-1]))
    observed = data.post_counts["W"].astype(float)
    predictive_mean = draws.mean(axis=0)
    lo50, hi50 = np.quantile(draws, [0.25, 0.75], axis=0)
    lo89, hi89 = np.quantile(draws, [0.055, 0.945], axis=0)
    rows: list[dict[str, Any]] = []
    for label, mask in (
        ("all", np.ones(data.n_obs, dtype=bool)),
        ("intervention", data.G == 1),
        ("control", data.G == 0),
    ):
        rows.append(
            {
                "arm": label,
                "n": int(mask.sum()),
                "observed_mean_items": float(observed[mask].mean()),
                "posterior_predictive_mean_items": float(predictive_mean[mask].mean()),
                "mean_absolute_prediction_error_items": float(
                    np.mean(np.abs(predictive_mean[mask] - observed[mask]))
                ),
                "coverage_50": float(
                    np.mean(
                        (observed[mask] >= lo50[mask])
                        & (observed[mask] <= hi50[mask])
                    )
                ),
                "coverage_89": float(
                    np.mean(
                        (observed[mask] >= lo89[mask])
                        & (observed[mask] <= hi89[mask])
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_missingness_summary(
    frame: pd.DataFrame,
    *,
    trace_path: str | Path | None = None,
    require_converged: bool = True,
) -> tuple[str, ...]:
    """Return fail-closed schema, count, grid and trace-binding errors."""

    required = {
        "scenario",
        "scenario_class",
        "estimand_class",
        "target_population",
        "delta_intervention_items",
        "delta_control_items",
        "effect_items_median",
        "effect_items_lo50",
        "effect_items_hi50",
        "effect_items_lo89",
        "effect_items_hi89",
        "intervention_mean_items_median",
        "intervention_mean_items_lo50",
        "intervention_mean_items_hi50",
        "intervention_mean_items_lo89",
        "intervention_mean_items_hi89",
        "control_mean_items_median",
        "control_mean_items_lo50",
        "control_mean_items_hi50",
        "control_mean_items_lo89",
        "control_mean_items_hi89",
        "prob_effect_positive",
        "clipped_intervention_fraction",
        "clipped_control_fraction",
        "randomised_n",
        "randomised_intervention_n",
        "randomised_control_n",
        "observed_intervention_n",
        "observed_control_n",
        "missing_intervention_n",
        "missing_control_n",
        "lost_to_follow_up_n",
        "within_archive_w_missing_n",
        "n_trials",
        "source_sha256",
        "converged",
        "trace_file",
        "trace_sha256",
    }
    errors: list[str] = []
    missing = sorted(required - set(frame.columns))
    if missing:
        return (f"missing columns: {', '.join(missing)}",)
    if len(frame) != 3 + len(DEFAULT_DELTA_ITEMS) ** 2:
        errors.append("unexpected scenario count")
    scenarios = frame["scenario"].astype(str)
    if not scenarios.is_unique:
        errors.append("scenario identifiers are duplicated")
    required_named = set(MISSINGNESS_SCENARIOS[:3])
    if not required_named.issubset(set(frame["scenario"].astype(str))):
        errors.append("bridge, MAR or jump-to-reference scenario is absent")
    expected_classes = {
        "screening_model_observed_profiles": "bridge",
        "mar_all_57": "missing_at_random",
        "jump_to_reference_intervention_nonstarter": "reference_based",
    }
    for scenario, expected_class in expected_classes.items():
        rows = frame.loc[scenarios == scenario]
        if len(rows) == 1 and str(rows.iloc[0]["scenario_class"]) != expected_class:
            errors.append(f"{scenario} has the wrong scenario class")
    common_rows = frame.loc[
        scenarios.isin(
            {"screening_model_observed_profiles", "mar_all_57"}
        )
    ]
    if not common_rows["estimand_class"].astype(str).eq(
        "common_profile_standardisation"
    ).all():
        errors.append("bridge or all-57 MAR has the wrong estimand class")
    grid = frame.loc[frame["scenario_class"] == "arm_specific_delta_grid"]
    completion_rows = frame.loc[
        (scenarios == "jump_to_reference_intervention_nonstarter")
        | frame["scenario_class"].astype(str).eq("arm_specific_delta_grid")
    ]
    if not completion_rows["estimand_class"].astype(str).eq(
        "randomised_arm_factual_completion"
    ).all():
        errors.append("J2R or delta row has the wrong factual-completion estimand")
    pairs = set(
        zip(
            pd.to_numeric(grid["delta_intervention_items"], errors="coerce"),
            pd.to_numeric(grid["delta_control_items"], errors="coerce"),
            strict=True,
        )
    )
    if pairs != expected_delta_pairs():
        errors.append("arm-specific delta grid is incomplete or duplicated")
    finite_columns = [
        column
        for column in frame.columns
        if column.startswith(
            ("effect_items_", "intervention_mean_items_", "control_mean_items_")
        )
    ]
    finite_values = frame[finite_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(finite_values.to_numpy(dtype=float)).all():
        errors.append("scenario estimates contain non-finite values")
    probabilities = pd.to_numeric(frame["prob_effect_positive"], errors="coerce")
    if not probabilities.between(0.0, 1.0, inclusive="both").all():
        errors.append("direction probabilities lie outside [0, 1]")
    for column in ("clipped_intervention_fraction", "clipped_control_fraction"):
        clipped = pd.to_numeric(frame[column], errors="coerce")
        if not clipped.between(0.0, 1.0, inclusive="both").all():
            errors.append(f"{column} lies outside [0, 1]")
    count_contract = {
        "randomised_n": RANDOMISED_N,
        "randomised_intervention_n": RANDOMISED_INTERVENTION_N,
        "randomised_control_n": RANDOMISED_CONTROL_N,
        "observed_intervention_n": OBSERVED_INTERVENTION_N,
        "observed_control_n": OBSERVED_CONTROL_N,
        "missing_intervention_n": 1,
        "missing_control_n": 3,
        "lost_to_follow_up_n": LOST_TO_FOLLOW_UP_N,
        "within_archive_w_missing_n": WITHIN_ARCHIVE_W_MISSING_N,
        "n_trials": WORD_READING_N,
    }
    for column, expected in count_contract.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.eq(expected).all():
            errors.append(f"{column} does not equal {expected} in every row")
    if not frame["source_sha256"].astype(str).eq(RLI_ARCHIVE_CSV_SHA256).all():
        errors.append("source archive hash is not the registered UKDS file")
    if require_converged and not frame["converged"].map(
        lambda value: str(value).strip().casefold() in {"true", "1", "yes"}
    ).all():
        errors.append("screening-baseline sub-fit failed or was not checked")
    if not frame["trace_file"].astype(str).eq(MISSINGNESS_TRACE_FILENAME).all():
        errors.append("summary does not bind the registered sub-fit trace")
    trace_hashes = set(frame["trace_sha256"].dropna().astype(str))
    if len(trace_hashes) != 1:
        errors.append("summary does not carry one trace hash")
    if trace_path is not None:
        path = Path(trace_path)
        if not path.is_file():
            errors.append("registered sub-fit trace is missing")
        elif trace_hashes != {sha256_file(path)}:
            errors.append("registered sub-fit trace hash does not match the summary")
    delta_zero = grid.loc[
        pd.to_numeric(grid["delta_intervention_items"], errors="coerce").eq(0.0)
        & pd.to_numeric(grid["delta_control_items"], errors="coerce").eq(0.0)
    ]
    if len(delta_zero) != 1:
        errors.append("zero-delta factual-arm MAR completion is absent")
    return tuple(errors)


def run_missingness_subfit(
    ctx: Any,
    archive_path: str | Path,
    *,
    plan: Any,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    """Fit, persist and summarise the screening-baseline sensitivity sub-fit."""

    from language_reading_predictors.statistical_models.artifacts import (
        record_artifact,
        save_table,
    )
    from language_reading_predictors.statistical_models.subfits import (
        refresh_subfit_trace_hash,
    )

    if plan is None:
        raise ValueError("the resolved ITT missingness plan is absent")
    contract = {
        "source_csv_sha256": RLI_ARCHIVE_CSV_SHA256,
        "source_doi": RLI_ARCHIVE_DOI,
        "local_wide_sha256": RLI_LOCAL_WIDE_SHA256,
        "reconciliation_digest": RLI_RECONCILIATION_DIGEST,
        "screening_covariates": SCREENING_COVARIATES,
        "randomised_n": RANDOMISED_N,
        "randomised_intervention_n": RANDOMISED_INTERVENTION_N,
        "randomised_control_n": RANDOMISED_CONTROL_N,
        "observed_intervention_n": OBSERVED_INTERVENTION_N,
        "observed_control_n": OBSERVED_CONTROL_N,
        "lost_to_follow_up_n": LOST_TO_FOLLOW_UP_N,
        "within_archive_w_missing_n": WITHIN_ARCHIVE_W_MISSING_N,
        "word_reading_n": WORD_READING_N,
        "delta_items": DEFAULT_DELTA_ITEMS,
        "scenarios": MISSINGNESS_SCENARIOS,
        "common_estimand_class": "common_profile_standardisation",
        "completion_estimand_class": "randomised_arm_factual_completion",
        "intercept_prior_anchor": "mean_all_57_screening_word_reading_logit",
        "intercept_prior_sigma": SCREENING_ALPHA_SIGMA,
        "prior_predictive_draws": MISSINGNESS_PRIOR_DRAWS,
        "trace_filename": MISSINGNESS_TRACE_FILENAME,
        "summary_filename": MISSINGNESS_SUMMARY_FILENAME,
        "ppc_filename": MISSINGNESS_PPC_FILENAME,
        "prior_check_filename": MISSINGNESS_PRIOR_FILENAME,
        "provenance_filename": MISSINGNESS_PROVENANCE_FILENAME,
    }
    mismatched = [
        name
        for name, expected in contract.items()
        if getattr(plan, name, None) != expected
    ]
    if mismatched:
        raise ValueError(
            "resolved ITT missingness plan disagrees with the executable contract: "
            + ", ".join(mismatched)
        )
    data = load_randomised_w_archive(
        archive_path,
        expected_sha256=plan.source_csv_sha256,
        expected_local_sha256=plan.local_wide_sha256,
        expected_reconciliation_digest=plan.reconciliation_digest,
    )
    built = build_screening_w_model(data)
    prior_samples = sample_missingness_prior_predictive(
        built,
        draws=MISSINGNESS_PRIOR_DRAWS,
        random_seed=ctx.sampling.random_seed,
    )
    result = runner(
        ctx,
        built,
        label=MISSINGNESS_SUBFIT_LABEL,
        role="sensitivity",
        posterior_predictive=["y_post"],
        trace_filename=plan.trace_filename,
    )
    trace_path = Path(ctx.output_dir) / MISSINGNESS_TRACE_FILENAME
    attach_missingness_prior_groups(result.trace, prior_samples)
    # ``run_subfit`` persists before this family-owned prior draw exists. Rewrite
    # the same registered trace so its hash binds posterior, posterior-predictive,
    # prior and prior-predictive groups as one auditable object.
    result.trace.to_netcdf(trace_path)
    trace_sha256 = refresh_subfit_trace_hash(
        ctx,
        label=MISSINGNESS_SUBFIT_LABEL,
        trace_filename=MISSINGNESS_TRACE_FILENAME,
    )
    summary = summarise_missingness_sensitivity(
        result.trace,
        data,
        delta_items=plan.delta_items,
    )
    summary["converged"] = result.converged
    summary["max_rhat"] = result.convergence.get("max_rhat")
    summary["min_ess"] = result.convergence.get("min_ess")
    summary["min_bfmi"] = result.convergence.get("min_bfmi")
    summary["n_divergences"] = result.convergence.get("n_divergences")
    summary["trace_file"] = MISSINGNESS_TRACE_FILENAME
    summary["trace_sha256"] = trace_sha256
    errors = validate_missingness_summary(
        summary,
        trace_path=trace_path,
        require_converged=False,
    )
    if errors:
        raise RuntimeError("invalid ITT missingness bundle: " + "; ".join(errors))
    save_table(
        ctx,
        "itt_missingness_sensitivity",
        summary,
        filename=MISSINGNESS_SUMMARY_FILENAME,
    )
    ppc = missingness_ppc_summary(result.trace, data)
    save_table(
        ctx,
        "itt_missingness_ppc",
        ppc,
        filename=MISSINGNESS_PPC_FILENAME,
    )
    prior_check = missingness_prior_check(prior_samples, data)
    prior_errors = validate_missingness_prior_check(prior_check)
    if prior_errors:
        raise RuntimeError(
            "invalid ITT missingness prior check: " + "; ".join(prior_errors)
        )
    save_table(
        ctx,
        "itt_missingness_prior_check",
        prior_check,
        filename=MISSINGNESS_PRIOR_FILENAME,
    )
    summary_sha256 = sha256_file(Path(ctx.output_dir) / MISSINGNESS_SUMMARY_FILENAME)
    ppc_sha256 = sha256_file(Path(ctx.output_dir) / MISSINGNESS_PPC_FILENAME)
    prior_sha256 = sha256_file(Path(ctx.output_dir) / MISSINGNESS_PRIOR_FILENAME)

    provenance = {
        "model_id": str(ctx.spec.model_id),
        "role": "mandatory_secondary_missing_data_sensitivity",
        "source": {
            "title": "Reading and language intervention for children with Down syndrome: Experimental data",
            "doi": RLI_ARCHIVE_DOI,
            "landing_url": RLI_ARCHIVE_URL,
            "zip_url": RLI_ARCHIVE_ZIP_URL,
            "zip_sha256": RLI_ARCHIVE_ZIP_SHA256,
            "csv_name": RLI_ARCHIVE_CSV_NAME,
            "csv_sha256": data.data_sha256,
            "local_wide_sha256": data.local_wide_sha256,
            "reconciled_included_n": data.reconciled_included_n,
            "reconciliation_digest": data.reconciliation_digest,
            "rights_owner": "Down Syndrome Education International",
            "licence_note": (
                "The ReShare item is open access but its item-level licence field is blank. "
                "The source file is supplied at run time and is not redistributed in this repository."
            ),
        },
        "analysis": {
            "observed_outcome_n": data.n_obs,
            "target_profile_n": RANDOMISED_N,
            "randomised_by_arm": {"intervention": 29, "control": 28},
            "observed_outcome_by_arm": {"intervention": 28, "control": 25},
            "lost_to_follow_up_n": LOST_TO_FOLLOW_UP_N,
            "within_archive_word_reading_missing_n": WITHIN_ARCHIVE_W_MISSING_N,
            "screening_covariates": list(data.covariate_names),
            "covariate_scalers": data.covariate_scalers,
            "coefficient_priors": {
                "alpha": (
                    "Normal(mean all-57 pre-randomisation screening-W logit, 1.0)"
                ),
                "beta_screening_age": "Normal(0, 0.3)",
                "beta_screening_word": "Normal(0, 1.0)",
            },
            "intercept_anchor_logit": data.covariate_scalers[
                "screening_word_reading"
            ]["mean"],
            "intercept_anchor_uses_t2_outcome": False,
            "delta_items_grid": list(plan.delta_items),
            "delta_grid_status": "diagnostic_not_probabilistically_calibrated",
            "pattern_completion": (
                "Bridge/MAR average both arm surfaces over common profiles. "
                "J2R/delta instead complete the factual randomised arms using "
                "29 intervention and 28 control profiles, modifying only the "
                "unavailable outcome(s) in that arm; counterfactual response "
                "status is not modelled."
            ),
        },
        "trace": {
            "file": MISSINGNESS_TRACE_FILENAME,
            "sha256": trace_sha256,
            "converged": result.converged,
            **result.convergence,
        },
        "outputs": {
            "summary_file": MISSINGNESS_SUMMARY_FILENAME,
            "summary_sha256": summary_sha256,
            "ppc_file": MISSINGNESS_PPC_FILENAME,
            "ppc_sha256": ppc_sha256,
            "prior_check_file": MISSINGNESS_PRIOR_FILENAME,
            "prior_check_sha256": prior_sha256,
        },
    }
    provenance_path = Path(ctx.output_dir) / MISSINGNESS_PROVENANCE_FILENAME
    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    record_artifact(
        ctx,
        "itt_missingness_provenance",
        filename=MISSINGNESS_PROVENANCE_FILENAME,
        kind="json",
    )
    return {
        "status": "complete",
        "source_sha256": data.data_sha256,
        "trace_file": MISSINGNESS_TRACE_FILENAME,
        "trace_sha256": trace_sha256,
        "summary_file": MISSINGNESS_SUMMARY_FILENAME,
        "summary_sha256": summary_sha256,
        "ppc_file": MISSINGNESS_PPC_FILENAME,
        "ppc_sha256": ppc_sha256,
        "prior_check_file": MISSINGNESS_PRIOR_FILENAME,
        "prior_check_sha256": prior_sha256,
        "provenance_file": MISSINGNESS_PROVENANCE_FILENAME,
        "observed_outcome_n": data.n_obs,
        "target_profile_n": RANDOMISED_N,
    }


def missingness_source_path(option: str | None) -> Path | None:
    """Resolve the explicitly supplied source path without hidden downloading."""

    if option is None:
        return None
    path = Path(os.path.expanduser(option)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--rli-randomised-archive does not exist: {path}")
    return path
