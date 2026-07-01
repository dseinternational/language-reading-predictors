# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fit the first Bayesian historical growth model.

Model ``rlmhg01`` estimates BAS word-reading growth over the first
three waves of the Byrne, MacDonald, and Buckley reading-language-memory study.
It deliberately uses the same per-measure complete-case subset as
``scripts/replicate_reading_language_memory.py`` reproduces for the paper's
Table 2, and writes that observed complete-case baseline beside the posterior
summaries.

The model is descriptive/natural-history evidence, not an intervention-effect
model:

    y_it ~ BetaBinomial(n=87, p_it, kappa)
    logit(p_it) = eta[group_i, wave_t] + subject_offset_i

where ``eta`` gives the population group-by-wave expected score and
``subject_offset`` captures stable between-child reading level.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import freeze_support
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTENSOR_COMPILEDIR = REPO_ROOT / "tmp" / "pytensor"
NUMBA_CACHE_DIR = REPO_ROOT / "tmp" / "numba"

# PyTensor reads compilation flags when PyMC is imported.
os.environ.setdefault(
    "PYTENSOR_FLAGS",
    f"base_compiledir={PYTENSOR_COMPILEDIR.as_posix()}",
)
os.environ.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR.as_posix())

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402

from replicate_reading_language_memory import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR as AUDIT_OUTPUT_DIR,
    GROUP_LABELS,
    MEASURE_LABELS,
    PAPER_WAVES,
    complete_case_summary,
    load_data,
)


MODEL_ID = "rlmhg01"
MEASURE = "basread"
BAS_WORD_READING_MAX = 87
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "reading_language_memory" / "models" / MODEL_ID

CONFIG_DEFAULTS: dict[str, dict[str, int | float]] = {
    "dev": {"draws": 500, "tune": 500, "chains": 2, "target_accept": 0.9},
    "test": {"draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.9},
    "reporting": {"draws": 2000, "tune": 2000, "chains": 4, "target_accept": 0.95},
}
KEY_DIAGNOSTIC_VARS: tuple[str, ...] = (
    "eta_group_wave",
    "sigma_subject",
    "kappa",
    "growth_1_2_items",
    "growth_2_3_items",
    "growth_1_3_items",
    "growth_1_3_average_minus_down_syndrome",
    "growth_1_3_reading_matched_minus_down_syndrome",
    "growth_1_3_average_minus_reading_matched",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the RLMHG01 historical BAS word-reading growth model.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Long-format historical CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=AUDIT_OUTPUT_DIR,
        help=f"Replication audit output dir (default: {AUDIT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Model output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--config",
        choices=sorted(CONFIG_DEFAULTS),
        default="dev",
        help="Sampling preset.",
    )
    parser.add_argument("--draws", type=int, default=None, help="Override draws.")
    parser.add_argument("--tune", type=int, default=None, help="Override tuning steps.")
    parser.add_argument("--chains", type=int, default=None, help="Override chains.")
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Sampler cores. Defaults to 1 for reliable Windows CLI runs.",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=None,
        help="Override NUTS target_accept.",
    )
    parser.add_argument("--seed", type=int, default=20260701, help="Random seed.")
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="Do not write trace.nc. Useful only for temporary smoke tests.",
    )
    return parser.parse_args()


def sampling_options(args: argparse.Namespace) -> dict[str, int | float]:
    defaults = CONFIG_DEFAULTS[args.config]
    return {
        "draws": int(args.draws or defaults["draws"]),
        "tune": int(args.tune or defaults["tune"]),
        "chains": int(args.chains or defaults["chains"]),
        "target_accept": float(args.target_accept or defaults["target_accept"]),
    }


def prepare_complete_case_data(df: pd.DataFrame) -> pd.DataFrame:
    paper_df = df[df["time"].isin(PAPER_WAVES)].copy()
    wide = paper_df.pivot(index="subject_id", columns="time", values=MEASURE)
    complete_subjects = wide.dropna(subset=list(PAPER_WAVES)).index
    model_df = paper_df[paper_df["subject_id"].isin(complete_subjects)].copy()
    model_df = model_df.dropna(subset=[MEASURE, "readgrp", "time"])
    model_df["wave_zero"] = model_df["time"] - min(PAPER_WAVES)

    counts = model_df.groupby("subject_id")[MEASURE].size()
    incomplete = counts[counts != len(PAPER_WAVES)]
    if not incomplete.empty:
        raise ValueError(
            "Complete-case model input has subjects without all three waves: "
            f"{incomplete.to_dict()}"
        )
    if int(model_df[MEASURE].max()) > BAS_WORD_READING_MAX:
        raise ValueError(
            f"Observed {MEASURE} exceeds configured maximum {BAS_WORD_READING_MAX}."
        )
    return model_df.sort_values(["readgrp", "subject_id", "time"]).reset_index(drop=True)


def read_or_make_audit_summary(df: pd.DataFrame, audit_dir: Path) -> pd.DataFrame:
    path = audit_dir / "table2_complete_case_summary.csv"
    if path.exists():
        audit = pd.read_csv(path)
    else:
        audit = complete_case_summary(df, [MEASURE])

    audit = audit[audit["measure"] == MEASURE].copy()
    if audit.empty:
        raise ValueError(f"No {MEASURE!r} rows found in complete-case audit summary.")
    return audit


def build_model(
    model_df: pd.DataFrame,
) -> tuple[pm.Model, dict[str, Any]]:
    group_codes = [int(code) for code in sorted(model_df["readgrp"].unique())]
    group_labels = [GROUP_LABELS[int(code)] for code in group_codes]
    group_index = {code: idx for idx, code in enumerate(group_codes)}
    wave_codes = list(PAPER_WAVES)
    wave_index = {wave: idx for idx, wave in enumerate(wave_codes)}
    subject_ids = model_df["subject_id"].drop_duplicates().tolist()
    subject_index = {subject_id: idx for idx, subject_id in enumerate(subject_ids)}

    group_idx = model_df["readgrp"].map(group_index).to_numpy(dtype=int)
    wave_idx = model_df["time"].map(wave_index).to_numpy(dtype=int)
    subject_idx = model_df["subject_id"].map(subject_index).to_numpy(dtype=int)
    observed = model_df[MEASURE].to_numpy(dtype=int)
    subject_group = (
        model_df.drop_duplicates("subject_id")
        .set_index("subject_id")
        .loc[subject_ids, "readgrp"]
        .map(group_index)
        .to_numpy(dtype=int)
    )

    coords = {
        "group": group_labels,
        "wave": wave_codes,
        "subject": subject_ids,
        "obs": np.arange(len(model_df)),
    }

    with pm.Model(coords=coords) as model:
        eta_group_wave = pm.Normal(
            "eta_group_wave",
            mu=0.0,
            sigma=1.5,
            dims=("group", "wave"),
        )
        sigma_subject = pm.HalfNormal("sigma_subject", sigma=1.0)
        z_subject = pm.Normal("z_subject", mu=0.0, sigma=1.0, dims="subject")
        z_group_mean = pm.math.stack(
            [
                z_subject[subject_group == group_i].mean()
                for group_i in range(len(group_codes))
            ],
        )
        subject_offset = pm.Deterministic(
            "subject_offset",
            (z_subject - z_group_mean[subject_group]) * sigma_subject,
            dims="subject",
        )
        kappa = pm.HalfNormal("kappa", sigma=50.0)

        eta_obs = eta_group_wave[group_idx, wave_idx] + subject_offset[subject_idx]
        p_obs = pm.math.sigmoid(eta_obs)
        alpha = p_obs * kappa
        beta = (1.0 - p_obs) * kappa

        pm.BetaBinomial(
            "score",
            n=BAS_WORD_READING_MAX,
            alpha=alpha,
            beta=beta,
            observed=observed,
            dims="obs",
        )
        pm.Deterministic(
            "fitted_mean_items_obs",
            BAS_WORD_READING_MAX * p_obs,
            dims="obs",
        )

        mean_items = pm.Deterministic(
            "mean_items",
            BAS_WORD_READING_MAX * pm.math.sigmoid(eta_group_wave),
            dims=("group", "wave"),
        )
        pm.Deterministic(
            "growth_1_2_items",
            mean_items[:, 1] - mean_items[:, 0],
            dims="group",
        )
        pm.Deterministic(
            "growth_2_3_items",
            mean_items[:, 2] - mean_items[:, 1],
            dims="group",
        )
        growth_1_3 = pm.Deterministic(
            "growth_1_3_items",
            mean_items[:, 2] - mean_items[:, 0],
            dims="group",
        )

        pm.Deterministic(
            "growth_1_3_average_minus_down_syndrome",
            growth_1_3[1] - growth_1_3[0],
        )
        pm.Deterministic(
            "growth_1_3_reading_matched_minus_down_syndrome",
            growth_1_3[2] - growth_1_3[0],
        )
        pm.Deterministic(
            "growth_1_3_average_minus_reading_matched",
            growth_1_3[1] - growth_1_3[2],
        )

    meta = {
        "group_codes": group_codes,
        "group_labels": group_labels,
        "wave_codes": wave_codes,
        "subject_ids": subject_ids,
        "n_obs": len(model_df),
        "n_subjects": len(subject_ids),
    }
    return model, meta


def summarize_vector(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "q2_5": float(np.quantile(values, 0.025)),
        "q50": float(np.quantile(values, 0.5)),
        "q97_5": float(np.quantile(values, 0.975)),
        "p_gt_0": float(np.mean(values > 0)),
    }


def posterior_samples(idata: az.InferenceData, variable: str) -> Any:
    return idata.posterior[variable].stack(sample=("chain", "draw"))


def fitted_cell_samples(
    idata: az.InferenceData,
    model_df: pd.DataFrame,
    group: str,
    wave: int,
) -> np.ndarray:
    fitted = posterior_samples(idata, "fitted_mean_items_obs")
    obs_idx = model_df.index[
        (model_df["readgrp_label"] == group) & (model_df["time"] == wave)
    ].to_numpy()
    if len(obs_idx) == 0:
        raise ValueError(f"No observations for group={group!r}, wave={wave!r}.")
    return fitted.isel(obs=obs_idx).mean(dim="obs").values


def make_cell_summary(
    idata: az.InferenceData,
    audit: pd.DataFrame,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    population_mean = posterior_samples(idata, "mean_items")
    rows: list[dict[str, Any]] = []
    for group in population_mean.coords["group"].values:
        audit_row = audit[audit["readgrp_label"] == group]
        if len(audit_row) != 1:
            raise ValueError(f"Expected one audit row for group {group!r}.")
        audit_row = audit_row.iloc[0]
        for wave in population_mean.coords["wave"].values:
            wave = int(wave)
            values = fitted_cell_samples(idata, model_df, str(group), wave)
            summary = summarize_vector(values)
            pop_values = population_mean.sel(group=group, wave=wave).values
            pop_summary = summarize_vector(pop_values)
            observed_mean = float(audit_row[f"time_{int(wave)}_mean"])
            observed_sd = float(audit_row[f"time_{int(wave)}_sd"])
            rows.append(
                {
                    "measure": MEASURE,
                    "measure_label": MEASURE_LABELS[MEASURE],
                    "readgrp_label": group,
                    "time": int(wave),
                    "n_complete": int(audit_row["n_complete"]),
                    "observed_complete_case_mean": observed_mean,
                    "observed_complete_case_sd": observed_sd,
                    "posterior_mean_minus_observed_mean": summary["mean"] - observed_mean,
                    "posterior_population_mean": pop_summary["mean"],
                    "posterior_population_q2_5": pop_summary["q2_5"],
                    "posterior_population_q97_5": pop_summary["q97_5"],
                    **{f"posterior_{k}": v for k, v in summary.items()},
                }
            )
    return pd.DataFrame(rows)


def make_growth_summary(idata: az.InferenceData, model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start_wave, end_wave, label in [
        (1, 2, "Wave 1 to wave 2"),
        (2, 3, "Wave 2 to wave 3"),
        (1, 3, "Wave 1 to wave 3"),
    ]:
        variable = f"growth_{start_wave}_{end_wave}_items"
        for group in GROUP_LABELS.values():
            start = fitted_cell_samples(idata, model_df, group, start_wave)
            end = fitted_cell_samples(idata, model_df, group, end_wave)
            summary = summarize_vector(end - start)
            rows.append(
                {
                    "quantity": variable,
                    "label": label,
                    "readgrp_label": group,
                    **summary,
                }
            )

    total_growth = {
        group: (
            fitted_cell_samples(idata, model_df, group, 3)
            - fitted_cell_samples(idata, model_df, group, 1)
        )
        for group in GROUP_LABELS.values()
    }
    for variable, label, values in [
        (
            "growth_1_3_average_minus_down_syndrome",
            "Total growth: average readers minus Down syndrome",
            total_growth["Average readers"] - total_growth["Down syndrome"],
        ),
        (
            "growth_1_3_reading_matched_minus_down_syndrome",
            "Total growth: reading-matched minus Down syndrome",
            total_growth["Reading-matched"] - total_growth["Down syndrome"],
        ),
        (
            "growth_1_3_average_minus_reading_matched",
            "Total growth: average readers minus reading-matched",
            total_growth["Average readers"] - total_growth["Reading-matched"],
        ),
    ]:
        rows.append(
            {
                "quantity": variable,
                "label": label,
                "readgrp_label": "",
                **summarize_vector(values),
            }
        )
    return pd.DataFrame(rows)


def diagnostic_stats(summary: pd.DataFrame, prefix: str) -> list[dict[str, float | str]]:
    r_hat = summary.get("r_hat", pd.Series(dtype=float)).dropna()
    ess_bulk = summary.get("ess_bulk", pd.Series(dtype=float)).dropna()
    ess_tail = summary.get("ess_tail", pd.Series(dtype=float)).dropna()
    return [
        {
            "metric": f"{prefix}_max_r_hat",
            "value": float(r_hat.max()) if not r_hat.empty else math.nan,
        },
        {
            "metric": f"{prefix}_min_ess_bulk",
            "value": float(ess_bulk.min()) if not ess_bulk.empty else math.nan,
        },
        {
            "metric": f"{prefix}_min_ess_tail",
            "value": float(ess_tail.min()) if not ess_tail.empty else math.nan,
        },
    ]


def make_diagnostics(
    idata: az.InferenceData,
    parameter_summary: pd.DataFrame,
) -> pd.DataFrame:
    sample_stats = idata.sample_stats
    divergences = int(sample_stats["diverging"].sum().item())
    key_summary = az.summary(idata, var_names=list(KEY_DIAGNOSTIC_VARS))
    rows: list[dict[str, float | str]] = [
        {"metric": "divergences", "value": divergences},
        *diagnostic_stats(key_summary, "key_estimands"),
        *diagnostic_stats(parameter_summary, "all_posterior"),
    ]
    return pd.DataFrame(rows)


def write_model_config(
    output_dir: Path,
    args: argparse.Namespace,
    sampling: dict[str, int | float],
    meta: dict[str, Any],
) -> None:
    config = {
        "model_id": MODEL_ID,
        "title": "RLMHG01: historical BAS word-reading growth, waves 1-3",
        "data": str(args.data),
        "audit_dir": str(args.audit_dir),
        "measure": MEASURE,
        "measure_label": MEASURE_LABELS[MEASURE],
        "n_trials": BAS_WORD_READING_MAX,
        "waves": list(PAPER_WAVES),
        "likelihood": "BetaBinomial on bounded BAS word-reading count",
        "estimand": (
            "Descriptive group-by-wave expected BAS word-reading score and "
            "within-group growth over the first two annual intervals."
        ),
        "causal_interpretation": "None; historical natural-history growth model.",
        "config": args.config,
        "sampling": sampling,
        "random_seed": args.seed,
        "n_obs": meta["n_obs"],
        "n_subjects": meta["n_subjects"],
        "groups": dict(zip(meta["group_codes"], meta["group_labels"], strict=True)),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def write_trace_or_warning(
    idata: az.InferenceData,
    output_dir: Path,
) -> None:
    path = output_dir / "trace.nc"
    warning_path = output_dir / "trace_write_warning.json"
    try:
        idata.to_netcdf(path)
    except Exception as exc:  # pragma: no cover - environment dependent
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        warning = {
            "trace_path": str(path),
            "warning": (
                "Posterior sampling completed, but trace.nc could not be written. "
                "Install h5py/netCDF support in the fitting environment to persist "
                "the full ArviZ trace."
            ),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        warning_path.write_text(json.dumps(warning, indent=2), encoding="utf-8")


def fit(args: argparse.Namespace) -> None:
    sampling = sampling_options(args)
    df = load_data(args.data)
    model_df = prepare_complete_case_data(df)
    audit = read_or_make_audit_summary(df, args.audit_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_df.to_csv(args.output_dir / "model_input.csv", index=False)
    audit.to_csv(args.output_dir / "observed_complete_case_baseline.csv", index=False)

    model, meta = build_model(model_df)
    write_model_config(args.output_dir, args, sampling, meta)

    with model:
        idata = pm.sample(
            draws=int(sampling["draws"]),
            tune=int(sampling["tune"]),
            chains=int(sampling["chains"]),
            cores=args.cores,
            target_accept=float(sampling["target_accept"]),
            random_seed=args.seed,
            return_inferencedata=True,
        )

    parameter_summary = az.summary(idata)
    parameter_summary.to_csv(args.output_dir / "posterior_parameter_summary.csv")
    make_cell_summary(idata, audit, model_df).to_csv(
        args.output_dir / "posterior_cell_summary.csv",
        index=False,
        float_format="%.6g",
    )
    make_growth_summary(idata, model_df).to_csv(
        args.output_dir / "posterior_growth_summary.csv",
        index=False,
        float_format="%.6g",
    )
    make_diagnostics(idata, parameter_summary).to_csv(
        args.output_dir / "diagnostics_summary.csv",
        index=False,
        float_format="%.6g",
    )
    if not args.skip_trace:
        write_trace_or_warning(idata, args.output_dir)

    print(f"{MODEL_ID}: wrote outputs to {args.output_dir}")
    print(f"Rows: {meta['n_obs']}; subjects: {meta['n_subjects']}")


def main() -> None:
    freeze_support()
    fit(parse_args())


if __name__ == "__main__":
    main()
